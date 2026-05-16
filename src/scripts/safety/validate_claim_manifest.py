"""validate_claim_manifest.py — schema validator for claim_manifest.yaml.

Per TEST_SAFETY_FEEDBACK_DESIGN.md §2 + §2.5 (v4):

  Reads a claim_manifest.yaml (stdlib yaml.safe_load), checks structural
  conformance + cross-field invariants derived from mode_dispatch.py.

  NO regex on prose. NO LLM judgment. Mechanical checks only.

Exit codes:
  0  PASS — manifest is schema-valid
  1  FAIL — at least one ValidationError raised
  2  usage error / file not found

Usage:
  python3 validate_claim_manifest.py --manifest <path> [--repo-root <path>]

Author: T31.4 (P0h.0 implementation).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "validate_claim_manifest: PyYAML not installed. `pip install pyyaml`.\n"
    )
    sys.exit(2)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mode_dispatch import (  # noqa: E402
    CLAIM_MANIFEST_VERSION,
    SCHEMA_VERSION,
    SELF_CHALLENGE_MIN_PASSED,
    ArtifactRole,
    EvidenceType,
    Mode,
    Outcome,
    required_artifacts,
    required_evidence,
    required_validation_level,
)


@dataclass
class ValidationError(Exception):
    """One specific schema violation; collected by validate_manifest."""

    code: str
    description: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.description}"


@dataclass
class ValidationResult:
    errors: List[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        if self.ok:
            return "validate_claim_manifest: PASS\n"
        out = ["validate_claim_manifest: FAIL"]
        for err in self.errors:
            out.append(f"  - {err}")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------


def _require(data: dict, key: str, errors: List[ValidationError]) -> Any:
    if key not in data:
        errors.append(ValidationError("schema-field-missing", f"top-level key '{key}' missing"))
        return None
    return data[key]


def _is_isoformat(s: str) -> bool:
    """Cheap ISO8601 sanity (YYYY-MM-DDThh:mm:ss... or YYYY-MM-DD)."""
    import re
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}.*)?$", s))


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------


def _check_versions(data: dict, errors: List[ValidationError]) -> None:
    v = data.get("claim_manifest_version")
    if v != CLAIM_MANIFEST_VERSION:
        errors.append(ValidationError(
            "claim-manifest-version-mismatch",
            f"claim_manifest_version expected {CLAIM_MANIFEST_VERSION!r}, got {v!r}",
        ))
    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(ValidationError(
            "schema-version-mismatch",
            f"schema_version expected {SCHEMA_VERSION!r}, got {sv!r}",
        ))


def _check_mode(data: dict, errors: List[ValidationError]) -> Mode | None:
    raw = _require(data, "mode", errors)
    if raw is None:
        return None
    try:
        return Mode(raw)
    except ValueError:
        errors.append(ValidationError(
            "mode-not-canonical",
            f"mode={raw!r} not in {[m.value for m in Mode]}",
        ))
        return None


def _check_outcome(data: dict, errors: List[ValidationError]) -> Outcome | None:
    raw = _require(data, "outcome", errors)
    if raw is None:
        return None
    try:
        return Outcome(raw)
    except ValueError:
        errors.append(ValidationError(
            "outcome-canonical",
            f"outcome={raw!r} not canonical; allowed: {[o.value for o in Outcome]}; "
            f"prose variants (A-with-note, A-w-note, a_with_note) FORBIDDEN — "
            f"use the Python enum form (A_WITH_NOTE etc.)",
        ))
        return None


def _check_validation_level(data: dict, mode: Mode | None, errors: List[ValidationError]) -> None:
    raw = data.get("validation_level")
    if raw is None:
        errors.append(ValidationError("validation-level-missing", "validation_level required"))
        return
    if not isinstance(raw, int):
        errors.append(ValidationError(
            "validation-level-type",
            f"validation_level must be int (L1..L5 as 1..5); got {type(raw).__name__}",
        ))
        return
    if not 1 <= raw <= 5:
        errors.append(ValidationError(
            "validation-level-range",
            f"validation_level must be 1..5; got {raw}",
        ))
        return
    if mode is not None:
        min_lvl = required_validation_level(mode)
        if raw < min_lvl:
            errors.append(ValidationError(
                "validation-level-too-low",
                f"mode={mode.value} requires validation_level ≥ {min_lvl}; got {raw}",
            ))


def _collect_evidence_types(data: dict, errors: List[ValidationError]) -> Set[EvidenceType]:
    """Walk evidence[] and collect the EvidenceType set actually present."""
    present: Set[EvidenceType] = set()
    raw_list = data.get("evidence", [])
    if not isinstance(raw_list, list):
        errors.append(ValidationError("evidence-type-list", "evidence must be a list"))
        return present
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict) or "type" not in item:
            errors.append(ValidationError(
                "evidence-item-malformed", f"evidence[{i}] missing 'type'"))
            continue
        try:
            et = EvidenceType(item["type"])
        except ValueError:
            errors.append(ValidationError(
                "evidence-type-invalid",
                f"evidence[{i}].type={item['type']!r} not in "
                f"{[t.value for t in EvidenceType]}",
            ))
            continue
        present.add(et)
        # Per-type sanity:
        if et is EvidenceType.NUMERIC_METRIC:
            v = item.get("value")
            if v is None:
                errors.append(ValidationError(
                    "evidence-numeric-missing",
                    f"evidence[{i}] numeric_metric.value is null — "
                    "outcome with numeric_metric required can't have null value",
                ))
    return present


def _check_evidence_required(outcome: Outcome | None,
                             present: Set[EvidenceType],
                             errors: List[ValidationError]) -> None:
    if outcome is None:
        return
    required = required_evidence(outcome)
    missing = required - present
    if missing:
        errors.append(ValidationError(
            "evidence-required-missing",
            f"outcome={outcome.value} requires evidence types "
            f"{sorted(t.value for t in required)}; missing: "
            f"{sorted(t.value for t in missing)}",
        ))


def _collect_artifact_roles(data: dict, errors: List[ValidationError]) -> Set[ArtifactRole]:
    present: Set[ArtifactRole] = set()
    raw_list = data.get("artifacts", [])
    if not isinstance(raw_list, list):
        errors.append(ValidationError("artifacts-type-list", "artifacts must be a list"))
        return present
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict) or "role" not in item:
            errors.append(ValidationError(
                "artifact-role-missing", f"artifacts[{i}] missing 'role'"))
            continue
        try:
            role = ArtifactRole(item["role"])
        except ValueError:
            errors.append(ValidationError(
                "artifact-role-invalid",
                f"artifacts[{i}].role={item['role']!r} not in "
                f"{[r.value for r in ArtifactRole]}",
            ))
            continue
        present.add(role)
    return present


def _check_mode_artifact_combo(mode: Mode | None,
                               present: Set[ArtifactRole],
                               errors: List[ValidationError]) -> None:
    if mode is None:
        return
    required = required_artifacts(mode)
    missing = required - present
    if missing:
        errors.append(ValidationError(
            "mode-artifact-mismatch",
            f"mode={mode.value} requires artifact roles "
            f"{sorted(r.value for r in required)}; missing: "
            f"{sorted(r.value for r in missing)}",
        ))
    # Cross-mode forbiddens — e.g. issue_only must NOT have shim_module
    forbidden_by_mode: Dict[Mode, Set[ArtifactRole]] = {
        Mode.ISSUE_ONLY: {ArtifactRole.SHIM_MODULE, ArtifactRole.PR_MATERIAL},
    }
    forbidden = forbidden_by_mode.get(mode, set()) & present
    if forbidden:
        errors.append(ValidationError(
            "mode-artifact-mismatch",
            f"mode={mode.value} forbids artifact roles "
            f"{sorted(r.value for r in forbidden)}",
        ))


def _check_self_challenge(data: dict, errors: List[ValidationError]) -> None:
    sc = data.get("self_challenge")
    if sc is None:
        errors.append(ValidationError(
            "self-challenge-required",
            "self_challenge block required; agent must run /porting-self-challenge "
            "and merge result here before claim emission",
        ))
        return
    if not isinstance(sc, dict):
        errors.append(ValidationError(
            "self-challenge-type", "self_challenge must be a dict"))
        return
    passed = sc.get("patterns_passed")
    if not isinstance(passed, list):
        errors.append(ValidationError(
            "self-challenge-patterns-list",
            "self_challenge.patterns_passed must be a list",
        ))
        return
    if len(passed) < SELF_CHALLENGE_MIN_PASSED:
        errors.append(ValidationError(
            "self-challenge-too-few-passed",
            f"self_challenge.patterns_passed has {len(passed)} entries; "
            f"need ≥ {SELF_CHALLENGE_MIN_PASSED}",
        ))


def _check_repo_root_anchor(data: dict,
                            repo_root: Path | None,
                            errors: List[ValidationError]) -> None:
    """Verify the anchor file actually exists in repo_root.

    repo_root_anchor is a relative path (from repo root) to a file that
    must exist — used by scanner to confirm the repo_root path passed in
    really points to OUR repo.
    """
    anchor = data.get("repo_root_anchor")
    if not anchor:
        errors.append(ValidationError(
            "repo-root-anchor-missing",
            "repo_root_anchor field required (relative path to anchor file "
            "in repo root; e.g. 'README.md')",
        ))
        return
    if repo_root is None:
        # Validator can be called without repo_root context — that's OK,
        # but the anchor field itself must be present.
        return
    anchor_path = (repo_root / anchor).resolve()
    if not anchor_path.exists():
        errors.append(ValidationError(
            "repo-root-anchor-missing",
            f"repo_root_anchor={anchor!r} does not exist under repo_root="
            f"{repo_root}; either you pointed at the wrong tree, or anchor "
            f"was deleted",
        ))


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def validate_manifest(data: dict, repo_root: Path | None = None) -> None:
    """Raise ValidationError on first failure; or all-errors via ValidationResult.

    For test ergonomics, this raises on first error (per codex review fixture
    spec: pytest.raises(ValidationError) with substring match). Use
    validate_manifest_collect() to collect all errors.
    """
    result = validate_manifest_collect(data, repo_root=repo_root)
    if not result.ok:
        # Surface first one; full set available on result.errors.
        raise result.errors[0]


def validate_manifest_collect(data: dict,
                              repo_root: Path | None = None) -> ValidationResult:
    """Collect ALL errors (don't fail-fast); used for CLI + sanity suite."""
    errors: List[ValidationError] = []

    if not isinstance(data, dict):
        errors.append(ValidationError(
            "manifest-not-mapping", "manifest must be a YAML mapping"))
        return ValidationResult(errors)

    _check_versions(data, errors)
    _check_repo_root_anchor(data, repo_root, errors)
    mode = _check_mode(data, errors)
    outcome = _check_outcome(data, errors)
    _check_validation_level(data, mode, errors)
    evidence_present = _collect_evidence_types(data, errors)
    _check_evidence_required(outcome, evidence_present, errors)
    artifact_present = _collect_artifact_roles(data, errors)
    _check_mode_artifact_combo(mode, artifact_present, errors)
    _check_self_challenge(data, errors)

    return ValidationResult(errors)


def _main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Path to claim_manifest.yaml")
    p.add_argument("--repo-root", default=None, help="Repo root (for anchor check)")
    args = p.parse_args(argv)

    mp = Path(args.manifest)
    if not mp.exists():
        sys.stderr.write(f"manifest not found: {mp}\n")
        return 2
    try:
        with mp.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        sys.stderr.write(f"YAML parse error in {mp}: {e}\n")
        return 2

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    result = validate_manifest_collect(data, repo_root=repo_root)
    sys.stdout.write(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
