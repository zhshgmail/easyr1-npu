"""scan_outcome_claims.py — schema-level validation + M2 anti-cycle check.

Per TEST_SAFETY_FEEDBACK_DESIGN.md v4 §4.2:

  Job: validate claim_manifest.yaml schema + check verifier_scripts are
  data-flow-independent of the claim (M2 anti-cycle).

  M2 dynamic forbidden-path set: derive from THIS manifest's
  evidence[].path + artifacts[].paths + artifacts[].path. Any verifier
  script that reads any of those paths is rejected as self-citing
  (a5_ops `WORKER-SELF-CITING-VERIFIER` attack pattern, generalized).

  NO regex on prose. NO hardcoded literals. Forbidden set is per-manifest.

  M5 NOT implemented here per codex review #2 (deferred to DEBT-6 +
  P0e workflow_critic with PreToolUse tool-use log capture).

Exit codes:
  0 PASS
  1 FAIL — at least one error
  2 usage error

Usage:
  python3 scan_outcome_claims.py --manifest <p> --repo-root <r>

Author: T31.7 (P0i implementation).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

try:
    import yaml
except ImportError:
    sys.stderr.write("scan_outcome_claims: PyYAML not installed.\n")
    sys.exit(2)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from validate_claim_manifest import (  # noqa: E402
    ValidationError,
    validate_manifest_collect,
)


@dataclass
class ScanError:
    code: str
    description: str

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.code}] {self.description}"


@dataclass
class ScanResult:
    errors: List[ScanError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        if self.ok:
            return "scan_outcome_claims: PASS\n"
        out = ["scan_outcome_claims: FAIL"]
        for err in self.errors:
            out.append(f"  - {err}")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# M2 anti-cycle dynamic check
# ---------------------------------------------------------------------------


def _collect_forbidden_paths(manifest: dict) -> Set[str]:
    """Derive the M2 forbidden-path set from THIS manifest.

    These are paths the verifier MUST NOT read (or the verifier becomes
    data-flow-dependent on the claim it is verifying).
    """
    forbidden: Set[str] = set()

    for ev in manifest.get("evidence", []) or []:
        if not isinstance(ev, dict):
            continue
        for key in ("path", "source_log"):
            v = ev.get(key)
            if v and isinstance(v, str):
                forbidden.add(v)

    for art in manifest.get("artifacts", []) or []:
        if not isinstance(art, dict):
            continue
        p = art.get("path")
        if p and isinstance(p, str):
            forbidden.add(p)
        for p in art.get("paths", []) or []:
            if isinstance(p, str):
                forbidden.add(p)

    # Also forbid the manifest file itself (clearest cycle).
    mrel = manifest.get("_manifest_relative_to_repo_root")
    if mrel and isinstance(mrel, str):
        forbidden.add(mrel)

    return forbidden


def _check_anti_cycle_dynamic(
    repo_root: Path,
    manifest: dict,
    errors: List[ScanError],
) -> None:
    """For each verifier_script: check existence, exec bit, path within
    repo_root, and that body does NOT reference any forbidden path."""
    forbidden = _collect_forbidden_paths(manifest)
    verifier_scripts = manifest.get("verifier_scripts", []) or []
    if not isinstance(verifier_scripts, list):
        errors.append(ScanError(
            "verifier-scripts-type", "verifier_scripts must be a list"))
        return

    repo_root_resolved = repo_root.resolve()
    for i, v in enumerate(verifier_scripts):
        if not isinstance(v, dict) or "path" not in v:
            errors.append(ScanError(
                "verifier-script-malformed",
                f"verifier_scripts[{i}] missing 'path'",
            ))
            continue
        rel = v["path"]
        if not isinstance(rel, str):
            errors.append(ScanError(
                "verifier-script-path-type",
                f"verifier_scripts[{i}].path must be string; got {type(rel).__name__}",
            ))
            continue
        # Path resolution: explicit repo-root-relative (v4 codex fix).
        script = (repo_root / rel).resolve()
        try:
            within = script.is_relative_to(repo_root_resolved)
        except AttributeError:
            # Python < 3.9 fallback (shouldn't apply; 3.11 minimum here).
            within = str(script).startswith(str(repo_root_resolved))
        if not within:
            errors.append(ScanError(
                "verifier-escapes-repo-root",
                f"verifier_scripts[{i}].path={rel!r} resolves to {script} "
                f"outside repo_root={repo_root_resolved}",
            ))
            continue
        if not script.exists():
            errors.append(ScanError(
                "verifier-missing",
                f"verifier_scripts[{i}].path={rel!r} does not exist "
                f"(resolved: {script})",
            ))
            continue
        if not os.access(script, os.X_OK):
            errors.append(ScanError(
                "verifier-not-executable",
                f"verifier_scripts[{i}].path={rel!r} not executable",
            ))
            # Don't return — still check anti-cycle on body
        try:
            body = script.read_text(errors="replace")
        except OSError as e:
            errors.append(ScanError(
                "verifier-unreadable",
                f"verifier_scripts[{i}].path={rel!r}: {e}",
            ))
            continue

        # M2 anti-cycle: body must not reference any forbidden path.
        for fp in forbidden:
            if not fp:
                continue
            if fp in body:
                errors.append(ScanError(
                    "anti-cycle",
                    f"verifier_scripts[{i}].path={rel!r} body references "
                    f"manifest-claim path {fp!r} (M2 anti-cycle: verifier "
                    f"must be data-flow-independent of the claim it verifies)",
                ))


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def scan(manifest_path: Path, repo_root: Path) -> ScanResult:
    """Run schema validation + M2 anti-cycle check."""
    errors: List[ScanError] = []

    if not manifest_path.exists():
        errors.append(ScanError("manifest-missing", f"file not found: {manifest_path}"))
        return ScanResult(errors)

    try:
        with manifest_path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(ScanError("manifest-yaml-error", f"yaml parse: {e}"))
        return ScanResult(errors)

    # Schema validation first.
    schema_result = validate_manifest_collect(data, repo_root=repo_root)
    for ve in schema_result.errors:
        # Promote ValidationError into ScanError with same code.
        errors.append(ScanError(ve.code, ve.description))

    # If schema is clean enough, run M2 anti-cycle.
    if isinstance(data, dict):
        _check_anti_cycle_dynamic(repo_root, data, errors)

    return ScanResult(errors)


def _main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--repo-root", required=True)
    args = p.parse_args(argv)
    result = scan(Path(args.manifest), Path(args.repo_root))
    sys.stdout.write(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
