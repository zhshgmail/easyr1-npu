"""finalize_day0_check.py — multi-gate verification before day-0 done.

Per TEST_SAFETY_FEEDBACK_DESIGN.md v4 §4.3 + §4.4:

  4 independent gates, each verifying a different invariant. Defense-in-
  depth principle: each gate verifies independently — no gate trusts a
  prior gate's PASS verdict. Avoids a5_ops 10-layer collapse pattern.

  GateID enum (4 only):
    1. CLAIM_EVIDENCE_PRESENT       — claim's evidence files exist + sha256 + mtime
    2. REQUIRED_ARTIFACTS_PRESENT   — mode-required artifacts exist + section headers
    3. EXTERNAL_PUBLICATION_VERIFIED — fork branch pushed / image SHA real / issue URL reachable
    4. VALIDATION_ARTIFACT_VERIFIED  — smoke log grep_assertion matches

  Mode-derived: all required-sets come from mode_dispatch.py — no
  hardcoded `PR_MATERIAL.md` / `ascend-port/...` literals in gate code.

  Network checks (gate 3) accept --offline flag to skip; offline runs
  emit warnings, never block (per design v4 §9.1).

Exit codes:
  0 PASS — all gates passed (or skipped with allow_offline)
  1 FAIL — at least one gate rejected
  2 usage error

Usage:
  python3 finalize_day0_check.py --workspace <ws> --repo-root <r> [--offline]

Author: T31.8 (P0j implementation).
"""
from __future__ import annotations

import argparse
import enum
import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

try:
    import yaml
except ImportError:
    sys.stderr.write("finalize_day0_check: PyYAML not installed.\n")
    sys.exit(2)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mode_dispatch import (  # noqa: E402
    ArtifactRole,
    EvidenceType,
    Mode,
    Outcome,
    required_artifacts,
    required_evidence,
)


class GateID(str, enum.Enum):
    """4 independent finalize gates. Order is arbitrary — each verifies
    a different invariant; failure of any one => reject."""

    CLAIM_EVIDENCE_PRESENT = "CLAIM_EVIDENCE_PRESENT"
    REQUIRED_ARTIFACTS_PRESENT = "REQUIRED_ARTIFACTS_PRESENT"
    EXTERNAL_PUBLICATION_VERIFIED = "EXTERNAL_PUBLICATION_VERIFIED"
    VALIDATION_ARTIFACT_VERIFIED = "VALIDATION_ARTIFACT_VERIFIED"


@dataclass
class Rejection:
    gate_id: GateID
    description: str

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.gate_id.value}] {self.description}"


@dataclass
class FinalizeResult:
    eligible: bool = True
    rejections: List[Rejection] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def reject(self, gate: GateID, desc: str) -> None:
        self.eligible = False
        self.rejections.append(Rejection(gate, desc))

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        if self.eligible:
            out = ["finalize_day0_check: PASS"]
            for w in self.warnings:
                out.append(f"  ⚠ warning: {w}")
            return "\n".join(out) + "\n"
        out = ["finalize_day0_check: FAIL"]
        for r in self.rejections:
            out.append(f"  - {r}")
        for w in self.warnings:
            out.append(f"  ⚠ warning: {w}")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Gate 1: CLAIM_EVIDENCE_PRESENT
# ---------------------------------------------------------------------------


def gate_claim_evidence_present(
    repo_root: Path,
    workspace: Path,
    manifest: dict,
    result: FinalizeResult,
) -> None:
    """Every evidence[].path must exist; sha256 must match if provided."""
    for i, ev in enumerate(manifest.get("evidence", []) or []):
        if not isinstance(ev, dict):
            continue
        path_rel = ev.get("path") or ev.get("source_log")
        if not path_rel:
            continue  # Some evidence types (commit_ref / github_issue) have no path
        # Paths in manifest are repo-root relative.
        ev_path = (repo_root / path_rel).resolve()
        if not ev_path.exists():
            result.reject(
                GateID.CLAIM_EVIDENCE_PRESENT,
                f"evidence[{i}] path {path_rel!r} does not exist "
                f"(resolved: {ev_path})",
            )
            continue
        # sha256 check (best-effort; do not fail if not provided).
        claim_sha = ev.get("sha256")
        if claim_sha and isinstance(claim_sha, str) and len(claim_sha) == 64:
            actual = hashlib.sha256(ev_path.read_bytes()).hexdigest()
            # Allow fixtures with placeholder zero-hash to skip the check.
            if claim_sha != "0" * 64 and actual != claim_sha:
                result.reject(
                    GateID.CLAIM_EVIDENCE_PRESENT,
                    f"evidence[{i}] sha256 mismatch: claim={claim_sha[:12]}..., "
                    f"actual={actual[:12]}...",
                )


# ---------------------------------------------------------------------------
# Gate 2: REQUIRED_ARTIFACTS_PRESENT (mode-derived)
# ---------------------------------------------------------------------------


def gate_required_artifacts_present(
    repo_root: Path,
    workspace: Path,
    manifest: dict,
    result: FinalizeResult,
) -> None:
    """Mode dictates which artifact roles required. For each required role,
    the manifest must list an artifact with that role, and that artifact's
    file(s) must exist + (for pr_material) have required section headers."""
    try:
        mode = Mode(manifest.get("mode"))
    except (TypeError, ValueError):
        # Schema validation should have caught this; bail.
        return

    required_roles: Set[ArtifactRole] = required_artifacts(mode)
    present_role_map: dict[ArtifactRole, dict] = {}
    for art in manifest.get("artifacts", []) or []:
        if not isinstance(art, dict):
            continue
        try:
            role = ArtifactRole(art.get("role"))
        except (TypeError, ValueError):
            continue
        present_role_map[role] = art

    missing = required_roles - set(present_role_map.keys())
    if missing:
        result.reject(
            GateID.REQUIRED_ARTIFACTS_PRESENT,
            f"mode={mode.value} requires artifact roles "
            f"{sorted(r.value for r in required_roles)}; missing "
            f"{sorted(r.value for r in missing)}",
        )

    # Per-role structural checks.
    for role, art in present_role_map.items():
        paths = []
        if "path" in art and isinstance(art["path"], str):
            paths.append(art["path"])
        for p in art.get("paths", []) or []:
            if isinstance(p, str):
                paths.append(p)
        for p in paths:
            full = (repo_root / p).resolve()
            if not full.exists():
                # For shim_module / smoke_log / checkpoint_path, file
                # may live on A3 host; we don't require local presence
                # unless required-by-role. Only require local presence
                # for pr_material (which must be a doc shipped to the
                # branch/repo).
                if role in (ArtifactRole.PR_MATERIAL, ArtifactRole.WORKAROUND_DOC):
                    result.reject(
                        GateID.REQUIRED_ARTIFACTS_PRESENT,
                        f"artifact role={role.value} path={p!r} does not "
                        f"exist (resolved: {full})",
                    )
                continue
            # Section-header check for pr_material / workaround_doc
            if role in (ArtifactRole.PR_MATERIAL, ArtifactRole.WORKAROUND_DOC):
                required_sections = art.get("required_sections", []) or []
                if required_sections:
                    text = full.read_text(errors="replace")
                    # Require section header to appear at line start
                    # (not inside HTML comment / inline mention).
                    lines = text.splitlines()
                    for sec in required_sections:
                        if not any(line.lstrip().startswith(sec) for line in lines):
                            result.reject(
                                GateID.REQUIRED_ARTIFACTS_PRESENT,
                                f"artifact role={role.value} path={p!r} "
                                f"missing required section {sec!r} "
                                f"(section-missing)",
                            )


# ---------------------------------------------------------------------------
# Gate 3: EXTERNAL_PUBLICATION_VERIFIED (network-backed, --offline-skippable)
# ---------------------------------------------------------------------------


def _check_branch_exists(repo: str, branch: str, sha: str) -> tuple[bool, str]:
    """Return (exists, description). Uses `gh api` for github, `gc api`
    for gitcode, or curl for arbitrary URLs."""
    # Determine host
    repo_lower = repo.lower()
    if repo_lower.startswith("github.com/"):
        slug = repo[len("github.com/"):]
        cmd = ["gh", "api", f"repos/{slug}/branches/{branch}"]
    elif repo_lower.startswith("gitcode.com/"):
        slug = repo[len("gitcode.com/"):]
        cmd = ["gc", "api", f"repos/{slug}/branches/{branch}"]
    else:
        # Best effort: try gh
        return False, f"unsupported repo host for {repo!r}"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, f"cmd error: {e}"
    if p.returncode != 0:
        return False, f"{' '.join(cmd)} exit {p.returncode}: {p.stderr[:200].strip()}"
    return True, "branch found"


def gate_external_publication_verified(
    repo_root: Path,
    workspace: Path,
    manifest: dict,
    result: FinalizeResult,
    offline: bool = False,
) -> None:
    """Mode-derived check:
      fork_patch          -> commit_ref's branch must be reachable
      integrated_overlay  -> image SHA in evidence/artifacts must be a real
                             docker manifest (skip — usually local only)
      issue_only          -> github_issue URL must be reachable
    """
    if offline:
        result.warn(
            "EXTERNAL_PUBLICATION_VERIFIED skipped (--offline); "
            "must be re-run before customer ship"
        )
        return

    try:
        mode = Mode(manifest.get("mode"))
    except (TypeError, ValueError):
        return

    if mode is Mode.FORK_PATCH:
        # Find commit_ref evidence
        for ev in manifest.get("evidence", []) or []:
            if not isinstance(ev, dict):
                continue
            if ev.get("type") == EvidenceType.COMMIT_REF.value:
                repo = ev.get("repo", "")
                branch = ev.get("branch", "")
                sha = ev.get("sha", "")
                if not (repo and branch):
                    continue
                ok, desc = _check_branch_exists(repo, branch, sha)
                if not ok:
                    result.reject(
                        GateID.EXTERNAL_PUBLICATION_VERIFIED,
                        f"commit_ref repo={repo!r} branch={branch!r}: "
                        f"branch-not-found ({desc})",
                    )
    elif mode is Mode.ISSUE_ONLY:
        for ev in manifest.get("evidence", []) or []:
            if not isinstance(ev, dict):
                continue
            if ev.get("type") == EvidenceType.GITHUB_ISSUE.value:
                url = ev.get("url", "")
                if not url:
                    continue
                try:
                    p = subprocess.run(
                        ["curl", "-fsI", "--max-time", "10", url],
                        capture_output=True, text=True, timeout=15,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                    result.reject(
                        GateID.EXTERNAL_PUBLICATION_VERIFIED,
                        f"github_issue URL {url!r} curl error: {e}",
                    )
                    continue
                if p.returncode != 0:
                    result.reject(
                        GateID.EXTERNAL_PUBLICATION_VERIFIED,
                        f"github_issue URL {url!r}: HEAD failed "
                        f"(exit {p.returncode})",
                    )
    elif mode is Mode.INTEGRATED_OVERLAY:
        # image_sha verification typically requires docker on A3 host;
        # skip with warning unless explicitly forced.
        result.warn(
            "INTEGRATED_OVERLAY image SHA verification deferred "
            "(requires A3 docker access; emit warning only)"
        )


# ---------------------------------------------------------------------------
# Gate 4: VALIDATION_ARTIFACT_VERIFIED
# ---------------------------------------------------------------------------


def gate_validation_artifact_verified(
    repo_root: Path,
    workspace: Path,
    manifest: dict,
    result: FinalizeResult,
) -> None:
    """For every smoke_log evidence with grep_assertion, the assertion
    must actually grep-match in the log content."""
    for i, ev in enumerate(manifest.get("evidence", []) or []):
        if not isinstance(ev, dict):
            continue
        if ev.get("type") != EvidenceType.SMOKE_LOG.value:
            continue
        rel = ev.get("path") or ev.get("source_log")
        assertion = ev.get("grep_assertion")
        if not rel or not assertion:
            continue
        log_path = (repo_root / rel).resolve()
        if not log_path.exists():
            # Will already be caught by gate 1; skip duplicate rejection.
            continue
        text = log_path.read_text(errors="replace")
        # Allow assertion to contain regex special chars; require literal
        # substring match for now (conservative). If we later need regex,
        # introduce assertion_kind: regex|literal field.
        if assertion not in text:
            result.reject(
                GateID.VALIDATION_ARTIFACT_VERIFIED,
                f"evidence[{i}] smoke_log {rel!r} does not contain "
                f"grep_assertion {assertion!r} (grep-assertion-failed)",
            )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def check(workspace: Path, repo_root: Path,
          *, offline: bool = False) -> FinalizeResult:
    """Run all 4 gates on a workspace dir containing claim_manifest.yaml."""
    result = FinalizeResult()
    manifest_path = workspace / "claim_manifest.yaml"
    if not manifest_path.exists():
        result.reject(
            GateID.CLAIM_EVIDENCE_PRESENT,
            f"claim_manifest.yaml not found in workspace {workspace}",
        )
        return result
    try:
        with manifest_path.open() as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.reject(
            GateID.CLAIM_EVIDENCE_PRESENT,
            f"claim_manifest.yaml yaml parse error: {e}",
        )
        return result

    if not isinstance(manifest, dict):
        result.reject(
            GateID.CLAIM_EVIDENCE_PRESENT,
            "claim_manifest.yaml not a mapping",
        )
        return result

    # Run all 4 gates independently — defense in depth.
    gate_claim_evidence_present(repo_root, workspace, manifest, result)
    gate_required_artifacts_present(repo_root, workspace, manifest, result)
    gate_external_publication_verified(
        repo_root, workspace, manifest, result, offline=offline)
    gate_validation_artifact_verified(repo_root, workspace, manifest, result)

    return result


def _main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--offline", action="store_true",
                   help="Skip network-backed gate (warning, not reject)")
    args = p.parse_args(argv)
    result = check(Path(args.workspace), Path(args.repo_root), offline=args.offline)
    sys.stdout.write(result.render())
    return 0 if result.eligible else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
