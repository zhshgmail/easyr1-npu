"""diff_snapshot.py — compare baseline.yaml vs current state.

Per TEST_SAFETY_FEEDBACK_DESIGN.md v4 §5 + P0k:

  Reads baseline YAML (created earlier by snapshot_current.py) and a
  fresh current snapshot, and prints regressions.

  A "regression" is one of:
    - branch in baseline disappeared from current ledger
    - status_summary changed in a way that flips outcome category
      (e.g., loses 'PASS' marker; gains 'BLOCKED' marker)
    - image_sha256 changed (for /integrated-overlay-build)
    - skill row removed entirely

  auto_advance_branches list in baseline OPT-IN allows SHA / status text
  drift for specific branches (e.g., vllm-main keeps advancing).

Usage:
  python3 diff_snapshot.py --baseline <YAML> --repo-root <r>

Author: T31.10 (P0k implementation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError:
    sys.stderr.write("diff_snapshot: PyYAML not installed.\n")
    sys.exit(2)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from snapshot_current import snapshot as snapshot_current  # noqa: E402


POSITIVE_MARKERS = ["PASS", "DONE", "outcome A", "outcome A_WITH_NOTE", "shipped"]
NEGATIVE_MARKERS = ["BLOCKED", "FAIL", "REGRESSION", "abandoned"]


def _by_skill(baselines: List[dict]) -> Dict[str, dict]:
    return {b.get("skill", ""): b for b in baselines if b.get("skill")}


def diff(baseline: dict, current: dict) -> List[str]:
    regressions: List[str] = []

    base_by_skill = _by_skill(baseline.get("baselines", []))
    cur_by_skill = _by_skill(current.get("baselines", []))
    auto_advance = set(baseline.get("auto_advance_branches", []) or [])

    for skill in sorted(set(base_by_skill) - set(cur_by_skill)):
        regressions.append(
            f"REGRESSION: skill {skill} present in baseline but missing from current"
        )

    for skill in sorted(set(base_by_skill) & set(cur_by_skill)):
        b = base_by_skill[skill]
        c = cur_by_skill[skill]
        branch = b.get("branch", "")

        if "image_sha256" in b:
            b_sha = b.get("image_sha256")
            c_sha = c.get("image_sha256")
            if b_sha and c_sha and b_sha != c_sha:
                regressions.append(
                    f"REGRESSION: {skill} image_sha256 changed "
                    f"{b_sha!r} -> {c_sha!r}"
                )

        b_branch = b.get("branch", "")
        c_branch = c.get("branch", "")
        if b_branch != c_branch and not (b_branch == "n/a" or c_branch == "n/a"):
            regressions.append(
                f"REGRESSION: {skill} branch changed {b_branch!r} -> {c_branch!r}"
            )

        if branch not in auto_advance:
            b_status = b.get("status_summary", "")
            c_status = c.get("status_summary", "")
            for pos in POSITIVE_MARKERS:
                if pos in b_status and pos not in c_status:
                    regressions.append(
                        f"REGRESSION: {skill} status lost positive marker {pos!r}; "
                        f"baseline: {b_status[:60]!r}; current: {c_status[:60]!r}"
                    )
            for neg in NEGATIVE_MARKERS:
                if neg not in b_status and neg in c_status:
                    regressions.append(
                        f"REGRESSION: {skill} status gained negative marker {neg!r}; "
                        f"current: {c_status[:120]!r}"
                    )

    return regressions


def _main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--repo-root", required=True)
    args = p.parse_args(argv)

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        sys.stderr.write(f"baseline not found: {baseline_path}\n")
        return 2
    repo_root = Path(args.repo_root).resolve()

    with baseline_path.open() as f:
        baseline = yaml.safe_load(f) or {}
    current = yaml.safe_load(snapshot_current(repo_root)) or {}

    regressions = diff(baseline, current)
    if not regressions:
        sys.stdout.write("diff_snapshot: PASS (no regressions vs baseline)\n")
        return 0
    sys.stdout.write("diff_snapshot: REGRESSIONS DETECTED\n")
    for r in regressions:
        sys.stdout.write(f"  - {r}\n")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
