#!/usr/bin/env python3
"""KB-driven cold-drive test for vllm-ascend port-expert.

Given a target vllm commit (the drift to adapt to), reproduce Phase A/B
of the skill deterministically from KB:

1. Detect which symbols, classes, methods, signatures changed in the
   target vllm commit vs the current vllm_ascend tree is known against.
2. Match each change to a KB family (F1-F8) by symptom shape.
3. For each matched family, emit the fix template + the specific
   call-sites in vllm_ascend that need the fix.
4. Write a proposal file the worker can then patch against.

This is the KB-validation harness. If KB_INDEX.md symptom-routing table
and vllm-api-drift.md family templates are complete, this script should
produce an actionable patch plan without any human guidance.

Usage:
  kb_drive_test.py --vllm-ref <SHA_OR_TAG> \
                   --vllm-path <PATH_TO_VLLM_CHECKOUT> \
                   --vllm-ascend-path <PATH_TO_VLLM_ASCEND_CHECKOUT> \
                   --kb-dir <PATH_TO_references/> \
                   [--out <OUTPUT_DIR>]

Exit codes:
  0 = no drift detected, or all drifts matched to a KB family
  1 = drift detected but one or more symptoms did NOT match any family
  2 = usage error / missing inputs
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Drift:
    kind: str                  # "removed_symbol" / "renamed" / "sig_change" / ...
    vllm_path: str             # path inside vllm repo
    symbol: str                # symbol name that changed
    detail: str = ""
    matched_family: str = ""   # F1..F8 or "" if unmatched
    ascend_callsites: list[tuple[str, int, str]] = field(default_factory=list)


@dataclass
class FamilyMatchRule:
    family: str
    kind: str          # matches Drift.kind
    description: str


FAMILY_RULES = [
    FamilyMatchRule("F1", "removed_symbol",
                    "Removed symbol (import) — Upstream deleted a public symbol vllm-ascend imports"),
    FamilyMatchRule("F2", "renamed",
                    "Renamed type/class — Upstream renamed a class/type vllm-ascend imports or subclasses"),
    FamilyMatchRule("F3", "sig_change",
                    "Signature change — Upstream added/removed/reordered args on a function vllm-ascend calls"),
    FamilyMatchRule("F4", "return_type_change",
                    "Return-type migration — Scalar → NamedTuple/dict"),
    FamilyMatchRule("F5", "buffer_api_migration",
                    "Buffer API migration — CpuGpuBuffer ↔ plain tensor"),
    FamilyMatchRule("F6", "kv_cache_contract",
                    "kv_cache tensor-vs-list contract change"),
    FamilyMatchRule("F7", "new_attr_required",
                    "New required attribute on NPU integration class"),
    FamilyMatchRule("F8", "new_method_required",
                    "New required method on NPU integration class"),
]


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def detect_removed_symbols(vllm_path: Path, ref: str) -> list[Drift]:
    """Look at target ref's diff vs main — files that moved/deleted and
    symbols that were exported but are no longer."""
    drifts: list[Drift] = []
    diff = run_git(vllm_path, "show", "--stat", "--format=", ref)
    if not diff:
        return drifts

    deleted_files = re.findall(r"delete mode \d+ (.+)$", diff, re.MULTILINE)
    for df in deleted_files:
        if df.endswith(".py"):
            module = df.replace("/", ".").removesuffix(".py")
            drifts.append(Drift(
                kind="removed_symbol",
                vllm_path=df,
                symbol=module.split(".")[-1],
                detail=f"File deleted; any `from {module} import ...` will fail",
            ))
    return drifts


def detect_class_removals(vllm_path: Path, ref: str) -> list[Drift]:
    """Look for `class Xyz` or module-level `def xyz` that existed at
    ref^ but not at ref."""
    drifts: list[Drift] = []
    files_changed = run_git(vllm_path, "diff", "--name-only", f"{ref}^..{ref}")
    for f in files_changed.strip().splitlines():
        if not f.endswith(".py") or "/tests/" in f:
            continue
        diff = run_git(vllm_path, "diff", f"{ref}^..{ref}", "--", f)
        removed_classes = re.findall(r"^-class (\w+)", diff, re.MULTILINE)
        added_classes = re.findall(r"^\+class (\w+)", diff, re.MULTILINE)
        truly_removed_classes = set(removed_classes) - set(added_classes)
        for cls in truly_removed_classes:
            if cls.startswith("_"):
                continue
            drifts.append(Drift(
                kind="removed_symbol",
                vllm_path=f,
                symbol=cls,
                detail=f"class {cls} removed in {ref}",
            ))

        removed_defs = re.findall(r"^-def (\w+)\s*\(", diff, re.MULTILINE)
        added_defs = re.findall(r"^\+def (\w+)\s*\(", diff, re.MULTILINE)
        truly_removed_defs = set(removed_defs) - set(added_defs)
        for fn in truly_removed_defs:
            if fn.startswith("_"):
                continue
            drifts.append(Drift(
                kind="removed_symbol",
                vllm_path=f,
                symbol=fn,
                detail=f"function {fn} removed in {ref}",
            ))
    return drifts


def grep_ascend_for_symbol(vllm_ascend_path: Path, symbol: str) -> list[tuple[str, int, str]]:
    """Return call sites in vllm-ascend that reference the symbol."""
    out = subprocess.run(
        ["grep", "-rn", "-w", symbol, "--include=*.py", str(vllm_ascend_path)],
        capture_output=True, text=True, check=False,
    )
    sites: list[tuple[str, int, str]] = []
    for line in out.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, lineno, text = parts
        rel = path.removeprefix(str(vllm_ascend_path) + "/")
        if "/tests/" in rel or "/test_" in rel:
            continue
        try:
            sites.append((rel, int(lineno), text.strip()))
        except ValueError:
            continue
    return sites


def match_drift_to_family(drift: Drift) -> str:
    for rule in FAMILY_RULES:
        if rule.kind == drift.kind:
            return rule.family
    return ""


def load_kb_family_template(kb_dir: Path, family: str) -> str:
    """Extract the fix-template section for the given family from the KB."""
    kb_file = kb_dir / "patterns" / "domains" / "vllm-api-drift.md"
    if not kb_file.exists():
        return f"[KB MISS] {kb_file} does not exist"
    content = kb_file.read_text()
    marker_re = re.compile(rf"^## {family} — .+?(?=^## [A-Z]\d |^## Cross-family|\Z)",
                           re.MULTILINE | re.DOTALL)
    m = marker_re.search(content)
    return m.group(0) if m else f"[KB MISS] no section for {family} in {kb_file}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-ref", required=True, help="target vllm commit SHA or tag")
    p.add_argument("--vllm-path", required=True, type=Path)
    p.add_argument("--vllm-ascend-path", required=True, type=Path)
    p.add_argument("--kb-dir", required=True, type=Path,
                   help="path to port-expert/references/")
    p.add_argument("--out", type=Path, default=Path("/tmp/kb_drive_out"))
    args = p.parse_args()

    for path in [args.vllm_path, args.vllm_ascend_path, args.kb_dir]:
        if not path.exists():
            print(f"ERROR: path does not exist: {path}", file=sys.stderr)
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] detecting drift in vllm ref {args.vllm_ref} ...")
    drifts: list[Drift] = []
    drifts += detect_removed_symbols(args.vllm_path, args.vllm_ref)
    drifts += detect_class_removals(args.vllm_path, args.vllm_ref)
    print(f"      found {len(drifts)} raw drifts")

    print(f"[2/4] locating call sites in vllm-ascend ...")
    for d in drifts:
        d.ascend_callsites = grep_ascend_for_symbol(args.vllm_ascend_path, d.symbol)
    impactful = [d for d in drifts if d.ascend_callsites]
    print(f"      {len(impactful)} drifts impact vllm-ascend")

    print(f"[3/4] matching impactful drifts to KB families ...")
    unmatched = []
    for d in impactful:
        d.matched_family = match_drift_to_family(d)
        if not d.matched_family:
            unmatched.append(d)
    print(f"      matched {len(impactful) - len(unmatched)}/{len(impactful)}")

    print(f"[4/4] writing proposal to {args.out}/proposal.md ...")
    lines = [
        f"# KB-driven port proposal — vllm ref {args.vllm_ref}\n",
        f"Raw drifts detected: {len(drifts)}",
        f"Impact vllm-ascend: {len(impactful)}",
        f"Matched to KB family: {len(impactful) - len(unmatched)}",
        f"Unmatched (KB gap): {len(unmatched)}",
        "\n---\n",
    ]
    for d in impactful:
        lines.append(f"## [{d.matched_family or 'UNMATCHED'}] {d.symbol} — {d.kind}\n")
        lines.append(f"- vllm path: `{d.vllm_path}`")
        lines.append(f"- detail: {d.detail}")
        lines.append("- vllm-ascend call sites:")
        for site in d.ascend_callsites:
            lines.append(f"  - `{site[0]}:{site[1]}` — {site[2][:120]}")
        if d.matched_family:
            template = load_kb_family_template(args.kb_dir, d.matched_family)
            lines.append("\n**KB fix template**:\n")
            lines.append(template)
        lines.append("\n---\n")

    proposal_path = args.out / "proposal.md"
    proposal_path.write_text("\n".join(lines))

    summary = {
        "vllm_ref": args.vllm_ref,
        "raw_drifts": len(drifts),
        "impact_ascend": len(impactful),
        "matched": len(impactful) - len(unmatched),
        "unmatched": len(unmatched),
        "proposal_path": str(proposal_path),
        "drifts": [
            {
                "symbol": d.symbol,
                "kind": d.kind,
                "vllm_path": d.vllm_path,
                "family": d.matched_family,
                "ascend_sites": len(d.ascend_callsites),
            }
            for d in impactful
        ],
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{json.dumps(summary, indent=2)}")

    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
