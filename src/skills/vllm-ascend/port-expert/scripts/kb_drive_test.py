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
    vllm_path: str             # path inside vllm repo (where removed/changed)
    symbol: str                # symbol name that changed
    detail: str = ""
    matched_family: str = ""   # F1..F8 or "" if unmatched
    ascend_callsites: list[tuple[str, int, str]] = field(default_factory=list)
    # new-location guesses (if the symbol moved rather than being deleted).
    # populated only for F1 hits where the fix requires knowing the new home.
    new_home_candidates: list[str] = field(default_factory=list)


@dataclass
class FamilyMatchRule:
    family: str
    kind: str          # matches Drift.kind
    description: str


# Only families that have actual detector functions implemented below
# are listed here. F4 (return-type migration), F5 (buffer API), F6
# (kv_cache contract), F7 (new-required-attr), F8 (new-required-method)
# are described in the KB but NOT automatically detected by this scanner —
# they require runtime / base-class inspection that is out of scope here.
# A clean scan output (impact_ascend=0) only means no F1/F2/F3 drift;
# the user must still manually check the KB §F4..F8 sections on each
# new vllm release.
FAMILY_RULES = [
    FamilyMatchRule("F1", "removed_symbol",
                    "Removed symbol (import) — Upstream deleted a public symbol vllm-ascend imports"),
    FamilyMatchRule("F2-rename", "renamed",
                    "Renamed type/class — Upstream renamed a class/type vllm-ascend imports or subclasses"),
    FamilyMatchRule("F3", "sig_change",
                    "Signature change — Upstream added/removed/reordered args on a function vllm-ascend calls"),
    FamilyMatchRule("F5-suspect", "buffer_api_migration",
                    "F5 SUSPECT: class with buffer-shaped methods removed; verify by reading KB §F5"),
]

# Families documented in the KB but NOT yet scanner-detected. Surfaced
# in the final report so the user knows they still need manual inspection.
# F5 gets a weak detector above (name + body hints) but real F5 often
# manifests as per-method F3 (e.g. .copy_to_gpu added num_reqs arg) — that
# shape is caught by the F3 detector, just classified under F3.
UNDETECTED_FAMILIES = [
    ("F4", "return-type migration (scalar → NamedTuple/dict)"),
    ("F6", "kv_cache tensor-vs-list contract"),
    ("F7", "new required attribute on NPU subclass"),
    ("F8", "new required method on NPU subclass"),
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
        if not f.endswith(".py"):
            continue
        if f.startswith("tests/") or "/tests/" in f or "/test_" in f or f.startswith("test_"):
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


def ascend_defines_symbol(vllm_ascend_path: Path, symbol: str) -> bool:
    """True if vllm-ascend itself defines a class or top-level def with
    this name. When true, any call site referencing `symbol` may be to
    vllm-ascend's own definition, not to vllm — likely a false positive."""
    out = subprocess.run(
        ["grep", "-rln", "-E", rf"^(class|def) {symbol}\b",
         "--include=*.py", str(vllm_ascend_path)],
        capture_output=True, text=True, check=False,
    )
    for line in out.stdout.splitlines():
        rel = line.removeprefix(str(vllm_ascend_path) + "/")
        if "/tests/" not in rel and "/test_" not in rel:
            return True
    return False


def grep_ascend_for_symbol(vllm_ascend_path: Path, symbol: str,
                           require_vllm_import: bool = True
                           ) -> list[tuple[str, int, str]]:
    """Return call sites in vllm-ascend that reference the symbol.

    If require_vllm_import, only count sites in files that also have
    `from vllm...` import of the symbol — avoids false positives when
    vllm-ascend has its own class of the same name."""
    out = subprocess.run(
        ["grep", "-rn", "-w", symbol, "--include=*.py", str(vllm_ascend_path)],
        capture_output=True, text=True, check=False,
    )
    raw_sites: list[tuple[str, int, str]] = []
    for line in out.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, lineno, text = parts
        rel = path.removeprefix(str(vllm_ascend_path) + "/")
        if "/tests/" in rel or "/test_" in rel:
            continue
        # `/compat/` is the conventional home of forward-compat shims;
        # symbol references there are intentional, not broken call sites.
        if "/compat/" in rel or rel.startswith("compat/"):
            continue
        try:
            raw_sites.append((rel, int(lineno), text.strip()))
        except ValueError:
            continue

    if not require_vllm_import:
        return raw_sites

    files_with_vllm_import: set[str] = set()
    for site in raw_sites:
        rel, _, _ = site
        if rel in files_with_vllm_import:
            continue
        full = vllm_ascend_path / rel
        try:
            content = full.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        # Only count if `from vllm...` import actually contains the symbol.
        # Support single-line and parenthesized multi-line imports.
        # Note: `vllm` must be standalone or followed by `.` — avoid matching `vllm_ascend...`
        patterns = [
            rf"^from vllm(\.[\w\.]+)?\s+import\s+[^\n#(]*\b{symbol}\b",
            rf"^from vllm(\.[\w\.]+)?\s+import\s*\([^)]*\b{symbol}\b[^)]*\)",
        ]
        for pat in patterns:
            if re.search(pat, content, re.MULTILINE | re.DOTALL):
                files_with_vllm_import.add(rel)
                break

    return [s for s in raw_sites if s[0] in files_with_vllm_import]


def detect_buffer_api_migration(vllm_path: Path, ref: str) -> list[Drift]:
    """F5 detector. Upgrades an F1 removed_symbol to F5 suspect when:
      - class name looks buffer-ish (ends with Buffer, contains Buffer)
      - OR the diff's deletion body contains `def copy_to_gpu` or a
        `.np` attribute assignment typical of CpuGpuBuffer-shaped classes.

    Works off the same `diff` scan that detect_class_removals uses.
    """
    drifts: list[Drift] = []
    files_changed = run_git(vllm_path, "diff", "--name-only", f"{ref}^..{ref}")
    for f in files_changed.strip().splitlines():
        if not f.endswith(".py"):
            continue
        if f.startswith("tests/") or "/tests/" in f or "/test_" in f or f.startswith("test_"):
            continue
        diff = run_git(vllm_path, "diff", f"{ref}^..{ref}", "--", f)
        removed_classes = re.findall(r"^-class (\w+)", diff, re.MULTILINE)
        added_classes = re.findall(r"^\+class (\w+)", diff, re.MULTILINE)
        truly_removed = set(removed_classes) - set(added_classes)
        for cls in truly_removed:
            if cls.startswith("_"):
                continue
            name_hint = "Buffer" in cls
            # Body hint: find the `-class <cls>` block in diff, look for
            # buffer-shaped member names within the adjacent `-` lines.
            # Conservative: if the diff near the class removal shows
            # a `-    def copy_to_gpu(` or a `-    np = ` style line, flag.
            body_pattern = re.compile(
                rf"-class {cls}\b.*?(?=^-class |\Z)",
                re.MULTILINE | re.DOTALL,
            )
            body_m = body_pattern.search(diff)
            body_hint = False
            if body_m:
                body = body_m.group(0)
                if re.search(r"^-\s+def copy_to_gpu\s*\(", body, re.MULTILINE):
                    body_hint = True
                elif re.search(r"^-\s+np\s*[:=]", body, re.MULTILINE):
                    body_hint = True

            if not (name_hint or body_hint):
                continue

            hints = []
            if name_hint: hints.append("name contains 'Buffer'")
            if body_hint: hints.append("body had copy_to_gpu / .np")

            drifts.append(Drift(
                kind="buffer_api_migration",
                vllm_path=f,
                symbol=cls,
                detail=f"F5 suspect: {cls} removed ({', '.join(hints)})",
            ))
    return drifts


def detect_renames(vllm_path: Path, ref: str) -> list[Drift]:
    """Detect class or def renamed within same file in a single commit.
    Heuristic: `-class Old` + `+class New` in same file, where `New` is
    not in the old version. Not rigorous (same refactor may delete +
    re-add unrelated names) — we score by file-pair locality."""
    drifts: list[Drift] = []
    files_changed = run_git(vllm_path, "diff", "--name-only", f"{ref}^..{ref}")
    for f in files_changed.strip().splitlines():
        if not f.endswith(".py"):
            continue
        if f.startswith("tests/") or "/tests/" in f or "/test_" in f or f.startswith("test_"):
            continue
        diff = run_git(vllm_path, "diff", f"{ref}^..{ref}", "--", f)
        removed = set(re.findall(r"^-class (\w+)", diff, re.MULTILINE))
        added = set(re.findall(r"^\+class (\w+)", diff, re.MULTILINE))
        truly_removed = removed - added
        truly_added = added - removed

        if len(truly_removed) == 1 and len(truly_added) == 1:
            old = next(iter(truly_removed))
            new = next(iter(truly_added))
            if old.startswith("_") or new.startswith("_"):
                continue
            drifts.append(Drift(
                kind="renamed",
                vllm_path=f,
                symbol=old,
                detail=f"class {old} → {new} in {f}",
            ))
    return drifts


def detect_sig_changes(vllm_path: Path, ref: str) -> list[Drift]:
    """Detect function signature changes: `-def name(old_args)` +
    `+def name(new_args)` where args differ."""
    drifts: list[Drift] = []
    files_changed = run_git(vllm_path, "diff", "--name-only", f"{ref}^..{ref}")
    for f in files_changed.strip().splitlines():
        if not f.endswith(".py"):
            continue
        if f.startswith("tests/") or "/tests/" in f or "/test_" in f or f.startswith("test_"):
            continue
        diff = run_git(vllm_path, "diff", f"{ref}^..{ref}", "--", f)
        removed_sigs = dict(re.findall(r"^-def (\w+)\s*\(([^)]*)\)", diff, re.MULTILINE))
        added_sigs = dict(re.findall(r"^\+def (\w+)\s*\(([^)]*)\)", diff, re.MULTILINE))
        # Only top-level (module-level) defs. Crude check: the `-def`
        # match starts at column 0 in the diff after the `-` prefix.
        top_level = set()
        for m in re.finditer(r"^-(def \w+\s*\()", diff, re.MULTILINE):
            top_level.add(re.match(r"def (\w+)", m.group(1)).group(1))
        for name in removed_sigs:
            if name.startswith("_") or name not in added_sigs:
                continue
            if name not in top_level:
                continue
            if removed_sigs[name].strip() == added_sigs[name].strip():
                continue
            drifts.append(Drift(
                kind="sig_change",
                vllm_path=f,
                symbol=name,
                detail=f"def {name}({removed_sigs[name]}) → def {name}({added_sigs[name]})",
            ))
    return drifts


def find_new_home_candidates(vllm_path: Path, ref: str, symbol: str,
                              exclude_path: str) -> list[str]:
    """Grep the target ref's tree for where `symbol` is defined now.

    Excludes the path the symbol was removed from (to avoid showing the
    old home in its current state). Returns paths relative to the vllm
    repo root; up to 5 candidates."""
    if not symbol or symbol.startswith("_"):
        return []
    # Use git grep against the target ref so we don't have to actually
    # check the working tree out.
    r = subprocess.run(
        ["git", "-C", str(vllm_path), "grep", "-l", "-E",
         rf"^(class|def) {symbol}( |\(|:)",
         ref, "--", "vllm/"],
        capture_output=True, text=True, check=False,
    )
    hits = []
    for line in r.stdout.splitlines():
        # Output like "<ref>:vllm/foo/bar.py"
        if ":" not in line:
            continue
        path = line.split(":", 1)[1]
        if path == exclude_path:
            continue
        if "/tests/" in path or path.startswith("tests/"):
            continue
        if path.endswith(".pyi") or path.endswith(".pyi.in"):
            continue
        hits.append(path)
    return sorted(set(hits))[:5]


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
    drifts += detect_buffer_api_migration(args.vllm_path, args.vllm_ref)
    drifts += detect_renames(args.vllm_path, args.vllm_ref)
    drifts += detect_sig_changes(args.vllm_path, args.vllm_ref)
    print(f"      found {len(drifts)} raw drifts "
          f"(removed={sum(1 for d in drifts if d.kind == 'removed_symbol')}, "
          f"buffer_api={sum(1 for d in drifts if d.kind == 'buffer_api_migration')}, "
          f"renamed={sum(1 for d in drifts if d.kind == 'renamed')}, "
          f"sig_change={sum(1 for d in drifts if d.kind == 'sig_change')})")

    print(f"[2/4] locating call sites in vllm-ascend ...")
    filtered_out: list[Drift] = []
    for d in drifts:
        if ascend_defines_symbol(args.vllm_ascend_path, d.symbol):
            # vllm-ascend has its own class/def with this name — likely
            # a false positive. Still grep but require a `from vllm...`
            # import in the same file.
            d.ascend_callsites = grep_ascend_for_symbol(
                args.vllm_ascend_path, d.symbol, require_vllm_import=True)
            if not d.ascend_callsites:
                filtered_out.append(d)
        else:
            d.ascend_callsites = grep_ascend_for_symbol(
                args.vllm_ascend_path, d.symbol, require_vllm_import=False)
    impactful = [d for d in drifts if d.ascend_callsites]
    print(f"      {len(impactful)} drifts impact vllm-ascend "
          f"({len(filtered_out)} filtered as internal-name collision)")

    print(f"[3/4] finding new-home candidates for F1 drifts ...")
    for d in impactful:
        if d.kind == "removed_symbol":
            d.new_home_candidates = find_new_home_candidates(
                args.vllm_path, args.vllm_ref, d.symbol, d.vllm_path)

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
        "",
        "## ⚠ Detector coverage",
        "",
        "This scanner only catches the following drift families automatically:",
    ]
    for rule in FAMILY_RULES:
        lines.append(f"- **{rule.family}** — {rule.description}")
    lines.append("")
    lines.append("NOT auto-detected (still require manual KB §F{N} inspection):")
    for fid, desc in UNDETECTED_FAMILIES:
        lines.append(f"- **{fid}** — {desc}")
    lines.append("")
    lines.append("A clean `impact_ascend: 0` therefore does NOT mean no drift — only no F1/F2-rename/F3 drift.")
    lines.append("\n---\n")
    for d in impactful:
        lines.append(f"## [{d.matched_family or 'UNMATCHED'}] {d.symbol} — {d.kind}\n")
        lines.append(f"- vllm path (removed/changed): `{d.vllm_path}`")
        lines.append(f"- detail: {d.detail}")
        if d.new_home_candidates:
            lines.append("- **new-home candidates on target ref** (for F1 fallback import):")
            for p in d.new_home_candidates:
                lines.append(f"  - `{p}`")
        elif d.kind == "removed_symbol":
            lines.append("- new-home candidates: NONE FOUND (symbol truly gone — may be F1-real-removal, not F2-path-move)")
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
        "detector_covers": [r.family for r in FAMILY_RULES],
        "undetected_families": [f for f, _ in UNDETECTED_FAMILIES],
        "proposal_path": str(proposal_path),
        "drifts": [
            {
                "symbol": d.symbol,
                "kind": d.kind,
                "vllm_path": d.vllm_path,
                "family": d.matched_family,
                "ascend_sites": len(d.ascend_callsites),
                "new_home_candidates": d.new_home_candidates,
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
