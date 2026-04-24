#!/usr/bin/env python3
"""Refined drift detector for torch private-API imports in torch_npu.

For each (mod, sym) pair that torch_npu imports from `torch._*`:
1. Check if sym exists at mod's original path in the BASELINE torch tag
   (the tag torch_npu currently builds against).
2. Check if sym exists at mod's original path in the TARGET torch tag.
3. If exists in baseline but not in target -> DRIFT.
4. If drifted, locate new home in target tree via grep.

Usage:
  check_drift.py --pt-repo <pytorch-community-checkout> \\
                 --pairs-file <output-of-extract_imports.py> \\
                 --baseline-tag v2.11.0 \\
                 --target-tag v2.12.0-rc3 \\
                 --out /tmp/drift_scan.json

Exit codes:
  0 = no drifts
  1 = drifts found (proceed to per-family classification + shim)
  2 = usage / checkout error
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict


def mod_to_fspath(mod):
    """torch._inductor.utils -> _inductor/utils"""
    return mod[len("torch."):].replace(".", "/")


def read_pairs(path):
    """Read output of extract_imports.py into {mod: [sym, ...]}."""
    pairs = defaultdict(list)
    with open(path) as f:
        current_mod = None
        for line in f:
            line = line.rstrip()
            m = re.match(r"##\s+(\S+)", line)
            if m:
                current_mod = m.group(1)
                continue
            m = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)\s+\(", line)
            if m and current_mod:
                pairs[current_mod].append(m.group(1))
    return pairs


def checkout(repo, ref):
    r = subprocess.run(["git", "checkout", ref], cwd=repo, check=False,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR: git checkout {ref} failed in {repo}: {r.stderr}",
              file=sys.stderr)
        sys.exit(2)


def symbol_exists(repo, mod, sym):
    """Return 'at-original' | 'submodule' | 'not-here' | 'mod-gone'."""
    rel = mod_to_fspath(mod)
    file_py = os.path.join(repo, "torch", rel + ".py")
    dir_init = os.path.join(repo, "torch", rel, "__init__.py")
    submod_py = os.path.join(repo, "torch", rel, sym + ".py")
    submod_dir = os.path.join(repo, "torch", rel, sym, "__init__.py")

    if os.path.exists(submod_py) or os.path.exists(submod_dir):
        return "submodule"

    for path in [file_py, dir_init]:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        patterns = [
            rf"^class\s+{sym}\b",
            rf"^def\s+{sym}\s*\(",
            rf"^async\s+def\s+{sym}\s*\(",
            rf"^{sym}\s*=",
            rf"^{sym}\s*:\s*[^=\n]+=",
            rf"^from\s+\S+\s+import\s+.*\b{sym}\b",
            rf"^from\s+\S+\s+import\s+\([^)]*\b{sym}\b",
            rf"^import\s+\S+\s+as\s+{sym}\b",
        ]
        for p in patterns:
            if re.search(p, text, re.MULTILINE):
                return "at-original"

    if not (os.path.exists(file_py) or os.path.exists(dir_init)):
        return "mod-gone"
    return "not-here"


def scan(repo, ref, pairs):
    checkout(repo, ref)
    results = {}
    for mod, syms in pairs.items():
        for sym in syms:
            key = f"{mod}::{sym}"
            results[key] = symbol_exists(repo, mod, sym)
    return results


def find_new_home_via_grep(repo, sym):
    """Find files in torch/ where `sym` is class/def/assignment."""
    r = subprocess.run(
        ["grep", "-rlE",
         rf"^(class|def|async def) {sym}( |\(|:)",
         os.path.join(repo, "torch"), "--include=*.py"],
        capture_output=True, text=True, timeout=30,
    )
    return [h for h in r.stdout.splitlines() if h]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt-repo", required=True,
                   help="community pytorch checkout")
    p.add_argument("--pairs-file", required=True,
                   help="output of extract_imports.py")
    p.add_argument("--baseline-tag", required=True,
                   help="torch tag torch_npu was built against (e.g. v2.11.0)")
    p.add_argument("--target-tag", required=True,
                   help="torch tag to port to (e.g. v2.12.0-rc3)")
    p.add_argument("--out", default="/tmp/drift_scan.json")
    args = p.parse_args()

    pairs = read_pairs(args.pairs_file)
    print(f"# Scanning {args.baseline_tag} ...", file=sys.stderr)
    res_base = scan(args.pt_repo, args.baseline_tag, pairs)
    print(f"# Scanning {args.target_tag} ...", file=sys.stderr)
    res_target = scan(args.pt_repo, args.target_tag, pairs)

    drifts = []
    for key, sb in res_base.items():
        st = res_target[key]
        if sb in ("at-original", "submodule") and st not in ("at-original", "submodule"):
            drifts.append({"key": key, "baseline": sb, "target": st})

    for d in drifts:
        _mod, sym = d["key"].split("::")
        d["new_homes"] = find_new_home_via_grep(args.pt_repo, sym)

    total = sum(len(v) for v in pairs.values())
    print(f"\n# Drift scan {args.baseline_tag} -> {args.target_tag}")
    print(f"Total (mod,sym) pairs scanned: {total}")
    print(f"Drifts (in-baseline but out-of-target): {len(drifts)}\n")

    for d in drifts:
        print(f"- `{d['key']}`  ({args.baseline_tag}:{d['baseline']} -> "
              f"{args.target_tag}:{d['target']})")
        for h in sorted(set(d.get("new_homes", [])))[:6]:
            short = h.replace(args.pt_repo + "/torch/", "torch/")
            print(f"    - {short}")

    with open(args.out, "w") as f:
        json.dump(drifts, f, indent=2)
    print(f"\n# Wrote {args.out}", file=sys.stderr)

    return 0 if not drifts else 1


if __name__ == "__main__":
    sys.exit(main())
