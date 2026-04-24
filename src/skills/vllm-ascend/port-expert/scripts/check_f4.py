#!/usr/bin/env python3
"""F4 detector — return-type annotation changes on functions/methods
that vllm-ascend calls.

Uses AST to parse function signatures at baseline vs target tag, diffs
the `returns` annotation. Common F4 shapes:
  - scalar (e.g. `int`, `float`, `Tensor`) -> NamedTuple / dataclass
  - None -> something
  - Tuple[...] -> NamedTuple with same shape

Since annotation-level changes may be purely cosmetic (alias renaming),
each finding is flagged F4-suspect.

Usage:
  check_f4.py --vllm-path <community-vllm> \\
              --vllm-ascend-path <vllm-ascend> \\
              --baseline-tag v0.20.0 \\
              --target-tag origin/main \\
              --out /tmp/f4_scan.json

Exit codes:
  0 = no F4-suspect findings
  1 = findings exist
  2 = usage error
"""
from __future__ import annotations
import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run_git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False).stdout


def show_file_at_ref(vllm_path, ref, path):
    return run_git(vllm_path, "show", f"{ref}:{path}")


def scan_ascend_for_vllm_calls(vllm_ascend_path):
    """Return {module: {func, ...}} — map of vllm modules and function
    names imported by vllm-ascend source."""
    out = subprocess.run(
        ["grep", "-rhE",
         r"^from vllm(\.[A-Za-z0-9_.]+)? import",
         "--include=*.py",
         str(Path(vllm_ascend_path) / "vllm_ascend")],
        capture_output=True, text=True, check=False,
    )
    imports = defaultdict(set)
    for line in out.stdout.splitlines():
        m = re.match(r"from (vllm(?:\.[\w.]+)?) import (.+)", line)
        if not m:
            continue
        mod, rhs = m.group(1), m.group(2)
        rhs = re.sub(r"[()]", "", rhs).strip().rstrip("\\")
        for part in rhs.split(","):
            name = part.split(" as ")[0].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                imports[mod].add(name)
    return imports


def ann_string(node):
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


def normalize_ann(s):
    s = re.sub(r"Union\[([^\[\]]+)\]",
               lambda m: "|".join(x.strip() for x in m.group(1).split(",")),
               s)
    s = re.sub(r"Optional\[([^\[\]]+)\]",
               lambda m: f"{m.group(1).strip()}|None",
               s)
    s = re.sub(r"\s*\|\s*", "|", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def collect_function_returns(vllm_path, ref, modules_to_funcs):
    findings = {}
    for mod, funcs in modules_to_funcs.items():
        rel = mod[len("vllm."):].replace(".", "/") if mod != "vllm" else ""
        candidates = []
        if rel:
            candidates.append(f"vllm/{rel}.py")
            candidates.append(f"vllm/{rel}/__init__.py")
        else:
            candidates.append("vllm/__init__.py")
        for path in candidates:
            src = show_file_at_ref(vllm_path, ref, path)
            if not src:
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in funcs:
                        findings[f"{mod}::{node.name}"] = ann_string(node.returns)
            break
    return findings


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-path", required=True, type=Path)
    p.add_argument("--vllm-ascend-path", required=True, type=Path)
    p.add_argument("--baseline-tag", required=True)
    p.add_argument("--target-tag", required=True)
    p.add_argument("--out", type=Path, default=Path("/tmp/f4_scan.json"))
    args = p.parse_args()

    imports = scan_ascend_for_vllm_calls(args.vllm_ascend_path)
    print(f"# {sum(len(v) for v in imports.values())} (module, func) pairs "
          f"from vllm-ascend imports", file=sys.stderr)

    print(f"# Baseline ({args.baseline_tag}) ...", file=sys.stderr)
    base = collect_function_returns(args.vllm_path, args.baseline_tag, imports)
    print(f"# Target   ({args.target_tag}) ...", file=sys.stderr)
    target = collect_function_returns(args.vllm_path, args.target_tag, imports)

    findings = []
    for key, b in base.items():
        t = target.get(key)
        if t is None:
            continue
        if b == t:
            continue
        if normalize_ann(b) == normalize_ann(t):
            continue
        findings.append({
            "key": key,
            "baseline_return": b,
            "target_return": t,
        })

    print(f"\n# F4 scan {args.baseline_tag} -> {args.target_tag}")
    print(f"Function-pair annotations checked: {len(base)}")
    print(f"Return-annotation changes:         {len(findings)}\n")

    for f in findings:
        print(f"- `{f['key']}`")
        print(f"    baseline: {f['baseline_return']}")
        print(f"    target:   {f['target_return']}")

    with open(args.out, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\n# Wrote {args.out}", file=sys.stderr)
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
