#!/usr/bin/env python3
"""F7/F8 detector — new attributes / methods on vllm base classes that
vllm-ascend subclasses.

Uses Python's `ast` module to parse each subclassed parent class at
baseline vs target tag. Reports additions of PUBLIC attrs and methods,
which are the ones NPU subclasses may need to override.

Usage:
  check_f7_f8.py --vllm-path <community-vllm> \\
                 --vllm-ascend-path <vllm-ascend> \\
                 --baseline-tag <e.g. v0.20.0> \\
                 --target-tag <e.g. origin/main> \\
                 --out /tmp/f78_scan.json

Exit codes:
  0 = no F7/F8 adds on subclassed classes
  1 = one or more additions detected
  2 = usage / harness error
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
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True, check=False)
    return r.stdout


def show_file_at_ref(vllm_path, ref, path):
    return run_git(vllm_path, "show", f"{ref}:{path}")


def get_ascend_subclassed_parents(vllm_ascend_path):
    parents = set()
    out = subprocess.run(
        ["grep", "-rhE", r"^class [A-Za-z_]+\(", "--include=*.py",
         str(Path(vllm_ascend_path) / "vllm_ascend")],
        capture_output=True, text=True, check=False,
    )
    for line in out.stdout.splitlines():
        m = re.match(r"class [A-Za-z_]+\(([^)]+)\)", line)
        if not m:
            continue
        for p in m.group(1).split(","):
            p = p.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p):
                if p in ("object", "ABC", "Enum", "Exception", "str", "int",
                         "list", "dict", "tuple", "BaseModel", "NamedTuple"):
                    continue
                if p.startswith(("Ascend", "Npu", "NPU")):
                    continue
                parents.add(p)
    return parents


def find_class_definitions(vllm_path, ref, class_names):
    if not class_names:
        return {}
    pattern = r"^class (" + "|".join(re.escape(c) for c in class_names) + r")\b"
    r = subprocess.run(
        ["git", "-C", str(vllm_path), "grep", "-lE",
         pattern, ref, "--", "vllm/"],
        capture_output=True, text=True, check=False,
    )
    found = defaultdict(list)
    for line in r.stdout.splitlines():
        if ":" not in line:
            continue
        path = line.split(":", 1)[1]
        if "/tests/" in path or path.startswith("tests/"):
            continue
        if path.endswith(".pyi") or path.endswith(".pyi.in"):
            continue
        src = show_file_at_ref(vllm_path, ref, path)
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in class_names:
                found[node.name].append((path, node))
    return found


def class_members(class_node):
    attrs = set()
    methods = set()
    for stmt in class_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = stmt.name
            if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            methods.add(name)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    attrs.add(t.id)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and not stmt.target.id.startswith("_"):
                attrs.add(stmt.target.id)
    return attrs, methods


def diff_class(baseline_members, target_members):
    base_attrs, base_methods = baseline_members
    target_attrs, target_methods = target_members
    return target_attrs - base_attrs, target_methods - base_methods


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-path", required=True, type=Path)
    p.add_argument("--vllm-ascend-path", required=True, type=Path)
    p.add_argument("--baseline-tag", required=True)
    p.add_argument("--target-tag", required=True)
    p.add_argument("--out", type=Path, default=Path("/tmp/f78_scan.json"))
    args = p.parse_args()

    parents = get_ascend_subclassed_parents(args.vllm_ascend_path)
    print(f"# {len(parents)} parent classes subclassed by vllm-ascend",
          file=sys.stderr)

    print(f"# Parsing baseline ({args.baseline_tag}) ...", file=sys.stderr)
    base_defs = find_class_definitions(args.vllm_path, args.baseline_tag, parents)
    print(f"# Parsing target ({args.target_tag}) ...", file=sys.stderr)
    target_defs = find_class_definitions(args.vllm_path, args.target_tag, parents)

    findings = []
    for cls in sorted(parents):
        base_hits = base_defs.get(cls, [])
        target_hits = target_defs.get(cls, [])
        if not base_hits or not target_hits:
            continue
        _, base_node = base_hits[0]
        target_path, target_node = target_hits[0]
        base_members = class_members(base_node)
        target_members = class_members(target_node)
        new_attrs, new_methods = diff_class(base_members, target_members)
        if new_attrs or new_methods:
            findings.append({
                "class": cls,
                "target_path": target_path,
                "new_attrs": sorted(new_attrs),
                "new_methods": sorted(new_methods),
            })

    print(f"\n# F7/F8 scan {args.baseline_tag} -> {args.target_tag}")
    print(f"Parent classes compared: {len(parents)}")
    print(f"Classes with additions:  {len(findings)}\n")

    for f in findings:
        print(f"- {f['class']}  @  {f['target_path']}")
        if f["new_attrs"]:
            print(f"    F7 new attrs: {', '.join(f['new_attrs'])}")
        if f["new_methods"]:
            print(f"    F8 new methods: {', '.join(f['new_methods'])}")

    with open(args.out, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\n# Wrote {args.out}", file=sys.stderr)
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
