#!/usr/bin/env python3
"""F7/F8 detector for torch_npu — new attrs / methods on torch base
classes that torch_npu subclasses.

Sister of vllm-ascend port-expert's check_f7_f8.py, same AST approach,
adapted for torch private-API classes.

Usage:
  check_f7_f8.py --torch-path <community-pytorch-checkout> \\
                 --torch-npu-path <torch-npu-checkout> \\
                 --baseline-tag v2.11.0 \\
                 --target-tag v2.12.0-rc3 \\
                 --out /tmp/torch_f78_scan.json

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


def show_file_at_ref(torch_path, ref, path):
    return run_git(torch_path, "show", f"{ref}:{path}")


def get_torch_npu_subclassed_parents(torch_npu_path):
    """Return parent class names that torch_npu subclasses AND imports
    from torch (not redefined locally).

    Scanning name-only yields false positives when torch_npu has a
    local shadow class with the same name as an upstream torch class
    (example: torch_npu/npu/_graph_tree.py:706 has its own `class
    OutputAliasInfo` that is unrelated to `torch._functorch.*.OutputAliasInfo`).
    """
    parents = set()
    out = subprocess.run(
        ["grep", "-rhE", r"^class [A-Za-z_]+\(", "--include=*.py",
         str(Path(torch_npu_path) / "torch_npu")],
        capture_output=True, text=True, check=False,
    )
    subclassed_names = set()
    for line in out.stdout.splitlines():
        m = re.match(r"class [A-Za-z_]+\(([^)]+)\)", line)
        if not m:
            continue
        for p in m.group(1).split(","):
            p = p.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p):
                if p in ("object", "ABC", "Enum", "Exception", "str", "int",
                         "list", "dict", "tuple", "NamedTuple"):
                    continue
                if p.startswith(("Ascend", "Npu", "NPU")):
                    continue
                subclassed_names.add(p)

    # Now require evidence the name is imported from torch (not locally defined).
    # Scan Python files, handle both single-line and parenthesized multi-line imports.
    root = Path(torch_npu_path) / "torch_npu"
    import_sources = defaultdict(set)
    for py in root.rglob("*.py"):
        try:
            src = py.read_text(errors="replace")
        except OSError:
            continue
        # Single-line: from torch.X import A, B, C
        for m in re.finditer(
            r"^from (torch(?:\.[A-Za-z0-9_.]+)?)\s+import\s+([^\n#(]+)",
            src, re.MULTILINE,
        ):
            mod = m.group(1)
            names = [n.split(" as ")[0].strip() for n in m.group(2).split(",")]
            for n in names:
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n):
                    import_sources[n].add(mod)
        # Multi-line: from torch.X import (A, B, C)
        for m in re.finditer(
            r"^from (torch(?:\.[A-Za-z0-9_.]+)?)\s+import\s*\(([^)]*)\)",
            src, re.MULTILINE | re.DOTALL,
        ):
            mod = m.group(1)
            body = re.sub(r"#.*", "", m.group(2))
            names = [n.split(" as ")[0].strip() for n in body.split(",")]
            for n in names:
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n):
                    import_sources[n].add(mod)

    for name in subclassed_names:
        if name in import_sources:
            parents.add(name)
    return parents


def find_class_definitions(torch_path, ref, class_names):
    if not class_names:
        return {}
    pattern = r"^class (" + "|".join(re.escape(c) for c in class_names) + r")\b"
    r = subprocess.run(
        ["git", "-C", str(torch_path), "grep", "-lE",
         pattern, ref, "--", "torch/"],
        capture_output=True, text=True, check=False,
    )
    found = defaultdict(list)
    for line in r.stdout.splitlines():
        if ":" not in line:
            continue
        path = line.split(":", 1)[1]
        if "/test/" in path or path.startswith("test/"):
            continue
        if path.endswith(".pyi") or path.endswith(".pyi.in"):
            continue
        src = show_file_at_ref(torch_path, ref, path)
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
    p.add_argument("--pt-repo", dest="torch_path", required=True, type=Path,
                   help="community pytorch checkout (alias: --torch-path)")
    p.add_argument("--torch-path", dest="torch_path", type=Path,
                   help=argparse.SUPPRESS)  # accepted for backward-compat
    p.add_argument("--torch-npu-path", required=True, type=Path,
                   help="torch_npu checkout")
    p.add_argument("--baseline-tag", required=True)
    p.add_argument("--target-tag", required=True)
    p.add_argument("--out", type=Path, default=Path("/tmp/torch_f78_scan.json"))
    args = p.parse_args()

    parents = get_torch_npu_subclassed_parents(args.torch_npu_path)
    print(f"# {len(parents)} parent classes subclassed by torch_npu",
          file=sys.stderr)

    print(f"# Parsing baseline ({args.baseline_tag}) ...", file=sys.stderr)
    base_defs = find_class_definitions(args.torch_path, args.baseline_tag, parents)
    print(f"# Parsing target ({args.target_tag}) ...", file=sys.stderr)
    target_defs = find_class_definitions(args.torch_path, args.target_tag, parents)

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
