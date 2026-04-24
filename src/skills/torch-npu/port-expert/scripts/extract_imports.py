#!/usr/bin/env python3
"""Extract (private_module, symbol) pairs imported by torch_npu.

Handles:
  from torch._x.y import A
  from torch._x.y import A, B, C
  from torch._x.y import (A, B, C)

Usage:
  extract_imports.py [--root <torch_npu-tree>]

Default ROOT is the local A3-repo checkout; override for CI or other hosts.
"""
import argparse
import os
import re
from collections import defaultdict

_ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--root", default=os.environ.get(
    "TORCH_NPU_ROOT",
    "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu",
))
_args = _ap.parse_args()
ROOT = _args.root

# Match "from torch._xxx import ..." capturing (module, rest)
# rest may be "A", "A, B", "(\nA,\nB,\n)"
IMPORT_RE = re.compile(r"^\s*from\s+(torch\._[A-Za-z0-9_.]*)\s+import\s+(.*?)$", re.MULTILINE)
# For multiline parenthesized imports, we need to glue
PAREN_IMPORT_RE = re.compile(
    r"^\s*from\s+(torch\._[A-Za-z0-9_.]*)\s+import\s*\((.*?)\)",
    re.MULTILINE | re.DOTALL,
)

pairs = defaultdict(lambda: defaultdict(list))  # module -> symbol -> [files]

for dirpath, _, files in os.walk(ROOT):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        fp = os.path.join(dirpath, fn)
        with open(fp, "r", errors="replace") as f:
            text = f.read()

        # First get paren imports
        consumed_spans = []
        for m in PAREN_IMPORT_RE.finditer(text):
            mod = m.group(1)
            body = m.group(2)
            # strip comments and whitespace
            body = re.sub(r"#.*", "", body)
            syms = [s.strip() for s in body.split(",")]
            syms = [s.split(" as ")[0].strip() for s in syms if s.strip()]
            for s in syms:
                if s.isidentifier():
                    pairs[mod][s].append(fp)
            consumed_spans.append((m.start(), m.end()))

        # Now single-line imports on remaining (simple: run same regex and filter dupes)
        for m in IMPORT_RE.finditer(text):
            # Skip if inside a paren span
            inside = any(s <= m.start() < e for s, e in consumed_spans)
            if inside:
                continue
            mod = m.group(1)
            body = m.group(2).strip()
            # Must not contain open paren (that was handled above)
            if "(" in body:
                continue
            # Strip trailing comment
            body = re.sub(r"\s*#.*", "", body)
            # Strip trailing backslash continuation (we'd lose lines, but count what we see)
            body = body.rstrip("\\ ").strip()
            if not body:
                continue
            syms = [s.strip() for s in body.split(",")]
            syms = [s.split(" as ")[0].strip() for s in syms if s.strip()]
            for s in syms:
                if s.isidentifier():
                    pairs[mod][s].append(fp)

# Output: sorted by module, with symbol counts
for mod in sorted(pairs.keys()):
    sym_counts = pairs[mod]
    n_files = sum(len(set(files)) for files in sym_counts.values())
    print(f"\n## {mod} (unique symbols: {len(sym_counts)}, total sites: {n_files})")
    for sym in sorted(sym_counts.keys()):
        files = sorted(set(sym_counts[sym]))
        print(f"  {sym}  ({len(files)} file{'s' if len(files) != 1 else ''})")
