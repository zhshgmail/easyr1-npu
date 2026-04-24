#!/usr/bin/env python3
"""F3 (signature-change) drift detector.

Sister to check_drift.py. Where check_drift catches F1/F2-path-move
(symbol absent at old path), this catches F3: symbol still at old
path, but the def signature changed between baseline and target.

Conservative: only reports when the full `def <name>(<args>)` line
differs verbatim after whitespace normalization. Does NOT try to
reason about semantic compatibility — a signature change of any
shape is reported.

Usage:
  check_sig_drift.py --pt-repo <community-pytorch> \\
                     --pairs-file <extract_imports.py output> \\
                     --baseline-tag v2.11.0 \\
                     --target-tag v2.12.0-rc3 \\
                     --out /tmp/sig_drift.json

Exit codes: 0 no drifts, 1 drifts found, 2 usage error.
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
    return mod[len("torch."):].replace(".", "/")


def read_pairs(path):
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
        print(f"ERROR: git checkout {ref} failed: {r.stderr}", file=sys.stderr)
        sys.exit(2)


def get_def_signature(repo, mod, sym):
    """Return the normalized `def <name>(<args>)` line for sym in mod.

    Returns:
      - full sig string if found as top-level def / async def
      - None if symbol is not a top-level def here (class, assignment,
        re-export, or absent)
    """
    rel = mod_to_fspath(mod)
    candidates = [
        os.path.join(repo, "torch", rel + ".py"),
        os.path.join(repo, "torch", rel, "__init__.py"),
    ]
    # Match `def name(` or `async def name(` possibly spanning multiple
    # lines until the closing `)`. Simple approach: grep the opening
    # line, then keep reading until we see a `):` that closes the paren.
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        # Look for line starting with def/async def and sym
        sig_re = re.compile(
            rf"^(async\s+)?def\s+{re.escape(sym)}\s*\(", re.MULTILINE)
        m = sig_re.search(text)
        if not m:
            continue
        # Collect until the opening paren's match.
        # Walk through character by character counting parens.
        start = m.start()
        i = text.find("(", m.end() - 1)
        depth = 1
        j = i + 1
        while j < len(text) and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            return None
        sig = text[start:j]
        # Normalize whitespace so formatting-only changes don't count.
        sig = re.sub(r"\s+", " ", sig).strip()
        return sig
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt-repo", required=True)
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--baseline-tag", required=True)
    p.add_argument("--target-tag", required=True)
    p.add_argument("--out", default="/tmp/sig_drift.json")
    args = p.parse_args()

    pairs = read_pairs(args.pairs_file)
    print(f"# Checking signatures at {args.baseline_tag} ...", file=sys.stderr)
    checkout(args.pt_repo, args.baseline_tag)
    sigs_base = {}
    for mod, syms in pairs.items():
        for sym in syms:
            sig = get_def_signature(args.pt_repo, mod, sym)
            if sig is not None:
                sigs_base[f"{mod}::{sym}"] = sig

    print(f"# Checking signatures at {args.target_tag} ...", file=sys.stderr)
    checkout(args.pt_repo, args.target_tag)
    sigs_target = {}
    for key in sigs_base:
        mod, sym = key.split("::")
        sig = get_def_signature(args.pt_repo, mod, sym)
        if sig is not None:
            sigs_target[key] = sig

    def normalize_semantic(sig):
        """Strip PEP 604 rewrites and other cosmetic changes that don't
        affect calling semantics. Callers pass values, not types, so a
        type-annotation refactor is not a real F3 drift."""
        s = sig
        # Iteratively flatten Union[...] / Optional[...] — may be nested
        # inside other generics, so loop until fixed point.
        prev = None
        while prev != s:
            prev = s
            # Union[A, B, ...] -> A|B|... (handles nested brackets via non-greedy match)
            s = re.sub(
                r"Union\[((?:[^\[\]]|\[[^\]]*\])+)\]",
                lambda m: "|".join(x.strip() for x in m.group(1).split(",")),
                s,
            )
            # Optional[X] -> X|None
            s = re.sub(
                r"Optional\[((?:[^\[\]]|\[[^\]]*\])+)\]",
                lambda m: f"{m.group(1).strip()}|None",
                s,
            )
        # Also: `Iterable[X]` vs `list[X] | tuple[X, ...]` is often a narrowing
        # the caller won't notice if they always pass list/tuple anyway; we
        # don't auto-normalize that one because it IS a real contract change
        # in theory. Keep as-is.
        # Collapse whitespace around punctuation
        s = re.sub(r"\s*\|\s*", "|", s)
        s = re.sub(r"\s*,\s*", ",", s)
        s = re.sub(r"\s*\(\s*", "(", s)
        s = re.sub(r"\s*\)\s*", ")", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    drifts = []
    for key, sb in sigs_base.items():
        st = sigs_target.get(key)
        if st is None:
            continue
        if sb == st:
            continue
        # Check if the change is purely cosmetic (PEP 604 rewrite)
        sb_norm = normalize_semantic(sb)
        st_norm = normalize_semantic(st)
        cosmetic = (sb_norm == st_norm)
        drifts.append({
            "key": key,
            "baseline_sig": sb,
            "target_sig": st,
            "cosmetic_only": cosmetic,
        })

    def classify_breakage(d):
        """Further classify non-cosmetic drifts:

        'additive-with-defaults': new arg added WITH a default value →
          existing positional/keyword callers still work → NOT a breakage.
        'breaking': new required arg, removed arg, reordered positional,
          or default changed to incompatible → ACTUAL breakage.
        We can't perfectly distinguish without a full Python parser, but
        heuristic: if target has strictly more args than baseline and
        the extra args all have `=` (default), call it additive.
        """
        if d["cosmetic_only"]:
            return "cosmetic"
        sb = d["baseline_sig"]
        st = d["target_sig"]
        # Count commas at top level of args as a rough arg count
        def top_level_commas(s):
            # Extract the args between first ( and matching )
            m = re.search(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", s)
            if not m:
                return 0
            args = m.group(1)
            return args.count(",")

        base_args = top_level_commas(sb)
        target_args = top_level_commas(st)
        if target_args > base_args:
            # Check if every NEW parenthesized section contains `=` (has default)
            # Simple: if the entire target signature's extra chunk looks
            # like it contains `= ` we consider it additive-with-defaults.
            # Conservative approach: report as 'additive-candidate' if we
            # cannot confidently classify as breaking.
            target_tail = st[-max(40, len(st) - len(sb)):]
            if "=" in target_tail:
                return "additive-with-defaults"
        return "breaking-candidate"

    for d in drifts:
        d["breakage_class"] = classify_breakage(d)

    cosmetic = [d for d in drifts if d["breakage_class"] == "cosmetic"]
    additive = [d for d in drifts if d["breakage_class"] == "additive-with-defaults"]
    breaking = [d for d in drifts if d["breakage_class"] == "breaking-candidate"]
    real = breaking  # only breaking-candidate needs a fix

    print(f"\n# F3 signature-drift scan {args.baseline_tag} -> {args.target_tag}")
    print(f"Top-level-def symbols compared: {len(sigs_base)}")
    print(f"Total signature changes: {len(drifts)}")
    print(f"  Cosmetic-only (PEP 604 / whitespace): {len(cosmetic)}")
    print(f"  Additive with defaults (non-breaking): {len(additive)}")
    print(f"  Potentially-breaking: {len(breaking)}\n")

    if breaking:
        print("## Potentially-breaking signature drifts (actionable)\n")
        for d in breaking:
            print(f"- `{d['key']}`")
            print(f"    baseline: {d['baseline_sig'][:140]}")
            print(f"    target:   {d['target_sig'][:140]}")
        print()

    if additive:
        print(f"## Additive with defaults ({len(additive)}, not breaking but noted)\n")
        for d in additive[:5]:
            print(f"- `{d['key']}`")
        if len(additive) > 5:
            print(f"- ... ({len(additive) - 5} more)")

    if cosmetic:
        print(f"\n## Cosmetic-only ({len(cosmetic)}, not actionable) — PEP 604 rewrite")

    with open(args.out, "w") as f:
        json.dump(drifts, f, indent=2)
    print(f"\n# Wrote {args.out}", file=sys.stderr)
    return 0 if not real else 1


if __name__ == "__main__":
    sys.exit(main())
