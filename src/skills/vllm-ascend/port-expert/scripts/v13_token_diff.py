#!/usr/bin/env python3
"""V1.3 semantic-vs-formal check — token-by-token diff against baseline.

The existing smoke_validate.sh V1.3 path only does a marker grep
('V1.3 ROLLOUT SMOKE PASSED'). This is a 'formal' check — it proves
that the rollout exited cleanly, NOT that the generated tokens are
correct. On 2026-04-24 we saw a vllm 0.20 overlay pass the marker
while producing nonsense tokens (KV cache was silently wrong).

This script adds a 'semantic' check: run a deterministic set of V1.3
prompts, capture the tokens, and compare against a baseline file.

Usage:
  # first-time baseline capture:
  v13_token_diff.py --mode capture \\
    --log <rollout-stdout-log> \\
    --out <baseline-tokens.json>

  # subsequent runs — diff against baseline:
  v13_token_diff.py --mode diff \\
    --log <new-rollout-stdout-log> \\
    --baseline <baseline-tokens.json> \\
    [--tolerance <float>]

The "baseline-tokens.json" schema:
  {
    "prompt_1": ["tok_a", "tok_b", ..., "tok_N"],
    "prompt_2": [...],
    ...
  }

Diff result:
  - PASS when every (prompt, position) matches identically.
  - PASS-WITH-NOISE when mismatch rate <= tolerance.
  - FAIL otherwise. Exit 1.

Integration: can be called right after smoke_validate.sh V1.3 PASS
as an additional gate. When invoked from the skill, a FAIL at this
stage means: marker passed but output is garbage — classify as
C-patch NOT outcome A.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path


def extract_tokens_per_prompt(log_path):
    """Parse rollout stdout into {prompt_text: [token_strings...]}.

    V1.3 smoke uses a specific log format — the parser below is lenient
    and supports the two forms we've seen:
      (1) `Prompt: <p>\\nGenerated: <tok_a tok_b ...>`
      (2) structured JSON lines: `{"prompt": ..., "completion": ...}`

    Extend this when new smoke harness formats appear.
    """
    text = Path(log_path).read_text(errors="replace")
    per_prompt = {}

    # Format 2: JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = rec.get("prompt")
        c = rec.get("completion") or rec.get("output") or rec.get("generated")
        if p and c:
            tokens = c.split() if isinstance(c, str) else list(c)
            per_prompt[p] = tokens

    if per_prompt:
        return per_prompt

    # Format 1: Prompt/Generated pairs
    pattern = re.compile(
        r"Prompt:\s*(.+?)\s*\n.*?Generated:\s*(.+?)(?:\n\n|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        p = m.group(1).strip()
        tokens = m.group(2).split()
        per_prompt[p] = tokens

    return per_prompt


def diff_against_baseline(new_tokens, baseline, tolerance=0.0):
    """Return (outcome, report_dict).

    outcome in {PASS, PASS-WITH-NOISE, FAIL}.
    """
    total = 0
    mismatches = 0
    missing_prompts = []
    per_prompt_result = {}

    for prompt, baseline_toks in baseline.items():
        if prompt not in new_tokens:
            missing_prompts.append(prompt)
            continue
        new_toks = new_tokens[prompt]
        length = min(len(baseline_toks), len(new_toks))
        prompt_mm = 0
        for i in range(length):
            total += 1
            if baseline_toks[i] != new_toks[i]:
                mismatches += 1
                prompt_mm += 1
        # length-mismatch counts as mismatches too
        len_diff = abs(len(baseline_toks) - len(new_toks))
        total += len_diff
        mismatches += len_diff
        per_prompt_result[prompt] = {
            "baseline_len": len(baseline_toks),
            "new_len": len(new_toks),
            "mismatches": prompt_mm,
            "length_diff": len_diff,
        }

    rate = mismatches / total if total else 1.0

    if missing_prompts:
        outcome = "FAIL"
    elif mismatches == 0:
        outcome = "PASS"
    elif rate <= tolerance:
        outcome = "PASS-WITH-NOISE"
    else:
        outcome = "FAIL"

    report = {
        "outcome": outcome,
        "total_positions": total,
        "mismatches": mismatches,
        "mismatch_rate": round(rate, 4),
        "tolerance": tolerance,
        "missing_prompts": missing_prompts,
        "per_prompt": per_prompt_result,
    }
    return outcome, report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["capture", "diff"], required=True)
    p.add_argument("--log", required=True, type=Path,
                   help="rollout stdout log to parse")
    p.add_argument("--out", type=Path,
                   help="capture mode: write baseline here")
    p.add_argument("--baseline", type=Path,
                   help="diff mode: read baseline from here")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="diff mode: accept up to this mismatch rate")
    args = p.parse_args()

    if not args.log.exists():
        print(f"ERROR: log not found: {args.log}", file=sys.stderr)
        return 2

    tokens = extract_tokens_per_prompt(args.log)
    if not tokens:
        print(f"ERROR: no prompts/tokens found in {args.log}. "
              "Log format may not be supported yet.", file=sys.stderr)
        return 2
    print(f"# Parsed {len(tokens)} prompts, "
          f"{sum(len(v) for v in tokens.values())} total tokens "
          f"from {args.log.name}", file=sys.stderr)

    if args.mode == "capture":
        if not args.out:
            print("ERROR: --mode capture requires --out", file=sys.stderr)
            return 2
        args.out.write_text(json.dumps(tokens, indent=2, ensure_ascii=False))
        print(f"Baseline written: {args.out}")
        return 0

    # diff mode
    if not args.baseline or not args.baseline.exists():
        print("ERROR: --mode diff requires existing --baseline", file=sys.stderr)
        return 2
    baseline = json.loads(args.baseline.read_text())

    outcome, report = diff_against_baseline(tokens, baseline, args.tolerance)
    print(f"\nV1.3 token-diff outcome: {outcome}")
    print(f"  positions: {report['total_positions']}")
    print(f"  mismatches: {report['mismatches']} "
          f"(rate {report['mismatch_rate']}, tol {report['tolerance']})")
    if report["missing_prompts"]:
        print(f"  MISSING prompts: {len(report['missing_prompts'])}")
        for p in report["missing_prompts"][:3]:
            print(f"    - {p[:80]}")
    print()
    return 0 if outcome in ("PASS", "PASS-WITH-NOISE") else 1


if __name__ == "__main__":
    sys.exit(main())
