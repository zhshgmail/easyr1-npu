---
id: tilelang-001
date: 2026-05-15
layer: tilelang
title: bishengir "ub overflow" is 30s into compile; CheckUBBudget pass surfaces it in <1s with per-alloc breakdown
trigger:
  - "tilelang program compile time-to-failure > 20s with cryptic 'ub overflow'"
  - "tuning block_M / block_H without knowing which alloc is over budget"
  - "porting H=64 / D_V=512 attention to Ascend A3 64KB UB"
  - "writing a CheckUBBudget pass for tilelang-mlir-ascend"
symptom_in_wild:
  - "bishengir-compile errors out after 30s with `ub overflow: requested N exceeds cap`"
  - "no breakdown of which scope/alloc contributed how many KB"
  - "user must guess block_M/H combinations and retry"
  - "compile cost dominates the porting iteration loop"
root_cause: >
  bishengir-compile only checks UB cap at the lowering stage, by which point
  passes have already run for ~30 seconds. The check has no per-allocation
  breakdown or block_M suggestion. For a real-shape DSv4-Flash port that
  requires iterating block sizes 5-10 times, this kills the iteration loop.

  The fix is a tilelang-side Python pass that runs immediately after
  `LowerOpaqueBlock` (so UB scope info is available) and statically estimates
  total UB occupancy.
mistake_pattern: "compiler reports failure too late in pipeline; consumer iteration loop dies of latency"
correction:
  - "PR tile-ai/tilelang-mlir-ascend#80 adds `CheckUBBudget` pass after `LowerOpaqueBlock`"
  - "Counts allocations in `{local, local.fragment}` scopes (`_UB_BACKED_SCOPES`); skips `wmma.matrix_a/_b/_accumulator` because those are mixcv (not UB-backed) and would false-trigger"
  - "Two-tier thresholds: soft budget (80% cap) -> log warning only, never raise; catastrophic (≥2× cap) -> raise with per-alloc breakdown + suggested block_M. Single threshold approach was rejected by user: see memory `feedback_capacity_check_calibration.md`."
  - "Per-alloc breakdown format: `alloc_name (scope) shape dtype size_kb` table sorted desc"
  - "Suggested block_M derived from current block_M * (cap / total). Always rounded down to power of 2."
  - "Test: `tests/passes/test_check_ub_budget.py` -- 4 cases: PASS small, WARN soft, RAISE catastrophic, IGNORE mixcv"
evidence:
  - "PR: https://github.com/tile-ai/tilelang-mlir-ascend/pull/80 (CI green 24m15s test PASS)"
  - "3-commit chain: `daea72f` (original) -> `d2d1871` (ruff F841 + delete stale import) -> `df7431e` (_UB_BACKED_SCOPES tighten + threshold catastrophic-only)"
  - "User feedback Discord 2026-05-15: catastrophic-only threshold avoids mixcv false-positive while still catching real overflows"
  - "miles real-shape port benefited: porting `sparse_mla_bwd` to fit 192 KB cap took 3 iterations with CheckUBBudget vs the prior 30+ minutes of bishengir compile cycles"
---

# tilelang-001 — CheckUBBudget early-fail pass

## Why this matters

The bishengir compile cost (~30s/iter) was the dominant tax during the miles tilelang port. Each block_M / block_H adjustment required a full compile to find out if UB fits. With CheckUBBudget running at npuir stage, the same diagnostic is delivered in <1s with actionable breakdown + suggestion.

This is the canonical example of "early-fail diagnostic at the right layer wins more than perfect downstream tooling": the bishengir error message could in principle have everything CheckUBBudget produces, but adding it there requires cross-binary collab; adding it at tilelang-mlir-ascend (Python) is one PR.

## Calibration lesson (don't skip)

The first version of this pass used a single 80% threshold and raised on overshoot. It caught the target case but false-positived on `wmma.*` allocs (mixcv) that aren't actually UB-backed. The fix:
- Tighten the scope set to `{local, local.fragment}` only (`_UB_BACKED_SCOPES`)
- Split threshold: 80% -> log-warn (informational), 200% -> raise (catastrophic-only)
- The middle band (80%-200%) lets bishengir handle tiling; we don't try to second-guess

Generalized rule: **diagnostic thresholds should be calibrated against real backend capacity, not spec-sheet UB caps**. See memory `feedback_capacity_check_calibration.md`.

## Future direction (not done yet)

If reviewers ask: similar early-fail passes for `LCM` (LCM bank conflict), `register spill`, and `instruction cache pressure` would each save minutes per iteration. The pattern (Python pass at npuir stage, two-tier threshold, per-alloc breakdown) replicates.
