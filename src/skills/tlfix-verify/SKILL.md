---
name: tlfix-verify
description: Phase 4 of /tilelang-fix. Regression-test a patched tilelang-mlir-ascend tree to confirm the recipe applied by /tlfix-patch did not break previously-PASSing ops. Reruns a sample (default N=5-10) plus the originally-failing op, then emits verify.json with diff vs the pre-patch sweep.
---

# /tlfix-verify — Phase 4 (regression test after patch)

Standalone runnable; called by `/tilelang-fix` orchestrator.

## Inputs

- `tilelang_dir`: tilelang-mlir-ascend checkout (already patched + rebuilt
  by `/tlfix-patch`)
- `target_op`: the originally-failing op the patch was meant to fix
  (required — confirms forward progress)
- `pre_sweep_results`: pre-patch `results.json` from `/tlfix-sweep`
  (used to pick the regression sample + detect new failures)
- `out_dir`: where to write `verify.json`
- `--sample-size N`: number of previously-PASS ops to re-run as regression
  sample (default 8). Sample is stratified across `examples/`,
  `testing/npuir/`, `unittest/npuir/` if those subdirs exist.
- `--sample-seed S`: deterministic seed (default 0) so reruns are
  reproducible across sessions
- `--mode {expert,developer,both}`: same semantics as `/tlfix-sweep`

## Verdict rules

| Verdict | Criteria |
|---------|----------|
| `PASS` | target_op now PASS AND no PASS→FAIL regression in sample |
| `PARTIAL` | target_op now PASS AND ≥1 sample op flipped PASS→FAIL |
| `NO_PROGRESS` | target_op still FAIL with same signature |
| `WORSE` | target_op FAIL with different signature OR new compile error in unrelated op |

Any verdict other than `PASS` blocks `/tlfix-kb-update` from
sedimenting the recipe — orchestrator must surface it back to the user
for manual review.

## Output schema

`out_dir/verify.json`:
```jsonc
{
  "target_op": "examples/deepseek_v4/example_fp8_gemm_kernel.py",
  "pre_status": "COMPILE_FAIL_MLIR_VERIFIER",
  "post_status": "PASS",
  "verdict": "PASS",
  "sample": [
    {"op": "examples/elementwise/vec_add_2d.py",
     "pre": "PASS", "post": "PASS"},
    {"op": "testing/npuir/test_layernorm.py",
     "pre": "PASS", "post": "PASS"},
    ...
  ],
  "regressions": [],
  "elapsed_s": 412
}
```

## How to invoke

```bash
bash run.sh <tilelang_dir> <target_op> <pre_sweep_results> <out_dir> \
  [--sample-size N] [--sample-seed S] [--mode {expert,developer,both}]
```

## See also

- `tlfix-patch/SKILL.md` — produces our input (patched tree)
- `tlfix-kb-update/SKILL.md` — gated by our verdict
- `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md` §13.2 — BOTH-mode
  regression discipline this skill enforces
