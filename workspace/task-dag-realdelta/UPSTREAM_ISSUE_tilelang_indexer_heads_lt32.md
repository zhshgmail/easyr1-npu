# [DRAFT — held for owner to file at tile-ai/tilelang-mlir-ascend; no agent signature]

**Title**: `fp8_lighting_indexer` produces wrong output for head dim `h < 32` (assert_close fails)

## Description

`examples/fp8_lighting_indexer.py` fails its own precision check (`assert_close` vs the PyTorch
reference) when the head dim `--h` is below 32. It passes at the default `h=32`. The mismatch is a
consistent ~24.7% of elements across `h ∈ {4, 8, 16}`.

## Reproduction

Environment: MLIR/bishengir backend (`target='npuir'`), Ascend A3 (Ascend910_9382), CANN 8.5.2,
tilelang-mlir-ascend v0.1.1.030. Defaults: b=2, m=2048, n=4096, k=64, bs=64.

```
python examples/fp8_lighting_indexer.py          # h=32 default -> PASS
python examples/fp8_lighting_indexer.py --h 16   # FAIL: ~24.7% elements mismatch
python examples/fp8_lighting_indexer.py --h 8    # FAIL: ~24.7%
python examples/fp8_lighting_indexer.py --h 4    # FAIL: ~24.8%
```

The pass/fail boundary is between h=16 (fail) and h=32 (pass). The mismatch fraction is roughly
constant (~24.7%) for all tested h<32, which suggests a fixed block of output lanes is mis-handled
rather than a magnitude-scaling error.

## Impact

Any indexer use with head dim < 32 returns incorrect output. Unlike a silent-wrong case, the example's
own `assert_close` catches this — so a consumer using the example's validation will at least see the
failure rather than trusting bad output.

## Note

A sibling head-count-dependent correctness issue exists in `sparse_mla_fwd` (wrong for heads<16,
silently — separate issue). The two have different thresholds (sparse_mla: <16; indexer: <32) and
different severity (sparse_mla is silent), suggesting independent root causes in their respective
head-padding / tiling paths rather than one shared bug.

## Suggested fix

Investigate the h<32 tiling / head-padding path; either correct the output for h<32 or assert-refuse
unsupported head dims rather than computing a wrong (assert-failing) result.

---

## Diagnostic update (2026-06-02) — NOT yet fixed; root cause harder than sparse_mla

Unlike sparse_mla (clean output-store over-write, fixed), the indexer's failure is **strongly
shape-dependent**, which rules out a simple per-head error:
- h=16, m=2048/n=4096 (default): 24.7% mismatch
- h=16, m=128/n=512: **99.1% mismatch**

The fraction-wrong scaling with problem shape points at the kernel **work distribution**
(`T.ceildiv(m_num*n_num, nums_kernel)` task assignment) and/or workspace aliasing interacting with H,
rather than the head-reduction itself. Fewer logic-kernels (smaller shape) → more output wrong. This
is a deeper tiling/work-assignment bug; a confident fix needs analysis of the cube/vector task split +
workspace layout for H<32. Not fixed yet — partial diagnosis only.
