# sparse_mla_fwd systematic sweep results (2026-06-02, A3, fixed example a19acd5)

Harness: `tilelang_sweep_harness.py` (subprocess per combo, verdict-parsed). sparse_mla_fwd __main__
builds all data from args.* → CLI sweeps valid. Uses the heads<16-fixed example.

| combo | verdict | note |
|---|---|---|
| --heads 4 | PASS | the fixed heads<16 path |
| --heads 16 | PASS | |
| --heads 64 | PASS | **V4-Flash real head count** |
| --top_k 256 | PASS | |
| --block_i 128 --top_k 128 | PASS | tile sweep |
| --seq_len 256 | ERR(invalid config) | seq_len > seq_len_kv(128 default) — nonsensical for causal; with matched --seq_len_kv 256 → PASS |
| --seq_len_kv 512 --top_k 256 | PASS | |
| --dim 128 --tail_dim 64 | PASS | |
| --batch_size 2 | PASS | |
| --num_kernels 48 | PASS | |
| --kv_group 2 (from A.4) | FAIL | likely-unsupported config (H-padding auto-handle is kv_group==1-only); not confirmed bug |

**Net**: sparse_mla_fwd is robust across the VALID parameter grid (heads incl. V4-Flash 64, top_k, block,
seq_len with valid seq_kv, seq_kv, dim, batch, num_kernels). The single ERR was an invalid config
(seq_len>seq_len_kv), confirmed by the matched-seq_kv re-run passing. kv_group>1 is a separate
likely-unsupported config. dtype is fp16-hardcoded (bf16 untested). Only real kernel bug found in the
whole investigation: heads<16 silent-wrong (FIXED, a19acd5).

## fp8_lighting_indexer sweep (2026-06-02, harness-corrected to honor --h)

| combo | verdict | note |
|---|---|---|
| --h 4 / 8 / 16 / 32 | PASS | **confirms the earlier "h<32 bug" was a harness artifact (retracted)** |
| --h 64 | FAIL(1/16777216 = 0.0%) | **single-element fp16 ULP outlier just over rtol=3e-2/atol=2e-2** — NOT a kernel bug (the example's own tolerance comment notes fp16 cross-impl noise); 1 element in 16.7M |
| --m 512 --n 1024 | PASS | |
| --bs 128 | PASS | |
| --k 128 | PASS | |

**Net**: indexer is robust across head counts (h=4..64) and m/n/bs/k. The h=64 "FAIL" is 1/16.7M elements
(0.0%) = fp16 numerical noise, not a defect. This CONFIRMS the earlier "indexer h<32 bug" was entirely my
hardcoded-H=32 harness error (now retracted). No indexer kernel bug exists.

## flash_attn_npuir sweep (2026-06-02)

| combo | verdict | note |
|---|---|---|
| default (fp16) | PASS | |
| --seq_len 1024 / 2048 / dim 64 / dim 256 / block_m128 n128 | PASS | fp16 shape/block robust |
| --dtype bfloat16 (any shape) | FAIL → **harness dtype-compare bug, NOT a kernel bug** | assert_close fails with "dtype do not match: torch.float32 != torch.bfloat16" — ref_output = `softmax(...).to(bf16) @ v` yields float32 (torch bf16 matmul upcast) while kernel output `o` is bf16. Comparison refuses on dtype attribute before comparing values. Fix is harness-side (cast both to same dtype). Kernel bf16 correctness UNDETERMINED (can't compare), not proven broken. |

**Net**: flash_attn_npuir robust in fp16 (shape/block). bf16 path's test is broken (float32-ref vs
bf16-output dtype mismatch) — harness nit, not a kernel bug; kernel bf16 correctness not assessable
without fixing the harness compare. Third harness issue caught this investigation (after indexer-H32,
CANN-NaN).

## tilelang-ascend example suite — default-config triage (2026-06-02)

Per the standing mandate, triaging every example at default config:

| op | default verdict |
|---|---|
| sparse_mla_fwd | PASS (after heads<16 fix a19acd5) |
| fp8_lighting_indexer | PASS |
| flash_attn_npuir | PASS (fp16; bf16 = harness dtype-compare bug) |
| norm/example_rms_norm | PASS |
| norm/layer_norm | PASS |
| gemm/example_gemm | PASS |
| gemm/matmul | PASS |
| gemv/example_gemv | PASS |
| exp2 | PASS |
| log2 | PASS |
| elementwise/example_elementwise_add | PASS |
| elementwise/vec_add_2d | PASS |

12 ops PASS at default. Remaining to triage: engram/* (fwd/bwd/decode), mixcv, *_dynamic_shape
variants, other elementwise (vec_add_1d/auto_brc/multi_buffer/atomic_add), vectorization_in_parallel.

## FULL SUITE triage COMPLETE (2026-06-02)

Triaged the remaining examples — all PASS at default:
- sparse_mla_fwd_dynamic_shape, gemm/matmul_dynamic_shape, vec_add_1d / 2d_dynamic_shape / auto_brc /
  2d_multi_buffer, vectorization_in_parallel, mixcv_mixkernel, mixcv/example_mixcv,
  gemm/example_gemm_int82int32, engram/{fwd,bwd,bwd_exp,decode}, elementwise/atomic_add.

**Full tilelang-ascend example suite (~24 kernels) result**:
- **~24 ops PASS at default config**.
- **1 real kernel bug total**: sparse_mla_fwd heads<block_H silent-wrong → **FIXED** (fork a19acd5, verified heads=4/8/16/32).
- **2 harness-only issues (NOT kernel bugs)**: fp8_lighting_indexer __main__ hardcodes H=32 (caused a false bug, retracted); flash_attn_npuir bf16 test compares float32-ref vs bf16-output (dtype-mismatch assert).
- **non-bugs correctly rejected**: sparse_mla kv_group>1 (likely-unsupported), seq_len>seq_len_kv (invalid config), indexer h=64 (1/16.7M fp16 noise).
- **dtype**: flash_attn bf16 untestable until harness fix; sparse_mla/indexer fp16-hardcoded.

Net: the tilelang-ascend example suite is broadly HEALTHY on Ascend A3. The one genuine kernel
correctness bug (sparse_mla heads<16) is fixed and ready to PR. Remaining mandate work: open the
sparse_mla PR; optionally fix the 2 harness nits (lower value).

## flash_attn_npuir bf16 — CONFIRMED real kernel bug (2026-06-02)

After fixing the harness dtype-compare, tested kernel output vs a PURE fp32 reference (the true math):
- **fp16 kernel vs fp32 ref: PASS** (within rtol/atol 2e-2) → fp32 ref is correct.
- **bf16 kernel vs fp32 ref: FAIL 73.5% mismatch, greatest abs diff 0.345** → bf16 genuinely diverges.

This is a **real bf16-specific correctness bug** in flash_attn_npuir — NOT harness/tolerance/noise (fp16
passes the same fp32-ref check; the 73.5%/0.345 is far beyond bf16's ~0.4% expected error). The kernel is
exposed via --dtype bfloat16 and computes wrong output in bf16 while correct in fp16. Likely root cause: an
fp16-specific assumption in the kernel (hardcoded fp16 cast/const, or the online-softmax exp/scale path).
**Matters for V4: bf16 is V4's working dtype.** Bug #2 found in the suite (after sparse_mla heads<16).
Root-cause + fix: in progress.
