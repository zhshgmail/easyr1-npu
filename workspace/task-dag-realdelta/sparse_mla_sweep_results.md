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
