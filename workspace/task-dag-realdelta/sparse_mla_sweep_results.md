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
