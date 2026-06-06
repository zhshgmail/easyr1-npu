# M3 CANN-native probe scripts (2026-06-05, A3 real-hardware)

These are the exact scripts blue ran on A3 (`115.190.166.102`, container `miles_v4_npu`,
davinci0, torch_npu 2.9.0) to validate the DSv4 CANN-native ops. Results +
findings in `../RESULT_M3_nsa_select_attention_e2e_2026-06-05.md`.

Run inside an NPU container: `cd <dir> && python3 m3_probe3.py` (etc).

| script | what it validates |
|---|---|
| `m3_probe3.py` | `npu_nsa_select_attention` fwd (TND layout) — attn + softmax max/sum, finite |
| `m3_bwd2.py` | `npu_nsa_select_attention_grad` bwd (correct arg-order) — dq/dk/dv finite |
| `m3_ci2.py` | `npu_nsa_compress_attention` fwd (TND) + `npu_lightning_indexer` fwd (BSND) |
| `m3_idx3.py` | indexer values diagnosis (`-inf` = correct mode-3 causal mask, not a bug) |
| `m3_more.py` | compress/indexer grad schemas + indexer tuned fwd |

NOTE (for the picking-up agent): the probe stage is DONE. Start from attn_sink adaptation
+ compress/indexer bwd + numerical-vs-reference + dispatcher + UT (see RESULT doc §"Remaining M3").
