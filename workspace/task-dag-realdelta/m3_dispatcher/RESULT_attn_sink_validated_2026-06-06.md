# RESULT — M3 attn_sink adapter VALIDATED on A3 (2026-06-06, scan-agent)

The #1 M3 risk (latest-main DSv4 passes `attn_sink[H]fp32` but native
`npu_nsa_select_attention` has no such param) is **solved + empirically validated**:
a closed-form post-hoc rescale of the three outputs the native op already returns.

## Adapter (in `nsa_attn_sink_adapter.py`)

```
o_sink   = o_native * sum_exp / (sum_exp + exp(attn_sink[h] - softmax_max))
lse_sink = softmax_max + log(sum_exp + exp(attn_sink[h] - softmax_max))
```
- `o_native=out0`, `softmax_max=out1`, `softmax_sum=out2` from `npu_nsa_select_attention`.
- **Take `[...,0]` of out1/out2** (the trailing 8-wide is replicated padding) — NEVER sum the 8.
- `attn_sink` is per-head, broadcast to the row dim. `sink-max` underflowing to 0 = no-sink, fine, no clamp.

## Validation (a5ops-a3-scan, A3 90.90.97.70, torch_npu 2.8.0)

Setup: select ALL blocks (topk=0..15) so native == dense attention over all 1024 keys →
clean apples-to-apples vs a hand fp32 dense reference (with & without sink). Script:
`validate_attn_sink_adapter.py`.

```
o_native (64,4,128)  smax (64,4,8)  ssum (64,4,8)
smax[0,0] (8-wide): all = 2.7304     ssum[0,0] (8-wide): all = 109.1344   (8-wide all-equal: True)
(b) o_native vs dense no-sink ref   maxabs: 7.13e-4   (bf16-level)
(a) smax[...,0] vs row-max          maxabs: 9.54e-7
(a) ssum[...,0] vs row-sumexp       maxabs: 9.92e-5
adapter(o_sink) vs dense-with-sink ref   maxabs: 7.11e-4   ✅
sink actually changes output (o_native vs o_adapter)  maxabs: 7.08e-4   (sink applied, not ignored)
```

Cross-checked with blue's 8.5-A3 softmax-state probe (smax/ssum 8-lane replication,
distinct-count=1) — both assumptions confirmed on two CANN versions.

## Status / next
- ✅ attn_sink fwd adapter — validated.
- TODO: bwd sink term (sink only enters denom → grad rescale), `npu_nsa_compress_grad` +
  `npu_lightning_indexer_grad` wiring, full dispatcher (q.is_npu + TND-flatten + cu_seqlens),
  high-coverage UT vs reference, e2e report → PR-bar → independent-agent verify.
- Branch `feat/m3-attn-sink-dispatcher` on `zhshgmail/easyr1-npu`.
