# RESULT — M3 fallback bwd VALIDATED on A3 (2026-06-09, scan-agent)

main decision 2026-06-09 (②): native `npu_nsa_select_attention_grad` is numerically unreliable vs
autograd (internal bwd P-reconstruction diverges from fwd P — 11 probes + source read, see
`m3_probes/FINDINGS_bwd_convention_2026-06-09.md`). Fallback adopted; native fused-grad → ROADMAP.

## Implementation (`nsa_attn_sink_autograd.py`)

`torch.autograd.Function NsaSelectSinkAttn`:
- **forward** = native `npu_nsa_select_attention` (fast, fused) + validated `apply_attn_sink` closed-form.
- **backward** = torch recompute of sparse-softmax-WITH-sink (`_masked_sink_fwd`, masked-dense form,
  math-equivalent to gather-selected) + `torch.autograd.grad` → dq/dk/dv/d(attn_sink).
- Public entry `nsa_select_sink_attention(q,k,v,topk,attn_sink,scale,head_num,sbs,sbc,aq,akv)`.

Selection semantics honored: topk `[T1,N_kv,sbc]` per-KV-head, shared across G=N1/N_kv q-heads,
block-level (block=sbs contiguous KV), stored order, per-q-head softmax over selected, sink in denom.

## Validation (`validate_fallback_bwd.py`, a5ops-a3-scan, torch_npu 2.8.0, REAL sparse topk)

```
(1) fwd: Function vs native+adapter      rel 0.00e+00   (fwd IS native kernel, exact)
         masked-sink-ref vs native+sink  rel 3.14e-03   (bf16)
(2) bwd: dq 2.13e-03  dk 2.51e-03  dv 3.35e-03  da 0.00e+00   (all bf16-level)
(3) gate: masked-sink-ref(no-sink) vs native o_native  rel 2.93e-03
          -> ref matches native fwd => its autograd is the gold-standard grad
```

All bf16-level (per-dtype tol: bf16 = 2e-2/2e-3). fwd exact; bwd correct dq/dk/dv/da.

## Status
- ✅ fwd attn_sink adapter (prior milestone, 2×2 verified).
- ✅ fallback bwd (this) — NPU fwd + correct sparse-sink autograd bwd, bf16-level.
- Pending: blue 8.5-A3 cross-verify + code review (per-milestone discipline) → close.
- Deferred → ROADMAP "native-grad P-recon alignment" (CANN op owner; evidence `89c939a` + source pin).
- Next M3: dispatcher wiring (q.is_npu, TND-flatten + cu_seqlens) + compress/indexer bwd + high-cov UT
  + e2e report → PR-bar.
