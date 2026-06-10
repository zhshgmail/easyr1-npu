# M3 — compress/indexer native-grad probe (2026-06-10, blue 8.5 A3, parallel lane)

scan-agent asked (during the dispatcher split): do `npu_nsa_compress_grad` / `npu_lightning_indexer_grad`
hit the same fwd-right/bwd-wrong native-grad issue as `npu_nsa_select_attention_grad` → native usable or fwd-match-ref fallback?

## Findings (blue, 8.5 A3 / torch_npu 2.9)
- Both grad ops EXIST in dir(torch_npu).
- **Schemas captured**:
  - `npu_lightning_indexer_grad(query, key, dy, sparse_indices, weights, *, actual_seq_lengths_query, actual_seq_lengths_key, layout="BSND", sparse_mode=3, pre_tokens, next_tokens) -> (T,T,T)`
  - `npu_nsa_compress_grad(grad, input, weight, compress_block_size, compress_stride, *, actual_seq_len) -> (T,T)` — NOTE: this is grad for `npu_nsa_compress` (the COMPRESSION/pooling op, returns d_input/d_weight), NOT for `npu_nsa_compress_attention`. The compress-ATTENTION bwd is likely another internal path (same family as select_attention).
- **indexer_grad hard to even CALL**: 561103 (AclNN_Parameter_Error) across dy bf16/fp32/[B,S,Nq,S] shapes — same arg-shape/convention rabbit hole as select_attention_grad's 561103 trap; needs multi-iteration RE to find the exact contract (not statically introspectable).

## Recommendation: USE THE FALLBACK (fwd-match-ref autograd) for compress + indexer bwd too
Reasoning (consistent with main's 2026-06-09 fallback ruling for select_attention):
1. select_attention_grad definitively needed fallback — root-caused (internal P-rebuild, 11 probes).
2. indexer/compress are the SAME NSA-family FA-style ops → very likely the same internal-recompute architecture → same fwd-right/bwd-wrong limitation.
3. The native grads are even hard to CALL correctly (561103) → more iteration risk for an op that likely needs fallback anyway.
4. The **fwd-match-ref autograd fallback is dimension-independent, mathematically correct, and proven** (attn_sink bwd already uses it; validated 2x2). It applies uniformly to compress/indexer: build a torch reference whose fwd matches the native fwd (to ~1e-3), autograd through it → correct dq/dk/dweights.

→ scan-agent: default compress/indexer bwd to fwd-match-ref autograd (same pattern as attn_sink bwd). Native compress/indexer grad alignment → ROADMAP perf-followup (same bucket as native select_attention grad, pending CANN op owner).
