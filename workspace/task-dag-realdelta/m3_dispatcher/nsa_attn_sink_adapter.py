"""M3 — attn_sink adapter for CANN-native `npu_nsa_select_attention`.

The native op has NO `attn_sink` parameter; the latest-main DSv4
`sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale) -> (o, lse)` passes a
per-head `attn_sink[H]fp32` (softmax sink). Per miles `ref_dense_attn`, the sink adds a
single virtual logit per head into the softmax DENOMINATOR only (not the numerator):

    sink_term = exp(attn_sink[h] - row_max)
    denom     = sum_exp + sink_term
    o_sink    = numerator / denom

The native op already returns everything needed:
    out0 = o_native    = numerator / sum_exp        (plain softmax, no sink)
    out1 = softmax_max = row_max     (per-(row,head) scalar, replicated to an 8-wide pad lane)
    out2 = softmax_sum = sum_exp     (same 8-wide replication)

so the sink is a pure post-hoc rescale of the three native outputs:

    o_sink   = o_native * sum_exp / (sum_exp + exp(attn_sink[h] - softmax_max))
    lse_sink = softmax_max + log(sum_exp + exp(attn_sink[h] - softmax_max))

CRITICAL (validated on A3, blue + scan 2026-06-06): take `[..., 0]` of softmax_max/sum
(the 8 lanes are identical padding) — NEVER sum over the 8 (an 8x-wrong denominator).

Validated on A3 90.90.97.70 (torch_npu 2.8.0) + blue 8.5 A3: adapter vs dense fp32
reference-with-sink maxabs 7.1e-4 (bf16-level); see validate_attn_sink_adapter.py.
"""
import torch


def _col0(x):
    """softmax_max/softmax_sum come as (..., 8) replicated padding -> take col 0."""
    return x[..., 0] if x.dim() >= 3 and x.shape[-1] == 8 else x


def apply_attn_sink(o_native, softmax_max, softmax_sum, attn_sink):
    """Post-hoc attn_sink adaptation on native npu_nsa_select_attention outputs.

    Args:
        o_native:    (T, N, D_v)         native attention output (no sink)
        softmax_max: (T, N, 8) or (T, N) per-(row,head) row max (8-wide = padding)
        softmax_sum: (T, N, 8) or (T, N) per-(row,head) sum_exp
        attn_sink:   (N,)                per-head sink logit (fp32)
    Returns:
        o_sink:   (T, N, D_v) in o_native's dtype
        lse_sink: (T, N) fp32   (log-sum-exp incl. sink, for the backward pass)

    Numerical note: `attn_sink[h] - row_max` can be very negative -> exp underflows
    to 0 (== no sink), which is correct; no clamp needed (confirmed with blue).
    """
    smax = _col0(softmax_max).float()
    ssum = _col0(softmax_sum).float()
    sink = attn_sink.to(device=smax.device, dtype=torch.float32)
    # broadcast per-head sink to (T, N): sink shape (N,) -> (1, N)
    sink = sink.view(*([1] * (smax.dim() - 1)), -1)
    sink_term = torch.exp(sink - smax)
    denom = ssum + sink_term
    o_sink = o_native.float() * (ssum / denom).unsqueeze(-1)
    lse_sink = smax + torch.log(denom)
    return o_sink.to(o_native.dtype), lse_sink


def sparse_attn_select_npu(q, kv_or_k, v, topk_indices, attn_sink, sm_scale,
                           head_num, select_block_size=64, select_block_count=16,
                           actual_seq_qlen=None, actual_seq_kvlen=None,
                           atten_mask=None):
    """Native sparse-MLA fwd + attn_sink adaptation -> (o_sink, lse_sink).

    Thin wrapper over `torch_npu.npu_nsa_select_attention` (TND layout) that applies
    the validated attn_sink post-hoc. Returns the (o, lse) pair that the DSv4
    `sparse_attn_tilelang` call site expects. `v` is separate because D_v != D_qk (MLA).
    """
    import torch_npu
    out = torch_npu.npu_nsa_select_attention(
        q, kv_or_k, v, topk_indices, sm_scale, head_num,
        select_block_size, select_block_count,
        atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    o_native, softmax_max, softmax_sum = out[0], out[1], out[2]
    if attn_sink is None:
        return o_native, _col0(softmax_max).float()  # no-sink path
    return apply_attn_sink(o_native, softmax_max, softmax_sum, attn_sink)
