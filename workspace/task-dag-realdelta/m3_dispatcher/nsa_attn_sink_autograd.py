"""M3 — sparse-MLA + attn_sink as a torch.autograd.Function (NPU fwd, fallback bwd).

main decision 2026-06-09 (②): native `npu_nsa_select_attention_grad` is numerically unreliable
vs autograd (its internal bwd P-reconstruction diverges from the fwd P; 11 probes + source read,
see m3_probes/FINDINGS_bwd_convention_2026-06-09.md). So:

  forward  = native `npu_nsa_select_attention` (fast, fused) + validated attn_sink closed-form
             (nsa_attn_sink_adapter.apply_attn_sink).  o vs dense/sparse ref ≤ 1.2e-3.
  backward = torch recompute of the sparse-softmax-WITH-sink (masked-dense form, math-equivalent
             to gather-selected since softmax-over-selected == masked softmax) + autograd.
             This IS the mathematically-correct dq/dk/dv/d(attn_sink); proven in Stage A (fp64 5.6e-16)
             and the fwd ref matches native fwd (so its autograd is the gold-standard grad).

native fused-grad alignment is deferred to ROADMAP (perf optimization, needs CANN op owner).

Selection semantics (from torch_npu doc + nsa_selected_attention kernel source):
  topk_indices [T1, N_kv, sbc] (per-KV-head, shared across G=N1/N_kv q-heads), values = block indices
  in [0, S2/sbs); each block = sbs contiguous KV; per-q-head softmax over the gathered selected KV.
"""
import torch
import torch_npu
from nsa_attn_sink_adapter import apply_attn_sink


def _masked_sink_fwd(q, k, v, topk, attn_sink, scale, sbs):
    """Differentiable sparse-softmax-with-sink (masked-dense form). Single batch (TND, one seq).

    q [Tq,Nq,Dqk], k [Tkv,Nkv,Dqk], v [Tkv,Nkv,Dv], topk [Tq,Nkv,sbc] int, attn_sink [Nq] (per q-head).
    q-head n maps to kv-head n // G (contiguous grouping), G = Nq // Nkv.
    Returns o_sink [Tq,Nq,Dv].
    """
    Tq, Nq, Dqk = q.shape
    Tkv, Nkv, Dv = v.shape
    G = Nq // Nkv
    nblk = Tkv // sbs

    # scores per q-head against its kv-head: s[t,n,j] = scale * <q[t,n], k[j, n//G]>
    kk = k[:, (torch.arange(Nq, device=q.device) // G), :]            # [Tkv, Nq, Dqk]
    s = scale * torch.einsum('tnd,jnd->tnj', q, kk)                   # [Tq, Nq, Tkv]

    # block-selection mask: for (t, kv-head g2), blocks in topk[t,g2,:] are valid -> expand to KV cols
    blk_valid = torch.zeros(Tq, Nkv, nblk, dtype=torch.bool, device=q.device)
    tt = torch.arange(Tq, device=q.device).view(Tq, 1, 1).expand_as(topk)
    gg = torch.arange(Nkv, device=q.device).view(1, Nkv, 1).expand_as(topk)
    blk_valid[tt.reshape(-1), gg.reshape(-1), topk.long().reshape(-1)] = True   # dup blocks idempotent (mask)
    col_valid = blk_valid.repeat_interleave(sbs, dim=2)              # [Tq, Nkv, Tkv]
    col_valid = col_valid[:, (torch.arange(Nq, device=q.device) // G), :]       # [Tq, Nq, Tkv]

    s = s.masked_fill(~col_valid, float('-inf'))
    m = s.max(dim=-1, keepdim=True).values.clamp(min=-1e30)
    e = torch.exp(s - m)
    Z = e.sum(-1)                                                    # [Tq,Nq]
    S = torch.exp(attn_sink.view(1, Nq) - m.squeeze(-1))            # sink in denom only
    num = torch.einsum('tnj,jnd->tnd', e, v[:, (torch.arange(Nq, device=q.device) // G), :])
    o = num / (Z + S).unsqueeze(-1)
    return o


class NsaSelectSinkAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk, attn_sink, scale, head_num, sbs, sbc, aq, akv):
        o_native, smax, ssum = torch_npu.npu_nsa_select_attention(
            q, k, v, topk, scale, head_num, sbs, sbc, actual_seq_qlen=aq, actual_seq_kvlen=akv)
        o_sink, _lse = apply_attn_sink(o_native, smax, ssum, attn_sink)
        ctx.save_for_backward(q, k, v, topk, attn_sink)
        ctx.scale, ctx.sbs = scale, sbs
        return o_sink.to(q.dtype)

    @staticmethod
    def backward(ctx, g):
        q, k, v, topk, attn_sink = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.float().detach().requires_grad_(True)
            kd = k.float().detach().requires_grad_(True)
            vd = v.float().detach().requires_grad_(True)
            ad = attn_sink.float().detach().requires_grad_(True)
            o = _masked_sink_fwd(qd, kd, vd, topk, ad, ctx.scale, ctx.sbs)
            dq, dk, dv, da = torch.autograd.grad(o, [qd, kd, vd, ad], g.float())
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, da.to(attn_sink.dtype),
                None, None, None, None, None, None)


def nsa_select_sink_attention(q, k, v, topk, attn_sink, scale, head_num, sbs=64, sbc=16, aq=None, akv=None):
    """Public entry: sparse-MLA select-attention + attn_sink, NPU fwd + correct fallback bwd."""
    return NsaSelectSinkAttn.apply(q, k, v, topk, attn_sink, scale, head_num, sbs, sbc, aq, akv)
