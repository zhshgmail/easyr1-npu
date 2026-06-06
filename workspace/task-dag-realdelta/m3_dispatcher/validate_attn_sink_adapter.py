"""M3 — validate the attn_sink closed-form adapter on A3 NPU.

Native `npu_nsa_select_attention` has NO attn_sink param. It returns
(o_native, softmax_max, softmax_sum). The attn_sink (per miles ref_dense_attn)
adds a virtual logit per head into the softmax DENOMINATOR only:
    sink_term = exp(attn_sink[h] - row_max);  denom = sum_exp + sink_term
    o_sink    = numerator / denom = o_native * sum_exp / (sum_exp + sink_term)

This script confirms, on real NPU, the two assumptions the adapter relies on:
  (a) softmax_max / softmax_sum are the final per-(row,head) reduced values
      (and the trailing 8-wide dim is alignment padding -> index 0 is real / all equal);
  (b) o_native == numerator/sum_exp  (plain softmax over the selected keys, no sink),
then checks the adapter output vs a dense fp32 reference WITH sink.

Setup: select ALL blocks (topk = 0..nblk-1) so native == dense attention over all
keys -> a clean apples-to-apples vs a hand fp32 dense reference (avoids the
block-vs-key selection-granularity matching subtlety for this first validation).
"""
import torch, torch_npu

torch.manual_seed(0)
dev = "npu:0"
op = torch_npu.npu_nsa_select_attention

print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
print("npu_available", torch.npu.is_available(), "count", torch.npu.device_count())

B, S_q, Nq, Nkv, D_qk, D_v, sbs, sbc, S_kv = 1, 64, 4, 1, 192, 128, 64, 16, 1024
T_q, T_kv = B * S_q, B * S_kv
nblk = S_kv // sbs  # 16
assert sbc == nblk, "this validation selects ALL blocks -> dense"

q = torch.randn(T_q, Nq, D_qk, dtype=torch.bfloat16, device=dev)
k = torch.randn(T_kv, Nkv, D_qk, dtype=torch.bfloat16, device=dev)
v = torch.randn(T_kv, Nkv, D_v, dtype=torch.bfloat16, device=dev)
# select ALL blocks for every (row, head) -> dense over all keys
topk = torch.arange(nblk, dtype=torch.int32, device=dev).view(1, 1, nblk).expand(T_q, Nq, sbc).contiguous()
scale = 1.0 / (D_qk ** 0.5)
aq, akv = [S_q], [S_kv]

out = op(q, k, v, topk, scale, Nq, sbs, sbc, actual_seq_qlen=aq, actual_seq_kvlen=akv)
o_native, smax, ssum = out[0], out[1], out[2]
print("shapes  o_native", tuple(o_native.shape), "smax", tuple(smax.shape), "ssum", tuple(ssum.shape))

# --- assumption (a): inspect the trailing 8-wide dim ---
sm00 = smax[0, 0].float().cpu()
ss00 = ssum[0, 0].float().cpu()
print("smax[0,0] (8-wide):", [round(x, 4) for x in sm00.tolist()])
print("ssum[0,0] (8-wide):", [round(x, 4) for x in ss00.tolist()])
print("smax 8-wide all-equal:", bool((sm00 == sm00[0]).all()), " ssum 8-wide all-equal:", bool((ss00 == ss00[0]).all()))
smax0 = smax[..., 0].float().cpu()   # take index 0
ssum0 = ssum[..., 0].float().cpu()

# --- dense fp32 reference (no sink) on the SAME (all-block) selection ---
qf, kf, vf = q.float().cpu(), k.float().cpu(), v.float().cpu()
scores = torch.einsum("thd,nd->thn", qf, kf[:, 0, :]) * scale          # [T_q, Nq, T_kv]
m = scores.max(dim=-1, keepdim=True).values                            # [T_q, Nq, 1]
e = torch.exp(scores - m)
se = e.sum(-1)                                                         # [T_q, Nq]
num = torch.einsum("thn,nd->thd", e, vf[:, 0, :])                      # [T_q, Nq, D_v]
o_ref_nosink = num / se.unsqueeze(-1)

print("\n--- assumption checks ---")
print("(b) o_native vs dense no-sink ref   maxabs:", (o_native.float().cpu() - o_ref_nosink).abs().max().item())
print("(a) smax[...,0] vs row-max          maxabs:", (smax0 - m.squeeze(-1)).abs().max().item())
print("(a) ssum[...,0] vs row-sumexp       maxabs:", (ssum0 - se).abs().max().item())

# --- attn_sink adapter vs dense fp32 reference WITH sink ---
attn_sink = torch.randn(Nq).float()                                    # per-head sink
sink_ref = torch.exp(attn_sink.view(1, Nq) - m.squeeze(-1))
o_ref_sink = num / (se + sink_ref).unsqueeze(-1)

sink_n = torch.exp(attn_sink.view(1, Nq) - smax0)                      # adapter uses native smax
o_adapter = o_native.float().cpu() * (ssum0 / (ssum0 + sink_n)).unsqueeze(-1)

print("\n--- attn_sink ADAPTER ---")
print("adapter(o_sink) vs dense-with-sink ref   maxabs:", (o_adapter - o_ref_sink).abs().max().item())
print("sink actually changes output (o_native vs o_adapter) maxabs:",
      (o_native.float().cpu() - o_adapter).abs().max().item())
print("\nDONE")
