"""Validate the fallback NsaSelectSinkAttn (NPU fwd + torch sparse-sink autograd bwd) on a5ops-a3-scan.

Checks, with REAL sparse topk (per-KV-head, stored order):
  (1) fwd: Function o_sink vs native o + apply_attn_sink, AND vs the masked-sink torch ref -> ≤1e-3.
  (2) bwd: Function dq/dk/dv/da vs a direct autograd through the masked-sink ref -> ~machine (proves wiring).
  (3) gradcheck-style sanity: the masked-sink ref fwd matches native fwd o (so its autograd = gold grad).
"""
import torch, torch_npu
from nsa_attn_sink_autograd import NsaSelectSinkAttn, _masked_sink_fwd
from nsa_attn_sink_adapter import apply_attn_sink

torch.manual_seed(0)
dev = "npu:0"
B, S_q, Nq, Nkv, D_qk, D_v, sbs, sbc, S_kv = 1, 64, 4, 1, 192, 128, 64, 16, 1024
T_q, T_kv = S_q, S_kv
nblk = S_kv // sbs
scale = 1.0 / (D_qk ** 0.5)
aq, akv = [S_q], [S_kv]

q = torch.randn(T_q, Nq, D_qk, dtype=torch.bfloat16, device=dev)
k = torch.randn(T_kv, Nkv, D_qk, dtype=torch.bfloat16, device=dev)
v = torch.randn(T_kv, Nkv, D_v, dtype=torch.bfloat16, device=dev)
# REAL sparse topk: per (query, kv-head) pick sbc distinct blocks out of nblk, stored order (no sort)
topk = torch.stack([torch.stack([torch.randperm(nblk, device=dev)[:sbc].int() for _ in range(Nkv)])
                    for _ in range(T_q)])                          # [T_q, Nkv, sbc]
attn_sink = torch.randn(Nq, dtype=torch.float32, device=dev) * 2.0
g = torch.randn(T_q, Nq, D_v, dtype=torch.bfloat16, device=dev)

def rel(x, y): return ((x - y).abs().max() / (y.abs().max() + 1e-6)).item()

# --- (1) fwd ---
qL = q.clone().requires_grad_(True); kL = k.clone().requires_grad_(True)
vL = v.clone().requires_grad_(True); aL = attn_sink.clone().requires_grad_(True)
o_fn = NsaSelectSinkAttn.apply(qL, kL, vL, topk, aL, scale, Nq, sbs, sbc, aq, akv)
o_native, smax, ssum = torch_npu.npu_nsa_select_attention(q, k, v, topk, scale, Nq, sbs, sbc,
                                                          actual_seq_qlen=aq, actual_seq_kvlen=akv)
o_sink_adapter, _ = apply_attn_sink(o_native, smax, ssum, attn_sink)
o_ref = _masked_sink_fwd(q.float(), k.float(), v.float(), topk, attn_sink, scale, sbs)
print("torch_npu", torch_npu.__version__)
print(f"(1) fwd: Function vs adapter(native) rel {rel(o_fn.float(), o_sink_adapter):.2e} | "
      f"masked-sink-ref vs native-sink-adapter rel {rel(o_ref, o_sink_adapter):.2e}")

# --- (2) bwd: Function grads vs direct autograd through masked-sink ref ---
o_fn.backward(g)
qd = q.float().detach().requires_grad_(True); kd = k.float().detach().requires_grad_(True)
vd = v.float().detach().requires_grad_(True); ad = attn_sink.float().detach().requires_grad_(True)
o2 = _masked_sink_fwd(qd, kd, vd, topk, ad, scale, sbs)
dq_r, dk_r, dv_r, da_r = torch.autograd.grad(o2, [qd, kd, vd, ad], g.float())
print(f"(2) bwd: dq rel {rel(qL.grad.float(), dq_r):.2e} | dk rel {rel(kL.grad.float(), dk_r):.2e} | "
      f"dv rel {rel(vL.grad.float(), dv_r):.2e} | da rel {rel(aL.grad.float(), da_r):.2e}")

# --- (3) the ref-fwd-matches-native gate (so autograd = gold) ---
print(f"(3) masked-sink-ref(no-sink) vs native o_native rel "
      f"{rel(_masked_sink_fwd(q.float(),k.float(),v.float(),topk,torch.full((Nq,),-1e30,device=dev),scale,sbs), o_native.float()):.2e}")
print("DONE")
