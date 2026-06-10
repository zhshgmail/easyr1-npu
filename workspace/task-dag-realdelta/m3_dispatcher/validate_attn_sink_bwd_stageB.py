"""M3 attn_sink BACKWARD — Stage B (real kernel, NPU a5ops-a3-scan).

Closes the loop Stage A opened: the closed-form using the REAL
`npu_nsa_select_attention_grad` (native bwd, no sink) + analytic sink corrections,
vs a torch fp32 autograd dense-with-sink reference.

  dout' = g·r;  dv = NativeGrad_v(dout');
  dq    = NativeGrad_q(dout') + β·(S/D²)·scale·Σ_n e_n k_n
  dk_n  = NativeGrad_k(dout') + β·(S/D²)·scale·e_n·q_i
  da_h  = −Σ_rows β·Z·S/D²

Select ALL blocks (topk = 0..nblk-1) -> native == dense over all keys, so a torch dense
reference is apples-to-apples. bf16 kernel -> compare at bf16 tolerance (2e-2 rel / 2e-3 abs).
Also reports first-order-only (no cross-term) to confirm it fails on real hw too (Stage A parity).

native bwd arg-order (blue's 561103 trap): (dout, q, k, v, attention_out, softmax_max,
softmax_sum, topk_indices, scale, head_num, sbs, sbc, *, actual_seq_qlen, actual_seq_kvlen).
"""
import torch, torch_npu

torch.manual_seed(0)
dev = "npu:0"
fwd = torch_npu.npu_nsa_select_attention
bwd = torch_npu.npu_nsa_select_attention_grad

B, S_q, Nq, Nkv, D_qk, D_v, sbs, sbc, S_kv = 1, 64, 4, 1, 192, 128, 64, 16, 1024
T_q, T_kv = B * S_q, B * S_kv
nblk = S_kv // sbs
assert sbc == nblk
scale = 1.0 / (D_qk ** 0.5)
aq, akv = [S_q], [S_kv]

q = torch.randn(T_q, Nq, D_qk, dtype=torch.bfloat16, device=dev)
k = torch.randn(T_kv, Nkv, D_qk, dtype=torch.bfloat16, device=dev)
v = torch.randn(T_kv, Nkv, D_v, dtype=torch.bfloat16, device=dev)
topk = torch.arange(nblk, dtype=torch.int32, device=dev).view(1, 1, nblk).expand(T_q, Nq, sbc).contiguous()
a = torch.randn(Nq, dtype=torch.float32, device=dev) * 2.0    # per-head sink, sizeable
g = torch.randn(T_q, Nq, D_v, dtype=torch.bfloat16, device=dev)   # upstream cotangent

# ---- native fwd ----
o_native, smax, ssum = fwd(q, k, v, topk, scale, Nq, sbs, sbc, actual_seq_qlen=aq, actual_seq_kvlen=akv)
m = smax[..., 0].float()            # (T,N)  row max
Z = ssum[..., 0].float()            # (T,N)  sum_exp
S = torch.exp(a.view(1, Nq) - m)    # (T,N)
D = Z + S
r = Z / D
beta = (g.float() * o_native.float()).sum(-1)         # <g,o_native>  (T,N)
dout_p = (g.float() * r.unsqueeze(-1)).to(torch.bfloat16)

# ---- native bwd (no sink) with scaled cotangent ----  arg-order per blue
dq_nat, dk_nat, dv_nat = bwd(dout_p, q, k, v, o_native, smax, ssum, topk, scale, Nq, sbs, sbc,
                             actual_seq_qlen=aq, actual_seq_kvlen=akv)

# ---- analytic sink corrections (fp32, recomputed e on the dense key set) ----
qf, kf, vf = q.float(), k.float()[:, 0, :], v.float()[:, 0, :]   # MQA single kv head
s = scale * torch.einsum('tnd,kd->tnk', qf, kf)        # (T,N,Tkv)
e = torch.exp(s - m.unsqueeze(-1))                      # align to native max
e = e * (Z / e.sum(-1)).unsqueeze(-1)                   # renorm so Σe = native Z (kernel-consistent)
coef = (beta * S / D ** 2)                              # β·S/D²   (T,N)
sum_e_k = torch.einsum('tnk,kd->tnd', e, kf)            # Σ_n e_n k_n  (T,N,Dqk)
corr_q = (coef.unsqueeze(-1) * scale * sum_e_k)         # (T,N,Dqk)

dq_cf = dq_nat.float() + corr_q
# dk: accumulate Σ_{t,n} coef·scale·e[t,n,key]·q[t,n,:]  -> (Tkv, Dqk)
dk_corr = scale * torch.einsum('tn,tnk,tnd->kd', coef, e, qf)
dk_cf = dk_nat.float()[:, 0, :] + dk_corr               # (Tkv,Dqk)
dv_cf = dv_nat.float()[:, 0, :]                         # (Tkv,Dv)
da_cf = -(beta * Z * S / D ** 2).sum(0)                 # (N,)

# ---- REFERENCE: torch fp32 autograd dense-with-sink ----
qr = qf.clone().requires_grad_(True)
kr = kf.clone().requires_grad_(True)
vr = vf.clone().requires_grad_(True)
ar = a.clone().requires_grad_(True)
sr = scale * torch.einsum('tnd,kd->tnk', qr, kr)
mr = sr.max(-1, keepdim=True).values
er = torch.exp(sr - mr)
Zr = er.sum(-1)
Sr = torch.exp(ar.view(1, Nq) - mr.squeeze(-1))
numr = torch.einsum('tnk,kd->tnd', er, vr)
o_sink_ref = numr / (Zr + Sr).unsqueeze(-1)
L = (g.float() * o_sink_ref).sum()
dq_ref, dk_ref, dv_ref, da_ref = torch.autograd.grad(L, [qr, kr, vr, ar])

def rel(x, y):
    return ((x - y).abs().max() / (y.abs().max() + 1e-6)).item()
print("torch", torch.__version__, "torch_npu", torch_npu.__version__, "| mean r", round(r.mean().item(), 4))
print(f"dq  full vs ref rel: {rel(dq_cf, dq_ref):.2e}   first-order-only: {rel(dq_nat.float(), dq_ref):.2e}")
print(f"dk  full vs ref rel: {rel(dk_cf, dk_ref):.2e}   first-order-only: {rel(dk_nat.float()[:,0,:], dk_ref):.2e}")
print(f"dv  full vs ref rel: {rel(dv_cf, dv_ref):.2e}")
print(f"da  full vs ref rel: {rel(da_cf, da_ref):.2e}")
print("DONE")
