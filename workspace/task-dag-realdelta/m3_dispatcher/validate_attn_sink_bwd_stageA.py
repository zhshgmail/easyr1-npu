"""M3 attn_sink BACKWARD — Stage A (pure torch, CPU): prove the closed-form decomposition.

Validates the bwd closed-form in DERIVATION_attn_sink_bwd.md against torch autograd:
  - "native" = softmax WITHOUT sink (o_native = num/Z); its VJP stands in for
    npu_nsa_select_attention_grad (Stage B swaps in the real kernel).
  - closed-form: dout'=g·r through native VJP + analytic Jacobian cross-term + da.
  - REFERENCE: autograd through the full dense-with-sink output.
Also demonstrates first-order-only (drop the cross-term) FAILS at large sink — the empirical
proof that the β·∂r/∂x cross-term is required (blue's key point).

Dense (all keys valid) so the math is exercised cleanly; MQA (single kv head shared across q heads).
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)  # CPU fp64: isolate math error from fp rounding

T, N, Dqk, Dv, Tkv = 8, 4, 16, 16, 12       # T rows, N q-heads, dense over Tkv keys
scale = 1.0 / Dqk ** 0.5

def run(sink_scale, tag):
    q = torch.randn(T, N, Dqk, requires_grad=True)
    k = torch.randn(Tkv, Dqk, requires_grad=True)   # MQA: shared kv
    v = torch.randn(Tkv, Dv, requires_grad=True)
    a = (torch.randn(N) * sink_scale).requires_grad_(True)   # per-head sink
    g = torch.randn(T, N, Dv)                                 # fixed upstream cotangent

    # ---- REFERENCE: autograd through full dense-with-sink ----
    s = scale * torch.einsum('tnd,kd->tnk', q, k)
    m = s.max(-1, keepdim=True).values
    e = torch.exp(s - m)
    Z = e.sum(-1)
    S = torch.exp(a.view(1, N) - m.squeeze(-1))
    num = torch.einsum('tnk,kd->tnd', e, v)
    o_sink = num / (Z + S).unsqueeze(-1)
    L = (g * o_sink).sum()
    dq_ref, dk_ref, dv_ref, da_ref = torch.autograd.grad(L, [q, k, v, a])

    # ---- CLOSED-FORM (reuse "native" = no-sink softmax) ----
    qd, kd, vd = q.detach().requires_grad_(True), k.detach().requires_grad_(True), v.detach().requires_grad_(True)
    s2 = scale * torch.einsum('tnd,kd->tnk', qd, kd)
    m2 = s2.max(-1, keepdim=True).values
    e2 = torch.exp(s2 - m2)
    Z2 = e2.sum(-1)
    S2 = torch.exp(a.detach().view(1, N) - m2.squeeze(-1))
    o_native = torch.einsum('tnk,kd->tnd', e2, vd) / Z2.unsqueeze(-1)
    r = (Z2 / (Z2 + S2))                              # per (row,head)
    beta = (g * o_native).sum(-1)                     # <g, o_native>
    dout_p = g * r.unsqueeze(-1)
    # NativeGrad_x(dout') — VJP of o_native (stand-in for npu_nsa_select_attention_grad)
    dq_nat, dk_nat, dv_nat = torch.autograd.grad(o_native, [qd, kd, vd], grad_outputs=dout_p)
    coef = beta * S2 / (Z2 + S2) ** 2                 # β·S/D²
    sum_e_k = torch.einsum('tnk,kd->tnd', e2, kd.detach())          # Σ_n e_n k_n  (for dq)
    dq_cf = dq_nat + coef.unsqueeze(-1) * scale * sum_e_k
    dk_cf = dk_nat + scale * torch.einsum('tn,tnk,tnd->kd', coef, e2, qd.detach())
    dv_cf = dv_nat
    da_cf = -(beta * Z2 * S2 / (Z2 + S2) ** 2).sum(0)

    def mae(x, y): return (x - y).abs().max().item()
    print(f"\n=== {tag} (sink_scale={sink_scale}, mean r={r.mean():.4f}) ===")
    print(f"  dq  full-form vs ref : {mae(dq_cf, dq_ref):.2e}   first-order-only: {mae(dq_nat, dq_ref):.2e}")
    print(f"  dk  full-form vs ref : {mae(dk_cf, dk_ref):.2e}   first-order-only: {mae(dk_nat, dk_ref):.2e}")
    print(f"  dv  full-form vs ref : {mae(dv_cf, dv_ref):.2e}")
    print(f"  da  full-form vs ref : {mae(da_cf, da_ref):.2e}")
    return mae(dq_cf, dq_ref), mae(dq_nat, dq_ref)

print("torch", torch.__version__)
# small sink: cross-term ~0, native grad already ~correct (sanity #1)
run(0.01, "SMALL sink (S<<Z)")
# large sink: cross-term first-order-significant -> full PASSES, first-order-only FAILS
full_err, fo_err = run(3.0, "LARGE sink (S~Z)")
print(f"\nCROSS-TERM PROOF (large sink): full-form dq err {full_err:.2e}  <<  first-order-only dq err {fo_err:.2e}")
print("PASS" if full_err < 1e-9 and fo_err > 1e-3 else "CHECK")
