# Equivalence harness: native torch_npu ops vs V4 complex-mul torch fallback.
# Gate before swapping. Must match within bf16 tolerance or we DO NOT swap.
import torch, torch_npu
torch.manual_seed(0)
dev="npu:0"; torch.npu.set_device(0)

def torch_rmsnorm(x, eps):
    xf=x.float(); var=xf.pow(2).mean(-1,keepdim=True)
    return (xf*torch.rsqrt(var+eps)).to(x.dtype)

# ---- 1. RMSNorm: npu_rms_norm (gamma=ones) vs torch fallback ----
N,D=8,128; eps=1e-6
x=torch.randn(N,D,dtype=torch.bfloat16,device=dev)
ref=torch_rmsnorm(x,eps)
gamma=torch.ones(D,dtype=torch.bfloat16,device=dev)
nat=torch_npu.npu_rms_norm(x,gamma,eps)[0]
err=(nat.float()-ref.float()).abs().max().item()
print(f"[rmsnorm] max_abs_err={err:.3e}  ref_rms={ref.float().abs().mean():.3f}")

# ---- 2. RoPE convention probe ----
# V4 fallback: complex-mul on reshape(..., rope_dim//2, 2)  (INTERLEAVED pairs)
rope_dim=64; H=4
def v4_rope(x, freqs_cis, positions):
    rp=x.float().contiguous()
    rc=torch.view_as_complex(rp.reshape(*rp.shape[:-1], rope_dim//2, 2))
    f_real=torch.view_as_real(freqs_cis)[positions]
    f=torch.view_as_complex(f_real.contiguous())
    while f.dim()<rc.dim(): f=f.unsqueeze(-2)
    rot=rc*f
    return torch.view_as_real(rot).reshape(*rp.shape).to(x.dtype)

# build freqs_cis [seqlen, rope_dim//2] complex
seqlen=8
inv=1.0/(10000**(torch.arange(0,rope_dim,2,device=dev).float()/rope_dim))
t=torch.arange(seqlen,device=dev).float()
ang=torch.outer(t,inv)  # [seqlen, rope_dim//2]
freqs_cis=torch.polar(torch.ones_like(ang),ang)
positions=torch.arange(seqlen,device=dev)
xr=torch.randn(seqlen,H,rope_dim,dtype=torch.bfloat16,device=dev)
ref_rope=v4_rope(xr,freqs_cis,positions)

# native rotary_mul half-mode: needs r1=cos r2=sin broadcast to [.., rope_dim] in rotate-half layout
# rotate-half splits [first half | second half], NOT interleaved => convention MISMATCH expected.
cos=torch.cos(ang); sin=torch.sin(ang)  # [seqlen, rope_dim//2]
cos_h=torch.cat([cos,cos],-1).to(torch.bfloat16)  # [seqlen, rope_dim]
sin_h=torch.cat([sin,sin],-1).to(torch.bfloat16)
r1=cos_h[:,None,:].expand(seqlen,H,rope_dim).contiguous()
r2=sin_h[:,None,:].expand(seqlen,H,rope_dim).contiguous()
nat_half=torch_npu.npu_rotary_mul(xr,r1,r2,rotary_mode="half")
err_half=(nat_half.float()-ref_rope.float()).abs().max().item()
print(f"[rope half ] max_abs_err vs v4-complex = {err_half:.3e}")

# interleave mode: cos/sin interleaved per pair => matches complex-on-pairs convention
cos_i=cos.repeat_interleave(2,dim=-1).to(torch.bfloat16)
sin_i=sin.repeat_interleave(2,dim=-1).to(torch.bfloat16)
r1i=cos_i[:,None,:].expand(seqlen,H,rope_dim).contiguous()
r2i=sin_i[:,None,:].expand(seqlen,H,rope_dim).contiguous()
try:
    nat_il=torch_npu.npu_rotary_mul(xr,r1i,r2i,rotary_mode="interleave")
    err_il=(nat_il.float()-ref_rope.float()).abs().max().item()
    print(f"[rope intlv] max_abs_err vs v4-complex = {err_il:.3e}")
except Exception as e:
    print("[rope intlv] FAILED:", repr(e)[:200])
