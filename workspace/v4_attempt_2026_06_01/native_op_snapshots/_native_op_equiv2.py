import torch, torch_npu
torch.manual_seed(0); torch.npu.set_device(0); dev="npu:0"

# ---- npu_apply_rotary_pos_emb signature/convention probe ----
# Typical sig: npu_apply_rotary_pos_emb(query, key, cos, sin, ...) expects [B,S,N,D] or similar.
rope_dim=64; H=4; seqlen=8
inv=1.0/(10000**(torch.arange(0,rope_dim,2,device=dev).float()/rope_dim))
t=torch.arange(seqlen,device=dev).float()
ang=torch.outer(t,inv)
freqs_cis=torch.polar(torch.ones_like(ang),ang)
positions=torch.arange(seqlen,device=dev)
def v4_rope(x):
    rp=x.float().contiguous()
    rc=torch.view_as_complex(rp.reshape(*rp.shape[:-1], rope_dim//2, 2))
    f=torch.view_as_complex(torch.view_as_real(freqs_cis)[positions].contiguous())
    while f.dim()<rc.dim(): f=f.unsqueeze(-2)
    return torch.view_as_real(rc*f).reshape(*rp.shape).to(x.dtype)
xr=torch.randn(seqlen,H,rope_dim,dtype=torch.bfloat16,device=dev)
ref=v4_rope(xr)
import inspect
try:
    print("sig apply_rotary:", inspect.signature(torch_npu.npu_apply_rotary_pos_emb))
except Exception as e: print("no sig:",e)
# try [B,N,S,D] interleaved cos/sin
cos_i=torch.cos(ang).repeat_interleave(2,-1).to(torch.bfloat16)  # [S, rope_dim]
sin_i=torch.sin(ang).repeat_interleave(2,-1).to(torch.bfloat16)
for layout in ["BSND","BNSD"]:
    try:
        q=xr.unsqueeze(0).clone()  # [1,S,H,D]
        if layout=="BNSD": q=q.transpose(1,2).contiguous()
        c=cos_i.view(1,seqlen,1,rope_dim) if layout=="BSND" else cos_i.view(1,1,seqlen,rope_dim)
        s=sin_i.view(1,seqlen,1,rope_dim) if layout=="BSND" else sin_i.view(1,1,seqlen,rope_dim)
        k=q.clone()
        torch_npu.npu_apply_rotary_pos_emb(q,k,c.contiguous(),s.contiguous())
        out=q[0] if layout=="BSND" else q[0].transpose(0,1)
        err=(out.float()-ref.float()).abs().max().item()
        print(f"[apply_rotary {layout}] err={err:.3e}")
    except Exception as e:
        print(f"[apply_rotary {layout}] FAIL {repr(e)[:120]}")

# ---- npu_clipped_swiglu vs torch silu_and_mul_clamp ----
# V4 moe.py silu_and_mul_clamp: split last dim, silu(a)*b with clamp
M,Dh=8,256
x=torch.randn(M,2*Dh,dtype=torch.bfloat16,device=dev)
a,b=x.chunk(2,-1)
limit=7.0; alpha=1.702
# sglang variant typically: clamp then gated. Inspect clipped_swiglu output vs a plausible ref.
try:
    out=torch_npu.npu_clipped_swiglu(x,dim=-1,alpha=alpha,limit=limit)
    print("[clipped_swiglu] out shape",tuple(out.shape),"dtype",out.dtype)
except Exception as e:
    print("[clipped_swiglu] FAIL",repr(e)[:160])
