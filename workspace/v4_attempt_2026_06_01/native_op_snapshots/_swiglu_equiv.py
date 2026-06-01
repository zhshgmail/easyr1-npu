import torch, torch_npu
torch.manual_seed(0); torch.npu.set_device(0); dev="npu:0"
M,N=8,128; lim=7.0
x=torch.randn(M,2*N,dtype=torch.bfloat16,device=dev)*3.0  # *3 to exercise clamp
def v4_ref(x):
    gate=x[...,:N]; up=x[...,N:]
    g=torch.nn.functional.silu(gate.float()).clamp(-lim,lim)
    u=up.float().clamp(-lim,lim)
    return (g*u).to(x.dtype)
ref=v4_ref(x)
# clipped_swiglu interleaved=True splits even/odd; interleaved=False = front/back like V4.
import itertools
best=None
for alpha,bias,interleaved in itertools.product([1.702,1.0],[1.0,0.0],[True,False]):
    try:
        out=torch_npu.npu_clipped_swiglu(x,dim=-1,alpha=alpha,limit=lim,bias=bias,interleaved=interleaved)
        if out.shape[-1]!=N: 
            print(f"a={alpha} b={bias} il={interleaved} -> shape {tuple(out.shape)} skip"); continue
        err=(out.float()-ref.float()).abs().max().item()
        print(f"a={alpha} b={bias} il={interleaved} -> max_abs_err={err:.3e}")
    except Exception as e:
        print(f"a={alpha} b={bias} il={interleaved} -> FAIL {repr(e)[:80]}")
