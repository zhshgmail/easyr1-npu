import torch, torch_npu
torch.manual_seed(1); torch.npu.set_device(0); dev="npu:0"
lim=7.0
for M,N,scale in [(8,128,3.0),(256,512,5.0),(4,2048,1.0)]:
    x=torch.randn(M,2*N,dtype=torch.bfloat16,device=dev)*scale
    gate=x[...,:N]; up=x[...,N:]
    # bf16 ref (matches the torch fallback dtype path)
    ref=(torch.nn.functional.silu(gate.float()).clamp(-lim,lim)*up.float().clamp(-lim,lim)).to(torch.bfloat16)
    nat=torch_npu.npu_clipped_swiglu(x,dim=-1,alpha=1.0,limit=lim,bias=0.0,interleaved=False)
    abs_err=(nat.float()-ref.float()).abs()
    rel=(abs_err/(ref.float().abs()+1e-3)).max().item()
    # how big is one bf16 ulp at this magnitude?
    print(f"M={M} N={N} s={scale}: max_abs={abs_err.max().item():.3e} mean_abs={abs_err.mean().item():.3e} max_rel={rel:.3e} ref_absmax={ref.float().abs().max():.2f}")
