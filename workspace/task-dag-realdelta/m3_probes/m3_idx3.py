import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
B=1;S=128;Nq=4;D=128
q=torch.randn(B,S,Nq,D,dtype=torch.bfloat16,device=dev)
k=torch.randn(B,S,1,D,dtype=torch.bfloat16,device=dev)
w=torch.randn(B,S,Nq,dtype=torch.bfloat16,device=dev)
idx,val=torch_npu.npu_lightning_indexer(q,k,w,layout_query="BSND",layout_key="BSND",sparse_count=S,sparse_mode=3,return_value=True)
vf=val.float().cpu()
print("values: total",vf.numel(),"nan",torch.isnan(vf).sum().item(),"posinf",torch.isposinf(vf).sum().item(),"neginf",torch.isneginf(vf).sum().item(),"finite",torch.isfinite(vf).sum().item())
print("finite-value sample (first 8 finite):", vf[torch.isfinite(vf)][:8].tolist())
# causal mode 3 => upper-triangle masked => masked slots = -inf is EXPECTED, not a bug
fin=torch.isfinite(vf)
print("finite fraction:", fin.float().mean().item())
# check indices for the masked rows: are non-finite values aligned with out-of-range/masked index sentinels?
idxc=idx.cpu()
print("idx range: min",idxc.min().item(),"max",idxc.max().item(),"(valid keys 0..%d)"%(S-1))
