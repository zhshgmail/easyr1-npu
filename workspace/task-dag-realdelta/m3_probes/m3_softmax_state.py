import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
op=torch_npu.npu_nsa_select_attention
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=16;S_kv=1024
Tq=64;Tkv=S_kv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
topk=torch.randint(0,S_kv//sbs,(Tq,Nq,sbc),dtype=torch.int32,device=dev)
scale=1.0/(D_qk**0.5)
attn,smax,ssum=op(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
print("attn",tuple(attn.shape),"smax",tuple(smax.shape),"ssum",tuple(ssum.shape))
sm=smax.float().cpu(); ss=ssum.float().cpu()
# Q1: is last-dim 8 padding (cols identical / mostly 0) or real distinct structure?
r0=sm[0,0,:]  # row0, head0, the 8 values
print("smax[0,0,:] (8 vals):", [round(x,4) for x in r0.tolist()])
print("ssum[0,0,:] (8 vals):", [round(x,4) for x in ss[0,0,:].tolist()])
# how many of the 8 are distinct / nonzero per row?
nz = (ss[:, :, :] != 0).sum(-1).float()  # nonzero count in last dim per (row,head)
print("ssum last-dim nonzero-count: min %.1f max %.1f mean %.2f"%(nz.min(),nz.max(),nz.mean()))
distinct = torch.tensor([[len(set(round(x,3) for x in sm[i,h,:].tolist())) for h in range(Nq)] for i in range(min(8,Tq))]).float()
print("smax last-dim distinct-count (first 8 rows): min %.1f max %.1f"%(distinct.min(),distinct.max()))
# Q2: which column holds the actual per-row max/sum? col0? or need reduce over 8?
# If 8=blocks, true row sum_exp = sum over the 8 (with max-rescale); if padding, col0 is the answer.
print("ssum[:,:,0] vs ssum.sum(-1): col0 mean %.3f, sum-over-8 mean %.3f"%(ss[...,0].mean(), ss.sum(-1).mean()))
print("ratio sum8/col0 (per elem) mean: %.3f"%((ss.sum(-1)/(ss[...,0]+1e-9)).mean()))
