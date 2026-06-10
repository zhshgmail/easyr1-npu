"""Isolate the native npu_nsa_select_attention_grad convention (no sink).

Stage B showed da matches but dq/dk/dv ~1.0 even at r~=1 (near no-sink). So the gap is the
NATIVE bwd convention, not the sink correction. Here: a=0 (pure no-sink), feed cotangent g,
compare native (dq,dk,dv) to torch autograd of the dense no-sink softmax, and probe layout/scale/order.
"""
import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; bwd=torch_npu.npu_nsa_select_attention_grad
Nq,Nkv,D_qk,D_v,sbs,sbc,S_kv=4,1,192,128,64,16,1024
S_q=64; T_q=S_q; T_kv=S_kv; nblk=S_kv//sbs; scale=1.0/(D_qk**0.5); aq=[S_q]; akv=[S_kv]
q=torch.randn(T_q,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(T_kv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(T_kv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
topk=torch.arange(nblk,dtype=torch.int32,device=dev).view(1,1,nblk).expand(T_q,Nq,sbc).contiguous()
g=torch.randn(T_q,Nq,D_v,dtype=torch.bfloat16,device=dev)

o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=aq,actual_seq_kvlen=akv)
dq_n,dk_n,dv_n=bwd(g,q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=aq,actual_seq_kvlen=akv)
print("shapes: dq",tuple(dq_n.shape),"dk",tuple(dk_n.shape),"dv",tuple(dv_n.shape))

# torch dense no-sink ref
qf=q.float().clone().requires_grad_(True); kf=k.float()[:,0,:].clone().requires_grad_(True); vf=v.float()[:,0,:].clone().requires_grad_(True)
s=scale*torch.einsum('tnd,kd->tnk',qf,kf); m=s.max(-1,keepdim=True).values
e=torch.exp(s-m); Z=e.sum(-1); o_ref=torch.einsum('tnk,kd->tnd',e,vf)/Z.unsqueeze(-1)
L=(g.float()*o_ref).sum(); dq_r,dk_r,dv_r=torch.autograd.grad(L,[qf,kf,vf])
def rel(x,y): return ((x-y).abs().max()/(y.abs().max()+1e-6)).item()
print("fwd o vs ref:", rel(o.float(),o_ref))
print("dv native vs ref:", rel(dv_n.float()[:,0,:],dv_r))
print("dq native vs ref:", rel(dq_n.float(),dq_r))
print("dk native vs ref:", rel(dk_n.float()[:,0,:],dk_r))
# probes: maybe outputs are returned in a different order, or dv lives in dq slot, etc.
print("-- cross-probe rel (native X vs ref Y) --")
print(" dq_n vs dv_r(pad):", rel(dq_n.float()[...,:D_v], dv_r[:T_q].reshape(-1)[:1].new_zeros(0)) if False else "skip")
print(" |dq_n| max", dq_n.float().abs().max().item(), "|dq_r| max", dq_r.abs().max().item())
print(" |dk_n| max", dk_n.float().abs().max().item(), "|dk_r| max", dk_r.abs().max().item())
print(" |dv_n| max", dv_n.float().abs().max().item(), "|dv_r| max", dv_r.abs().max().item())
# scale hypothesis: ratio of norms
print(" dq norm ratio n/r:", (dq_n.float().norm()/dq_r.norm()).item())
print(" dv norm ratio n/r:", (dv_n.float().norm()/dv_r.norm()).item())
# correlation (is it the same direction, just scaled?)
print(" dv cos:", torch.nn.functional.cosine_similarity(dv_n.float()[:,0,:].reshape(-1), dv_r.reshape(-1), dim=0).item())
print(" dq cos:", torch.nn.functional.cosine_similarity(dq_n.float().reshape(-1), dq_r.reshape(-1), dim=0).item())
print("DONE")
