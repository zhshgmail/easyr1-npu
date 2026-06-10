import torch, torch_npu
torch.manual_seed(3); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=8;S_kv=1024
nblocks=S_kv//sbs; Tq=32;Tkv=S_kv; G=Nq//Nkv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
scale=1.0/(D_qk**0.5)
topk=torch.stack([torch.stack([torch.randperm(nblocks,device=dev)[:sbc].sort().values for _ in range(Nkv)]) for _ in range(Tq)]).to(torch.int32)
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
qf=q.float().detach().requires_grad_(True);kf=k.float().detach().requires_grad_(True);vf=v.float().detach().requires_grad_(True)
sc=scale*torch.einsum('tnd,sd->tns',qf,kf[:,0,:])
bmask=torch.zeros(Tq,Nkv,nblocks,device=dev,dtype=torch.bool).scatter_(2,topk.long(),True).repeat_interleave(G,1).repeat_interleave(sbs,2)
sc=sc.masked_fill(~bmask,float('-inf')); P=torch.softmax(sc,-1)
oref=torch.einsum('tns,sd->tnd',P,vf[:,0,:])
g=torch.randn_like(oref); oref.backward(g)
dq_ref,dk_ref,dv_ref=qf.grad.clone(),kf.grad.clone(),vf.grad.clone()
dq,dk,dv=grad(g.to(torch.bfloat16),q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
# ratio test: element-wise native/ref where ref is not tiny
def ratio(name,nat,ref):
    nat=nat.float().flatten();ref=ref.float().flatten()
    m=ref.abs()>ref.abs().max()*0.05
    r=nat[m]/ref[m]
    print(f"  {name}: ratio mean={r.mean():.4f} std={r.std():.4f} median={r.median():.4f}  (scale={scale:.4f})  [const ratio => missing scalar]")
ratio("dv",dv[:,0,:],dv_ref[:,0,:])
ratio("dq",dq,dq_ref)
ratio("dk",dk[:,0,:],dk_ref[:,0,:])
# also: does native dv == manual P^T@dO * 0.5? check exact factor
dv_manual=torch.einsum('tns,tnd->sd',P,g)
fac=(dv[:,0,:].float()*dv_manual).sum()/(dv_manual*dv_manual).sum()
print("best-fit scalar native_dv = fac * manual_dv: fac=%.4f (resid cos stays 0.49 if not pure-scale)"%fac.item())
