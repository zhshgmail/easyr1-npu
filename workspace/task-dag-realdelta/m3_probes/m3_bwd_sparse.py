import torch, torch_npu
torch.manual_seed(1); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=4;S_kv=1024   # sbc=4 selected of 16 blocks => real sparse
nblocks=S_kv//sbs  # 16
Tq=64;Tkv=S_kv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
# real sparse: each (query,head) picks sbc distinct random blocks of nblocks
topk=torch.stack([torch.stack([torch.randperm(nblocks,device=dev)[:sbc].sort().values for _ in range(Nq)]) for _ in range(Tq)]).to(torch.int32)  # [Tq,Nq,sbc]
scale=1.0/(D_qk**0.5)
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])

# torch ref: per (t,n) gather the sbc selected blocks' keys/values, softmax over them
qf=q.float().detach().requires_grad_(True); kf=k.float().detach().requires_grad_(True); vf=v.float().detach().requires_grad_(True)
kk=kf[:,0,:].view(nblocks,sbs,D_qk); vv=vf[:,0,:].view(nblocks,sbs,D_v)  # [nb,sbs,D]
oref=torch.zeros(Tq,Nq,D_v,device=dev)
# build dense scores then mask to selected blocks (cleaner autograd)
sc=scale*torch.einsum('tnd,sd->tns',qf,kf[:,0,:])  # [Tq,Nq,Tkv]
mask=torch.zeros(Tq,Nq,nblocks,device=dev,dtype=torch.bool)
mask.scatter_(2, topk.long(), True)  # [Tq,Nq,nblocks]
keymask=mask.repeat_interleave(sbs,dim=2)  # [Tq,Nq,Tkv]
sc=sc.masked_fill(~keymask, float('-inf'))
P=torch.softmax(sc,-1)
oref=torch.einsum('tns,sd->tnd',P,vf[:,0,:])
print("fwd o vs sparse-torch-ref maxabs:", (o.float()-oref).abs().max().item())
g=torch.randn_like(oref); oref.backward(g)
dq_ref,dk_ref,dv_ref=qf.grad.clone(),kf.grad.clone(),vf.grad.clone()
def cmp(name,nat,ref):
    nat=nat.float();ref=ref.float()
    print(f"  {name}: rel={((nat-ref).norm()/(ref.norm()+1e-9)).item():.4f} cos={torch.nn.functional.cosine_similarity(nat.flatten(),ref.flatten(),0).item():.4f} nnorm={nat.norm():.2f} refnorm={ref.norm():.2f}")
print("=== native grad vs sparse ref ===")
try:
    dq,dk,dv=grad(g.to(torch.bfloat16),q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
    cmp("dq",dq,dq_ref); cmp("dk",dk,dk_ref); cmp("dv",dv,dv_ref)
except Exception as e: print("  ERR",str(e)[:200])
