import torch, torch_npu
torch.manual_seed(3); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=8;S_kv=1024   # select 8 of 16 blocks, distinct per (q,head)
nblocks=S_kv//sbs
Tq=32;Tkv=S_kv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
scale=1.0/(D_qk**0.5)
# distinct sorted blocks per (t,n)
topk=torch.stack([torch.stack([torch.randperm(nblocks,device=dev)[:sbc].sort().values for _ in range(Nq)]) for _ in range(Tq)]).to(torch.int32)
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])

# CORRECT sparse ref: dense scores, mask to selected-block ELEMENTS per (t,n), softmax, autograd
qf=q.float().detach().requires_grad_(True); kf=k.float().detach().requires_grad_(True); vf=v.float().detach().requires_grad_(True)
sc=scale*torch.einsum('tnd,sd->tns',qf,kf[:,0,:])  # [Tq,Nq,Tkv]
# block mask -> element mask
bmask=torch.zeros(Tq,Nq,nblocks,device=dev,dtype=torch.bool).scatter_(2,topk.long(),True)
emask=bmask.repeat_interleave(sbs,2)  # [Tq,Nq,Tkv]
sc=sc.masked_fill(~emask,float('-inf'))
P=torch.softmax(sc,-1)
oref=torch.einsum('tns,sd->tnd',P,vf[:,0,:])
fwd_err=(o.float()-oref).abs().max().item()
print("fwd o vs CORRECT sparse ref maxabs:", fwd_err)
g=torch.randn_like(oref); oref.backward(g)
dq_ref,dk_ref,dv_ref=qf.grad.clone(),kf.grad.clone(),vf.grad.clone()
def cmp(name,nat,ref):
    nat=nat.float();ref=ref.float()
    print(f"  {name}: rel={((nat-ref).norm()/(ref.norm()+1e-9)).item():.4f} cos={torch.nn.functional.cosine_similarity(nat.flatten(),ref.flatten(),0).item():.4f}")
if fwd_err < 0.02:
    print("  FWD MATCHES -> grad comparison meaningful")
    dq,dk,dv=grad(g.to(torch.bfloat16),q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
    cmp("dq",dq,dq_ref); cmp("dk",dk,dk_ref); cmp("dv",dv,dv_ref)
else:
    print("  FWD STILL OFF -> ref bug remains; selection semantics still not matched")
    print("  o[0,0,:4]:",[round(x,3) for x in o[0,0,:4].float().tolist()],"ref:",[round(x,3) for x in oref[0,0,:4].tolist()])
