import torch, torch_npu
torch.manual_seed(3); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=8;S_kv=1024
nblocks=S_kv//sbs; Tq=32;Tkv=S_kv; G=Nq//Nkv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
scale=1.0/(D_qk**0.5)
# CORRECT shape: topk [Tq, Nkv, sbc]  (per-KV-head selection)
topk=torch.stack([torch.stack([torch.randperm(nblocks,device=dev)[:sbc].sort().values for _ in range(Nkv)]) for _ in range(Tq)]).to(torch.int32)  # [Tq,Nkv,sbc]
print("topk shape:",tuple(topk.shape))
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
# ref: each q-head n maps to kv-head g=n//G... wait G=Nq/Nkv=4, so all 4 q-heads -> kv-head 0
qf=q.float().detach().requires_grad_(True); kf=k.float().detach().requires_grad_(True); vf=v.float().detach().requires_grad_(True)
sc=scale*torch.einsum('tnd,sd->tns',qf,kf[:,0,:])  # [Tq,Nq,Tkv]
# block mask from per-KV-head topk, broadcast to the G q-heads in that group
bmask=torch.zeros(Tq,Nkv,nblocks,device=dev,dtype=torch.bool).scatter_(2,topk.long(),True)  # [Tq,Nkv,nblocks]
bmask_q=bmask.repeat_interleave(G,dim=1)  # [Tq,Nq,nblocks]  (each kv-head's mask shared by G q-heads)
emask=bmask_q.repeat_interleave(sbs,2)  # [Tq,Nq,Tkv]
sc=sc.masked_fill(~emask,float('-inf')); P=torch.softmax(sc,-1)
oref=torch.einsum('tns,sd->tnd',P,vf[:,0,:])
fwd_err=(o.float()-oref).abs().max().item()
print("CORRECT-shape fwd o vs ref maxabs:", fwd_err)
if fwd_err<0.02:
    print("  *** FWD MATCHES *** -> grad now meaningful")
    g=torch.randn_like(oref); oref.backward(g)
    dq,dk,dv=grad(g.to(torch.bfloat16),q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
    def cmp(n,a,b): a=a.float();b=b.float(); print(f"  {n}: rel={((a-b).norm()/(b.norm()+1e-9)).item():.4f} cos={torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),0).item():.4f}")
    cmp("dq",dq,qf.grad); cmp("dk",dk,kf.grad); cmp("dv",dv,vf.grad)
else:
    err=(o.float()-oref).abs().amax(-1); print("  per-(t,n) rows err>0.05:",(err>0.05).sum().item(),"of",Tq*Nq)
    print("  o[0,0,:4]",[round(x,3) for x in o[0,0,:4].float().tolist()],"ref",[round(x,3) for x in oref[0,0,:4].tolist()])
