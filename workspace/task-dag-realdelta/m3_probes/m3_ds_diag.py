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
# fwd-matching P (float)
qf=q.float();kf=k[:,0,:].float();vf=v[:,0,:].float()
sc=scale*torch.einsum('tnd,sd->tns',qf,kf)
bmask=torch.zeros(Tq,Nkv,nblocks,device=dev,dtype=torch.bool).scatter_(2,topk.long(),True).repeat_interleave(G,1).repeat_interleave(sbs,2)
sc=sc.masked_fill(~bmask,float('-inf')); P=torch.softmax(sc,-1)  # [Tq,Nq,Tkv]
g=torch.randn(Tq,Nq,D_v,device=dev)  # dO
gb=g.to(torch.bfloat16)
dq,dk,dv=grad(gb,q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
def cos(a,b): a=a.float().flatten();b=b.float().flatten(); return torch.nn.functional.cosine_similarity(a,b,0).item()
# (1) manual dv = sum over q-heads of P_n^T @ g_n  (scatter-add into shared kv head)
dv_manual = torch.einsum('tns,tnd->sd', P, g)  # [Tkv,D_v], summed over t,n
print("dv: native vs manual P^T@dO  cos=%.4f  rel=%.4f"%(cos(dv[:,0,:],dv_manual), ((dv[:,0,:].float()-dv_manual).norm()/dv_manual.norm()).item()))
# (2) dS variants. dP = g @ v^T  [Tq,Nq,Tkv]
dP = torch.einsum('tnd,sd->tns', g, vf)
Di = (P*dP).sum(-1,keepdim=True)  # rowsum(dP*P) ... but FA uses rowsum(dO o O)=rowsum over D_v of (g*o)
Di2 = (g*o.float()).sum(-1,keepdim=True)  # [Tq,Nq,1] the FA D_i
dS_withDi  = P*(dP - Di)
dS_withDi2 = P*(dP - Di2)
dS_noDi    = P*dP
# dq = scale * dS @ K  (per head)
def dq_from(dS): return scale*torch.einsum('tns,sd->tnd', dS, kf)  # [Tq,Nq,D_qk]
for name,dS in [("P*(dP-rowsum(P*dP))",dS_withDi),("P*(dP-rowsum(g*o))",dS_withDi2),("P*dP(no Di)",dS_noDi)]:
    dqv=dq_from(dS); print("dq native vs %s: cos=%.4f"%(name,cos(dq,dqv)))
# also without scale
print("dq native vs (no-scale dS@K, P*(dP-Di2)): cos=%.4f"%cos(dq, torch.einsum('tns,sd->tnd',dS_withDi2,kf)))
