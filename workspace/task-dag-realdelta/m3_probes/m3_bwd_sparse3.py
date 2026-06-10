import torch, torch_npu
torch.manual_seed(3); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=8;S_kv=1024
nblocks=S_kv//sbs; Tq=32;Tkv=S_kv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
scale=1.0/(D_qk**0.5)
# HYPOTHESIS: all Nq heads share the SAME topk per query (GQA group shares selection)
topk_per_q=torch.stack([torch.randperm(nblocks,device=dev)[:sbc].sort().values for _ in range(Tq)])  # [Tq,sbc]
topk=topk_per_q.view(Tq,1,sbc).expand(Tq,Nq,sbc).contiguous().to(torch.int32)
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
qf=q.float().detach().requires_grad_(True); kf=k.float().detach().requires_grad_(True); vf=v.float().detach().requires_grad_(True)
sc=scale*torch.einsum('tnd,sd->tns',qf,kf[:,0,:])
bmask=torch.zeros(Tq,Nq,nblocks,device=dev,dtype=torch.bool).scatter_(2,topk.long(),True)
emask=bmask.repeat_interleave(sbs,2)
sc=sc.masked_fill(~emask,float('-inf')); P=torch.softmax(sc,-1)
oref=torch.einsum('tns,sd->tnd',P,vf[:,0,:])
fwd_err=(o.float()-oref).abs().max().item()
print("shared-topk fwd o vs ref maxabs:", fwd_err)
if fwd_err<0.02:
    print("  *** FWD MATCHES with SHARED topk -> grad now meaningful ***")
    g=torch.randn_like(oref); oref.backward(g)
    dq,dk,dv=grad(g.to(torch.bfloat16),q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
    def cmp(n,a,b): a=a.float();b=b.float(); print(f"  {n}: rel={((a-b).norm()/(b.norm()+1e-9)).item():.4f} cos={torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),0).item():.4f}")
    cmp("dq",dq,qf.grad); cmp("dk",dk,kf.grad); cmp("dv",dv,vf.grad)
else:
    # still off -> per-row diagnosis: which rows diverge?
    err=(o.float()-oref).abs().amax(dim=-1)  # [Tq,Nq]
    print("  per-(t,n) maxabs: max",err.max().item(),"; rows with err>0.05:", (err>0.05).sum().item(),"of",Tq*Nq)
