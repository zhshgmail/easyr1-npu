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
qf=q.float();kf=k[:,0,:].float();vf=v[:,0,:].float()
sc=scale*torch.einsum('tnd,sd->tns',qf,kf)
bmask=torch.zeros(Tq,Nkv,nblocks,device=dev,dtype=torch.bool).scatter_(2,topk.long(),True).repeat_interleave(G,1).repeat_interleave(sbs,2)
sc=sc.masked_fill(~bmask,float('-inf')); P=torch.softmax(sc,-1)
g=torch.randn(Tq,Nq,D_v,device=dev); gb=g.to(torch.bfloat16)
dq,dk,dv=grad(gb,q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
dv_manual=torch.einsum('tns,tnd->sd',P,g)  # [Tkv,D_v]
nv=dv[:,0,:].float(); mv=dv_manual.float()
def cos(a,b):a=a.float().flatten();b=b.float().flatten();return torch.nn.functional.cosine_similarity(a,b,0).item()
print("baseline dv cos:",round(cos(nv,mv),4))
# is dv nonzero only on SELECTED blocks? check sparsity structure
nv_blk = nv.view(nblocks,sbs,D_v).abs().sum(dim=(1,2))  # per-block magnitude
mv_blk = mv.view(nblocks,sbs,D_v).abs().sum(dim=(1,2))
print("native dv per-block nonzero blocks:", (nv_blk>1e-3).sum().item(),"of",nblocks)
print("manual dv per-block nonzero blocks:", (mv_blk>1e-3).sum().item(),"of",nblocks)
print("native dv block-mag[:8]:",[round(x,2) for x in nv_blk[:8].tolist()])
print("manual dv block-mag[:8]:",[round(x,2) for x in mv_blk[:8].tolist()])
# maybe native dv packs selected blocks CONTIGUOUSLY (compacted)? compare native[first sbc blocks] vs manual[selected blocks in topk order]
tk0=topk[0,0].tolist()  # NOTE topk is per-query; can't globally compact. but check row0's selected
print("topk[0,0]:",tk0)
# check: is native dv maybe = sum but only over a different head-grouping? try per-head dv (no sum)
# dv native is [Tkv,1,D]; manual summed all heads. try cos of native vs single-head manual
for h in range(Nq):
    mvh=torch.einsum('ts,td->sd',P[:,h,:],g[:,h,:])
    print(f"  native dv vs manual head{h} only: cos={cos(nv,mvh):.4f}")
