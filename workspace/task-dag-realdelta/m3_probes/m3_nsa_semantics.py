import torch, torch_npu
torch.manual_seed(2); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention
# minimal: 1 query, find which KV blocks actually influence o by topk selection
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=2;S_kv=1024  # select 2 of 16 blocks
nblocks=S_kv//sbs
Tq=1;Tkv=S_kv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
scale=1.0/(D_qk**0.5)
# select blocks 0 and 5 for all heads
topk=torch.tensor([0,5],dtype=torch.int32,device=dev).view(1,1,2).expand(Tq,Nq,sbc).contiguous()
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
print("o shape",tuple(o.shape))
# reference: dense softmax over ONLY blocks 0 and 5 (elements 0..63 and 320..383)
qf=q.float();kf=k[:,0,:].float();vf=v[:,0,:].float()
sel=torch.cat([torch.arange(0,64),torch.arange(5*64,6*64)]).to(dev)
sc=scale*torch.einsum('tnd,sd->tns',qf,kf[sel])  # [1,Nq,128]
P=torch.softmax(sc,-1)
oref=torch.einsum('tns,sd->tnd',P,vf[sel])
print("o[0,0,:4] native:",[round(x,3) for x in o[0,0,:4].float().tolist()])
print("o[0,0,:4] ref   :",[round(x,3) for x in oref[0,0,:4].tolist()])
print("maxabs o vs block-{0,5}-softmax ref:", (o.float()-oref).abs().max().item())
# also try: maybe selection is per-head independent but topk here same -> ok. try element-level? print smax
print("smax[0,0,0] (row max):", smax[0,0,0].item())
# what's the actual max over selected vs all?
print("dense-all max logit:", (scale*torch.einsum('tnd,sd->tns',qf,kf)).max().item())
print("block-{0,5} max logit:", sc.max().item())
