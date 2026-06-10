import torch, torch_npu
torch.manual_seed(3); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=8;S_kv=1024
nblocks=S_kv//sbs; Tq=32;Tkv=S_kv; G=Nq//Nkv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
scale=1.0/(D_qk**0.5)
# distinct blocks, NOT sorted (stored order)
topk=torch.stack([torch.stack([torch.randperm(nblocks,device=dev)[:sbc] for _ in range(Nkv)]) for _ in range(Tq)]).to(torch.int32)  # [Tq,Nkv,sbc] unsorted
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
# GATHER-based ref in stored order: for each (t), build selected KV index list = concat of blocks in topk order
kf=k[:,0,:].float(); vf=v[:,0,:].float()
qf=q.float()
# selected element indices per t: [Tq, sbc*sbs], in stored topk order
blk=topk[:,0,:].long()  # [Tq,sbc]
elem_idx = (blk.unsqueeze(-1)*sbs + torch.arange(sbs,device=dev)).view(Tq,sbc*sbs)  # [Tq, M] M=sbc*sbs, stored order
M=sbc*sbs
# gather K_sel,V_sel per t: [Tq,M,D]
Ksel = kf[elem_idx]   # [Tq,M,D_qk]
Vsel = vf[elem_idx]   # [Tq,M,D_v]
Ksel.requires_grad_(True); Vsel.requires_grad_(True)
# per q-head: scores = scale * q . Ksel ; softmax over M ; o = P @ Vsel
scg = scale*torch.einsum('tnd,tmd->tnm', qf, Ksel)  # [Tq,Nq,M]
Pg = torch.softmax(scg,-1)
og = torch.einsum('tnm,tmd->tnd', Pg, Vsel)  # [Tq,Nq,D_v]
print("gather-ref fwd o vs native maxabs:", (o.float()-og).abs().max().item())
g=torch.randn_like(og); og.backward(g)
dKsel, dVsel = Ksel.grad.clone(), Vsel.grad.clone()  # [Tq,M,D]  grads in gathered rep
# scatter back to full KV with AtomicAdd (index_add handles duplicates, but distinct here)
dk_ref=torch.zeros(Tkv,D_qk,device=dev); dv_ref=torch.zeros(Tkv,D_v,device=dev)
for t in range(Tq):
    dk_ref.index_add_(0, elem_idx[t], dKsel[t])
    dv_ref.index_add_(0, elem_idx[t], dVsel[t])
# native
dq,dk,dv=grad(g.to(torch.bfloat16),q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
def cos(a,b):a=a.float().flatten();b=b.float().flatten();return torch.nn.functional.cosine_similarity(a,b,0).item()
print("dk: native vs gather-scatter ref cos=%.4f"%cos(dk[:,0,:],dk_ref))
print("dv: native vs gather-scatter ref cos=%.4f"%cos(dv[:,0,:],dv_ref))
