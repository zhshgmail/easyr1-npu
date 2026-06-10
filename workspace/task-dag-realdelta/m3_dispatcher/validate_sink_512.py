import torch, torch_npu
from nsa_attn_sink_adapter import apply_attn_sink
torch.manual_seed(0); dev="npu:0"
# DSv4 MLA latent dims: D_qk=D_v=512 (kv_lora_rank). select-all-blocks -> dense, vs fp32 ref-with-sink.
B,S_q,Nq,Nkv,D_qk,D_v,sbs,sbc,S_kv = 1,64,4,1,512,512,64,16,1024
T_q,T_kv=S_q,S_kv; nblk=S_kv//sbs; scale=1.0/(D_qk**0.5); aq,akv=[S_q],[S_kv]
q=torch.randn(T_q,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(T_kv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(T_kv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
topk=torch.arange(nblk,dtype=torch.int32,device=dev).view(1,1,nblk).expand(T_q,Nq,sbc).contiguous()
attn_sink=torch.randn(Nq,dtype=torch.float32,device=dev)
out=torch_npu.npu_nsa_select_attention(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=aq,actual_seq_kvlen=akv)
o_native,smax,ssum=out
o_sink,lse=apply_attn_sink(o_native,smax,ssum,attn_sink)
# dense fp32 ref with sink
qf,kf,vf=q.float(),k.float()[:,0,:],v.float()[:,0,:]
s=scale*torch.einsum('tnd,kd->tnk',qf,kf); m=s.max(-1,keepdim=True).values
e=torch.exp(s-m); Z=e.sum(-1); num=torch.einsum('tnk,kd->tnd',e,vf)
S=torch.exp(attn_sink.view(1,Nq)-m.squeeze(-1)); o_ref=num/(Z+S).unsqueeze(-1)
def rel(x,y): return ((x-y).abs().max()/(y.abs().max()+1e-6)).item()
print("D_qk=D_v=512 (DSv4 MLA latent):")
print("  smax 8-wide all-equal:", bool((smax[0,0]==smax[0,0,0]).all().item()))
print("  o_native vs dense no-sink ref rel:", rel(o_native.float(), num/Z.unsqueeze(-1)))
print("  adapter o_sink vs dense-with-sink ref rel:", rel(o_sink, o_ref))
print("DONE")
