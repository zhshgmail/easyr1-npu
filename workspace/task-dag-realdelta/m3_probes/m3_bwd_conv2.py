import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=16;S_kv=1024
Tq=64;Tkv=S_kv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
topk=torch.arange(sbc,dtype=torch.int32,device=dev).view(1,1,sbc).expand(Tq,Nq,sbc).contiguous()
scale=1.0/(D_qk**0.5)
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
qf=q.float().detach().requires_grad_(True); kf=k.float().detach().requires_grad_(True); vf=v.float().detach().requires_grad_(True)
sc = scale * torch.einsum('tnd,sd->tns', qf, kf[:,0,:]); P=torch.softmax(sc,-1)
oref=torch.einsum('tns,sd->tnd',P,vf[:,0,:])
g=torch.randn_like(oref)
oref.backward(g)
dq_ref,dk_ref,dv_ref=qf.grad.clone(),kf.grad.clone(),vf.grad.clone()
def cmp(name,nat,ref):
    nat=nat.float();ref=ref.float()
    print(f"  {name}: rel={((nat-ref).norm()/(ref.norm()+1e-9)).item():.4f} cos={torch.nn.functional.cosine_similarity(nat.flatten(),ref.flatten(),0).item():.4f} nnorm={nat.norm():.2f} refnorm={ref.norm():.2f}")
gb=g.to(torch.bfloat16)
# dq is the most diagnostic (per-head, no aggregation ambiguity). Check it across variants.
print("=== B: smax/ssum [...,0:1] (keep dim, 1-wide) ===")
try:
    dq,dk,dv=grad(gb,q,k,v,o,smax[...,:1],ssum[...,:1],topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv]); cmp("dq",dq,dq_ref)
except Exception as e: print("  ERR",str(e)[:150])
print("=== C: dout fp32 ===")
try:
    dq,dk,dv=grad(g.float(),q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv]); cmp("dq",dq,dq_ref)
except Exception as e: print("  ERR",str(e)[:150])
# Is dq maybe correct up to the softmax-max/sum NOT being LSE? try feeding lse = max+log(sum) in both slots
print("=== D: feed lse=max+log(sum) as smax, ones as ssum ===")
try:
    lse=(smax[...,:1].float()+torch.log(ssum[...,:1].float())).to(smax.dtype).expand(-1,-1,8).contiguous()
    dq,dk,dv=grad(gb,q,k,v,o,lse,torch.ones_like(ssum),topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv]); cmp("dq",dq,dq_ref)
except Exception as e: print("  ERR",str(e)[:150])
# diagnostic: maybe native dq is correct but my ref scale is off — print ratio per-head
print("=== diag: native dq[0,0,:4] vs ref dq[0,0,:4] ===")
dqA,_,_=grad(gb,q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
print("  native:", [round(x,3) for x in dqA[0,0,:4].float().tolist()])
print("  ref   :", [round(x,3) for x in dq_ref[0,0,:4].tolist()])
print("  ratio :", [round((dqA[0,0,i].float()/(dq_ref[0,0,i]+1e-9)).item(),3) for i in range(4)])
