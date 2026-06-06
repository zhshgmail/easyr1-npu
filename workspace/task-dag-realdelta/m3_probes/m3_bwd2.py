import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
B=1;S_q=64;Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=16;S_kv=1024
T_q=B*S_q;T_kv=B*S_kv
q=torch.randn(T_q,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(T_kv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(T_kv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
topk=torch.randint(0,S_kv//sbs,(T_q,Nq,sbc),dtype=torch.int32,device=dev)
scale=1.0/(D_qk**0.5); aq=[S_q]; akv=[S_kv]
attn,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=aq,actual_seq_kvlen=akv)
dout=torch.randn_like(attn)
# CORRECT schema order: grad, query, key, value, attention_out, softmax_max, softmax_sum, topk_indices, scale, head_num, sbs, sbc
print("=== bwd correct-order ===",flush=True)
try:
    res=grad(dout,q,k,v,attn,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=aq,actual_seq_kvlen=akv)
    print("[OK] grad returns",len(res),"tensors")
    for i,t in enumerate(res):
        if torch.is_tensor(t): print(f"  d[{i}] shape={tuple(t.shape)} dtype={t.dtype} finite={torch.isfinite(t.float()).all().item()}")
except Exception as e:
    print("[ERR]",type(e).__name__,str(e)[:300])
