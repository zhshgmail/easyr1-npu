import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
op=torch_npu.npu_nsa_select_attention
# TND layout: query [T_q, N, D], T_q = B*S_q flattened; actual_seq_* = cumulative seqlens
B=1;S_q=64;Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=16;S_kv=1024
T_q=B*S_q; T_kv=B*S_kv
q=torch.randn(T_q,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(T_kv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(T_kv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
# topk_indices TND: [T_q, N, sbc]
topk=torch.randint(0,S_kv//sbs,(T_q,Nq,sbc),dtype=torch.int32,device=dev)
scale=1.0/(D_qk**0.5)
aq=[S_q]; akv=[S_kv]  # one batch, cumulative seqlens
print("=== TND call ===", flush=True)
try:
    out=op(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=aq,actual_seq_kvlen=akv)
    print("[OK] returns",len(out),"tensors")
    for i,t in enumerate(out):
        if torch.is_tensor(t): print(f"  out[{i}] shape={tuple(t.shape)} dtype={t.dtype} finite={torch.isfinite(t.float()).all().item()}")
except Exception as e:
    print("[ERR]",type(e).__name__,str(e)[:400])
