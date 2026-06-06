import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
print("=== npu_lightning_indexer fwd (BSND) ===",flush=True)
try:
    B=1;S=128;Nq=4;D=128
    q=torch.randn(B,S,Nq,D,dtype=torch.bfloat16,device=dev)
    k=torch.randn(B,S,1,D,dtype=torch.bfloat16,device=dev)
    w=torch.randn(B,S,Nq,dtype=torch.bfloat16,device=dev)  # weights [B,S,Nq]
    out=torch_npu.npu_lightning_indexer(q,k,w,layout_query="BSND",layout_key="BSND")
    print("[OK] returns",len(out),"tensors")
    for i,t in enumerate(out):
        if torch.is_tensor(t): print(f"  out[{i}] shape={tuple(t.shape)} dtype={t.dtype} finite={torch.isfinite(t.float()).all().item()}")
except Exception as e:
    print("[ERR]",type(e).__name__,str(e)[:300])

print("=== npu_nsa_compress_attention fwd (TND) ===",flush=True)
try:
    Nq=4;Nkv=1;D_qk=192;D_v=128;cbs=32;cstride=16;sbs=64;sbc=16
    Tq=64;Tkv=1024
    q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
    # compressed kv length = (Tkv-cbs)/cstride+1; just give a compressed-size k/v
    Tc=(Tkv-cbs)//cstride+1
    k=torch.randn(Tc,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
    v=torch.randn(Tc,Nkv,D_v,dtype=torch.bfloat16,device=dev)
    scale=1.0/(D_qk**0.5)
    out=torch_npu.npu_nsa_compress_attention(q,k,v,scale,Nq,cbs,cstride,sbs,sbc,
            actual_seq_qlen=[Tq],actual_cmp_seq_kvlen=[Tc],actual_sel_seq_kvlen=[Tkv//sbs])
    print("[OK] returns",len(out),"tensors")
    for i,t in enumerate(out):
        if torch.is_tensor(t): print(f"  out[{i}] shape={tuple(t.shape)} dtype={t.dtype} finite={torch.isfinite(t.float()).all().item()}")
except Exception as e:
    print("[ERR]",type(e).__name__,str(e)[:300])
