import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
fwd=torch_npu.npu_lightning_indexer; grad=torch_npu.npu_lightning_indexer_grad
B=1;S=128;Nq=4;D=128
q=torch.randn(B,S,Nq,D,dtype=torch.bfloat16,device=dev)
k=torch.randn(B,S,1,D,dtype=torch.bfloat16,device=dev)
w=torch.randn(B,S,Nq,dtype=torch.bfloat16,device=dev)
idx,val=fwd(q,k,w,layout_query="BSND",layout_key="BSND",sparse_count=S,sparse_mode=3,return_value=True)
# try a few dy shapes/dtypes; val is [B,S,1,S]
for name,dy in [("dy=randn_like(val) bf16", torch.randn_like(val)),
                ("dy fp32", torch.randn_like(val).float()),
                ("dy [B,S,Nq,S]", torch.randn(B,S,Nq,S,dtype=torch.bfloat16,device=dev))]:
    try:
        res=grad(q,k,dy,idx,w,layout="BSND",sparse_mode=3)
        print(f"[OK] {name}: returns {len(res)}; finite={[torch.isfinite(t.float()).all().item() for t in res if torch.is_tensor(t)]}")
    except Exception as e:
        print(f"[ERR] {name}: {str(e)[:90]}")
