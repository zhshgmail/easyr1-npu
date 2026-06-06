import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
def sch(n):
    try:
        for s in torch._C._jit_get_schemas_for_operator(f"npu::{n}"): return str(s)
    except Exception as e: return f"(fail {e})"
print("compress_grad schema:", sch("npu_nsa_compress_attention_grad"), flush=True)
print("indexer_grad schema:", sch("npu_lightning_indexer_grad"), flush=True)

# --- indexer fwd with TUNED params (sparse_count=S, return_value=True) for clean values ---
print("=== indexer fwd tuned (sparse_count=S, return_value=True) ===", flush=True)
try:
    B=1;S=128;Nq=4;D=128
    q=torch.randn(B,S,Nq,D,dtype=torch.bfloat16,device=dev)
    k=torch.randn(B,S,1,D,dtype=torch.bfloat16,device=dev)
    w=torch.randn(B,S,Nq,dtype=torch.bfloat16,device=dev)
    out=torch_npu.npu_lightning_indexer(q,k,w,layout_query="BSND",layout_key="BSND",sparse_count=S,sparse_mode=3,return_value=True)
    for i,t in enumerate(out):
        if torch.is_tensor(t): print(f"  out[{i}] shape={tuple(t.shape)} dtype={t.dtype} finite={torch.isfinite(t.float()).all().item()}")
except Exception as e:
    print("[ERR]",type(e).__name__,str(e)[:250])
