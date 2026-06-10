import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
# indexer grad schema
def sch(n):
    try:
        for s in torch._C._jit_get_schemas_for_operator(f"npu::{n}"): return str(s)
    except Exception as e: return f"(fail {e})"
print("indexer_grad schema:", sch("npu_lightning_indexer_grad"))
print("compress_grad schema:", sch("npu_nsa_compress_grad"))
