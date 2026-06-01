"""V4 indexer -> CANN npu_lightning_indexer shim (NPU dispatch for batched_indexer_fwd)."""
import torch, torch_npu

def indexer_fwd_npu(q, k, weights, cu_ks=None, cu_ke=None):
    # miles batched_indexer_fwd(q[T1,H,D], k[T2,D], weights[T1,H], cu_ks, cu_ke) -> index_scores
    # CANN npu_lightning_indexer(query, key, weights, *, actual_seq_lengths_*, layout, sparse_count, sparse_mode)
    # layout BSND: query [B,S1,N1,D], key [B,S2,N2,D], weights [B,S1,N1]
    T1, H, D = q.shape
    T2 = k.shape[0]
    qb = q.view(1, T1, H, D).contiguous()
    kb = (k.view(1, T2, 1, D) if k.dim()==2 else k.view(1, T2, k.shape[1], D)).contiguous()
    wb = weights.view(1, T1, H).contiguous()
    aslq = torch.tensor([T1], dtype=torch.int32, device=q.device)
    aslk = torch.tensor([T2], dtype=torch.int32, device=q.device)
    out = torch_npu.npu_lightning_indexer(
        qb, kb, wb,
        actual_seq_lengths_query=aslq, actual_seq_lengths_key=aslk,
        layout_query="BSND", layout_key="BSND",
    )
    return out

if __name__ == "__main__":
    torch.npu.set_device(0); torch.manual_seed(0)
    T1, H, D, T2 = 128, 4, 128, 128
    q = torch.randn(T1, H, D, dtype=torch.float16, device="npu:0")
    k = torch.randn(T2, D, dtype=torch.float16, device="npu:0")
    w = torch.randn(T1, H, dtype=torch.float16, device="npu:0")
    try:
        o = indexer_fwd_npu(q, k, w)
        torch.npu.synchronize()
        oo = o[0] if isinstance(o,(tuple,list)) else o
        print(f"[V4 indexer shim] npu_lightning_indexer OK: out={tuple(oo.shape)} dev={oo.device} finite={torch.isfinite(oo.float()).all().item()}")
        print("=> miles batched_indexer_fwd can dispatch to CANN npu_lightning_indexer")
    except Exception as e:
        print("[V4 indexer shim] FAIL:", type(e).__name__, str(e)[:250])
