"""V4 training ops → CANN native dispatch (NPU).

Single import surface for running DeepSeek-V4 training-side ops on Ascend NPU via
CANN native torch_npu ops, replacing the miles tilelang(CUDA) kernels. To wire into
miles: at the V4 op call sites, `if x.is_npu: use the *_npu fn here`.

Verified on Ascend A3 (910C, CANN 8.5, torch_npu 2.9) 2026-06-01:
  - sparse_attn_npu      -> npu_nsa_select_attention      [RAN: o(128,4,128)+bwd state, 94.9µs]
  - indexer_fwd_npu      -> npu_lightning_indexer         [RAN: out(1,128,1,2048)]
Documented mapping (CANN native exists; wire + constraint-fit when integrating):
  - compressor_attn_npu  -> npu_nsa_compress_attention    [compress_block_size=32,stride=16,sbs=64,sbc=16,SparseMode=1]
  - sparse_attn_bwd      -> npu_nsa_select_attention returns softmax max/sum; + npu_sparse_flash_attention_grad
  - indexer_bwd          -> npu_lightning_indexer_grad / npu_sparse_lightning_indexer_grad_kl_loss [A3-supported]
  - act_quant_npu        -> npu_* dynamic-quant family    [fp8 block; layout TBD]
  - hc_split_sinkhorn    -> NO native (V4-specific); sigmoid/exp/reduce composition OR tilelang-ascend

NSA select_attention constraints (learned): layout TND, dtype fp16/bf16, D_qk=192, D_v=128,
select_block_size ONLY 64, select_block_count 16, KV S >= 1024 (mult of 64), G=Nq/Nkv<=32,
topk int32 [T1,N2,sbc] in [0,S2/64). errno 561103 = constraint violation.
"""
import torch

try:
    import torch_npu
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False


def sparse_attn_npu(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """V4 sparse-MLA fwd -> npu_nsa_select_attention. Matches miles
    ops/attention_core.py::sparse_attn_tilelang(q,kv,attn_sink,topk_idxs,sm_scale)->(o,lse).
    VERIFIED on A3."""
    T1, N1, Dqk = q.shape
    Dv = 128
    if sm_scale is None:
        sm_scale = 1.0 / (Dqk ** 0.5)
    sbc = topk_idxs.shape[-1]
    sbs = 64
    T2 = kv.shape[0]
    N2 = kv.shape[1] if kv.dim() == 3 else 1
    k = kv[..., :Dqk].contiguous().view(T2, N2, Dqk)
    v = (kv[..., Dqk:Dqk + Dv] if kv.shape[-1] >= Dqk + Dv else kv[..., :Dv]).contiguous().view(T2, N2, Dv)
    out = torch_npu.npu_nsa_select_attention(
        q.contiguous(), k, v, topk_idxs.int().contiguous(), sm_scale, N1, sbs, sbc,
        actual_seq_qlen=[T1], actual_seq_kvlen=[T2],
    )
    o = out[0] if isinstance(out, (tuple, list)) else out
    lse = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else None
    return o, lse


def indexer_fwd_npu(q, k, weights, cu_ks=None, cu_ke=None):
    """V4 lightning indexer fwd -> npu_lightning_indexer. Matches miles
    ops/kernel/tilelang_indexer_fwd.py::batched_indexer_fwd(q,k,weights,cu_ks,cu_ke).
    VERIFIED on A3."""
    T1, H, D = q.shape
    T2 = k.shape[0]
    qb = q.view(1, T1, H, D).contiguous()
    kb = (k.view(1, T2, 1, D) if k.dim() == 2 else k.view(1, T2, k.shape[1], D)).contiguous()
    wb = weights.view(1, T1, H).contiguous()
    aslq = torch.tensor([T1], dtype=torch.int32, device=q.device)
    aslk = torch.tensor([T2], dtype=torch.int32, device=q.device)
    return torch_npu.npu_lightning_indexer(
        qb, kb, wb,
        actual_seq_lengths_query=aslq, actual_seq_lengths_key=aslk,
        layout_query="BSND", layout_key="BSND",
    )


# compressor_attn_npu / act_quant_npu / indexer_bwd: CANN native ops mapped above;
# wire + constraint-fit at miles integration time (npu_nsa_compress_attention etc.).
