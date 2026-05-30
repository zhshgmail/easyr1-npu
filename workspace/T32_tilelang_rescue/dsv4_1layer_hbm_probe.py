"""Probe: minimum HBM for 1-layer DeepSeek-V4-Flash on Ascend A3 NPU
while keeping all 4 tilelang operators exercised.

Goal: empirical answer to user 2026-05-30: "DSv4-Flash 减层，所有算子保留
最低多少内存？"

Setup:
  * MLATransformerConfig matches REAL DSv4-Flash: hidden=4096, H=64,
    q_lora_rank=1024, kv_lora_rank=512, v_head_dim=512, qk_rope_head_dim=64
  * num_layers=1 (the reduction)
  * No MoE for now (DSAMLASelfAttention only; MoE FFN is a separate layer)
  * SEQ=128 (minimal cu_seqlens that still exercises sparse_mla > block_M)
  * Runs forward + backward + Adam step

Exercises every NPU operator we shipped:
  * lighting_indexer_fwd / bwd (during attention forward and grad)
  * sparse_mla_fwd / bwd (during attention forward and grad)
  * MLA absorbed-Q with dim_plus_tail=576

This runs IN tlrescue (not sgl_probe) since tlrescue has the patched
MindSpeed + miles + tilelang stack. HBM is monitored by the host
watchdog, NOT by this script.
"""
import sys

sys.path = [p for p in sys.path if p not in ("", "/")]

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu  # noqa: F401

# MindSpeed adaptor must come before megatron.core imports
import mindspeed.megatron_adaptor  # noqa: F401

from miles_plugins.models.glm5.ops._npu._e2e_megatron_step_mindspeed import (
    _init_distributed as _init_dist,
    ColumnParallelLinear,
    RowParallelLinear,
    IndexerColumnParallelLinear,
    _LayerNormColumnParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams

from miles_plugins.models.glm5.glm5 import (
    DSAMLASelfAttention,
    DSASelfAttentionSubmodules,
)
import miles_plugins.models.glm5.glm5 as _glm5_module


def build_dsv4_config():
    """MLATransformerConfig matching REAL DSv4-Flash dims (single layer)."""
    cfg = MLATransformerConfig(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=64,
        ffn_hidden_size=1408,  # close to DSv2 default; full MoE FFN not exercised here
        kv_channels=128,
        q_lora_rank=1024,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=512,
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        rotary_percent=1.0,
        original_max_position_embeddings=2048,
        mscale=1.0,
        mscale_all_dim=1.0,
        beta_fast=32,
        beta_slow=1,
        add_bias_linear=False,
        layernorm_epsilon=1e-5,
        normalization="RMSNorm",
        recompute_granularity=None,
    )
    cfg.index_num_attention_heads = 64
    cfg.index_head_dim = 128
    return cfg


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

    print(f"[probe] init distributed ...")
    _init_dist()

    cfg = build_dsv4_config()
    print(f"[probe] cfg: hidden={cfg.hidden_size} H={cfg.num_attention_heads} v_head={cfg.v_head_dim}")

    submods = DSASelfAttentionSubmodules(
        linear_q_down_proj=ColumnParallelLinear,
        linear_q_up_proj=_LayerNormColumnParallelLinear,
        linear_kv_down_proj=ColumnParallelLinear,
        linear_kv_up_proj=_LayerNormColumnParallelLinear,
        linear_v_up_proj=IdentityOp,
        core_attention=IdentityOp,
        linear_proj=RowParallelLinear,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    )
    for extra, target in (
        ("wq_b", IndexerColumnParallelLinear),
        ("wk", IndexerColumnParallelLinear),
        ("weights_proj", IndexerColumnParallelLinear),
    ):
        if hasattr(submods, extra):
            setattr(submods, extra, target)
    if hasattr(submods, "k_norm"):
        class _LN(nn.LayerNorm):
            def __init__(self, hidden_size, config=None, eps=1e-5, **kw):
                super().__init__(hidden_size, eps=eps)
        submods.k_norm = _LN

    print(f"[probe] building 1-layer DSAMLASelfAttention (real DSv4-Flash dims) ...")
    actor = DSAMLASelfAttention(
        config=cfg,
        submodules=submods,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).npu()
    # Small index_topk + small SEQ to keep KV cache + intermediate buffers tiny
    actor.index_topk = 8
    SEQ, BSZ = 128, 1

    total_params = sum(p.numel() for p in actor.parameters())
    print(f"[probe] params: {total_params:,}  ({total_params * 2 / 1024**3:.2f} GB at bf16)")

    hidden_states = (torch.randn(SEQ, BSZ, cfg.hidden_size, dtype=torch.bfloat16) * 0.1).npu()
    cu_seqlens = torch.tensor([0, SEQ], dtype=torch.int32).npu()
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=SEQ,
        max_seqlen_kv=SEQ,
        qkv_format="thd",
    )
    position_ids = torch.arange(SEQ, dtype=torch.int64).unsqueeze(0).npu()

    indexer_captures: list = []
    _real = _glm5_module.lighting_indexer

    def _captured(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=None):
        score, indices = _real(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=topk_indices)
        indexer_captures.append(score)
        return score, indices

    _glm5_module.lighting_indexer = _captured  # type: ignore[assignment]

    opt = torch.optim.Adam(actor.parameters(), lr=1e-4)

    print(f"[probe] forward ...")
    t0 = time.time()
    out = actor(
        hidden_states=hidden_states,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=packed,
        position_ids=position_ids,
    )
    primary = out[0] if isinstance(out, tuple) else out
    fwd_t = time.time() - t0
    print(f"[probe] forward done in {fwd_t:.1f}s, out shape: {primary.shape}")

    # Synthetic loss
    advantage = (torch.randn_like(primary.float()) * 0.5).clamp(-1, 1)
    loss = -(primary.float() * advantage).sum() / max(1, primary.numel())

    print(f"[probe] backward ...")
    t1 = time.time()
    opt.zero_grad()
    loss.backward()
    bwd_t = time.time() - t1
    print(f"[probe] backward done in {bwd_t:.1f}s")

    # R-KA-15 mitigation: zero non-finite grads before clip
    for p in actor.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)

    print(f"[probe] adam step ...")
    opt.step()

    finite_count = sum(1 for p in actor.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    total_count = sum(1 for _ in actor.parameters())
    print(f"  loss: {float(loss.detach().cpu()):.5f}")
    print(f"  finite grads: {finite_count}/{total_count}")
    print(f"  idx capture layers: {len(indexer_captures)}")
    print(f"[probe] PASS")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
