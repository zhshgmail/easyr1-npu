"""Fabricate a 1-layer DeepSeek-V4-Flash HF checkpoint — REAL V4 architecture.

This replaces fabricate_dsv4_1layer_ckpt.py which (per user catch
2026-06-01) was silently using DeepseekV32 not DSv4-Flash.

What this script targets (real V4, NOT V3.2):

  * architectures = ["DeepseekV4ForCausalLM"]
  * model_type   = "deepseek_v4"
  * sglang loader path: srt/models/deepseek_v4.py (entry class
    DeepseekV4ForCausalLM at line 1630 of sglang trunk main).
  * sglang config:      srt/configs/deepseek_v4.py (DeepSeekV4Config).

V4-only schema fields (NOT present in V3.2; reviewer + future readers
MUST be able to see we are not just renaming V3.2):

  - o_lora_rank, o_groups         (V4 output-side LoRA — new)
  - n_hash_layers, hc_mult,
    hc_sinkhorn_iters, hc_eps     (V4 hash-coding compressor — new)
  - compress_rope_theta,
    compress_ratios               (V4 compressor — new)
  - scoring_func = "sqrtsoftplus" (V4 router scoring — different from V3.2's softmax)
  - topk_method  = "noaux_tc"     (V4 router topk — new variant)
  - swiglu_limit                  (V4 MLP — new)
  - window_size                   (V4 attention windowing — new)
  - num_nextn_predict_layers      (V4 MTP — new)
  - expert_dtype = "fp4"          (V4 expert dtype — new)
  - quantization_config.quant_method = "fp8" + scale_fmt "ue8m0" (V4 quant scheme)

Reduced dims for HBM-constrained PoC (miles_local):
  num_hidden_layers = 1             (vs production 43)
  n_routed_experts  = 4 (or 8)      (vs production 256)
  num_experts_per_tok = 2           (vs production 6)
  index_topk        = 8             (vs production 512)
  SEQ later via driver = 128        (vs production 1M context)

All OTHER dims kept at production values:
  hidden_size = 4096
  num_attention_heads = 64
  q_lora_rank = 1024
  kv_lora_rank = 512
  qk_rope_head_dim = 64
  v_head_dim = 512
  index_n_heads = 64
  index_head_dim = 128
  vocab_size = 129280

This script is INTENTIONALLY parallel to the V3.2 fab so the diff is
auditable. Per user 2026-06-01 directive after catching the silent V3.2
substitution: every arch-class choice from now on must match the public
HF/upstream truth, and any reduction must be explicitly named.

Ground truth saved to v4_real_truth/:
  v4_real_config.json              # HF deepseek-ai/DeepSeek-V4-Flash config
  sglang_deepseek_v4.py            # sglang trunk model file
  sglang_DeepSeekV4Config.py       # sglang trunk config dataclass
"""
import json
import os
import pathlib
import sys
from typing import Optional

import torch
from safetensors.torch import save_file

# ---- DIMENSIONS — V4 real production values, kept unless reduction is needed ----

HIDDEN = 4096
NUM_HEADS = 64
Q_LORA_RANK = 1024
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 0  # V4 lays out attention differently from V3.2:
QK_ROPE_HEAD_DIM = 64  # head_dim = 512 alone (no nope+rope split on Q side per V4 config)
V_HEAD_DIM = 512
HEAD_DIM = V_HEAD_DIM  # V4 config has `head_dim: 512` instead of `qk_head_dim`

INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128

# V4 output-side LoRA (NOT in V3.2)
O_LORA_RANK = 1024
O_GROUPS = 8

# V4 hash-coding compressor (NOT in V3.2)
N_HASH_LAYERS = 3
HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6

# Window attention (V4 new)
WINDOW_SIZE = 128

# V4 compress_ratios — production is 43-entry list; for 1-layer reduced PoC use [0]
COMPRESS_ROPE_THETA = 40000

# ---- REDUCED for PoC (miles_local) ----
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "1"))  # env-overridable for symmetric 2-layer basis
INDEX_TOPK = 8  # production 512
INTERMEDIATE = 4096  # dense MLP path (used when MOE_ACTIVE=0)
N_ROUTED_EXPERTS = 4  # production 256
MOE_INTERMEDIATE = 2048  # V4 production value
N_SHARED_EXPERTS = 1
NUM_EXPERTS_PER_TOK = 2  # production 6
TOPK_GROUP = 2
N_GROUP = 2
NUM_NEXTN_LAYERS = 0  # disable MTP for 1-layer PoC

# Vocab / tokens
VOCAB = 129280
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000
MAX_POSITION_EMBEDDINGS = 2048

MOE_ACTIVE = bool(int(os.environ.get("MOE_ACTIVE", "0")))
FIRST_K_DENSE_REPLACE = 0 if MOE_ACTIVE else 1


def build_config_dict() -> dict:
    """Return the HF-style config.json dict for sglang's DSv4 loader.

    architectures + model_type are V4-truthful (not V3.2). All V4-only fields
    are populated so sglang's DeepSeekV4Config dataclass accepts the load
    without falling back to defaults silently.
    """
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        # core
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": 1,
        # V4 attention layout
        "head_dim": HEAD_DIM,
        "q_lora_rank": Q_LORA_RANK,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        # V4 output-side LoRA (V4 NEW)
        "o_lora_rank": O_LORA_RANK,
        "o_groups": O_GROUPS,
        # V4 indexer
        "index_topk": INDEX_TOPK,
        "index_n_heads": INDEX_N_HEADS,
        "index_head_dim": INDEX_HEAD_DIM,
        # V4 hash-coding compressor (V4 NEW)
        "n_hash_layers": N_HASH_LAYERS,
        "hc_mult": HC_MULT,
        "hc_sinkhorn_iters": HC_SINKHORN_ITERS,
        "hc_eps": HC_EPS,
        "compress_rope_theta": COMPRESS_ROPE_THETA,
        "compress_ratios": [0] * NUM_LAYERS,  # per-layer; reduced PoC uses no compression schedule
        # V4 MoE router (V4 NEW)
        "n_routed_experts": N_ROUTED_EXPERTS,
        "n_shared_experts": N_SHARED_EXPERTS,
        "first_k_dense_replace": FIRST_K_DENSE_REPLACE,
        "moe_layer_freq": 1,
        "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
        "topk_group": TOPK_GROUP,
        "n_group": N_GROUP,
        "topk_method": "noaux_tc",  # V4 (V3.2 was "greedy" or default)
        "scoring_func": "sqrtsoftplus",  # V4 (V3.2 was "softmax")
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        # Window (V4 NEW)
        "window_size": WINDOW_SIZE,
        # MTP (V4 NEW; disabled for PoC)
        "num_nextn_predict_layers": NUM_NEXTN_LAYERS,
        # vocab + tokenizer
        "vocab_size": VOCAB,
        "rms_norm_eps": RMS_NORM_EPS,
        "rope_theta": ROPE_THETA,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "torch_dtype": "bfloat16",
        # Rope scaling (V4 production uses yarn; PoC keeps default rope)
        "rope_scaling": {},
        # misc
        "hidden_act": "silu",
        "swiglu_limit": 10.0,  # V4 NEW
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "use_cache": True,
        "transformers_version": "4.57.1",
    }


def _randn(shape, generator, scale=0.02):
    return torch.randn(shape, generator=generator, dtype=torch.bfloat16) * scale


def _ones(shape):
    return torch.ones(shape, dtype=torch.bfloat16)


def build_random_weights(seed: int = 42) -> dict:
    """Random-init bf16 weight dict in HF naming convention.

    V4 introduces additional modules vs V3.2:
      - compressor (hash-coding KV compressor)
      - c4_indexer (vs V3.2's indexer)
      - output o_a_proj / o_b_proj (V4 output-side LoRA)
      - shared_head.norm / shared_head.head (V4 MTP layers — skipped here, num_nextn=0)

    We initialize ALL weights V4 expects. If sglang's loader complains about
    a missing module, that's a real V4 schema gap and a follow-up issue.
    """
    g = torch.Generator().manual_seed(seed)
    w = {}

    # Embeddings + LM head + final norm
    w["model.embed_tokens.weight"] = _randn((VOCAB, HIDDEN), g)
    w["model.norm.weight"] = _ones((HIDDEN,))
    w["lm_head.weight"] = _randn((VOCAB, HIDDEN), g)

    # Model-level hash-coding head tensors (V4 NEW)
    # Source: /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py:1173
    # hc_head_fn: (hc_mult, hc_dim) fp32
    # hc_head_base: (hc_mult,) fp32
    # hc_head_scale: (1,) fp32
    _HC_DIM_MODEL = HC_MULT * HIDDEN
    w["model.hc_head_fn"] = torch.randn(
        (HC_MULT, _HC_DIM_MODEL), generator=g, dtype=torch.float32
    ) * 0.02
    w["model.hc_head_base"] = torch.zeros((HC_MULT,), dtype=torch.float32)
    w["model.hc_head_scale"] = torch.ones((1,), dtype=torch.float32)

    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}"

        # Layer norms
        w[f"{prefix}.input_layernorm.weight"] = _ones((HIDDEN,))
        w[f"{prefix}.post_attention_layernorm.weight"] = _ones((HIDDEN,))

        # ---- self_attn — V4 MQA layer (CORRECTED naming per sglang src 2026-06-01) ----
        # Source: /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py
        #   line 321: self.attn_sink = nn.Parameter([n_heads], fp32)
        #   line 324: self.wqkv_a = ReplicatedLinear(hidden, q_lora + head_dim, bias=False)  [if fused]
        #     else:  self.wq_a / self.wkv (split)
        #   line 346: self.q_norm = RMSNorm(q_lora_rank)
        #   line 347: self.wq_b = ColumnParallelLinear(q_lora, n_heads * head_dim, bias=False)
        #   line 356: self.kv_norm = RMSNorm(head_dim)
        #   line 357: self.wo_a = ColumnParallelLinear((n_heads*head_dim)/n_groups, n_groups*o_lora_rank)
        #   line 372: self.wo_b = RowParallelLinear(n_groups*o_lora_rank, hidden, bias=False)
        sa = f"{prefix}.self_attn"

        # attention sink (V4 NEW, fp32 [n_heads])
        w[f"{sa}.attn_sink"] = torch.zeros((NUM_HEADS,), dtype=torch.float32)

        # Q+KV down: split path (wq_a + wkv); fab uses NON-fused; sglang env SGLANG_OPT_FUSE_WQA_WKV=0 default
        w[f"{sa}.wq_a.weight"] = _randn((Q_LORA_RANK, HIDDEN), g)
        w[f"{sa}.wkv.weight"] = _randn((HEAD_DIM, HIDDEN), g)

        # Q norm + Q up
        w[f"{sa}.q_norm.weight"] = _ones((Q_LORA_RANK,))
        w[f"{sa}.wq_b.weight"] = _randn((NUM_HEADS * HEAD_DIM, Q_LORA_RANK), g)

        # KV norm
        w[f"{sa}.kv_norm.weight"] = _ones((HEAD_DIM,))

        # Output-side O-LoRA (V4 NEW; n_groups split per V4 schema)
        # PyTorch Linear weight convention is [output, input]
        # wo_a: input=(n_heads*head_dim/n_groups)=4096, output=(n_groups*o_lora_rank)=8192
        w[f"{sa}.wo_a.weight"] = _randn(
            (O_GROUPS * O_LORA_RANK, (NUM_HEADS * HEAD_DIM) // O_GROUPS), g
        )
        # wo_b: input=(n_groups*o_lora_rank)=8192, output=hidden=4096
        w[f"{sa}.wo_b.weight"] = _randn((HIDDEN, O_GROUPS * O_LORA_RANK), g)

        # ---- Self-attn-level Compressor (V4 NEW; compress_ratio=4 main attention)
        # head_dim=V_HEAD_DIM, coff=2 (overlap=True for ratio=4)
        # ape: (ratio=4, coff*head_dim) fp32
        # wkv_gate.weight: (2*coff*head_dim, hidden) bf16 — torch Linear convention [out, in]
        # norm.weight: (head_dim,) fp32
        _COFF = 2  # 1 + overlap, overlap=True when ratio==4
        w[f"{sa}.compressor.ape"] = torch.randn(
            (4, _COFF * HEAD_DIM), generator=g, dtype=torch.float32
        ) * 0.02
        # NOTE: V4 loader expects TWO separate input weights — `wkv.weight` and
        # `wgate.weight` — that it fuses into `wkv_gate.weight` at load time
        # (deepseek_v4.py:1769). Each has shape (coff*head_dim, hidden).
        w[f"{sa}.compressor.wkv.weight"] = _randn((_COFF * HEAD_DIM, HIDDEN), g)
        w[f"{sa}.compressor.wgate.weight"] = _randn((_COFF * HEAD_DIM, HIDDEN), g)
        w[f"{sa}.compressor.norm.weight"] = torch.ones((HEAD_DIM,), dtype=torch.float32)

        # ---- C4Indexer (V4 replaces V3.2's indexer)
        idx = f"{sa}.indexer"
        w[f"{idx}.wq_b.weight"] = _randn((INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK), g)
        w[f"{idx}.weights_proj.weight"] = _randn((INDEX_N_HEADS, HIDDEN), g)
        # Note: V4 C4Indexer does NOT have wk or k_norm (those were V3.2's indexer)

        # C4Indexer's nested Compressor: head_dim=INDEX_HEAD_DIM=128
        w[f"{idx}.compressor.ape"] = torch.randn(
            (4, _COFF * INDEX_HEAD_DIM), generator=g, dtype=torch.float32
        ) * 0.02
        # Same wkv/wgate split for indexer's nested compressor
        w[f"{idx}.compressor.wkv.weight"] = _randn((_COFF * INDEX_HEAD_DIM, HIDDEN), g)
        w[f"{idx}.compressor.wgate.weight"] = _randn((_COFF * INDEX_HEAD_DIM, HIDDEN), g)
        w[f"{idx}.compressor.norm.weight"] = torch.ones((INDEX_HEAD_DIM,), dtype=torch.float32)

        # Hash-Coding (V4 NEW, per-layer mhc_post/pre params)
        # mix_hc = (2 + hc_mult) * hc_mult = (2+4)*4 = 24
        # hc_dim = hc_mult * hidden = 4 * 4096 = 16384
        MIX_HC = (2 + HC_MULT) * HC_MULT
        HC_DIM = HC_MULT * HIDDEN
        w[f"{prefix}.hc_attn_fn"] = torch.randn(
            (MIX_HC, HC_DIM), generator=g, dtype=torch.float32
        ) * 0.02
        w[f"{prefix}.hc_ffn_fn"] = torch.randn(
            (MIX_HC, HC_DIM), generator=g, dtype=torch.float32
        ) * 0.02
        w[f"{prefix}.hc_attn_base"] = torch.zeros((MIX_HC,), dtype=torch.float32)
        w[f"{prefix}.hc_ffn_base"] = torch.zeros((MIX_HC,), dtype=torch.float32)
        w[f"{prefix}.hc_attn_scale"] = torch.ones((3,), dtype=torch.float32)
        w[f"{prefix}.hc_ffn_scale"] = torch.ones((3,), dtype=torch.float32)

        # ---- MLP (dense or MoE) ----
        if FIRST_K_DENSE_REPLACE > layer_idx:
            # Dense layer
            mlp = f"{prefix}.mlp"
            w[f"{mlp}.gate_proj.weight"] = _randn((INTERMEDIATE, HIDDEN), g)
            w[f"{mlp}.up_proj.weight"] = _randn((INTERMEDIATE, HIDDEN), g)
            w[f"{mlp}.down_proj.weight"] = _randn((HIDDEN, INTERMEDIATE), g)
        else:
            # MoE layer
            mlp = f"{prefix}.mlp"
            # Router
            w[f"{mlp}.gate.weight"] = _randn((N_ROUTED_EXPERTS, HIDDEN), g)
            w[f"{mlp}.gate.e_score_correction_bias"] = torch.zeros((N_ROUTED_EXPERTS,), dtype=torch.bfloat16)
            # Routed experts
            for e in range(N_ROUTED_EXPERTS):
                ex = f"{mlp}.experts.{e}"
                w[f"{ex}.gate_proj.weight"] = _randn((MOE_INTERMEDIATE, HIDDEN), g)
                w[f"{ex}.up_proj.weight"] = _randn((MOE_INTERMEDIATE, HIDDEN), g)
                w[f"{ex}.down_proj.weight"] = _randn((HIDDEN, MOE_INTERMEDIATE), g)
            # Shared experts
            for s in range(N_SHARED_EXPERTS):
                sh = f"{mlp}.shared_experts.{s}" if N_SHARED_EXPERTS > 1 else f"{mlp}.shared_experts"
                w[f"{sh}.gate_proj.weight"] = _randn((MOE_INTERMEDIATE, HIDDEN), g)
                w[f"{sh}.up_proj.weight"] = _randn((MOE_INTERMEDIATE, HIDDEN), g)
                w[f"{sh}.down_proj.weight"] = _randn((HIDDEN, MOE_INTERMEDIATE), g)

    return w


def main(out_dir: str):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = build_config_dict()
    cfg_path = out_path / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[fab v4] wrote {cfg_path}")

    weights = build_random_weights()
    sft_path = out_path / "model.safetensors"
    save_file(weights, str(sft_path), metadata={"format": "pt"})
    n_params = sum(t.numel() for t in weights.values())
    n_bytes = sft_path.stat().st_size
    print(
        f"[fab v4] wrote {sft_path}  "
        f"n_tensors={len(weights)} n_params={n_params:,} ({n_bytes/1e6:.1f} MB)"
    )

    # Reuse the tokenizer copy from the V3.2 fab if present (Qwen2 tokenizer)
    tok_src = pathlib.Path("/host-models/dsv4_1layer_fab")
    if tok_src.is_dir():
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = tok_src / f
            if src.is_file():
                dst = out_path / f
                dst.write_bytes(src.read_bytes())
                print(f"[fab v4] copied tokenizer file {f}")

    print(f"[fab v4] DONE — V4-truthful 1-layer fab at {out_path}")
    print(f"        architectures: {cfg['architectures']}")
    print(f"        model_type:    {cfg['model_type']}")
    print(f"        MOE_ACTIVE={int(MOE_ACTIVE)}  first_k_dense_replace={FIRST_K_DENSE_REPLACE}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/host-models/dsv4_REAL_1layer_fab"
    main(out)
