"""Fabricate a 1-layer DeepseekV32ForCausalLM HF checkpoint that sglang
can load on Ascend NPU via its DSA_NPU path.

Per the project goal note in MILES_DSV4_NPU_POC_REPORT.md, the rollout
model and training model MUST be the same architecture. miles trains
1-layer DSv4-Flash via Megatron + miles DSAMLA at miles_local config.
This script saves the same config + random-init weights in HF naming
convention so sglang's deepseek_v2.py can load it directly.

What sglang reads:
  * config.architectures[0] in {DeepseekV3ForCausalLM, DeepseekV32ForCausalLM,
                                GlmMoeDsaForCausalLM, ...}
  * config.index_topk -> triggers is_deepseek_nsa() == True -> DSA path
  * config.hidden_size, num_attention_heads, num_hidden_layers
  * config.q_lora_rank, kv_lora_rank
  * config.qk_nope_head_dim, qk_rope_head_dim (sum = qk_head_dim)
  * config.v_head_dim
  * config.n_routed_experts, moe_intermediate_size, n_shared_experts
  * config.first_k_dense_replace (layers before this index are dense MLP,
    after are MoE; for 1-layer set first_k_dense_replace=1 -> dense MLP)
  * config.vocab_size, rms_norm_eps, rope_theta, etc.
  * index_n_heads, index_head_dim (for the DSA indexer)

Weight tensors saved (HF naming, single .safetensors file):
  model.embed_tokens.weight                  [V, H]
  model.norm.weight                          [H]
  lm_head.weight                             [V, H]
  model.layers.0.input_layernorm.weight      [H]
  model.layers.0.post_attention_layernorm.weight [H]
  model.layers.0.self_attn.q_a_proj.weight   [q_lora, H]
  model.layers.0.self_attn.q_a_layernorm.weight [q_lora]
  model.layers.0.self_attn.q_b_proj.weight   [num_heads * qk_head_dim, q_lora]
  model.layers.0.self_attn.kv_a_proj_with_mqa.weight [kv_lora + qk_rope, H]
  model.layers.0.self_attn.kv_a_layernorm.weight [kv_lora]
  model.layers.0.self_attn.kv_b_proj.weight  [num_heads * (qk_nope + v_head_dim), kv_lora]
  model.layers.0.self_attn.o_proj.weight     [H, num_heads * v_head_dim]
  model.layers.0.self_attn.indexer.wq_b.weight     [index_n_heads * index_head_dim, q_lora]
  model.layers.0.self_attn.indexer.wk.weight       [index_head_dim, H]
  model.layers.0.self_attn.indexer.weights_proj.weight [index_n_heads, H]
  model.layers.0.self_attn.indexer.k_norm.weight   [index_head_dim]
  model.layers.0.self_attn.indexer.k_norm.bias     [index_head_dim]
  model.layers.0.mlp.gate_proj.weight        [intermediate, H]  -- dense (since first_k_dense_replace=1)
  model.layers.0.mlp.up_proj.weight          [intermediate, H]
  model.layers.0.mlp.down_proj.weight        [H, intermediate]
"""
import os
import sys
import json
import torch

sys.path = [p for p in sys.path if p not in ("", "/")]

import safetensors.torch as safetensors

OUTPUT_DIR = "/host-models/dsv4_1layer_fab"

# miles_local config dims (verbatim from §3.6 / MLATransformerConfig)
HIDDEN = 4096
NUM_HEADS = 64
NUM_LAYERS = 1
Q_LORA_RANK = 1024
KV_LORA_RANK = 512
QK_HEAD_DIM = 128       # Megatron name
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = QK_HEAD_DIM - QK_ROPE_HEAD_DIM  # = 64; sglang convention splits
V_HEAD_DIM = 512
INTERMEDIATE = 1408     # ffn_hidden_size in MLATransformerConfig
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 8

VOCAB = 151936          # matches Qwen2 tokenizer.json (we copy it so token
                        # ids fit in the embedding); not changing operator
                        # shape, only ensuring tokenizer compatibility
RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0
MAX_POSITION_EMBEDDINGS = 2048

# MoE: first_k_dense_replace=1 with num_hidden_layers=1 means our single
# layer is DENSE MLP (skips MoE path entirely). This keeps the fabricated
# checkpoint smaller and avoids needing experts. The MoE code path is
# separately tested in miles' production -- not the focus of this validation.
N_ROUTED_EXPERTS = 4      # tiny MoE expert pool; layer 0 stays dense via
                          # first_k_dense_replace=1 (so no MoE weights actually loaded);
                          # sglang's eplb math still wants a valid count
MOE_INTERMEDIATE = 1408
N_SHARED_EXPERTS = 1
FIRST_K_DENSE_REPLACE = 1  # all layers (= 1) are dense -> no MoE init needed


def build_config_dict():
    """Return the HF-style config.json dict for sglang + transformers to parse.

    architectures + model_type targets GlmMoeDsaForCausalLM (transformers 5.3
    recognizes this; sglang's is_deepseek_nsa() also accepts it). miles is
    built on GLM-5 which is the same architecture family as DSv4-Flash.
    Dims overridden to DSv4-Flash 1-layer (matching miles_local config in §3.6).
    """
    return {
        "architectures": ["DeepseekV32ForCausalLM"],
        "model_type": "deepseek_v3",
        # core dims
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        # MLA
        "q_lora_rank": Q_LORA_RANK,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        # MoE
        "n_routed_experts": N_ROUTED_EXPERTS,
        "n_shared_experts": N_SHARED_EXPERTS,
        "first_k_dense_replace": FIRST_K_DENSE_REPLACE,
        "moe_layer_freq": 1,           # MoE applies to every layer past first_k_dense_replace
        "num_experts_per_tok": 2,      # 'top-k' experts per token; small=2
        "n_group": 1,                  # expert grouping; 1 = no grouping
        "topk_group": 1,
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        # vocab + tokenizer-side
        "vocab_size": VOCAB,
        "rms_norm_eps": RMS_NORM_EPS,
        "rope_theta": ROPE_THETA,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "torch_dtype": "bfloat16",
        # DSA / NSA fields (trigger sglang's NSA DSA_NPU path)
        "index_topk": INDEX_TOPK,
        "index_n_heads": INDEX_N_HEADS,
        "index_head_dim": INDEX_HEAD_DIM,
        "index_topk_freq": 1,
        "indexer_rope_interleave": False,
        # Rope: newer rope_parameters dict + rope_scaling field
        "rope_parameters": {
            "rope_theta": ROPE_THETA,
            "rope_type": "default",
        },
        "rope_scaling": None,
        # quantization (not used; we ship bf16) -- omit fields entirely
        # so transformers doesn't try to serialize a None
        # misc
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "transformers_version": "4.51.0",
    }


def build_random_weights(seed=42):
    """Random-init bf16 weight dict in HF naming convention.

    All tensors are initialized with normal_(0, 0.02) — same scale as
    typical HF model init. RMSNorm scales init to 1.0.
    """
    g = torch.Generator().manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=g, dtype=torch.bfloat16) * 0.02

    def ones(shape):
        return torch.ones(shape, dtype=torch.bfloat16)

    def zeros(shape):
        return torch.zeros(shape, dtype=torch.bfloat16)

    weights = {}

    # Embeddings + LM head + final norm
    weights["model.embed_tokens.weight"] = randn((VOCAB, HIDDEN))
    weights["model.norm.weight"] = ones((HIDDEN,))
    weights["lm_head.weight"] = randn((VOCAB, HIDDEN))

    # Single transformer layer
    prefix = "model.layers.0"

    # Layer norms (RMSNorm at fp32 stored as bf16; scale only)
    weights[f"{prefix}.input_layernorm.weight"] = ones((HIDDEN,))
    weights[f"{prefix}.post_attention_layernorm.weight"] = ones((HIDDEN,))

    # MLA self_attn
    sa = f"{prefix}.self_attn"
    weights[f"{sa}.q_a_proj.weight"] = randn((Q_LORA_RANK, HIDDEN))
    weights[f"{sa}.q_a_layernorm.weight"] = ones((Q_LORA_RANK,))
    weights[f"{sa}.q_b_proj.weight"] = randn((NUM_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), Q_LORA_RANK))
    weights[f"{sa}.kv_a_proj_with_mqa.weight"] = randn((KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN))
    weights[f"{sa}.kv_a_layernorm.weight"] = ones((KV_LORA_RANK,))
    weights[f"{sa}.kv_b_proj.weight"] = randn((NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK))
    weights[f"{sa}.o_proj.weight"] = randn((HIDDEN, NUM_HEADS * V_HEAD_DIM))

    # DSA indexer
    idx = f"{sa}.indexer"
    weights[f"{idx}.wq_b.weight"] = randn((INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK))
    weights[f"{idx}.wk.weight"] = randn((INDEX_HEAD_DIM, HIDDEN))
    weights[f"{idx}.weights_proj.weight"] = randn((INDEX_N_HEADS, HIDDEN))
    weights[f"{idx}.k_norm.weight"] = ones((INDEX_HEAD_DIM,))
    weights[f"{idx}.k_norm.bias"] = zeros((INDEX_HEAD_DIM,))

    # Dense MLP (since first_k_dense_replace=1 >= num_layers=1, this layer is dense)
    mlp = f"{prefix}.mlp"
    weights[f"{mlp}.gate_proj.weight"] = randn((INTERMEDIATE, HIDDEN))
    weights[f"{mlp}.up_proj.weight"] = randn((INTERMEDIATE, HIDDEN))
    weights[f"{mlp}.down_proj.weight"] = randn((HIDDEN, INTERMEDIATE))

    return weights


def save_checkpoint(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    config = build_config_dict()
    print(f"[fab] writing config to {out_dir}/config.json ...")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[fab] building random weights ...")
    weights = build_random_weights()
    total_params = sum(t.numel() for t in weights.values())
    total_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    print(f"[fab] total: {total_params:,} params ({total_bytes/1024**2:.1f} MB on disk)")
    for k, t in weights.items():
        print(f"  {k:60s} shape={tuple(t.shape)} dtype={t.dtype}")

    # Save as safetensors -- sglang default loader
    sf_path = os.path.join(out_dir, "model.safetensors")
    print(f"[fab] writing safetensors to {sf_path} ...")
    safetensors.save_file(weights, sf_path, metadata={"format": "pt"})

    # Also write minimal tokenizer config -- sglang complains if missing
    # We use a tiny tokenizer.json (just enough fields). For our smoke
    # rollout the tokenization quality is irrelevant; we want plumbing.
    tok_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": MAX_POSITION_EMBEDDINGS,
        "padding_side": "right",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2)

    # Copy a real tokenizer.json from Qwen2-0.5B; the actual token mappings
    # don't matter for our validation, we just need a parseable file.
    # Sglang uses it for /generate's tokenize step.
    qwen_tok = "/host-models/Qwen2-0.5B-Instruct/tokenizer.json"
    if os.path.exists(qwen_tok):
        import shutil
        shutil.copy2(qwen_tok, os.path.join(out_dir, "tokenizer.json"))
        print(f"[fab] copied tokenizer.json from Qwen2-0.5B-Instruct")

    print(f"[fab] checkpoint saved to {out_dir}")
    print(f"[fab] PASS")


if __name__ == "__main__":
    save_checkpoint(OUTPUT_DIR)
