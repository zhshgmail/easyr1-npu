"""Megatron miles DSAMLA -> HF deepseek_v2 weight rename map.

Source: miles_plugins/models/glm5/glm5.py (DSAMLASelfAttention class)
Target: sglang's deepseek_v2.py expected HF naming.

This map lets us take a Megatron state_dict (saved per-step by our 5-step
RL driver) and rewrite it into an HF safetensors checkpoint that sglang's
deepseek_v2 model loader accepts via /update_weights_from_disk.

Megatron miles                            ->  HF deepseek_v2 (sglang)
================================================================
linear_q_down_proj.weight                 ->  self_attn.q_a_proj.weight
linear_q_up_proj.weight                   ->  self_attn.q_b_proj.weight
linear_q_up_proj.layer_norm_weight        ->  self_attn.q_a_layernorm.weight
linear_kv_down_proj.weight                ->  self_attn.kv_a_proj_with_mqa.weight
linear_kv_up_proj.weight                  ->  self_attn.kv_b_proj.weight
linear_kv_up_proj.layer_norm_weight       ->  self_attn.kv_a_layernorm.weight
linear_proj.weight                        ->  self_attn.o_proj.weight
wq_b.weight                               ->  self_attn.indexer.wq_b.weight
wk.weight                                 ->  self_attn.indexer.wk.weight
weights_proj.weight                       ->  self_attn.indexer.weights_proj.weight
k_norm.weight                             ->  self_attn.indexer.k_norm.weight
k_norm.bias                               ->  self_attn.indexer.k_norm.bias

Notes:
1. q_layernorm / kv_layernorm = IdentityOp in our miles_local setup;
   they HAVE no weight tensors to migrate. The HF layout DOES expect
   q_a_layernorm and kv_a_layernorm to be present though -- we read
   those from linear_q_up_proj.layer_norm_weight / linear_kv_up_proj.layer_norm_weight
   in our _LayerNormColumnParallelLinear shim (§3.5 PoC).

2. Embedding + lm_head + final norm + input_layernorm + post_attention_layernorm:
   our miles_local actor only has a single DSAMLA attention layer; the
   surrounding embedding/norm/mlp is NOT in actor.state_dict(). We need
   to keep them from the fab ckpt (random init) so sglang has a complete
   model to load. Workflow:
     a) Load /host-models/dsv4_1layer_fab base (random) state_dict
     b) Replace ONLY the self_attn.* keys with Megatron-renamed values
     c) Save merged state_dict back

3. MoE weights: in MOE_ACTIVE=1 mode we also have mlp.gate.weight,
   mlp.gate.e_score_correction_bias, mlp.experts.N.*, mlp.shared_experts.*
   These are NOT in miles DSAMLA actor (which only models attention).
   Keep them from fab ckpt unchanged across steps.
"""

# Layer-0 attention key rename map (Megatron suffix -> HF suffix).
ATTN_RENAME = {
    "linear_q_down_proj.weight":             "self_attn.q_a_proj.weight",
    "linear_q_up_proj.weight":               "self_attn.q_b_proj.weight",
    "linear_q_up_proj.layer_norm_weight":    "self_attn.q_a_layernorm.weight",
    "linear_kv_down_proj.weight":            "self_attn.kv_a_proj_with_mqa.weight",
    "linear_kv_up_proj.weight":              "self_attn.kv_b_proj.weight",
    "linear_kv_up_proj.layer_norm_weight":   "self_attn.kv_a_layernorm.weight",
    "linear_proj.weight":                    "self_attn.o_proj.weight",
    # DSA indexer
    "wq_b.weight":                           "self_attn.indexer.wq_b.weight",
    "wk.weight":                             "self_attn.indexer.wk.weight",
    "weights_proj.weight":                   "self_attn.indexer.weights_proj.weight",
    "k_norm.weight":                         "self_attn.indexer.k_norm.weight",
    "k_norm.bias":                           "self_attn.indexer.k_norm.bias",
}


def megatron_attn_to_hf_attn(megatron_state_dict, hf_prefix="model.layers.0"):
    """Given a Megatron miles DSAMLA actor state_dict (keys like
    'linear_q_down_proj.weight'), return a dict mapping HF keys to tensors.

    Caller must merge this into the surrounding fab ckpt (which carries
    embedding, layernorms, mlp, lm_head untouched).
    """
    out = {}
    for k, v in megatron_state_dict.items():
        suffix = k
        if suffix in ATTN_RENAME:
            new_key = f"{hf_prefix}.{ATTN_RENAME[suffix]}"
            out[new_key] = v
        else:
            # Unknown miles key -- leave it; surface as warning so we know
            # what's missing from the map.
            print(f"  [warn] Megatron key {k!r} has no HF rename -- skipping")
    return out


def merge_into_fab(fab_state_dict, attn_overrides, hf_prefix="model.layers.0"):
    """Take the fab base safetensors state_dict and overwrite only the
    layer-0 attention slots with attn_overrides.

    Returns a NEW state_dict (doesn't mutate fab_state_dict).
    """
    merged = dict(fab_state_dict)
    for k, v in attn_overrides.items():
        if k not in merged:
            print(f"  [warn] override key {k!r} not in fab base -- adding new")
        merged[k] = v
    return merged


if __name__ == "__main__":
    # Self-test: verify all map entries are well-formed
    print("=== Megatron -> HF rename map self-test ===")
    for k, v in ATTN_RENAME.items():
        print(f"  {k:50s} -> {v}")
    print(f"Total mapped keys: {len(ATTN_RENAME)}")
