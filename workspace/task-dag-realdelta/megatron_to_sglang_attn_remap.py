"""n2_remap SCAFFOLD — megatron -> sglang attn weight-name remap for the V4 real-delta bridge.

STATUS: template only. The megatron-side source names + exact shapes are captured
at runtime by n1_export (the megatron module's named_parameters() — they are NOT
grep-able from the shim files because the layer is built live). This scaffold fixes
the sglang TARGET names (known) + the lookup/transform STRATEGY; n1's trained_attn_delta.pt
provides the source side. Do NOT hardcode guessed megatron prefixes here — fill from n1.

sglang target names (confirmed, from _v4_rl_loop_tensor_PASS.py:78-82):
    model.layers.0.self_attn.wq_a.weight
    model.layers.0.self_attn.wq_b.weight
    model.layers.0.self_attn.wkv.weight
    model.layers.0.self_attn.wo_a.weight
    model.layers.0.self_attn.wo_b.weight

Bridge contract (cross-stack, KB cross-layer-012):
  - The training iteration runs in the megatron/mindspeed stack (tlrescue container);
    the RL loop runs in the sglang.Engine stack. The delta crosses that boundary as a
    plain dict of {sglang_name: torch.Tensor} loaded from trained_attn_delta.pt.
  - update_weights_from_tensor expects the FULL post-step tensor (not the delta) under
    the sglang name. n3 reconstructs full = sglang_current + remap(delta) OR pushes the
    post-step weight directly if n1 exported the absolute post-step value.
  - Layout: verify per-tensor whether a transpose is needed (megatron ColumnParallel /
    RowParallel linears may store [out,in] vs sglang's expectation). Shape-assert against
    the sglang model's current tensor before pushing — a silent transpose corrupts the
    rollout without erroring.
"""

# sglang side is fixed; the megatron source name is resolved from n1's export keys.
SGLANG_ATTN_TENSORS = [
    "model.layers.0.self_attn.wq_a.weight",
    "model.layers.0.self_attn.wq_b.weight",
    "model.layers.0.self_attn.wkv.weight",
    "model.layers.0.self_attn.wo_a.weight",
    "model.layers.0.self_attn.wo_b.weight",
]


def build_remap(trained_delta_keys):
    """Given the megatron-side keys actually present in trained_attn_delta.pt (n1 output),
    return {sglang_name: megatron_name} by matching the wq_a/wq_b/wkv/wo_a/wo_b suffix.

    Raises if any of the 5 sglang tensors cannot be matched — fail loud, never silently
    drop an attn tensor (a dropped tensor means that weight never syncs and the RL loop
    secretly runs on stale weights for it).
    """
    suffixes = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]
    remap = {}
    for sgl_name, suf in zip(SGLANG_ATTN_TENSORS, suffixes):
        matches = [k for k in trained_delta_keys if k.endswith(f".{suf}.weight") or k.endswith(f".{suf}")]
        if len(matches) != 1:
            raise ValueError(
                f"remap: sglang {sgl_name} (suffix {suf}) matched {len(matches)} megatron keys "
                f"{matches} in n1 export — expected exactly 1. Resolve before bridging."
            )
        remap[sgl_name] = matches[0]
    return remap


def transform(sgl_name, tensor, sgl_current_shape):
    """Per-tensor layout fixup. Default: identity. If megatron stores transposed relative
    to sglang, transpose here AFTER confirming against sgl_current_shape. n3 must shape-
    assert the returned tensor matches sgl_current_shape before update_weights_from_tensor.
    """
    if tuple(tensor.shape) == tuple(sgl_current_shape):
        return tensor
    if tuple(tensor.shape[::-1]) == tuple(sgl_current_shape):
        return tensor.t().contiguous()
    raise ValueError(
        f"transform: {sgl_name} megatron shape {tuple(tensor.shape)} neither matches nor "
        f"transposes to sglang shape {tuple(sgl_current_shape)} — manual layout fixup needed."
    )
