---
id: sglang-002
date: 2026-05-30
layer: sglang
title: sgl_kernel_npu fused_split_qk_norm directly reads RMSNorm.bias; some NPU RMSNorm instances have no bias attribute and crash on init
trigger:
  - "sglang Engine init for DeepSeek-V3 / V3.2 / DSv4-Flash family on NPU"
  - "AttributeError: 'RMSNorm' object has no attribute 'bias' inside fused_split_qk_norm"
  - "porting any model that uses fused_split_qk_norm path on Ascend"
symptom_in_wild:
  - "AttributeError: 'RMSNorm' object has no attribute 'bias' at fused_split_qk_norm.py:118"
  - "Engine init crashes during weight load — before any generate() call"
  - "Same model on GPU loads fine; only NPU path errors"
root_cause: >
  `sgl_kernel_npu/norm/fused_split_qk_norm.py` reads `q_a_layernorm.bias` /
  `k_a_layernorm.bias` directly in 4 places (lines 117-127). Some NPU
  RMSNorm instances are constructed without a bias parameter — only weight.
  The instance is a valid Python object; access to the unset attribute raises
  AttributeError.

  Fix: read with `getattr(..., 'bias', None)`. The downstream consumer
  already handles None correctly.
mistake_pattern: "direct attribute access on optional torch nn.Module parameter; assumed always-present"
correction:
  - "PR sgl-project/sgl-kernel-npu#531: 4-line patch replacing `q_a_layernorm.bias` -> `getattr(q_a_layernorm, 'bias', None)` (and 3 similar sites)"
  - "Patch file: `workspace/T32_tilelang_rescue/sgl_kernel_npu_rmsnorm_bias_patch.diff`"
  - "Verification: Engine init on 1-layer DSv4-Flash fab ckpt passes (engine init 22-26s, generate 5.6s) after patch; without patch crashes immediately"
  - "Works for both DeepseekV32 (current) and DeepseekV3 (older); the attribute pattern is the same"
evidence:
  - "PR: https://github.com/sgl-project/sgl-kernel-npu/pull/531 (OPEN, REVIEW_REQUIRED)"
  - "Author: zhshgmail <zhengshencn@gmail.com>, commit 3c08165"
  - "Reproducer: workspace/T32_tilelang_rescue/test_load_dsv4_fab.py (loads fab ckpt with `architectures=['DeepseekV32ForCausalLM']`)"
  - "Filed 2026-05-30 17:30 Beijing during the sglang DSv4-Flash same-arch milestone chain"
---

# sglang-002 — RMSNorm.bias attribute access pattern

## Why this matters

Without this fix sglang cannot load any DeepseekV3-family model on NPU. It blocks the entire "use sglang as miles' default rollout engine on NPU" path. Once patched, the rest of the same-arch milestone chain (R3 plumbing, weight sync) works end-to-end.

## Pattern check

When you see `AttributeError: '<NnModule>' object has no attribute 'bias'` in NPU code:
1. Find the access site: `grep -n '\.bias' <path>/<file>.py`
2. Check whether the upstream construction always sets `bias`. For RMSNorm, the answer is "no — only when `elementwise_affine=True` and `bias=True`"
3. Fix with `getattr(obj, 'bias', None)`; verify downstream consumer handles None
4. File at the package where the access happens (here: `sgl-kernel-npu`), not at the package that constructed the RMSNorm (the model code)

## Related patterns

The same anti-pattern appears in:
- `LayerNorm.bias` (not present when `elementwise_affine=False`)
- `Linear.bias` (not present when `bias=False`)

Any NPU port that uses bare `<layer>.bias` instead of `getattr(<layer>, 'bias', None)` is a latent crash for some downstream model variant.
