# fix(norm): use `getattr` for optional `bias` attribute on layernorm in `fused_split_qk_norm`

## Summary

`sgl_kernel_npu/norm/fused_split_qk_norm.py:118+` reads `q_a_layernorm.bias` and `kv_a_layernorm.bias` unconditionally. When the caller passes RMSNorm modules (which only have `.weight`, no `.bias`), this raises `AttributeError` mid-forward and crashes the scheduler.

This is the exact configuration that sglang's `srt/models/deepseek_v2.py` uses on the DSA_NPU path: `q_a_layernorm` and `kv_a_layernorm` are constructed as `RMSNorm`, not LayerNorm.

## Reproduction

1. sglang main, NPU backend
2. Load any DeepSeek-V3.2 / DeepseekV32ForCausalLM / GlmMoeDsa model with `device='npu'` + `attention_backend='ascend'`
3. First forward pass triggers `fused_split_qk_norm` -> `AttributeError: 'RMSNorm' object has no attribute 'bias'`

Discovered while loading a fabricated 1-layer DSv4-Flash random-init HF checkpoint into `sgl.Engine(device='npu', ...)` on `lmsysorg/sglang:main-cann8.5.0-a3` (image built 2026-05-25; sglang `0.5.12.post2.dev434+gb13d3d18c`, sgl-kernel-npu `2026.5.1`).

Full traceback:
```
File "sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1754, in forward_prepare
    inner_state = forward_dsa_prepare_npu(...)
File "sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py", line 364, in forward_dsa_prepare_npu
    q_lora, k_nope, k_pe = fused_split_qk_norm(...)
File "sgl_kernel_npu/norm/fused_split_qk_norm.py", line 118, in fused_split_qk_norm
    q_a_layernorm.bias,
File "torch/nn/modules/module.py", line 1962, in __getattr__
    raise AttributeError(...)
AttributeError: 'RMSNorm' object has no attribute 'bias'
```

## Fix

Use `getattr(layernorm, 'bias', None)` so the kernel correctly receives `None` when bias is absent. Same change for the `is not None` callsites a few lines below.

```python
        fused_split_qk_norm_kernel[(B,)](
            ...
            q_a_layernorm.weight,
-           q_a_layernorm.bias,
+           getattr(q_a_layernorm, "bias", None),
            kv_a_layernorm.weight,
-           kv_a_layernorm.bias,
+           getattr(kv_a_layernorm, "bias", None),
            ...
-           q_a_layernorm.bias is not None,
-           kv_a_layernorm.bias is not None,
+           getattr(q_a_layernorm, "bias", None) is not None,
+           getattr(kv_a_layernorm, "bias", None) is not None,
            ...
        )
```

The kernel implementation already gates on the runtime constant flag (`bias_exists`), so passing `None` is harmless when no bias is present.

## Verification

After the patch on the same image, the same model loads + forwards + generates without error on NPU:

```
[test] engine init in 22.0s
[test] one-shot generate: " ..."
[test] PASS
```

(Full repro at: https://github.com/zhshgmail/easyr1-npu/blob/main/workspace/T32_tilelang_rescue/sgl_kernel_npu_rmsnorm_bias_patch.diff)

## Related

- `sgl-project/sglang` `srt/models/deepseek_v2.py` constructs `q_a_layernorm` / `kv_a_layernorm` as RMSNorm. No change needed there.
- Active norm/MLA-rope kernel cleanup wave: #212, #282, #290, #390, #404, #503. None of those address this specific `.bias` access pattern.
