# 🛡️ PROTECTED — V4 flash/attention WORKING on NPU (do NOT lose)

Owner 2026-06-01 20:24Z: "flash 的注意保存保护，不要弄丢". This is the verified-working V4
flash/attention-on-NPU result. All artifacts committed to git (origin/main) — protected.

## The PROVEN result (re-verifiable)
**A single `DeepSeekV4Attention` megatron layer (MLA + sparse/flash attention + indexer + compressor)
does a full FORWARD + BACKWARD training step on Ascend A3 NPU:**
```
V4 ATTENTION LAYER FORWARD OK on NPU:  out=(64,1,512) finite=True
V4 ATTENTION LAYER BACKWARD OK on NPU: loss=0.0353 grad_norm=0.173 params_with_grad=8
```
flash/attention is NOT a blocker — it WORKS (single-layer training step verified).

## Committed artifacts (git origin/main, all PROTECTED)
- `v4_e2e_megatron_layer_forward_npu.py` — the working fwd+bwd training-step driver (commit a577cc0)
- `v4_sparse_attn_nsa_shim.py` — V4 sparse-MLA → CANN npu_nsa_select_attention (verified)
- `v4_indexer_lightning_shim.py` — indexer → npu_lightning_indexer (verified)
- `v4_npu_native_dispatch.py` — consolidated CANN-native dispatch
- `miles_attention_core_npu_patched.py` / `miles_qat_npu_patched.py` / `miles_hyper_connection_npu_patched.py` — the patched miles V4 ops (lazy tilelang + is_npu→CANN-native)
- `megatron_npu_patches/` — the `all_reduce_grad_fp32` Megatron-LM-miles fork patch (PR-ready)
- `v4_attention_layer_npu_construct.py` — standalone layer construct+fwd+bwd

## The working stack (recipe to reproduce — see project_v4_megatron_layer_boundary memory)
MindSpeed `core_r0.16.0` + `import mindspeed.megatron_adaptor` + MLATransformerConfig (V4 MLA dims) +
the patched CANN-native ops + `MEGATRON_SPARSE_ATTN_IMPL=sparse` + the all_reduce_grad_fp32 megatron
patch + attn_sink fp32 + parallel_state/model_parallel_cuda_manual_seed. PYTHONPATH=/opt/miles_v4:
/home/z00637938/workspace/Megatron-LM-miles. tlrescue container, A3.

## What is NOT done (honest, distinct from the protected flash result)
The full REAL DSV4 model (4096 hidden / 64 heads / **256-expert MoE** / 43 layers) reduced to 1 layer
is blocked separately on the MoE path needing the full `miles` package (`miles.utils`) — a missing
training-framework dependency, NOT flash/attention. That's a different (MoE/env) workstream; the flash
result above stands independently and is protected here.
