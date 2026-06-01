# DRAFT — sgl-project/sglang issue (HELD — do NOT post yet)

> **DECISION (user 2026-06-01 08:46Z)**: do NOT post upstream now. Keep the V4
> NPU-adapter patches in the **fork repo** (sglang fork branch), wait until the
> whole miles+V4 RL loop is end-to-end working, THEN submit the upstream PR(s)
> in one batch with full e2e evidence. (Matches CLAUDE.md upstream-PR discipline:
> "充分验证后再开 PR".) This draft is the PR-prep material; it becomes the issue
> body / PR description when the batch goes up. Until then it stays in our repo.
>
> Target (eventual): https://github.com/sgl-project/sglang/issues/new (bug report template)
> Title: `[Bug] DeepseekV4ForCausalLM on device=npu: AscendAttnBackend missing V4 dispatch hooks + bf16-path kernels JIT-compile CUDA`
> Labels: follow repo defaults (do not add agent labels). NO agent/Claude signature in body.
> Language: English (template requires it).

---

### Checklist
- [x] I searched related issues but found no solution. (Closest: #26794 MoE `/update_weights_from_disk` reload narrow — separate.)
- [x] The bug persists in the latest version. (Reproduced on `0.5.12.post2.dev434+gb13d3d18c`.)
- [x] Issues without environment info and a minimal reproducible demo are hard to resolve.
- [x] This is a bug report, not a general question.
- [x] English.

### Describe the bug

Running `DeepseekV4ForCausalLM` (DeepSeek-V4-Flash architecture) with `device="npu"` (Ascend 910C / A3) on the bf16 path (`SGLANG_OPT_FP8_WO_A_GEMM=0`) cannot complete `generate()` out of the box. The V4 forward path hits a series of NPU-adapter gaps. None are fp8-specific — they are all on the **bf16** path. Two classes:

**Class 1 — `AscendAttnBackend` is missing V4 dispatch hooks** that `deepseek_v4.py` / the V4 attention modules call:
| Hook called by V4 | Site (V4 model/attn) | NPU backend |
|---|---|---|
| `_maybe_upgrade_forward_metadata` | `deepseek_v4.py` MQALayer | AttributeError |
| `forward_c4_indexer` | `layers/attention/dsv4/indexer.py:549` | AttributeError |
| `forward_core_compressor` | `deepseek_v4.py:528,651` | AttributeError |
| `store_cache` | `deepseek_v4.py:630` | AttributeError |
| `init_forward_metadata_indexer` | `deepseek_v4.py:1439` | AttributeError |
| `forward_compress` | `layers/attention/dsv4/compressor.py:381` | AttributeError |
| `forward_extend(..., compress_ratio=...)` | `deepseek_v4.py:729` | TypeError (unexpected kwarg) |
| `forward_decode(..., compress_ratio=...)` | decode path | TypeError (unexpected kwarg) |

`AscendAttnBackend` was written for V3.x classic MLA; V4 introduces these dispatch hooks the NPU backend does not implement.

**Class 2 — bf16-path kernels JIT-compile CUDA C++ (no nvcc on NPU host):**
`deepseek_v4.py` bf16 path calls, unconditionally (NOT inside `if _FP8_WO_A_GEMM`):
- `jit_kernel/dsv4/elementwise.py::fused_q_norm_rope` (`deepseek_v4.py:423`)
- `fused_norm_rope_inplace` (`deepseek_v4.py:467`, comment: "Bf16-kv path")
- `fused_rope_inplace` (`deepseek_v4.py:733`)
- `jit_kernel/dsv4/moe.py::silu_and_mul_clamp` (`deepseek_v2.py:364`)
- `jit_kernel/dsv4/gemm.py::linear_bf16_fp32` uses `torch.mm(..., out_dtype=fp32)` (not supported on torch_npu)

These call `load_jit(... cuda_files=...)` → `_find_cuda_home()` → `RuntimeError: Could not find CUDA installation` on an NPU host (no CUDA, by design).

**Other:**
- `DeepSeekV4TokenToKVPool.get_key_buffer/get_value_buffer` raise `NotImplementedError`, but `AscendAttnBackend.forward_extend` (classic-MLA path) calls them.
- torch_npu `aclnnIndex` does not support `complex64` (indexing `freqs_cis[positions]` on the rope path) — `error code 161002 / DT_COMPLEX64 not in support list`.

### Reproduction

1-layer reduced `DeepseekV4ForCausalLM` fab (real V4 schema, bf16), all `SGLANG_OPT_*` fusions disabled, `mhc` stubbed (no tilelang on NPU):
```python
import sglang as sgl
llm = sgl.Engine(model_path="<1-layer DeepseekV4ForCausalLM fab>",
                 dtype="bfloat16", device="npu", tp_size=1,
                 max_total_tokens=65536, swa_full_tokens_ratio=0.5,
                 disable_radix_cache=True, disable_cuda_graph=True)
# env: SGLANG_OPT_FP8_WO_A_GEMM=0 + all SGLANG_OPT_USE_* fusions=0
llm.generate(input_ids=[[1,2,3]], sampling_params={"temperature":0,"max_new_tokens":2})
```
Each gap above surfaces in turn (Engine init + KV pool allocation succeed; the gaps are in the forward path). With local no-op stubs for the missing hooks + torch fallbacks for the JIT kernels, `generate()` completes and returns output — i.e. each gap is an isolated, additive NPU-adapter task, not a fundamental blocker.

### Environment
```
sglang:    0.5.12.post2.dev434+gb13d3d18c
torch:     2.8.0   (torch_npu 2.8.0.post2)
device:    Ascend 910C (A3), SOC Ascend910_9382, CANN 8.5.0
torch.npu.is_available(): True   torch.cuda.is_available(): False
```

### Suggested direction
1. Add the 6 missing V4 dispatch hooks to `AscendAttnBackend` as safe defaults (no-op / dense fallback) so V4 attention degrades gracefully on NPU.
2. Accept V4 kwargs (`compress_ratio`, `attn_sink`) via `**kwargs` in `forward_extend`/`forward_decode`.
3. bf16-path elementwise kernels (`fused_q_norm_rope` etc.): provide a torch_npu native path (`torch_npu.npu_apply_rotary_pos_emb` + `npu_rms_norm` + `npu_clipped_swiglu` + `npu_kv_rmsnorm_rope_cache_v2` all exist) instead of unconditionally JIT-compiling CUDA.
4. `linear_bf16_fp32`: NPU branch dropping the `out_dtype=` kwarg.

Happy to provide the full reproducer fab + the local-stub patch set as evidence.
