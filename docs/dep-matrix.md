# EasyR1 → A3 dependency matrix

Synthesis of EasyR1 master `requirements.txt`, veRL master `requirements.txt` + `requirements-npu.txt` + `setup.py` extras, and the actual pip freezes of `verl-8.5.0-a3` and `verl-8.5.2-a3`.

Last updated: 2026-04-17.

## Gap classifications

- **V** = version-bump only — same package is there, version delta needs to fit EasyR1's pin
- **P** = port needed — compiled code, needs NPU-aware build or source change
- **R** = replace with CANN-provided alternative — functionality exists on NPU under a different package/backend
- **A** = add, likely clean — pure Python or CUDA-free binary, should install cleanly into the A3 image
- **D** = drop on NPU — dev/test-only or optional
- **N/A** = not applicable — EasyR1 doesn't use this; listed for completeness

## Table

| Package | EasyR1 pin | veRL core pin | veRL NPU pin | 8.5.0 image | 8.5.2 image | Gap |
|---|---|---|---|---|---|---|
| accelerate | unpinned | unpinned | unpinned | 1.13.0 | 1.13.0 | **—** |
| bytecode | — | — | unpinned | 0.17.0 | 0.17.0 | N/A (used by veRL tracing) |
| codetiming | unpinned | unpinned | unpinned | 1.4.0 | 1.4.0 | — |
| compressed_tensors | — (transitive via vllm) | — | — | 0.12.2 | 0.13.0 | — |
| datasets | unpinned | unpinned | unpinned | 4.8.4 | 4.8.4 | — |
| dill | — (EasyR1 doesn't use) | unpinned | unpinned | 0.4.1 | 0.4.1 | N/A |
| einops | — (transitive?) | — | unpinned | 0.8.2 | 0.8.2 | — |
| fastapi | — | unpinned | — | present | present | — |
| **flash-attn** | `>=2.4.3` | — (in `gpu` extra only) | — | **absent** | **absent** | **R** → replace with CANN attention (torch_npu FA ops / vllm_ascend backend) |
| hf_transfer | — | — | unpinned | 0.1.9 | 0.1.9 | — |
| huggingface_hub | — (transitive) | — | — | 0.36.2 | **1.11.0** | — (check 1.x API compat in EasyR1 direct calls) |
| hydra-core | — (EasyR1 uses omegaconf directly) | unpinned | unpinned | 1.3.2 | 1.3.2 | omegaconf 2.3.0 present in both |
| latex2sympy2_extended | — | unpinned | — | present | present | — |
| **liger-kernel** | unpinned | unpinned (veRL core) | — (dropped on NPU) | **absent** | **absent** | **R/D** → evaluate triton-ascend backend; else drop (runs training without it) |
| math_verify | — | unpinned | — | present | present | — |
| mathruler | unpinned | — | unpinned | 0.1.0 | 0.1.0 | — |
| numpy | unpinned | `<2.0.0` | `<2.0.0` | 1.26.4 | 1.26.4 | — |
| omegaconf | unpinned | — (transitive via hydra) | — (transitive) | 2.3.0 | 2.3.0 | — |
| packaging | — | `>=20.0` | — | present | present | — |
| pandas | unpinned | unpinned | unpinned | 3.0.2 | 3.0.2 | — |
| peft | unpinned | unpinned | `>=0.15.2` | 0.19.1 | 0.19.1 | — |
| pillow | unpinned | — | — | 12.2.0 | 12.2.0 | — (both images ship it; no action needed) |
| pyarrow | `>=15.0.0` | `>=19.0.0` | `>=15.0.0` | 23.0.1 | 23.0.1 | — |
| pybind11 | — | unpinned | unpinned | 3.0.3 | 3.0.3 | — |
| pylatexenc | unpinned | unpinned | unpinned | 2.10 | 2.10 | — |
| qwen-vl-utils / qwen_vl_utils | unpinned | — (in `geo` extra) | unpinned | 0.0.14 | 0.0.14 | — (underscore vs hyphen — both forms are the same pkg) |
| ray[default] | unpinned | unpinned | unpinned | 2.55.0 | 2.55.0 | — |
| safetensors | — (transitive) | — | — | 0.7.0 | 0.7.0 | — |
| sentencepiece | — (transitive) | — | — | 0.2.1 | 0.2.1 | — |
| tensorboard | — | unpinned | — | present (8.5.0) / present | present | — |
| tensordict | unpinned | `>=0.8.0,<=0.10.0,!=0.9.0` | `>=0.8.0,<=0.10.0,!=0.9.0` | 0.10.0 | 0.10.0 | — (EasyR1 should inherit the pin) |
| tokenizers | — (transitive) | — | — | 0.22.2 | 0.22.2 | — |
| **torch** | — (transitive) | — (transitive) | — (installed out-of-band) | **2.8.0+cpu** | **2.9.0+cpu** | — (NPU uses torch_npu plugin on top) |
| torch_npu | — (not in EasyR1) | — | — (installed out-of-band) | **2.8.0** | **2.9.0** | — (must match torch major/minor) |
| torchdata | unpinned | unpinned | unpinned | 0.11.0 | 0.11.0 | — |
| torchvision | — | — (in `geo` extra) | — | 0.23.0+cpu | 0.24.0+cpu | — |
| **transformers** | `>=4.54.0,<5.0.0` | unpinned | — (installed out-of-band) | **4.57.6** ✅ | **5.3.0.dev0** ❌ | **V** for 8.5.2: EasyR1's `<5.0.0` excludes it. Options: widen bound, pin 4.57.x into 8.5.2, or target 8.5.0 |
| **triton** | — (transitive via vllm/flash-attn) | — | — | absent | **3.6.0** | — |
| triton-ascend | — | — | `==3.2.0` | 3.2.0 | 3.2.0 | — |
| uvicorn | — | unpinned | — | present | present | — |
| **vllm** | `>=0.8.0` | — (in `vllm` extra, `>=0.8.5,<=0.12.0`) | — | **0.13.0+empty** | **0.18.0+empty** | **R** → actual rollout is `vllm_ascend`; vllm itself is a shim. 8.5.0's 0.13.0 satisfies EasyR1's `>=0.8.0` |
| vllm_ascend | — | — | — (installed out-of-band) | 0.13.1.dev18 | 0.17.0rc2.dev109 | — (matches vllm major/minor per image) |
| wandb | unpinned | unpinned | unpinned | 0.26.0 | 0.26.0 | — |

## Hidden direct imports (not in `requirements.txt`)

EasyR1 source imports these at runtime but doesn't declare them. Both A3 images ship them today, but the matrix is only accurate if we note them explicitly:

| Package | Image 8.5.0 | Image 8.5.2 | Used at | Risk |
|---|---|---|---|---|
| jinja2 | 3.1.6 | 3.1.6 | `verl/utils/dataset.py:24` | low — stable API |
| psutil | 7.2.2 | 7.2.2 | `verl/workers/fsdp_workers.py:23` | low |
| pyyaml | 6.0.3 | 6.0.3 | `verl/utils/py_functional.py:27` | low |

Action: recommend the EasyR1 ascend-port branch list these explicitly in `requirements.txt` so we don't rely on transitive availability. Low-risk, cheap to land.

## Code-path blockers (NOT dependency gaps)

The dep matrix alone does **not** capture the port surface. EasyR1 has CUDA-only runtime assumptions in core paths that no package swap will fix. These are the actual blockers for v1 bring-up:

| Call site | Issue | Fix direction |
|---|---|---|
| `verl/workers/fsdp_workers.py:83` — `dist.init_process_group(backend="nccl")` | NCCL is CUDA-only. NPU uses HCCL. | Make backend configurable, default to `hccl` when `torch_npu` is available |
| `verl/workers/fsdp_workers.py:123,125,131,387` — `init_device_mesh("cuda", ...)` | Device-mesh device type hardcoded | Resolve from torch accelerator detection (`"npu"` on Ascend) |
| `verl/workers/fsdp_workers.py:215,225` — `attn_implementation="flash_attention_2"` | Hardcoded FA2 in `from_pretrained` | Dispatch on device: `sdpa` (NPU-compatible) or a CANN-backed impl |
| `verl/workers/fsdp_workers.py:216` — `device_map="cuda"` | CUDA hardcoded | Resolve to `"npu"` on Ascend |
| `verl/workers/sharding_manager/fsdp_vllm.py:69,72-74,128,164,176,194-195,203,205,210,214-215` — `torch.cuda.*` RNG, empty_cache, mem_get_info | CUDA namespace API | Replace with device-neutral accessor (pattern: `get_device_module() = torch.npu if is_npu_available() else torch.cuda`) |
| `verl/workers/fsdp_workers.py:317,555,575,578,637,673,702,724` — `torch.cuda.current_device()`, `max_memory_allocated`, `max_memory_reserved` | Same | Same |
| `verl/models/transformers/flash_attention_utils.py:35` — `from flash_attn import flash_attn_func, flash_attn_varlen_func` | Direct flash-attn kernel import used as the attention forward | Replace forward with torch_npu FA op or vllm_ascend-backed impl |
| `verl/models/monkey_patch.py:45` — `ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward` | Registers FA2 as the default attention on supported models | Re-register with an NPU-aware forward when on Ascend |
| `verl/workers/actor/dp_actor.py:40`, `verl/workers/critic/dp_critic.py:38` — `from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input` | Flash-attn *utility* helpers (not the kernel) used for variable-length packing | Port padding helpers — these are pure-torch logic, just vendored by flash-attn |
| `verl/utils/torch_functional.py:31` — `from flash_attn.ops.triton.cross_entropy import cross_entropy_loss` | Flash-attn's Triton CE kernel | Fall back to upstream `torch.nn.functional.cross_entropy` if flash-attn absent |
| `verl/utils/flops_counter.py:37` — `torch.cuda.get_device_name()` | CUDA-only call | Device-neutral accessor |
| `verl/utils/model_utils.py:34` — `torch.cuda.mem_get_info()` | Same | Same |
| `verl/single_controller/base/worker.py:132,136` — CUDA device selection on `CUDA_VISIBLE_DEVICES` | Env-var assumption | Handle `ASCEND_RT_VISIBLE_DEVICES` in parallel |
| `verl/utils/checkpoint/*.py`, `verl/utils/fsdp_utils.py` — `torch.cuda.empty_cache()`, `torch.cuda.*rng*` | Misc CUDA calls | Device-neutral accessor |

Roughly **35+ call sites across 10 files**. Some are genuinely one-line swaps (`torch.cuda.empty_cache()` → `torch.npu.empty_cache()`), others require design work (attention backend selection).

## Summary of real v1 blockers

### Dependency-level

1. **flash-attn (R)** — not in image, but the bigger issue is the code-path usage above (kernel + padding helpers + triton CE). Not just a "replace package" — a "replace attention backend and vendor the padding helpers."
2. **transformers ceiling vs 8.5.2 image (V)** — EasyR1 `<5.0.0` fits 8.5.0 (4.57.6) ✅, excludes 8.5.2 (5.3.0.dev0) ❌.
3. **liger-kernel (D)** — optional; drop for v1.

### Code-path level (separate from dep matrix; see table above)

4. **Distributed backend** — hardcoded `nccl` → make `hccl`-compatible.
5. **Device-mesh device type** — hardcoded `"cuda"` → device-neutral.
6. **Attention implementation** — hardcoded `flash_attention_2` → device-aware selection.
7. **`device_map="cuda"`** → device-neutral.
8. **`torch.cuda.*` calls** → replace with device-module accessor (torch.npu on Ascend).
9. **Flash-attn padding helpers + triton CE** → vendor the padding helpers (pure torch), fall back to `F.cross_entropy`.

### Verification needed

10. **huggingface_hub 1.11.0 (8.5.2 only)** — grep for `hf_hub_download` / `snapshot_download` / `HfApi` in EasyR1 was empty on first pass; not urgent for 8.5.0 target.
11. **vllm_ascend model coverage** — check supported model list + attention backend coverage against EasyR1's recipes.

## Prioritization

- **15 of EasyR1's 20 declared runtime deps** match or are superseded in both A3 images without action.
- **Only 2–3 genuine dep-level gaps** (flash-attn, transformers-ceiling-for-8.5.2, liger-kernel) — but the flash-attn one is larger than a package swap.
- **10+ CUDA-specific code paths** must also be abstracted before the port boots on NPU.
- **8.5.0 image is the lowest-friction target**: skips the transformers 5.x + huggingface_hub 1.x migrations that 8.5.2 would add.
- **Strategy**: port EasyR1 against the 8.5.0 image first. 8.5.2 migration is a follow-on task set.

## Next steps (feed into `porting-journal.md`)

1. [TODO] EasyR1 `ascend-port` branch: device-module accessor utility; replace all `torch.cuda.*` call sites.
2. [TODO] EasyR1 `ascend-port` branch: distributed backend selection (nccl ↔ hccl), device-mesh device type, `device_map` resolution.
3. [TODO] EasyR1 `ascend-port` branch: attention backend selection + `ALL_ATTENTION_FUNCTIONS` registration for NPU.
4. [TODO] EasyR1 `ascend-port` branch: vendor flash-attn `bert_padding` helpers (pure torch); fall back for triton CE.
5. [TODO] EasyR1 `ascend-port` branch: declare `jinja2`, `psutil`, `pyyaml` in `requirements.txt`; optionally tighten `tensordict` pin.
6. [TODO] Decide v1 milestone scope: text-only PPO/GRPO, no VLM/video, default loggers (see design doc).
7. [TODO, BLOCKED] A3 runtime: rollout smoke test, short training run on 8.5.0 image with ascend-port branch.
