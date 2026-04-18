# verl-8.5.2-a3 — image inventory

Image: `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`
Size: 24.2 GB (≈10 GB larger than 8.5.0; mostly model weights in the image tag's "qwen3-5" variant)
Extracted: 2026-04-17

> **Executable on A3 hardware only.** Inspected via `docker create` + `docker cp`.
>
> **Matching upstream refs** in `repo/knowledge/upstream-refs.md`: torch-npu `v2.9.0-7.3.0`, transformers `main` (no tag yet for 5.3.0.dev0), vllm-ascend `main` (image is a pre-`releases/v0.18.0` snapshot), triton-ascend `release/3.2.x` (image also ships upstream `triton 3.6.0`).

## Runtime environment

- **Base**: Ubuntu 22.04
- **Python**: 3.11.14 at `/usr/local/python3.11.14`
- **CANN**: `8.5.1` at `/usr/local/Ascend/cann-8.5.1` — **note: image tag says 8.5.2 but the CANN version is 8.5.1.** The `8.5.2` is the verl-image revision, not CANN.
- **ATB**: `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1`
- **Entrypoint**: same 3-script source pattern as 8.5.0, with `cann-8.5.1` paths.
- **Additional env vars vs 8.5.0:**
  - `CMAKE_PREFIX_PATH=/usr/local/Ascend/cann-8.5.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-8.5.1/lib64/cmake`
  - `MAX_JOBS=1`, `MAKEFLAGS=-j1`, `CMAKE_BUILD_PARALLEL_LEVEL=1` — suggests the image also supports rebuilding extensions in-place with limited parallelism

## Core ML / RL stack

| Package | Version | vs 8.5.0 |
|---|---|---|
| **torch** | `2.9.0+cpu` | bumped from 2.8.0 |
| **torch_npu** | `2.9.0` | bumped from 2.8.0 |
| **triton** | `3.6.0` | **new — not in 8.5.0** |
| **triton_ascend** | `3.2.0` | same |
| **transformers** | `5.3.0.dev0` | **major bump** from 4.57.6 → 5.x dev |
| **vllm** | `0.18.0+empty` | bumped from 0.13.0 |
| **vllm_ascend** | `0.17.0rc2.dev109+g54879467c` | bumped from 0.13.1.dev18 |
| accelerate | 1.13.0 | same |
| tensordict | 0.10.0 | same |
| torchdata | 0.11.0 | same |
| torchvision | 0.24.0+cpu | bumped from 0.23.0 |
| **torchaudio** | `2.11.0+cpu` | **new** |
| peft | 0.19.1 | same |
| ray | 2.55.0 | same |
| datasets | 4.8.4 | same |
| numpy | 1.26.4 | same |
| pyarrow | 23.0.1 | same |
| safetensors | 0.7.0 | same |
| tokenizers | 0.22.2 | same |
| huggingface_hub | `1.11.0` | **major bump** from 0.36.2 (1.0 released in early 2026) |
| sentencepiece | 0.2.1 | same |
| compressed_tensors | 0.13.0 | bumped |

## EasyR1-relevant secondary deps

Unchanged vs 8.5.0: `einops 0.8.2`, `hf_transfer 0.1.9`, `hydra_core 1.3.2`, `omegaconf 2.3.0`, `mathruler 0.1.0`, `qwen_vl_utils 0.0.14`, `pandas 3.0.2`, `bytecode 0.17.0`, `codetiming 1.4.0`, `dill 0.4.1`, `pybind11 3.0.3`, `pylatexenc 2.10`, `wandb 0.26.0`.

## EasyR1-required packages **absent** (same as 8.5.0)

- **flash-attn** — replace with CANN attention
- **liger-kernel** — evaluate via triton-ascend (triton 3.6.0 is also present here, unlike 8.5.0), else drop
- **pillow** — add explicitly

## Packages dropped vs 8.5.0

The 8.5.2 image is **leaner**: 223 distributions vs 246, and drops a batch of Megatron/Ascend-Megatron tooling plus test harness:

- **Megatron stack**: `megatron_core`, `mindspeed`, `mbridge` (removed)
- **Test harness**: `pytest`, `pytest_cov`, `pytest_asyncio`, `pytest_mock`, `pytest_random_order`, `coverage`, `pluggy`, `iniconfig` (removed — closer to a runtime image)
- **Scientific stack**: `scikit_learn`, `gpytorch`, `linear_operator`, `pulp`, `joblib`, `threadpoolctl`, `tensorstore`, `zarr`, `numcodecs` (removed)
- **Long tail**: `aniso8601`, `donfig`, `Flask_RESTful`, `ml_dtypes`, `nltk`, `nvidia_modelopt`, `nvidia_ml_py`, `google_crc32c`, `pytz`, `sqlalchemy`, `greenlet` (removed)

## Packages added vs 8.5.0

- `triton` 3.6.0 — upstream Triton alongside triton-ascend. Suggests some paths now use upstream triton IR on CPU (possibly as an AOT compile step for later NPU consumption?). Worth a closer look.
- `torchaudio` 2.11.0+cpu — added to match torch 2.9
- `opentelemetry_exporter_otlp*`, `opentelemetry_semantic_conventions_ai` — telemetry exporters (observability)

## verl binary

- `verl-0.8.0.dev0` — same version string as in 8.5.0. The "8.5.2" in the image tag is a **Bytedance image revision**, not a verl package version; the package still reads as 0.8.0.dev0.

## Totals

- Python site-packages: **223 distributions** (vs 246 in 8.5.0)
- Apt packages: 197
- Image size on disk: 24.2 GB (extra bulk is weights in the `qwen3-5` tag)

## Critical observation for the dep matrix

**Both images install `transformers` out-of-band** (not via veRL's `requirements-npu.txt`). The version difference between images — `4.57.6` (8.5.0) → `5.3.0.dev0` (8.5.2) — is the single largest risk factor for EasyR1 compatibility:

- EasyR1 requires `transformers>=4.54.0,<5.0.0` → **compatible with 8.5.0, incompatible with 8.5.2.**
- To use the newer A3 stack (torch_npu 2.9, vllm_ascend 0.17) we would need EasyR1 to either (a) relax its transformers upper bound to `<6.0.0`, (b) downgrade transformers inside the 8.5.2 image, or (c) adopt a transformers 5.x-compatible EasyR1 branch.
- This is the first concrete porting task to land in `porting-journal.md`.

## Open questions

- Why does 8.5.2 ship upstream `triton 3.6.0` *and* `triton_ascend 3.2.0`? Is the upstream triton used as a lowering dialect for triton-ascend, or as a fallback for CPU paths in vllm?
- Is `huggingface_hub 1.11.0` a drop-in for users that only call the public surface? 1.0 was advertised as source-compatible but transformers 5.x may tighten the coupling.
- Does the lack of `pytest*` in 8.5.2 imply their CI runs against 8.5.0 but ships 8.5.2 to users? If so, our integration tests may need to target 8.5.0.
