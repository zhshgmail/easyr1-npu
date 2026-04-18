# verl-8.5.0-a3 — image inventory

Image: `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`
Size: 14.1 GB
Extracted: 2026-04-17

> **Executable on A3 hardware only.** This x86 host can inspect the filesystem but cannot run it. All facts below come from `docker create` + `docker cp`.
>
> **Matching upstream refs** for code review against this image live at `repo/knowledge/upstream-refs.md`: torch-npu `v2.8.0-7.3.0`, transformers `v4.57.6`, vllm-ascend `releases/v0.13.0` (image uses a post-branch dev build), triton-ascend `release/3.2.x`.

## Runtime environment

- **Base**: Ubuntu 22.04
- **Python**: 3.11.14 at `/usr/local/python3.11.14` (not the system python)
- **CANN**: `8.5.0` at `/usr/local/Ascend/cann-8.5.0`
- **ATB** (Ascend Transformer Boost): `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1`
- **Entrypoint**: sources three env scripts before exec'ing:
  1. `/usr/local/Ascend/ascend-toolkit/set_env.sh`
  2. `/usr/local/Ascend/cann-8.5.0/share/info/ascendnpu-ir/bin/set_env.sh`
  3. `/usr/local/Ascend/nnal/atb/set_env.sh`

Key env vars baked in:
- `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0`
- `ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0`
- `ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1`
- Several ATB tuning flags (stream sync, kernel cache, workspace alloc, matmul shuffle)
- `LCCL_DETERMINISTIC=0`, `LCCL_PARALLEL=0` (Ascend collective library)

## Core ML / RL stack (source of truth for NPU compatibility)

| Package | Version | Role |
|---|---|---|
| **torch** | `2.8.0+cpu` | Torch without CUDA (NPU goes through torch_npu plugin) |
| **torch_npu** | `2.8.0` | Ascend PyTorch backend — same major as torch |
| **triton_ascend** | `3.2.0` | Triton dialect for Ascend |
| **transformers** | `4.57.6` | |
| **vllm** | `0.13.0+empty` | Empty placeholder — real logic in vllm_ascend |
| **vllm_ascend** | `0.13.1.dev18+g2e5f72f92` | Dev build, matches vllm 0.13.x line |
| **accelerate** | `1.13.0` | |
| **tensordict** | `0.10.0` | Upper bound of veRL's pin |
| **torchdata** | `0.11.0` | |
| **torchvision** | `0.23.0+cpu` | |
| **peft** | `0.19.1` | |
| **ray** | `2.55.0` | |
| **datasets** | `4.8.4` | |
| **numpy** | `1.26.4` | Under 2.0 (veRL's pin) |
| **pyarrow** | `23.0.1` | |
| **safetensors** | `0.7.0` | |
| **tokenizers** | `0.22.2` | |
| **huggingface_hub** | `0.36.2` | |
| **sentencepiece** | `0.2.1` | |
| **compressed_tensors** | `0.12.2` | vllm dependency |

## EasyR1-relevant secondary deps (all present)

| Package | Version |
|---|---|
| einops | 0.8.2 |
| hf_transfer | 0.1.9 |
| hydra_core | 1.3.2 |
| omegaconf | 2.3.0 |
| mathruler | 0.1.0 |
| qwen_vl_utils | 0.0.14 |
| pandas | 3.0.2 |
| bytecode | 0.17.0 |
| codetiming | 1.4.0 |
| dill | 0.4.1 |
| pybind11 | 3.0.3 |
| pylatexenc | 2.10 |
| wandb | 0.26.0 |

## EasyR1-required packages **absent** from this image

These are the explicit gaps EasyR1 introduces on top of veRL-A3 8.5.0:

| Package | Why absent | Action |
|---|---|---|
| **flash-attn** | GPU-only kernel | Replace with CANN attention (via torch_npu / vllm_ascend backend) |
| **liger-kernel** | Triton GPU kernels | Evaluate via triton-ascend, else drop |
| **pillow** | Not a veRL-NPU dep | Likely pulls cleanly (pure Python+C, no CUDA); add |

`omegaconf` is present but pulled transitively via `hydra-core` — an explicit EasyR1 dep is still fine.

## veRL-specific tooling also present (not in EasyR1)

Informational only — EasyR1 doesn't depend on these but they're installed:

- `megatron_core` 0.12.1, `mindspeed` 0.12.1, `mbridge` 0.15.1 — Megatron/Ascend Megatron-adjacent tooling, pulled in because veRL supports mcore backend
- `nvidia_modelopt` 0.43.0 — note: present even on NPU image (CPU-only ops may still be useful?), worth verifying it doesn't pull nvidia-cuda-*
- `pytest`, `pytest-cov`, `pytest-asyncio` etc. — test harness shipped in image
- `scikit_learn`, `gpytorch`, `linear_operator`, `pulp`, `joblib` — analytic/optimization utilities used by some recipes
- `Flask_RESTful`, `aniso8601`, `sqlalchemy`, `tensorstore`, `zarr`, `ml_dtypes`, `numcodecs` — long tail

## verl binary

- `verl-0.8.0.dev0` installed — this is **verl, not EasyR1**. Confirms the image is a generic veRL-NPU runtime.

## apt packages (dpkg)

- Total: 197 packages (Ubuntu 22.04 base)
- Notable: `cmake`, `curl`, `git`, `wget`, `libatomic1`, `libgcc-11-dev`, `libstdc++-11-dev`, `libnuma1`, `libssl3`, `libgomp1`
- No openmpi / mpich at the apt level — if Ascend collectives need those, they come via Ascend driver packages

## Totals

- Python site-packages: **246 distributions**
- Apt packages: 197
- Image size on disk: 14.1 GB

## Open questions

- Does this image's `transformers==4.57.6` satisfy EasyR1's `transformers>=4.54.0,<5.0.0`? → **Yes** (4.57.6 ≥ 4.54.0, and < 5.0.0). Good.
- Can `vllm_ascend==0.13.1.dev18` service EasyR1's `vllm>=0.8.0` contract? → Needs checking against vllm-ascend's supported-model list and EasyR1's rollout code.
- Is `pillow` available transitively (via torchvision / datasets)? → Likely yes, but EasyR1 should not rely on transitives; verify in dep matrix.
