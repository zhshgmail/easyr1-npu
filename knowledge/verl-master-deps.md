# veRL master — declared dependencies

Source: `upstream/verl/` at April 2026 master tip. Used as GPU reference and as the baseline NPU port (since veRL-A3 images already exist).

## Files inspected

- `pyproject.toml` — PEP 621 metadata, Python `>=3.10`; deps dynamic (from `requirements.txt` + `setup.py` extras).
- `requirements.txt` — core dev deps.
- `requirements-npu.txt` — **NPU-specific requirement set** (the one relevant to us; this is what the A3 image installs from).
- `requirements-test.txt` — test-only.
- `setup.py` — defines `install_requires` and a large set of `extras_require` groups.

## `requirements.txt` (core, GPU or CPU-agnostic)

| Package | Pin |
|---|---|
| accelerate | unpinned |
| codetiming | unpinned |
| datasets | unpinned |
| dill | unpinned |
| hydra-core | unpinned |
| liger-kernel | unpinned |
| numpy | `<2.0.0` |
| pandas | unpinned |
| peft | unpinned |
| pyarrow | `>=19.0.0` |
| pybind11 | unpinned |
| pylatexenc | unpinned |
| pre-commit | unpinned |
| ray[default] | unpinned |
| tensordict | `>=0.8.0,<=0.10.0,!=0.9.0` |
| torchdata | unpinned |
| transformers | unpinned |
| vllm | commented out (`# vllm==0.8.4`) — managed via extras |
| wandb | unpinned |
| packaging | `>=20.0` |
| uvicorn | unpinned |
| fastapi | unpinned |
| latex2sympy2_extended | unpinned |
| math_verify | unpinned |
| tensorboard | unpinned |

## `requirements-npu.txt` (NPU-specific — **high-signal**)

This is the explicit NPU dep set in veRL. Diffs from `requirements.txt`:

| Package | Pin | Delta vs GPU `requirements.txt` |
|---|---|---|
| accelerate | unpinned | same |
| bytecode | unpinned | **NPU-only** |
| codetiming | unpinned | same |
| datasets | unpinned | same |
| dill | unpinned | same |
| hydra-core | unpinned | same |
| numpy | `<2.0.0` | same |
| pandas | unpinned | same |
| peft | `>=0.15.2` | **tighter pin than GPU** |
| pyarrow | `>=15.0.0` | **looser** than GPU's `>=19.0.0` (!) |
| pybind11 | unpinned | same |
| pylatexenc | unpinned | same |
| tensordict | `>=0.8.0,<=0.10.0,!=0.9.0` | same |
| ray[default] | unpinned | same |
| wandb | unpinned | same |
| mathruler | unpinned | **NPU-only** (also in EasyR1 — noteworthy) |
| torchdata | unpinned | same |
| einops | unpinned | **NPU-only** |
| qwen_vl_utils | unpinned | **NPU-only** (also in EasyR1) |
| hf_transfer | unpinned | **NPU-only** |
| triton-ascend | `==3.2.0` | **NPU-only; exact pin** |

Notably **absent** from NPU (present in GPU):
- `liger-kernel` — dropped on NPU (Triton-based kernels; likely replaced by CANN ops)
- `transformers` — not listed, implying it's installed out-of-band (likely from a vendored wheel in the image, tied to `torch_npu` compatibility)
- `pre-commit` — stripped (not needed at runtime)
- `vllm` — replaced by `vllm-ascend` (installed out-of-band in the image)
- `packaging`, `uvicorn`, `fastapi`, `latex2sympy2_extended`, `math_verify`, `tensorboard` — either pulled transitively or installed separately

Added on NPU:
- `bytecode`, `einops`, `hf_transfer`, `qwen_vl_utils` (EasyR1 overlap), `mathruler` (EasyR1 overlap), `triton-ascend==3.2.0`

## `setup.py` extras_require groups

| Extra | Packages |
|---|---|
| `test` | pytest, pre-commit, py-spy, pytest-asyncio, pytest-rerunfailures |
| `prime` | pyext |
| `geo` | mathruler, torchvision, qwen_vl_utils |
| `gpu` | liger-kernel, flash-attn |
| `math` | math-verify |
| `vllm` | tensordict pin, `vllm>=0.8.5,<=0.12.0` |
| `sglang` | tensordict pin, `sglang[srt,openai]==0.5.8`, `torch==2.9.1` |
| `trl` | `trl<=0.9.6` |
| `mcore` | mbridge |
| `trtllm` | `tensorrt-llm>=1.2.0rc6` |
| `transferqueue` | `TransferQueue==0.1.6` |

## Observations

- veRL has an explicit `requirements-npu.txt` — this is **the single most useful file** for our work. It encodes Bytedance/Ascend's own choices about what to install on NPU.
- `triton-ascend==3.2.0` is pinned exactly. That pin matters for `vllm-ascend` compatibility.
- veRL omits `transformers` from `requirements-npu.txt`, confirming it is pinned by the image (likely a specific version `torch_npu` is tested against). The docker inspection will confirm which.
- `liger-kernel` is explicitly **not** in the NPU list → EasyR1's use of `liger-kernel` is a known gap.
- `flash-attn` lives only in the `gpu` extra → EasyR1's hard dep on `flash-attn` in `requirements.txt` (not an extra) is also a known gap.
- veRL supports multiple rollout backends via extras (`vllm`, `sglang`, `trtllm`); EasyR1 commits to vllm-only. Simplifies our work — we only need the `vllm-ascend` path.
