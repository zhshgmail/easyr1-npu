# EasyR1 master — declared dependencies

Source: `upstream/EasyR1/` at April 2026 master tip.

## Files inspected

- `pyproject.toml` — minimal; declares project metadata only (interesting note: keeps `name = "verl"` internally, i.e. EasyR1 is installed as the `verl` Python package). Dynamic deps, sourced from `requirements.txt`.
- `requirements.txt` — the source of truth for runtime deps.
- `setup.py` — boilerplate that reads `requirements.txt`.
- No `requirements-npu.txt`, no `requirements-test.txt`, no `requirements-*.txt`.
- No `extras_require` / optional-deps groupings.

## Declared runtime deps (from `requirements.txt`)

| Package | Pin | Notes |
|---|---|---|
| accelerate | unpinned | |
| codetiming | unpinned | |
| datasets | unpinned | HuggingFace datasets |
| flash-attn | `>=2.4.3` | **GPU-only kernel** — NPU replacement required |
| liger-kernel | unpinned | Triton-based kernels; NPU path unclear (likely via triton-ascend or drop) |
| mathruler | unpinned | EasyR1-specific math reward util |
| numpy | unpinned | |
| omegaconf | unpinned | (EasyR1 uses omegaconf; veRL uses hydra-core which bundles omegaconf) |
| pandas | unpinned | |
| peft | unpinned | |
| pillow | unpinned | |
| pyarrow | `>=15.0.0` | |
| pylatexenc | unpinned | |
| qwen-vl-utils | unpinned | Qwen VL helper utilities |
| ray[default] | unpinned | Distributed runtime |
| tensordict | unpinned | |
| torchdata | unpinned | |
| transformers | `>=4.54.0,<5.0.0` | **Newer than what torch_npu may currently support — gap candidate** |
| vllm | `>=0.8.0` | On NPU → replace wholesale with `vllm-ascend` |
| wandb | unpinned | |

## Observations

- Surface is 20 packages vs veRL's ~30+ — EasyR1 is genuinely slimmer.
- **No torch version pin.** Torch comes via `flash-attn` / `vllm` / `tensordict` transitive constraints.
- **No test-deps file** — tests rely on whatever the dev installed.
- **No optional-dep groupings** (`extras_require`). Every dep in `requirements.txt` is required for any install.
- **No dev-tool deps** — no pre-commit, ruff, etc. declared here.

## Gap candidates flagged on first read

1. `flash-attn>=2.4.3` — must be replaced by CANN-provided attention ops on NPU (via `torch_npu` FA ops or `vllm-ascend`'s attention backend).
2. `transformers>=4.54.0` — the ceiling of what `torch_npu` supports needs to be verified; this may force a `torch_npu` patch or a `transformers` downgrade.
3. `vllm>=0.8.0` — replace with `vllm-ascend`. Need to check `vllm-ascend`'s covered vllm version range.
4. `liger-kernel` — Triton-based; on NPU either via `triton-ascend` or dropped.
5. `tensordict` — no version pin; veRL pins `>=0.8.0,<=0.10.0,!=0.9.0`. EasyR1 may accidentally pull a version incompatible with vllm-ascend's pins.

## What to compare against

- veRL master `requirements.txt` + `requirements-npu.txt` + `setup.py`'s `extras_require` groups → see `verl-master-deps.md`.
- verl-8.5.0-a3 and verl-8.5.2-a3 pip freeze → see `images/verl-*.md` (TBD).
