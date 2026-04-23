# easyr1-npu-torch211-fixc — Day-0 torch 2.11 + Fix C overlay inventory

Image: `easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423`
Size: ~28 GB (base + 3 thin overlays; vllm_ascend_C.so is ~472KB on top of Fix B+)
Built: 2026-04-23 (dusk session)

> **Production-ready Day-0 image** for running vllm + EasyR1 on
> community PyTorch 2.11 + pre-release torch_npu 2.11.0rc1. Validated
> end-to-end for both inference (V1.3 rollout) and training (V1.4 GRPO).

## How this image was built (3 overlay layers)

```
easyr1-npu-852:trans-upg-e2e-20260422-2200     (v2 base image)
  ↓ pip install --no-deps torch==2.11.0+cpu torch_npu==2.11.0rc1 ...
easyr1-npu-torch211:torch-day0-manual-20260423-0537     (torch layer)
  ↓ COPY patched utils.py, __init__.py, matmul.py      (Fix B+)
easyr1-npu-torch211-vllmascend-fixb:ascend-day0-torch211-20260423
  ↓ COPY rebuilt vllm_ascend_C.so + kernels + _cann_ops_custom  (Fix C)
easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423    ← this image
```

## Runtime environment

- **Base**: Ubuntu 22.04 (inherited from v2 image)
- **Python**: 3.11.14 at `/usr/local/python3.11.14`
- **CANN**: `8.5.1` at `/usr/local/Ascend/cann-8.5.1` (from base)
- **NPU**: Ascend 910 A3; SOC_VERSION `ascend910_9391`

## Day-0 overlay stack

| Package | Version | Origin |
|---|---|---|
| **torch** | `2.11.0+cpu` | community, download.pytorch.org |
| **torch_npu** | `2.11.0rc1` | PyPI pre-release wheel (2026-03-24) |
| **torchvision** | `0.26.0+cpu` | community |
| **torchaudio** | `2.11.0+cpu` | community |
| **vllm** | `0.18.0+empty` | unchanged from base |
| **vllm_ascend** | `0.17.0rc2.dev109+g54879467c + patches` | **branch `ascend-day0-torch211-20260423` on `zhshgmail/vllm-ascend`** (4 commits; `utils.py + __init__.py + matmul.py + CMakeLists.txt`) |
| **vllm_ascend_C.so** | rebuilt against torch 2.11 ABI | 472KB, generated in-container via `python3 setup.py build_ext --inplace` |
| **libvllm_ascend_kernels.so** | rebuilt | 423KB |
| **CANN** | `8.5.1` | base image (README-paired version is 8.5.0; +1 patch) |
| **triton_ascend** | `3.2.0` | from base |
| **transformers** | `5.3.0.dev0` | from base |

## Patches applied on `zhshgmail/vllm-ascend/ascend-day0-torch211-20260423`

1. `7c2078e7` — `vllm_ascend/utils.py`: add `_torch_abi_safe_for_custom_ops()`
   + guard in `enable_custom_op()` (auto-disable custom ops when torch
   ABI doesn't match the extension's build version)
2. `caa55fed` — `vllm_ascend/__init__.py`: auto-set
   `VLLM_BATCH_INVARIANT=1` at plugin entry when ABI mismatches
   (must run before any vllm module caches the env var)
3. `87b507ed` — `vllm_ascend/ops/triton/batch_invariant/matmul.py`:
   widen `linear_batch_invariant` wrapper to reshape 3D → 2D → restore
   (fixes V1.4 training path under batch-invariant mode)
4. `ab26a534` — `CMakeLists.txt`: widen `find_package(Torch)` version
   check to accept `torch 2.11.x` (was hard-pinned `2.9.0`; blocked
   rebuild)

## Validated smoke matrix

| Rung | Setup | Result |
|---|---|---|
| V1.3 rollout (Qwen2-0.5B, 1 chip, enforce_eager) | default (batch-invariant auto-on via Fix B+) | **PASS** — `V1.3 ROLLOUT SMOKE PASSED`; 3/3 prompts produce text |
| V1.4 training (Qwen2-0.5B GRPO + math12k, 2 chips, 2 steps) | `VLLM_BATCH_INVARIANT=0` (force native custom-op path on Fix C) | **PASS** — `entropy_loss=1.275` exact match to v2 baseline band `[1.21, 1.34]` |

Training path goes through native PrivateUse1 NPU backward (no
batch-invariant fallback, no CPU dispatch). Fix C rebuild is what
enabled this.

## Usage recipe

```dockerfile
FROM easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423
# your RL framework layer here
```

Run container:

```bash
docker run --rm -it --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /data/<your-user>/models:/data/<your-user>/models:ro \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423 \
    bash
```

Training invocations must set `VLLM_BATCH_INVARIANT=0` explicitly to
force the native custom-op path (Fix B+'s auto-set would otherwise
route to the slower batch-invariant fallback).

## Known not-tested on this image

- V1.1 device smoke (trivial but not run on Fix C)
- V1.5 (4-chip FSDP full shard)
- V2.1 (ulysses + padding-free, 4 chips)
- V2.2 (larger padding + ulysses, 4 chips)

These are training-ladder upper rungs; V1.4 PASS is the strongest
training-path signal, but higher-fidelity smokes haven't been run.
Cost: 4 chips × ~20 min each. Not run in this session per OL-05b
chip economy.

## Bug report routing

- Fix B+ / Fix C patches in our branch need refinement → issue against
  `easyr1-npu` with session tag `vllm-ascend-day0-*-20260423-*`
- Underlying vllm-ascend code needs changes beyond what our 4 commits
  cover → `github.com/vllm-project/vllm-ascend` issue + reference the
  personal-fork branch
- Ascend aclnn kernel not supporting `ascend910_93` for an op → Ascend
  team (separate workflow; our `_C_ascend.npu_add_rms_norm_bias`
  surfaced this earlier — AclNN kernel coverage gap, not Fix C scope)

## Cross-references

- `docs/examples/torch-2.11-day0.md` — user-facing 0-interaction skill-chain example
- `src/experts/torch-day0-expert/` — the Day-0 skill that produced the torch layer
- `src/experts/vllm-ascend/day0-expert/` — the Day-0 skill that produced Fix B+ / Fix C
- `workspace/vllm-ascend-day0-deploy-20260423-0655/` — session artifacts (Dockerfile, patched .py files, ONBOARDING, PR_MATERIAL)
- `knowledge/upstream-refs.md` § "Day-0 overlay combinations (2026-04-23)"
