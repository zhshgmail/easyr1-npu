# transformers-upgrade-expert Error Corrections

> Structured error → repair mappings for base-image upgrade failures.
> Load when: Phase C build / Phase D smoke fails on the target image.

All entries derived from the 2026-04-19 transformers 4.57→5.3 drill
(docs/transformers-upgrade-drill.md) plus subsequent finds. Each has a
concrete commit or dated observation as proof.

---

## EC-02: ImportError for `no_init_weights` from `transformers.modeling_utils`

**Error pattern**:
```
ImportError: cannot import name 'no_init_weights' from 'transformers.modeling_utils'
```
when importing `verl.workers.fsdp_workers` on target image with transformers ≥ 5.0.

**Root cause**: transformers 5.0 moved `no_init_weights` from
`transformers.modeling_utils` to `transformers.initialization`. The rest of
the API stayed compatible.

**Fix** (applied in `verl/workers/fsdp_workers.py`):
```python
try:
    from transformers.initialization import no_init_weights  # transformers >= 5.0
except ImportError:
    from transformers.modeling_utils import no_init_weights  # transformers <= 4.x
```

Verified on drill commit `55bb730`. Backward compatible: transformers 4.57
still imports via the fallback branch, so v1 smoke PASSes unchanged.

**Verify**: `python3 -c "from verl.workers import fsdp_workers"` must not
error inside the target-image container.

**Related**: drill report §"Code changes to EasyR1".

---

## EC-03: AttributeError: property 'eos_token_id' of SamplingParams has no setter

**Error pattern**:
```
AttributeError: property 'eos_token_id' of 'SamplingParams' object has no setter
```
raised in `verl/workers/rollout/vllm_rollout_spmd.py:update_sampling_params`
when running on target image with vllm ≥ 0.18.

**Root cause**: vllm 0.18 promoted several `SamplingParams` fields from
mutable attrs to read-only properties (`eos_token_id`, `stop_token_ids`,
`output_kind`). Code that `setattr`'d them on the fly now raises.

**Fix** (in `verl/workers/rollout/vllm_rollout_spmd.py`
`update_sampling_params`):
```python
# Detect read-only properties via descriptor introspection; skip them
# (the underlying default from SamplingParams.__init__ is correct for
# our rollout — no need to override per request).
for k, v in updates.items():
    cls_attr = getattr(type(sampling_params), k, None)
    if isinstance(cls_attr, property) and cls_attr.fset is None:
        continue
    setattr(sampling_params, k, v)
```

Alternative (simpler, but needs updating when vllm adds more read-only
props): explicit exclusion list.

Verified on drill commit `d213f01`. Backward compatible: on vllm ≤ 0.13
the descriptor check returns `None` / a mutable attr, so `setattr` runs
the original code path. v1 smoke PASSes unchanged.

**Verify**: `python3 -m verl.trainer.main ... ` reaches the first
`update_sampling_params` call without AttributeError.

**Related**: drill report §"Code changes to EasyR1"; NPU-CP-002
(vllm.lora.models rename) is a different vllm concern in the easyr1-expert
domain.

---

## EC-04: ImportError `Language` from `triton.backends.compiler` on target image

**Error pattern**:
```
ImportError: cannot import name 'Language' from 'triton.backends.compiler'
```
from `torch_npu` or downstream import chain.

**Root cause**: NPU-BUG-004. Target image ships upstream `triton` wheel with
`triton/backends/{amd,nvidia}/` subdirs that reference backend APIs
incompatible with triton-ascend's registration order.

**Fix**: prune the amd/nvidia backend dirs in Dockerfile.npu-\<target\>:
```dockerfile
RUN python3 -c '
import importlib.util, os, shutil
spec = importlib.util.find_spec("triton")
if spec and spec.origin:
    triton_dir = os.path.dirname(spec.origin)
    for backend in ["amd", "nvidia"]:
        p = os.path.join(triton_dir, "backends", backend)
        if os.path.isdir(p):
            shutil.rmtree(p)
'
```

Verified on drill commit `15f9450` / `a18d1f8`. Works on v2 drill image;
will need re-verification on any future base image that ships the same
upstream triton wheel variant.

**Verify**: `docker run --rm <image> python3 -c "import torch_npu, triton"` clean.

**Related**: PLATFORM_BUGS.md NPU-BUG-004.

---

## EC-10: pip install triton-ascend hangs

**Error pattern**: `docker build` appears stuck on `RUN pip install ... triton-ascend...`
for 50+ min, no progress.

**Root cause**: NPU-OPS-008. Huaweicloud ascend pypi mirror intermittently
empty / slow. Without a timeout, pip hangs indefinitely on TLS handshake
or an empty index.

**Fix**: use aliyun first with `--default-timeout=60`, huaweicloud as
fallback (see `patterns/domains/dockerfile-target.md` Stage 0 template).

**Verify**: `docker build` either completes or fails within ~2 min on the
triton-ascend step, not 50 min.

**Related**: OL-07; PLATFORM_BUGS.md NPU-OPS-008; drill report §"New
operational pits surfaced".

---

## Unclassified failure protocol

If a target-image failure doesn't match any EC here:

1. Document the full traceback in PROGRESS.md §"unclassified failures"
2. Check PLATFORM_BUGS.md for NPU-BUG-NN match
3. Check easyr1-expert's ERROR_CORRECTIONS.md — some platform bugs recur
4. If still unknown: record as a new finding, exit `stuck`, don't guess
   a fix and retry silently

New reproducible failures → add a new EC entry here for future sessions.
