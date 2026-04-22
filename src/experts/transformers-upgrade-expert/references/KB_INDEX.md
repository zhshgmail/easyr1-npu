# transformers-upgrade-expert KB — Search Index

> **Workers: grep this file FIRST when looking up a problem.**

## Decision frameworks (load unconditionally at Phase A)

| File | What | Keywords | When |
|---|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL rules (OL-01/02/04/04b/05/05b/06/07/09/10) | py_compile, session tag, chip precheck, GFW, mirror, provenance, triage | **MANDATORY — Phase A 第一读** |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | This expert's OL-03 (denylist) + OL-08 (edit scope) | ascend-port*, shim files | **MANDATORY — Phase A 第二读** |

## Pattern catalog (Phase B)

| File | When to load |
|---|---|
| [patterns/domains/image-diff-shim.md](patterns/domains/image-diff-shim.md) | Phase A pip-freeze diff + Phase B known shim apply (EC-02, EC-03, etc.) |
| [patterns/domains/dockerfile-target.md](patterns/domains/dockerfile-target.md) | Phase B writing new Dockerfile.npu-\<target\>; triton-ascend repair + NPU-BUG-004 upstream-triton-prune |
| [patterns/domains/validation-smoke.md](patterns/domains/validation-smoke.md) | Phase D writing the validation smoke config + picking the right baseline band |

## Error corrections (defer-load at Phase C/D failure)

| File | When |
|---|---|
| [ERROR_CORRECTIONS.md](ERROR_CORRECTIONS.md) | Build/smoke fail on the new image — match traceback to EC-NN |

Specific patterns indexed:

- **ImportError: cannot import name 'no_init_weights' from 'transformers.modeling_utils'** → EC-02 (transformers ≥5 moved to `transformers.initialization`)
- **AttributeError: property 'eos_token_id' of 'SamplingParams' object has no setter** → EC-03 (vllm ≥0.18 read-only properties)
- **ImportError: cannot import name 'Language' from 'triton.backends.compiler'** → EC-04 (upstream triton coexisting with triton-ascend; NPU-BUG-004)
- **Stack-import error from torch._inductor on NPU** → NPU-BUG-001 (triton-ascend broken install; recurs across images)
- **pip install hangs > 60s** → EC-10 (OL-07, aliyun first)

## Platform bugs (defer-load at failure)

| File | Keywords | When |
|---|---|---|
| [PLATFORM_BUGS.md](PLATFORM_BUGS.md) | NPU-BUG-001 (triton-ascend install), NPU-BUG-004 (upstream triton + triton-ascend coexist) | Build error / import error on new stack |

These two recur across base images. Every upgrade drill has to address both.

## Smoke baseline (Phase D)

| File | Keywords | When |
|---|---|---|
| [SMOKE_BASELINE.md](SMOKE_BASELINE.md) | v1 V1.4 = 0.991 [0.94, 1.04]; v2 V1.4 = 1.275 [1.21, 1.34] | Phase D assertion. **CRITICAL**: use TARGET image's band, not source's (G3). |

## Coverage audit

Stage 0 KB covers:
- Image inventory diff workflow (pip freeze source vs target)
- 2 shim patterns (no_init_weights move, SamplingParams read-only)
- 2 platform-bug workarounds that recur across images (NPU-BUG-001, NPU-BUG-004)
- Validation smoke baseline numerics per image family (v1, v2)

Gaps known at Stage 0 start (will grow with each upgrade session):
- transformers 6.x / vllm 0.20+ — not yet drilled
- CANN 8.5.2 → 8.6 — not yet drilled
- torch_npu 2.9 → 3.x — not yet drilled

Each gap becomes a new EC entry after a real drill verifies the fix.
