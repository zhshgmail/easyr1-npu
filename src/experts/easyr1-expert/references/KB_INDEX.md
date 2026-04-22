# EasyR1-port KB — Search Index

> **Workers: grep this file FIRST when looking up a problem.**
> Keywords/Aliases 列 designed for grep. Matched rows list the file to load next.

## How to search

```bash
grep -i "nccl\|hccl\|distributed" KB_INDEX.md
# → pointed at patterns/domains/device_dispatch.md
```

---

## Decision frameworks (load unconditionally at Phase A)

| File | What | Keywords/Aliases | When |
|---|---|---|---|
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | Meta rules + universal traps (OL-01..OL-10) | py_compile, dry-import, denylist, provenance, A3 shared host, github timeout, huaweicloud mirror, session tag | **MANDATORY — Phase A 首个加载** |

---

## Code-path pattern catalog (load at Phase B based on what you need to apply)

| File | Keywords/Aliases | When |
|---|---|---|
| [CODE_PATH_PATTERNS.md](CODE_PATH_PATTERNS.md) | **Master catalog** NPU-CP-001..007 all | Phase B start |
| [patterns/domains/device_dispatch.md](patterns/domains/device_dispatch.md) | `torch.cuda.*`, `init_device_mesh("cuda",...)`, `device_map="cuda"`, `nccl` backend, `CUDA_VISIBLE_DEVICES` | NPU-CP-001 fix |
| [patterns/domains/ray_integration.md](patterns/domains/ray_integration.md) | `num_gpus`, `ray.available_resources()["GPU"]`, `{"GPU": n}` placement bundle, `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO`, `ASCEND_RT_VISIBLE_DEVICES`, VLLM_ASCEND_ENABLE_NZ | NPU-CP-003 + NPU-BUG-002 + NPU-ENV-002 |
| [patterns/domains/attention_backend.md](patterns/domains/attention_backend.md) | `flash_attn`, `attn_implementation="flash_attention_2"`, `bert_padding`, `npu_flash_attention`, `npu_flash_attn_varlen_func`, padding_free | NPU-CP-007 |
| [patterns/domains/vllm_compat.md](patterns/domains/vllm_compat.md) | `vllm.lora.models`, `get_tensor_model_parallel_group`, `SamplingParams.eos_token_id`, vllm 0.13 rename, 0.18 read-only property | NPU-CP-002 + NPU-CP-004 |
| [patterns/domains/transformers_compat.md](patterns/domains/transformers_compat.md) | `no_init_weights`, `transformers.modeling_utils` vs `transformers.initialization`, transformers 4.x vs 5.x import | transformers upgrade shims |
| [patterns/domains/dockerfile.md](patterns/domains/dockerfile.md) | `Dockerfile.npu`, `triton-ascend force-reinstall`, aliyun pip mirror, huaweicloud unreliable, `pip --default-timeout`, `FROM quay.io/ascend/verl` | Phase C Dockerfile |

---

## Platform bugs (defer-load at Phase D failure)

| File | Keywords/Aliases | When |
|---|---|---|
| [PLATFORM_BUGS.md](PLATFORM_BUGS.md) | NPU-BUG-001 triton-ascend install incomplete; NPU-BUG-002 RAY_ACCEL override; NPU-BUG-003 inductor log_probs crash; NPU-BUG-004 triton 3.6 + triton-ascend 3.2 coexistence | Build/smoke errors — match to BUG-N |

---

## Error corrections (defer-load at Phase D failure — traceback → fix)

| File | Keywords/Aliases | When |
|---|---|---|
| [ERROR_CORRECTIONS.md](ERROR_CORRECTIONS.md) | EC-01..EC-NN mapping traceback patterns to root cause + fix | Smoke/build fails — match error signature to EC |

Specific error patterns indexed:

- **SyntaxError: invalid syntax (`from X import (` ... `from Y import ...`)** → EC-01
- **ImportError: cannot import name 'no_init_weights' from 'transformers.modeling_utils'** → EC-02
- **AttributeError: property 'eos_token_id' of 'SamplingParams' object has no setter** → EC-03
- **ImportError: cannot import name 'Language' from 'triton.backends.compiler'** → EC-04
- **ValueError: Total available GPUs 0 is less than total desired GPUs N** → EC-05
- **HFValidationError: Repo id must be in the form 'repo_name'** → EC-06
- **dcmi model initialized failed, because the device is used. ret is -8020** → EC-07
- **npu get board type failed. ret is -9005** → EC-08
- **uda_occupy_dev_by_ns ... Conflict open udevid** → EC-09
- **pip install triton-ascend hangs > 60s** → EC-10
- **V1.4 smoke_validate "no entropy_loss marker" but training ran end-to-end** → EC-11
- **V1.4 entropy_loss ≈ 1.27 (out of v1 band 0.94-1.04) despite correct port** → EC-12

---

## Smoke baseline (load at Phase D rung assertion)

| File | Keywords/Aliases | When |
|---|---|---|
| [SMOKE_BASELINE.md](SMOKE_BASELINE.md) | V1.1 / V1.3 / V1.4 / V1.5 / V2.1 / V2.2 per-image baseline (entropy_loss, grad_norm, step duration) | Phase D assert each rung |

Key numbers (for quick eye):

- V1.4 on v1 (8.5.0): step1 entropy_loss=**0.991**, step2=**1.263**
- V1.4 on v2 (8.5.2 drill): step1=**1.275**, step2=**0.895**
- V1.5 on v1 (4-chip): step1 entropy_loss=**1.127**, step2=**1.163**

---

## Coverage audit

Stage 0 KB covers these concerns:
- 5 archetype fixes (device / Ray / attn / vllm compat / Dockerfile)
- 10 known error signatures → fixes
- 4 platform bugs with workarounds
- 6 smoke rung baselines on 2 images

**Gaps known at Stage 0 start** (will grow via worker's Phase D findings):
- transformers 5/6 specific shims (beyond no_init_weights) — Stage 1+
- vllm 0.18+ further API changes — Stage 1+
- Multi-node HCCL tuning — not Stage 0 scope
