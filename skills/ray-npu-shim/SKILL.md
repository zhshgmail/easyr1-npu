---
name: ray-npu-shim
description: A drop-in Python module for any Ray-based trainer that needs to run on Ascend NPU. Wraps ray.init, actor options, and placement-group bundles with NPU-aware defaults (custom NPU resource, RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO, VLLM_ASCEND_ENABLE_NZ). Copy `ray_npu_shim.py` into the target project and swap 4-5 call sites. Generalizes NPU-CP-003 + NPU-BUG-002 + NPU-ENV-002 so future RL/ML framework ports don't re-derive them.
---

# ray-npu-shim

## Why

Every Ray-based RL / ML training framework ported to NPU has to solve the same three integration points:

1. **Ray doesn't auto-detect NPU as a resource.** `num_gpus` sugar is CUDA-only; NPU chips have to be registered with `resources={"NPU": n}` and claimed via `options["resources"]={"NPU": n}`. (`NPU-CP-003`)
2. **Ray 2.55+ wipes `ASCEND_RT_VISIBLE_DEVICES` on actor spawn** when `num_gpus` is 0/None (which is always, since we use the custom resource). Needs `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`. (`NPU-BUG-002`)
3. **vllm-ascend's FRACTAL_NZ weight layout corrupts param sync in RL.** Needs `VLLM_ASCEND_ENABLE_NZ=0` in the actor env. (`NPU-ENV-002`)

Doing these three by hand across multiple files of a host framework is error-prone and regresses on upgrades. This skill packages all three into one drop-in Python module.

## What you get

`skills/ray-npu-shim/ray_npu_shim.py` — ~100 LOC, no local imports, Apache-2.0.

Public API:

| Symbol | Purpose |
|---|---|
| `is_npu_available()` | True iff torch_npu loads and reports an NPU. lru_cached. |
| `get_ray_resource_name()` | `"NPU"` on NPU, `"GPU"` otherwise. Use for `available_resources()[...]` and resource-dict keys. |
| `ray_init_npu_aware(**kw)` | Forwards to `ray.init`; injects `resources={"NPU": device_count()}` on NPU, merges env-var defaults into `runtime_env.env_vars`. |
| `apply_actor_options(options, num_accel)` | Set `num_gpus=n` on CUDA, `resources={"NPU": n}` on NPU. |
| `placement_bundle(num_cpus, num_accel)` | Returns `{"CPU": ..., resource_name: ...}` where `resource_name` is active accelerator. |

## When to use

- You're porting a Ray-based trainer (veRL / EasyR1 / OpenRLHF / TRL-with-Ray / any custom Ray framework) to Ascend NPU.
- You want the integration to work on both CUDA and NPU from a single source.

## When not to use

- You're using a non-Ray trainer (`accelerate launch`, `torchrun`, deepspeed launcher). Those have their own accelerator-detection; this shim is Ray-specific.
- You're using Ray with a non-NPU accelerator (Habana, Neuron). The shim's NPU-specific env vars would be misleading; generalize the helper first.

## Integration recipe

This shim covers the **Ray-specific** integration points only. A framework port that uses CUDA-named APIs in its own code (`torch.cuda.*`, `"cuda"` device strings, nccl backend) still needs the `NPU-CP-001` sweep — see `repo/knowledge/npu-patterns.md` and run `repo/scripts/code-path-sweep.sh <source>` to find those sites. The shim + the sweep together are what a second Ray-based port needs.

Copy `ray_npu_shim.py` somewhere importable (e.g. `verl/utils/ray_npu_shim.py`), then:

1. Replace every `ray.init(...)` call:
   ```python
   # before
   ray.init(runtime_env=runtime_env)
   # after
   from .ray_npu_shim import ray_init_npu_aware
   ray_init_npu_aware(runtime_env=runtime_env)
   ```

2. Replace every `ray.available_resources().get("GPU", 0)` with:
   ```python
   from .ray_npu_shim import get_ray_resource_name
   ray.available_resources().get(get_ray_resource_name(), 0)
   ```

3. Replace every `options["num_gpus"] = n`:
   ```python
   from .ray_npu_shim import apply_actor_options
   apply_actor_options(options, num_accel=n)
   ```

4. Replace every `{"CPU": c, "GPU": 1}` placement-group bundle:
   ```python
   from .ray_npu_shim import placement_bundle
   bundles = [placement_bundle(num_cpus=c, num_accel=1) for _ in range(world_size)]
   ```

5. Run `repo/scripts/code-path-sweep.sh <your-source-tree>` to find the framework's own CUDA-named call sites — `torch.cuda.device_count()`, `torch.cuda.is_available()`, `backend="nccl"`, etc. Those are `NPU-CP-001` and need the `verl/utils/device.py`-style helper (separate from this Ray shim). Typical order: apply this shim first, then do the `NPU-CP-001` sweep, then test.

## Validation

After applying:
```python
import ray
from ray_npu_shim import ray_init_npu_aware, is_npu_available, get_ray_resource_name
ray_init_npu_aware()
print("is_npu_available:", is_npu_available())
print("resource name:", get_ray_resource_name())
print("available_resources:", ray.available_resources())
```

Expected on NPU: `is_npu_available: True`, `resource name: NPU`, `available_resources` contains `{"NPU": <device_count>}` and `accelerator_type:Ascend910_*`.

Spawn a probe actor and verify it inherits `ASCEND_RT_VISIBLE_DEVICES` and sees `torch.npu.is_available() == True`. (The EasyR1 port's `smoke_v11_device.py` is a model.)

## Versioning

This shim is pinned to **Ray 2.55+** (tested on 2.55.0 in the verl-8.5.0-a3 image). If Ray adds first-class NPU detection in a future release, fold the shim — check for `ray.available_resources()["NPU"]` being non-empty without the `resources=` kwarg.

## Related

- `repo/knowledge/npu-patterns.md` — `NPU-CP-003`, `NPU-BUG-002`, `NPU-ENV-002`.
- Reference implementation in the EasyR1 port: commit `fb1a223` (resource registration + placement bundles) + `59641d4` (RAY_ACCEL env) + `cc8e794` (VLLM_ASCEND_ENABLE_NZ). The shim merges those.
