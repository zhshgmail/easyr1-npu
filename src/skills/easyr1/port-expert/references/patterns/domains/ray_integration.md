# Domain — Ray integration (NPU-CP-003, NPU-BUG-002, NPU-ENV-002)

**Load when**: Phase B, fixing `num_gpus=`, `ray.available_resources()["GPU"]`,
`{"GPU": N}` placement bundles, `ray.init(...)` without NPU env injection, or
`ASCEND_RT_VISIBLE_DEVICES` clearing under Ray.

---

## The problem in one sentence

Ray has built-in sugar for CUDA (`num_gpus=N`, `"GPU"` in
`ray.available_resources()`), but NPUs have to be registered as a generic
custom resource (`{"NPU": N}`) and Ray then does NOT auto-set
`ASCEND_RT_VISIBLE_DEVICES` for worker actors — we must do it ourselves
(and prevent Ray from *clearing* it either).

## Fix — three pieces

### Piece 1: generic "accelerator resource name" lookup

Use `get_ray_resource_name()` from `verl/utils/device.py`
(see device_dispatch.md) — returns `"NPU"` or `"GPU"`.

### Piece 2: NPU registration shim at ray.init time (NPU-CP-003)

In `verl/single_controller/ray/base.py` (and anywhere `ray.init()` is called):

```python
from ...utils.device import is_npu_available, get_device_module


def _npu_ray_init_kwargs():
    """Inject {"resources": {"NPU": N}} when running on NPU so placement groups
    can allocate NPU chips. CUDA path returns {} — Ray's built-in GPU handling
    already works."""
    if not is_npu_available():
        return {}
    n_chips = get_device_module().device_count()
    return {"resources": {"NPU": n_chips}}
```

Merge into any `ray.init(...)` kwargs. Example:

```python
ray.init(runtime_env=..., **_npu_ray_init_kwargs())
```

### Piece 3: worker-side placement bundle uses `{"NPU": 1}` (not `{"GPU": 1}`)

In `verl/single_controller/ray/base.py` where placement group bundles are
built for worker actors:

```python
from ...utils.device import get_ray_resource_name

# old: bundles = [{"CPU": cpu, "GPU": gpu}, ...]
resource_key = get_ray_resource_name()
bundles = [{"CPU": cpu, resource_key: gpu}, ...]
```

And on the actor-options side:

```python
if use_gpu:
    # Ray's num_gpus sugar only understands CUDA. For NPU we request the
    # custom "NPU" resource. CUDA path unchanged.
    resource_key = get_ray_resource_name()
    if resource_key == "GPU":
        options["num_gpus"] = num_gpus
    else:
        options["resources"] = {resource_key: num_gpus}
```

### NPU-BUG-002 workaround — stop Ray from clearing `ASCEND_RT_VISIBLE_DEVICES`

Ray core clears any env var matching its "accelerator env var" list when
an actor boots. On NPU, Ray 2.x keys off `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`
to *skip* that reset. Set it at init time and also inside the actor:

```python
# in verl/single_controller/base/worker.py or ray_trainer.py actor init
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
```

If that env isn't honored by your Ray version, the second-line defense is:
capture `ASCEND_RT_VISIBLE_DEVICES` before Ray touches it, and re-set after:

```python
# at top of actor's __init__
_saved = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
if _saved is not None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = _saved
```

### NPU-ENV-002 — disable `VLLM_ASCEND_ENABLE_NZ` for RL rollout

The `NZ` tensor layout is faster for inference-only vllm but breaks RL
rollout weight-sync. Always set:

```bash
# in examples/qwen2_0_5b_math_grpo_npu_smoke.sh, before ray start:
export VLLM_ASCEND_ENABLE_NZ=0
```

Or in the actor's env (passed via `runtime_env`).

## Files typically touched

- `verl/single_controller/ray/base.py` — placement group + actor options (3-5 edits)
- `verl/single_controller/base/worker.py` — env setdefault for RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO
- `verl/trainer/main.py` — ray.init kwargs merge
- `verl/trainer/ray_trainer.py` — device-count lookup via helper
- `examples/qwen2_0_5b_math_grpo_npu_smoke.sh` — `VLLM_ASCEND_ENABLE_NZ=0` export

## Verify

```bash
# inside the running container (Phase D smoke):
python3 -c 'import ray, os; ray.init(); print(ray.available_resources())'
# → should show {"NPU": <n>, "CPU": ..., "node:...": ...}
# Specifically: NOT "GPU", and NPU count > 0.

# Also check ASCEND_RT_VISIBLE_DEVICES is preserved inside an actor:
python3 -c '
import ray, os
ray.init()
@ray.remote(resources={"NPU": 1})
class A:
    def vars(self):
        return os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
print(ray.get(A.remote().vars.remote()))
# → expect a single chip id like "0", not None or an empty string.'
```

## Evidence

Pattern verified in `ascend-port` branch commits:
- `fb1a223` "Ray: register NPU as custom resource so placement groups can claim chips"
- `59641d4` "Ray actor: stop Ray from clearing ASCEND_RT_VISIBLE_DEVICES"
- `cc8e794` "NPU-ENV-002: set VLLM_ASCEND_ENABLE_NZ=0 for RL scenarios"

V1.4 smoke reaches step 2 with this template; without it, Phase D fails at
"Total available GPUs 0 is less than total desired GPUs N" (EC-05).
