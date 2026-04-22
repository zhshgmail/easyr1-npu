# Domain — vllm compatibility (NPU-CP-002, NPU-CP-004, EC-03)

**Load when**: Phase B, fixing `from vllm.lora.models import ...`,
`from vllm.distributed.parallel_state import get_tensor_model_parallel_group`,
or any `SamplingParams.eos_token_id = ...` assignment.

---

## The problem

The `verl-8.5.0-a3` NPU image ships `vllm_ascend 0.13.1` which exposes the
post-0.13 vllm API. EasyR1 master was written against pre-0.13 API names.
Three breaking renames matter:

| Concern | pre-0.13 / ≤ 0.12 | ≥ 0.13 |
|---|---|---|
| LoRAModel module | `vllm.lora.models` | `vllm.lora.lora_model` |
| TP group getter | `get_tensor_model_parallel_group()` | `get_tp_group()` |
| SamplingParams.eos_token_id | mutable attr | read-only property (as of 0.18) |

Also: on NPU you can be running a build that is _between_ these renames
(e.g. drill image's 0.18.0.dev). So don't hard-switch — try new, fall back.

## Fix — three try/except patterns

### NPU-CP-002 — LoRAModel import

```python
# verl/utils/vllm_utils.py (and any other file doing the import)
try:
    from vllm.lora.lora_model import LoRAModel  # vllm >= 0.13
except ImportError:
    from vllm.lora.models import LoRAModel  # vllm <= 0.12
```

### NPU-CP-004 — TP group getter

```python
# verl/workers/sharding_manager/fsdp_vllm.py
from vllm.distributed import parallel_state as vllm_ps

# vllm 0.13 replaced get_tensor_model_parallel_group() with get_tp_group().
# The group object still exposes .device_group the same way.
if hasattr(vllm_ps, "get_tp_group"):
    self.tp_group = vllm_ps.get_tp_group().device_group
else:
    self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group
```

### EC-03 — SamplingParams.eos_token_id read-only

In `verl/workers/rollout/vllm_rollout_spmd.py` `update_sampling_params()`:

```python
# old (fails on vllm 0.18):
sampling_params.eos_token_id = tokenizer.eos_token_id

# new — skip read-only properties:
for k, v in updates.items():
    try:
        setattr(sampling_params, k, v)
    except AttributeError:
        # vllm 0.18+ made some of these read-only properties. Skip silently;
        # the underlying default from SamplingParams.__init__ is correct.
        pass
```

Or more surgically, exclude the known-read-only set:

```python
_READ_ONLY_IN_VLLM_018 = {"eos_token_id", "stop_token_ids", "output_kind"}
for k, v in updates.items():
    if k in _READ_ONLY_IN_VLLM_018:
        continue
    setattr(sampling_params, k, v)
```

## Related lazy-import hygiene

Don't do module-level `from vllm.lora.utils import ...` if your file is
imported before vllm_ascend's monkey-patching runs. Move those imports
inside the method that uses them (see `VLLMHijack.hijack()` in port-branch
`verl/utils/vllm_utils.py`).

## Files typically touched

- `verl/utils/vllm_utils.py` — LoRAModel import + hijack lazy-imports
- `verl/workers/sharding_manager/fsdp_vllm.py` — TP group getter
- `verl/workers/rollout/vllm_rollout_spmd.py` — SamplingParams update

## Verify

```bash
# inside container:
python3 -c '
import verl.utils.vllm_utils as v
print(type(v.LoRAModel).__module__)
# expect: vllm.lora.lora_model  (on 0.13+)
'

# TP group (requires a running torch.distributed group — defer to Phase D smoke):
# V1.3 (rollout smoke) exercises TP group init. PASS marker
# "V1.3 ROLLOUT SMOKE PASSED" confirms both imports work.
```

## Evidence

Port-branch commits:
- `87faff1` "vllm_utils: LoRAModel import compatible across vllm <=0.12 and >=0.13"
- `2d8ee2c` "fsdp_vllm: handle vllm 0.13 rename of get_tensor_model_parallel_group"
- `ecce71d` "[drill] vllm 0.18: skip read-only properties in update_sampling_params"

V1.3 PASS on ascend-port; V1.4 depends on V1.3 wiring so it also validates
this domain.
