# Domain — Device dispatch (NPU-CP-001, NPU-CP-006)

**Load when**: Phase B, fixing any `torch.cuda.*`, `init_device_mesh("cuda", ...)`,
`device_map="cuda"`, `torch.device("cuda", ...)`, `dist.init_process_group(backend="nccl")`,
`CUDA_VISIBLE_DEVICES` literal, or `torch.backends.cuda.matmul.*` callsite.

---

## Template — `verl/utils/device.py`

Create this file (it does not exist on `hiyouga/EasyR1 main`). Place it at
`verl/utils/device.py`. Every other file then imports from it. The helpers are
`@lru_cache`-gated so the detection runs once per process.

```python
"""Device-accessor helpers so the same code path works on CUDA and Ascend NPU.

Detection is done once at import time. The result is cached so every call
site sees a consistent answer without re-probing torch_npu.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        import torch
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


@lru_cache(maxsize=1)
def get_device_name() -> str:
    """Device-type string for torch.device, init_device_mesh, device_map."""
    return "npu" if is_npu_available() else "cuda"


def get_device_module():
    """torch.npu or torch.cuda — owns RNG, memory, cache ops."""
    import torch
    return torch.npu if is_npu_available() else torch.cuda


@lru_cache(maxsize=1)
def get_visible_devices_env() -> str:
    return "ASCEND_RT_VISIBLE_DEVICES" if is_npu_available() else "CUDA_VISIBLE_DEVICES"


@lru_cache(maxsize=1)
def get_dist_backend() -> str:
    return "hccl" if is_npu_available() else "nccl"


@lru_cache(maxsize=1)
def get_default_attn_implementation() -> str:
    """HF `attn_implementation` name for the active accelerator.

    On NPU we still pass "flash_attention_2": transformers 4.57+ registers
    the NPU-aware FA adapter (from transformers.integrations.npu_flash_attention)
    under ALL_ATTENTION_FUNCTIONS["flash_attention_2"] when
    apply_ulysses_patch() has run. Passing "sdpa" does NOT route to the NPU
    fused kernel on torch 2.8.0+cpu — it falls back to torch's math backend
    and drifts step-1 entropy_loss off-baseline (observed 2026-04-22 round 3/4).

    So: "flash_attention_2" on both CUDA and NPU.
    """
    return "flash_attention_2"


@lru_cache(maxsize=1)
def get_ray_resource_name() -> str:
    """Ray's `num_gpus` sugar only covers CUDA. For NPU use
    `resources={"NPU": n}`; `available_resources()` returns NPUs under "NPU".
    """
    return "NPU" if is_npu_available() else "GPU"
```

## Replacement rules — grep/sed table

| CUDA-only callsite | Replacement |
|---|---|
| `torch.cuda.device_count()` | `get_device_module().device_count()` |
| `torch.cuda.current_device()` | `get_device_module().current_device()` |
| `torch.cuda.max_memory_allocated()` | `get_device_module().max_memory_allocated()` |
| `torch.cuda.empty_cache()` | `get_device_module().empty_cache()` |
| `init_device_mesh("cuda", ...)` | `init_device_mesh(get_device_name(), ...)` |
| `device_map="cuda"` | `device_map=get_device_name()` |
| `torch.device("cuda", idx)` | `torch.device(get_device_name(), idx)` |
| `dist.init_process_group(backend="nccl", ...)` | `dist.init_process_group(backend=get_dist_backend(), ...)` |
| `os.environ["CUDA_VISIBLE_DEVICES"] = ...` | `os.environ[get_visible_devices_env()] = ...` |
| `<any_call>(..., device="cuda", ...)` | `<any_call>(..., device=get_device_name(), ...)` |
| `torch.empty_like(..., device="cuda")` / `torch.zeros(..., device="cuda")` etc. | swap `"cuda"` → `get_device_name()` |

**Kwarg-form `device="cuda"`** is easy to miss with a naive `grep torch.cuda.`. Explicitly run `grep -n 'device="cuda"' verl/` in Phase A; every hit is a replacement site unless inside a comment or string literal. 2026-04-22 round 4 + wet-run both re-discovered this in `fsdp_workers.py` `get_init_fn(model, device="cuda")` and in `seqlen_balancing.py` `torch.tensor(..., device="cuda")`.

## NPU-CP-006 — guard `torch.backends.cuda` knobs

These exist only on the CUDA path. Pattern:

```python
from .device import is_npu_available

if not is_npu_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
```

## Files typically touched (EasyR1 master)

- `verl/protocol.py` — 1 hit (`torch.cuda.current_device`)
- `verl/single_controller/base/worker.py` — 1 hit
- `verl/trainer/main.py` — init_device_mesh + CUDA_VISIBLE_DEVICES env
- `verl/trainer/ray_trainer.py` — device_count / empty_cache
- `verl/utils/checkpoint/checkpoint_manager.py` — RNG state via device_module
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py` — same
- `verl/utils/flops_counter.py` — device counter dispatch
- `verl/utils/seqlen_balancing.py` — **hard-coded `device="cuda"` at :255 in `num_micro_batches`** (found 2026-04-22 round 4 iter2; not in initial sweep because it's inside a helper that only fires on dynamic batching; must be routed through `get_device_name()`)
- `verl/utils/fsdp_utils.py` — init_device_mesh + nccl → hccl
- `verl/utils/model_utils.py` — device_map
- `verl/utils/seqlen_balancing.py` — device count
- `verl/workers/fsdp_workers.py` — multiple, including NPU-CP-006 knobs; **also a kwarg-form `device="cuda"` inside `FSDPWorker._build_fsdp_model` at `get_init_fn(model, device="cuda")` (~line 322)** (round 4 + wet-run both re-discovered this; fires whenever `worker.actor.fsdp.enable_rank0_init=true` which canonical V1.4 config DOES set. A naive `grep torch.cuda.` misses it because the literal is a keyword arg. Grep `device="cuda"` AND `get_init_fn` during Phase B.)

Total ≈ 35 callsite edits across ~10 files.

## Verify

After edits:

```bash
# G2 check
python3 src/skills/easyr1/port-expert/scripts/static_check.py \
  --files $(git diff --name-only main... | grep '\.py$') \
  --import-package verl

# Spot-check: no bare torch.cuda.<op>() left outside guards
grep -nrE 'torch\.cuda\.(device_count|current_device|empty_cache|max_memory)' verl/ \
  | grep -v 'is_npu_available\|get_device_module'
# → expected: no output (or only comments)
```

## Evidence

This template compiled + ran end-to-end on the `ascend-port` branch
(commit `7187b51` "route torch.cuda.* through device-module accessor"
and successors). V1.4 step-1 `entropy_loss=0.991` measured with this
template in place.
