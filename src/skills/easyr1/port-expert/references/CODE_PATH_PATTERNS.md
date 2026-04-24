# Code Path Patterns — NPU-CP-001..007

> Master catalog. Each pattern has a `patterns/domains/<name>.md` for deep detail.

---

## NPU-CP-001: torch.cuda.* / "cuda" / nccl → device-agnostic dispatch

**Callsite signatures**:
- `torch.cuda.device_count()`, `torch.cuda.current_device()`, `torch.cuda.max_memory_allocated()` etc.
- `init_device_mesh("cuda", ...)`
- `device_map="cuda"`
- `torch.device("cuda", ...)`
- `dist.init_process_group(backend="nccl")`
- `CUDA_VISIBLE_DEVICES` (string literal)

**Count in EasyR1 master**: 35 hits across ~10 files.

**Fix pattern**: create `verl/utils/device.py` helper, replace callsites with helper calls.

See `patterns/domains/device_dispatch.md` for full template + file list.

---

## NPU-CP-002: vllm.lora.models → vllm.lora.lora_model (vllm 0.13+ rename)

**Callsite signature**: `from vllm.lora.models import <anything>`

**Fix**: try/except import:
```python
try:
    from vllm.lora.lora_model import ...  # vllm >= 0.13
except ImportError:
    from vllm.lora.models import ...       # vllm < 0.13
```

See `patterns/domains/vllm_compat.md`.

---

## NPU-CP-003: Ray num_gpus / "GPU" resource → NPU custom resource

**Callsite signatures**:
- `num_gpus=N`
- `ray.available_resources().get("GPU", 0)`
- `{"GPU": 1}` placement-bundle
- `ray.init(...)` without NPU env injection

**Fix**: Ray doesn't auto-detect NPU. Use `resources={"NPU": N}` scheme + inject `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` env.

See `patterns/domains/ray_integration.md` (includes drop-in `ray_npu_shim.py`).

---

## NPU-CP-004: vllm get_tensor_model_parallel_group → get_tp_group (vllm 0.13+ rename)

**Callsite signature**: `from vllm.distributed.parallel_state import get_tensor_model_parallel_group`

**Fix**: hasattr gate:
```python
from vllm.distributed import parallel_state as ps
if hasattr(ps, "get_tp_group"):
    tp_group = ps.get_tp_group()
else:
    tp_group = ps.get_tensor_model_parallel_group()
```

See `patterns/domains/vllm_compat.md`.

---

## NPU-CP-005: flash_attn imports → transformers NPU FA integration

**Callsite signatures**:
- `from flash_attn import flash_attn_func, flash_attn_varlen_func`
- `from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis`
- `attn_implementation="flash_attention_2"` in `from_pretrained`

**Fix pattern**:
- Attention kernel: switch to `transformers.integrations.npu_flash_attention` on NPU (transformers 4.57+ ships it)
- `bert_padding` helpers: NPU doesn't have them via flash_attn (flash_attn is CUDA-only). Vendor pure-torch versions.
- `attn_implementation`: default to `sdpa` on NPU, `flash_attention_2` on CUDA (dispatch via `get_default_attn_implementation()`)

See `patterns/domains/attention_backend.md`.

---

## NPU-CP-006: torch.backends.cuda knobs → guarded

**Callsite signatures**:
- `torch.backends.cuda.matmul.allow_tf32 = ...`
- `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = ...`

**Fix**: guard with `if not is_npu_available():` — these are CUDA-only settings; torch_npu path doesn't have equivalents on these knobs.

See `patterns/domains/device_dispatch.md`.

---

## NPU-CP-007: padding_free + transformers NPU FA varlen

**Callsite signature**: EasyR1's `padding_free=True` path invokes flash_attn varlen

**Fix**: route varlen through `transformers.integrations.npu_flash_attention.npu_flash_attn_varlen_func` on NPU. Drop NPU raises from `apply_ulysses_patch` and `_build_model_optimizer`.

See `patterns/domains/attention_backend.md`.

---

## Coverage summary

| NPU-CP-ID | Files touched | Priority | See |
|---|---|---|---|
| 001 | 10+ (fsdp_workers.py, protocol.py, fsdp_utils.py, checkpoint_manager.py, seqlen_balancing.py:255, etc.) | Critical | patterns/domains/device_dispatch.md |
| 002 | vllm_utils.py | High | patterns/domains/vllm_compat.md |
| 003 | ray/base.py, ray_trainer.py, trainer/main.py | Critical | patterns/domains/ray_integration.md |
| 004 | sharding_manager/fsdp_vllm.py | High | patterns/domains/vllm_compat.md |
| 005 | flash_attention_utils.py (+ vendored npu_flash_attn_utils.py) | Critical | patterns/domains/attention_backend.md |
| 006 | fsdp_workers.py (2 lines) | Low | patterns/domains/device_dispatch.md |
| 007 | dp_actor.py, dp_critic.py, monkey_patch.py | Critical (for padding_free) | patterns/domains/attention_backend.md |

**Stage 0 minimum set to reach V1.4 PASS**: CP-001, CP-002, CP-003, CP-004, CP-005, CP-006. CP-007 needed for V2.1.
