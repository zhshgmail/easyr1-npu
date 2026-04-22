# Domain — vllm / vllm-ascend rename & read-only-property catalog

**Load when**: Phase A analyzing source-vs-target vllm version delta;
Phase B applying shims; Phase D matching a traceback.

Per-version API-change ledger for vllm / vllm-ascend on NPU. Every row
has concrete evidence (commit / drill / session).

---

## Catalog

### vllm 0.13 (baseline on v1 image)

No shims needed — this is the baseline.

- `vllm.lora.models.LoRAModel` at its original home
- `vllm.distributed.parallel_state.get_tensor_model_parallel_group` exists
- `SamplingParams` — mutable `eos_token_id`, `stop_token_ids`, `output_kind`

### vllm 0.13 → ≥ 0.13.1 (NPU-CP-002: LoRAModel module rename)

**Change**: `vllm.lora.models.LoRAModel` moved to `vllm.lora.lora_model`.

**Shim** in `verl/utils/vllm_utils.py`:
```python
try:
    from vllm.lora.lora_model import LoRAModel  # vllm >= 0.13.1
except ImportError:
    from vllm.lora.models import LoRAModel  # vllm <= 0.13.0 (legacy)
```

**Evidence**: commit `87faff1` on `ascend-port` (2026-04-17). Backward-compat
holds; fallback branch works on older vllm.

### vllm 0.13 → ≥ 0.13.0.post (NPU-CP-004: TP group rename)

**Change**: `get_tensor_model_parallel_group()` renamed to `get_tp_group()`
in `vllm.distributed.parallel_state`.

**Shim** in `verl/workers/sharding_manager/fsdp_vllm.py`:
```python
from vllm.distributed import parallel_state as vllm_ps
if hasattr(vllm_ps, "get_tp_group"):
    self.tp_group = vllm_ps.get_tp_group().device_group
else:
    self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group
```

**Evidence**: commit `2d8ee2c` on `ascend-port` (2026-04-17). Backcompat
holds — older vllm has `get_tensor_model_parallel_group` but NOT
`get_tp_group`.

### vllm 0.17 → 0.18 (EC-03: SamplingParams read-only)

**Change**: `SamplingParams.eos_token_id`, `.stop_token_ids`, `.output_kind`
promoted from mutable attrs to read-only `@property`. Code doing
`sampling_params.eos_token_id = x` raises `AttributeError`.

**Shim** (introspection variant — preferred):
```python
# In verl/workers/rollout/vllm_rollout_spmd.py update_sampling_params (or equivalent)
for k, v in updates.items():
    cls_attr = getattr(type(sampling_params), k, None)
    if isinstance(cls_attr, property) and cls_attr.fset is None:
        continue  # read-only; default is correct
    setattr(sampling_params, k, v)
```

**Evidence**: drill commit `d213f01` (2026-04-19). E2E wet-run
(2026-04-22) re-confirmed. Backcompat: introspection returns
non-property on older vllm, normal path runs.

### vllm ≥ 0.20 (hypothetical, not yet hit)

**Watch for**:
- Continued `SamplingParams` field freezes — add to `_READ_ONLY` set or
  rely on the introspection variant above (it's future-proof).
- `LLM.generate` signature drift — `prompts=` vs `inputs=` vs
  `prompt_token_ids=` have moved around. Today `vllm_rollout_spmd.py:225`
  uses `prompts=`. Verify on target.
- `compressed_tensors` transitive changes when vllm moves.

**Action when observed**: add a row to this catalog BEFORE writing the
shim. Cite the evidence (changelog URL + reproducer). Then shim.

---

## VLLMHijack method verification per target

`verl/utils/vllm_utils.py` defines `VLLMHijack` that monkey-patches vllm
LoRA adapter loading. Any vllm bump MUST re-verify these targets resolve:

| Target | vllm module path | Verify (inside target image) |
|---|---|---|
| `LRUCacheWorkerLoRAManager._load_adapter` | `vllm.lora.worker_manager` | `python -c 'from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager as C; assert hasattr(C, "_load_adapter")'` |
| `get_adapter_absolute_path` | `vllm.lora.utils` | `python -c 'from vllm.lora.utils import get_adapter_absolute_path'` |
| `PEFTHelper` | `vllm.lora.peft_helper` | `python -c 'from vllm.lora.peft_helper import PEFTHelper'` |

If any resolve differently → add a new row to this catalog, update the
hijack signature, cite the vllm commit URL.

---

## Verification protocol for a new vllm version

1. Phase A (inside target image via docker run):
   ```
   docker run --rm $TARGET_IMAGE python -c 'import vllm; print(vllm.__version__)'
   docker run --rm $TARGET_IMAGE python -c '
   from vllm.lora.lora_model import LoRAModel  # CP-002 new
   ' || echo "CP-002: need legacy fallback"
   docker run --rm $TARGET_IMAGE python -c '
   from vllm.distributed import parallel_state as p
   print("get_tp_group" if hasattr(p, "get_tp_group") else "legacy")
   '
   ```
2. Grep consumer source for all imports the catalog lists
3. Apply shims per the catalog; don't over-shim (only sites that
   actually appear in consumer source)
4. Phase B static_check py_compile must PASS
5. (Optional) container dry-import via
   `static_check.py --container-import-image $TARGET_IMAGE
   --container-import-live-source /home/.../upstream/<consumer>
   --import-package verl` to catch module-init regressions
6. Phase D — V1.3 rollout marker + V1.4 step-1 entropy_loss in band
