# vllm-upgrade-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL rules | Phase A |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | This expert's OL-03 + OL-08 | Phase A |
| [patterns/domains/vllm-rename-catalog.md](patterns/domains/vllm-rename-catalog.md) | Per-version vllm API rename/move ledger + VLLMHijack targets | Phase A/B |

## Known shim sites

| Callsite | Shim class | File |
|---|---|---|
| `from vllm.lora.models import LoRAModel` | CP-002 try/except fallback | `verl/utils/vllm_utils.py:20` |
| `get_tensor_model_parallel_group()` | CP-004 hasattr gate | `verl/workers/sharding_manager/fsdp_vllm.py` |
| `setattr(SamplingParams, ...)` for read-only properties | EC-03 descriptor check | `verl/workers/rollout/vllm_rollout_spmd.py` `update_sampling_params` |
| `VLLMHijack` monkey-patch targets | Verify target methods still exist on new vllm | `verl/utils/vllm_utils.py` `VLLMHijack.hijack()` |

## Quick-lookup errors

- **`ImportError: cannot import name 'LoRAModel' from 'vllm.lora.models'`** → CP-002 arm (vllm ≥ 0.13.1 moved to `vllm.lora.lora_model`)
- **`AttributeError: module 'vllm.distributed.parallel_state' has no attribute 'get_tensor_model_parallel_group'`** → CP-004 arm (vllm ≥ 0.13.0.post switched to `get_tp_group`)
- **`AttributeError: property 'eos_token_id' of 'SamplingParams' has no setter`** → EC-03 (vllm ≥ 0.18 froze several SamplingParams fields)
- **`AttributeError: type object 'LRUCacheWorkerLoRAManager' has no attribute '_load_adapter'`** → VLLMHijack target moved; consult catalog §"VLLMHijack methods to verify"

## Related error corrections from other experts

- `transformers/port-expert/_legacy-upgrade/references/ERROR_CORRECTIONS.md` — EC-02/03/04/10
  are shared across upgrade experts (EC-03 explicitly documented there)
- `easyr1/port-expert/references/ERROR_CORRECTIONS.md` — EC-01..14 general EasyR1
  port corrections; EC-11/12/13 cover V1.4 smoke corner cases you may
  see in Phase D
