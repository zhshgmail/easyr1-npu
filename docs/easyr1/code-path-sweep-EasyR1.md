# Code-path sweep for EasyR1

Generated: 2026-04-21
Source tree: `/home/z00637938/workspace/easyr1-npu/upstream/EasyR1`
Python files scanned: 79

Total hits: 59

Pattern IDs link to the canonical catalog at `repo/knowledge/npu-patterns.md`.

---

## NPU-CP-001 — torch.cuda.* namespace call

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/single_controller/base/worker.py` | 132 | `if "AMD" in torch.cuda.get_device_name():` | get_device_module().<op>() |
| `verl/single_controller/base/worker.py` | 136 | `torch.cuda.set_device(int(cuda_visible_devices))` | get_device_module().<op>() |
| `verl/protocol.py` | 337 | `NOTE: remember to use torch.cuda.synchronize() after self.to("cpu") to avoid weird number` | get_device_module().<op>() |
| `verl/protocol.py` | 741 | `data.batch = data.batch.cuda(device=torch.cuda.current_device())` | get_device_module().<op>() |
| `verl/utils/fsdp_utils.py` | 116 | `torch.cuda.empty_cache()` | get_device_module().<op>() |
| `verl/utils/fsdp_utils.py` | 150 | `torch.cuda.empty_cache()` | get_device_module().<op>() |
| `verl/utils/fsdp_utils.py` | 185 | `torch.cuda.empty_cache()` | get_device_module().<op>() |
| `verl/utils/checkpoint/checkpoint_manager.py` | 97 | `"cuda": torch.cuda.get_rng_state(),` | get_device_module().<op>() |
| `verl/utils/checkpoint/checkpoint_manager.py` | 106 | `torch.cuda.set_rng_state(rng_state["cuda"])` | get_device_module().<op>() |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | 148 | `torch.cuda.empty_cache()` | get_device_module().<op>() |
| `verl/utils/model_utils.py` | 34 | `free_mem, total_mem = torch.cuda.mem_get_info()` | get_device_module().<op>() |
| `verl/utils/flops_counter.py` | 37 | `device_name = torch.cuda.get_device_name()` | get_device_module().<op>() |
| `verl/workers/fsdp_workers.py` | 317 | `device_id=torch.cuda.current_device(),` | get_device_module().<op>() |
| `verl/workers/fsdp_workers.py` | 555 | `data = data.to(torch.cuda.current_device())` | get_device_module().<op>() |
| `verl/workers/fsdp_workers.py` | 575 | `torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes` | get_device_module().<op>() |
| `verl/workers/fsdp_workers.py` | 578 | `torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes` | get_device_module().<op>() |
| `verl/workers/fsdp_workers.py` | 637 | `data = data.to(torch.cuda.current_device())` | get_device_module().<op>() |
| `verl/workers/fsdp_workers.py` | 673 | `data = data.to(torch.cuda.current_device())` | get_device_module().<op>() |
| `verl/workers/fsdp_workers.py` | 702 | `data = data.to(torch.cuda.current_device())` | get_device_module().<op>() |
| `verl/workers/fsdp_workers.py` | 724 | `data = data.to(torch.cuda.current_device())` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 69 | `self.torch_random_states = torch.cuda.get_rng_state()` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 72 | `torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 73 | `self.gen_random_states = torch.cuda.get_rng_state()` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 74 | `torch.cuda.set_rng_state(self.torch_random_states)` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 128 | `torch.cuda.empty_cache()` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 164 | `torch.cuda.empty_cache()` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 169 | `# NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 176 | `torch.cuda.empty_cache()` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 194 | `self.torch_random_states = torch.cuda.get_rng_state()` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 195 | `torch.cuda.set_rng_state(self.gen_random_states)` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 203 | `free_bytes_before_sleep = torch.cuda.mem_get_info()[0]` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 205 | `free_bytes_after_sleep = torch.cuda.mem_get_info()[0]` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 210 | `torch.cuda.empty_cache()  # add empty cache after each compute` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 214 | `self.gen_random_states = torch.cuda.get_rng_state()` | get_device_module().<op>() |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 215 | `torch.cuda.set_rng_state(self.torch_random_states)` | get_device_module().<op>() |

Sub-total: 35

## NPU-CP-001 — tensor .cuda() method

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/protocol.py` | 741 | `data.batch = data.batch.cuda(device=torch.cuda.current_device())` | .to(torch.device(get_device_name(), idx)) |

Sub-total: 1

## NPU-CP-001 — string "cuda" device spec

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/utils/seqlen_balancing.py` | 255 | `num_micro_batches = torch.tensor([num_micro_batches], device="cuda")` | get_device_name() returns "cuda" or "npu" |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | 141 | `cuda_device = torch.device("cuda")` | get_device_name() returns "cuda" or "npu" |
| `verl/workers/fsdp_workers.py` | 123 | `self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))` | get_device_name() returns "cuda" or "npu" |
| `verl/workers/fsdp_workers.py` | 305 | `param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None` | get_device_name() returns "cuda" or "npu" |
| `verl/workers/fsdp_workers.py` | 387 | `rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp...` | get_device_name() returns "cuda" or "npu" |

Sub-total: 5

## NPU-CP-001 — CUDA_VISIBLE_DEVICES literal

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/single_controller/base/worker.py` | 74 | `"CUDA_VISIBLE_DEVICES",` | get_visible_devices_env() — "CUDA_VISIBLE_DEVICES" or "ASCEND_RT_VISIBLE_DEVICES" |
| `verl/single_controller/base/worker.py` | 133 | `os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("ROCR_VISIBLE_DEVICES")` | get_visible_devices_env() — "CUDA_VISIBLE_DEVICES" or "ASCEND_RT_VISIBLE_DEVICES" |
| `verl/single_controller/base/worker.py` | 179 | `cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "not set")` | get_visible_devices_env() — "CUDA_VISIBLE_DEVICES" or "ASCEND_RT_VISIBLE_DEVICES" |

Sub-total: 3

## NPU-CP-001 — nccl distributed backend literal

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/workers/fsdp_workers.py` | 83 | `dist.init_process_group(backend="nccl")` | get_dist_backend() — "nccl" or "hccl" |

Sub-total: 1

## NPU-CP-002 — vllm.lora.models import (pre-0.13 only)

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/utils/vllm_utils.py` | 20 | `from vllm.lora.models import LoRAModel` | try: from vllm.lora.lora_model import LoRAModel; except ImportError: from vllm.lora.models import LoRAModel |

Sub-total: 1

## NPU-CP-003 — Ray available_resources GPU lookup or bundle

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/single_controller/ray/base.py` | 101 | `{"CPU": self.max_colocate_count, "GPU": 1} if self.use_gpu else {"CPU": self.max_colocate_count}` | use get_ray_resource_name() for lookups; placement bundles via placement_bundle helper |
| `verl/trainer/ray_trainer.py` | 111 | `gpus_available = ray.available_resources().get("GPU", 0)` | use get_ray_resource_name() for lookups; placement bundles via placement_bundle helper |

Sub-total: 2

## NPU-CP-003 — Ray actor num_gpus option

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/single_controller/ray/base.py` | 262 | `num_gpus = 1 / resource_pool.max_colocate_count` | on NPU use options["resources"]={"NPU": n} via apply_actor_options (ray-npu-shim) |
| `verl/single_controller/ray/base.py` | 296 | `placement_group=pg, placement_group_bundle_idx=local_rank, use_gpu=use_gpu, num_gpus=num_gpus` | on NPU use options["resources"]={"NPU": n} via apply_actor_options (ray-npu-shim) |
| `verl/trainer/ray_trainer.py` | 682 | `num_gpus = self.resource_pool_manager.get_num_gpus()` | on NPU use options["resources"]={"NPU": n} via apply_actor_options (ray-npu-shim) |
| `verl/trainer/ray_trainer.py` | 685 | `metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))` | on NPU use options["resources"]={"NPU": n} via apply_actor_options (ray-npu-shim) |

Sub-total: 4

## NPU-CP-004 — vllm get_tensor_model_parallel_group (renamed in 0.13)

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/workers/sharding_manager/fsdp_vllm.py` | 62 | `self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group` | vllm 0.13 renamed to get_tp_group(); hasattr gate |

Sub-total: 1

## NPU-CP-005 — flash_attn import (CUDA-only kernel)

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/models/transformers/flash_attention_utils.py` | 35 | `from flash_attn import flash_attn_func, flash_attn_varlen_func` | CUDA-only; import-guard behind try/except or is_npu_available(); see attention_utils.py |
| `verl/utils/torch_functional.py` | 31 | `from flash_attn.ops.triton.cross_entropy import cross_entropy_loss` | CUDA-only; import-guard behind try/except or is_npu_available(); see attention_utils.py |
| `verl/workers/critic/dp_critic.py` | 38 | `from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input` | CUDA-only; import-guard behind try/except or is_npu_available(); see attention_utils.py |
| `verl/workers/actor/dp_actor.py` | 40 | `from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input` | CUDA-only; import-guard behind try/except or is_npu_available(); see attention_utils.py |

Sub-total: 4

## NPU-CP-005 — liger_kernel import (CUDA-only Triton)

No hits.

## NPU-CP-006 — torch.backends.cuda.* knob (CUDA-only)

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| `verl/workers/fsdp_workers.py` | 86 | `torch.backends.cuda.matmul.allow_tf32 = False` | wrap behind `if not is_npu_available():` |
| `verl/workers/fsdp_workers.py` | 87 | `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` | wrap behind `if not is_npu_available():` |

Sub-total: 2

---

## Summary

- Python files scanned: 79
- Total hits: 59

See `repo/knowledge/npu-patterns.md` for the fix pattern behind each ID.
