## Summary

After a successful initial `Engine.__init__` load of a DeepseekV32 model on NPU, the **second** load of the same checkpoint via `update_weights_from_disk` crashes the scheduler subprocess with:

```
RuntimeError: start (0) + length (4096) exceeds dimension size (1408).
```

The error path is in `srt/layers/moe/fused_moe_triton/layer.py:482` `_load_w13`, called for the `mlp.experts.N.gate_proj.weight` tensors. First-pass load works because it takes a different code path; the second-pass reload hits a `narrow(0, intermediate_size_per_partition * 2)` against the on-disk tensor whose first dim is `intermediate_size_per_partition`.

This blocks RL post-training flows that use `/update_weights_from_disk` to push trained actor weights back to the rollout engine each step (e.g. miles, the framework this was found in).

## Reproduction

Image: `lmsysorg/sglang:main-cann8.5.0-a3` (sglang `0.5.12.post2.dev434+gb13d3d18c`, sgl-kernel-npu `2026.5.1`, transformers `5.8.1`).

Random-init 1-layer DeepseekV32 HF checkpoint built with:

```
hidden_size=4096
num_attention_heads=64
num_hidden_layers=1
first_k_dense_replace=0    # MoE on layer 0
n_routed_experts=4
n_shared_experts=1
moe_intermediate_size=1408
num_experts_per_tok=2
q_lora_rank=1024
kv_lora_rank=512
qk_nope_head_dim=64
qk_rope_head_dim=64
v_head_dim=512
index_topk=8
```

Safetensors saved with `mlp.experts.{0..3}.{gate_proj,up_proj,down_proj}.weight` separate (not pre-fused as `gate_up_proj`).

Driver:

```python
import sglang as sgl
llm = sgl.Engine(
    model_path="/path/to/fab_ckpt",
    dtype="bfloat16", device="npu",
    mem_fraction_static=0.10, tp_size=1, disable_cuda_graph=True,
    enable_return_routed_experts=True,
)
# Initial load: succeeds
llm.generate(["Hi"], sampling_params={"temperature":0.0,"max_new_tokens":4})

# Trigger: reload same ckpt
llm.update_weights_from_disk(model_path="/path/to/fab_ckpt")  # CRASH
```

Full traceback in the scheduler subprocess:

```
File ".../sglang/srt/managers/scheduler.py", line 1638, in process_input_requests
    output = self._request_dispatcher(recv_req)
File ".../sglang/srt/managers/scheduler_components/weight_updater.py", line 62, in update_weights_from_disk
    success, message = self.tp_worker.update_weights_from_disk(recv_req)
File ".../sglang/srt/managers/tp_worker.py", line 97, in update_weights_from_disk
    success, message = self.model_runner.update_weights_from_disk(recv_req)
File ".../sglang/srt/model_executor/model_runner.py", line 1634, in update_weights_from_disk
    self.model = model_load_weights(self.model, iter)
File ".../sglang/srt/model_executor/model_runner.py", line 1616, in model_load_weights
    loader.load_weights_and_postprocess(model, iter, target_device)
File ".../sglang/srt/model_loader/loader.py", line 735, in load_weights_and_postprocess
    model.load_weights(weights)
File ".../sglang/srt/models/deepseek_v2.py", line 2645, in load_weights
    self.do_load_weights(weights, is_nextn)
File ".../sglang/srt/models/deepseek_common/deepseek_weight_loader.py", line 373, in do_load_weights
    future.result()
File ".../sglang/srt/layers/moe/fused_moe_triton/layer.py", line 662, in weight_loader
    self._weight_loader_physical(...)
File ".../sglang/srt/layers/moe/fused_moe_triton/layer.py", line 692, in _weight_loader_physical
    self._weight_loader_impl(...)
File ".../sglang/srt/layers/moe/fused_moe_triton/layer.py", line 950, in _weight_loader_impl
    self._load_model_weight_or_group_weight_scale(...)
File ".../sglang/srt/layers/moe/fused_moe_triton/layer.py", line 398, in _load_model_weight_or_group_weight_scale
    self._load_w13(...)
File ".../sglang/srt/layers/moe/fused_moe_triton/layer.py", line 482, in _load_w13
    loaded_weight = loaded_weight.narrow(0, ?, ?)
RuntimeError: start (0) + length (4096) exceeds dimension size (1408).
```

After the crash the scheduler dies (SIGQUIT), so even with try/except around the user-facing call, the engine is gone.

## What I think is happening

In `fused_moe_triton/layer.py:_load_w13`, the reload path computes the source slice with the expectation that the on-disk weight is pre-fused (gate+up packed along dim 0 = `2 * intermediate_size`). When the on-disk weight is separate (`gate_proj.weight` shape `[intermediate_size, hidden_size]`), the narrow tries to read `2 * intermediate_size_per_partition` from a tensor whose dim 0 is `intermediate_size`, hence `length (4096) > dim_size (1408)` -- here `4096 = 2 * 2048` (sglang's internal `intermediate_size_per_partition` likely rounded up or counted both halves) and `1408 = moe_intermediate_size`.

In the initial `Engine.__init__` load, `stacked_params_mapping` in `deepseek_common/deepseek_weight_loader.py` (lines 122-125):
```
("gate_up_proj", "gate_proj", 0),
("gate_up_proj", "up_proj", 1),
```
correctly handles separate names, so the first load works. The reload path doesn't appear to honor the same mapping.

## Workarounds

- Pre-fuse `gate_proj` + `up_proj` into a single `gate_up_proj` tensor of shape `[2 * intermediate_size, hidden_size]` when saving the HF checkpoint, then `update_weights_from_disk` succeeds (cumbersome for downstream trainers).
- Use `update_weights_from_tensor` (in-memory) instead, which bypasses the on-disk reload path. (Requires Engine to be co-located with the trainer process or HCCL group setup.)
- Avoid MoE in the layer that gets reloaded.

## Related

- #22784 (gpt-oss-120b EP=8 crashes with tensor size mismatch in weight_loader_fused) -- similar shape-mismatch family but with EP setup as trigger, not specific to reload.
- #20516 (KTransformers-MoE narrow regressions).

## Environment

- platform: Ascend A3 (910C / dav-c220), CANN 8.5.0
- image: `lmsysorg/sglang:main-cann8.5.0-a3` (built 2026-05-25)
- sglang: `0.5.12.post2.dev434+gb13d3d18c`
- sgl-kernel-npu: `2026.5.1`
- torch: `2.8.0+cpu`, torch_npu: `2.8.0.post2`
- transformers: `5.8.1`
- attention_backend: `ascend`
- tp_size=1, ep_size=1

## Question for maintainers

Is the reload path in `_load_w13` expected to honor `stacked_params_mapping` like the initial load? If yes this looks like a regression; if no, what's the recommended on-disk format for reloading FusedMoE checkpoints (pre-fused `gate_up_proj` or something else)? Happy to test a patch.
