# V4 megatron-layer 全路径 reduced-2层 训练步在 NPU 上跑通

**日期**: 2026-06-02。owner 指令"发现往前走的路线就继续"。
**结果**: REAL DeepSeek-V4(hidden=4096, 64h, 256-expert MoE, MLA)reduced-2层,走**真正的 `get_dsv4_spec` megatron spec builder**(不是 npu_native_shims 旁路),8.84B 参数 TransformerBlock,**forward + loss + backward 全通,0 NaN 梯度,grad_norm=4.25e-2**。

```
REAL V4 2-LAYER FORWARD OK on NPU: out=(64,1,4096) finite=True
params with NAN grad: 0 ; with FINITE grad: 1051
REAL V4 2-LAYER BACKWARD OK on NPU: loss=1.0000 grad_norm=4.251e-02 params_with_grad=1051/1051
```

这过了 handover 里记的 `TESpecProvider` 老边界 —— spec builder 真正工作,实例化了 TransformerLayer → MoELayer → router → MLA attention。

## 环境(关键 —— 之前的 V4 NPU 运行不在任何现存容器里)

- 老容器 `miles_v4_train_probe`:有 dsv4 miles 模型(`/opt/miles/miles_plugins`)+ workspace 挂载,但 **`docker inspect` device list = `[]`(没挂 NPU 设备)**,`torch.npu.is_available()=False`。handover 的"V4 NPU 训练跑通"不在这容器。
- **新建容器 `miles_v4_npu`**(verl 镜像 `verl-8.5.2-a3-...-qwen3-5`):`--device /dev/davinci0 + davinci_manager + devmm_svm + hisi_hdc`,挂 `/home/z00637938` + driver + hccn.conf,shm 32g。NPU avail True count 1。
- dsv4 miles 模型 cp 到共享挂载 `_miles_dsv4_preserved/`(脱离容器生命周期)。

**运行命令**:
```
docker exec miles_v4_npu bash -lc 'cd /home/z00637938/workspace && \
  export PYTHONPATH=/home/z00637938/workspace/_miles_stub:/home/z00637938/workspace/_miles_dsv4_preserved:\
/home/z00637938/workspace/tilelang-mlir-ascend:/home/z00637938/workspace/MindSpeed-clone:\
/home/z00637938/workspace/Megatron-LM-miles:$PYTHONPATH && \
  python3 _v4_real_2layer_fixed.py'
```

## 连过的 9 个真边界(每个都是 fix,patches 在 `_v4_megatron_fixes/`)

1. **triton 包冲突**:mainline triton 3.6 + triton-ascend 3.2 混装,namespace 互覆盖(`ImportError: Language`)。修:杀 race 的 pip → 清 site-packages/triton → 本地下 54MB wheel(A3 PyPI 28kB/s 太慢)→ scp → 离线 `pip install --no-deps`。[[feedback_triton_vs_triton_ascend_packaging_conflict]]
2. **容器无 NPU 设备**:见上,建 `miles_v4_npu`。
3. **dsv4 import 需 tilelang**:加 `tilelang-mlir-ascend` 到 PYTHONPATH(torch 2.9 兼容)。
4. **`T.bfloat16` 缺失**(`patch 03`):tilelang.language 无 bfloat16 属性(只有 float16)。dsv4 kernel 只把它当 dtype 值用(`dtype=T.bfloat16` / `==`),且 act_quant.py 本就用 `"bfloat16"` 字符串 → 加 `bfloat16 = "bfloat16"` string alias。注:底层 TVM ir_builder 缺 `BFloat16` PrimExpr FFI builder(只有 Float16),所以不能 func_gen,string alias 是对的。
5. **`PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` / `TL_ENABLE_FAST_MATH` 缺失**(`patch 04`):CUDA-centric perf flag,NPU pass pipeline 忽略 → 加到 enum 当 no-op,让 dsv4 kernel 加载。
6. **`miles.utils.replay_base` 缺失**(stub `_miles_stub/`):完整 miles RL 框架不在 host(只抽了 miles_plugins)。MoE router `routing_replay_manager.get_topk_fn(_compute_topk)` 只在 RL replay 用 → stub `get_topk_fn` 做 identity pass-through(返回原 topk fn)。**fwd+bwd 验证不需要 RL replay;跑完整 RL loop 时换真 miles 包**。
7. **dsv4 attention 返回 arity**(`patch 01`):`DeepSeekV4Attention.forward` 返回裸 `output`,但这版 Megatron-LM-miles `transformer_layer.py:671` 要 `(output, bias)` 契约 → 改 `return output, None`(dsv4 attn proj bias 是 fused,无单独 bias)。
8. **softmax-with-sink NaN 梯度**(`patch 02`,**最关键**):`sparse_attn_torch` 全 masked 行 `scores_max=max(-inf,...)=-inf` → `exp(-inf-(-inf))=NaN` → 282 attention 参数 NaN 梯度。`set_detect_anomaly` 定位到 `attention_core.py` `ExpBackward0`。第一版 clamp `-1e30` 修错方向(`exp(sink+1e30)=inf`→还是 NaN)。正确修:`scores_max = maximum(scores_max, attn_sink)` —— sink 纳入 max 归一化,`exp(scores-m)` 和 `exp(sink-m)` 都 ≤1 有界。这是数值稳定 softmax-with-sink 标准做法。修后 0 NaN。

## 诚实边界(不过度声称)

- ✅ megatron-layer **reduced-2层 fwd+loss+bwd 训练步**真在 NPU 跑通,梯度全 finite(0 NaN / 1051)。
- ✅ **用真 miles 包验证过(不是 stub)**:从本地已缓存的 `radixark/miles:deepseek-v4` 镜像(64.3GB)docker cp 出完整 `miles` 包(1.4MB Python 源,含 `utils/replay_base.py`)→ `_miles_real/`。换掉 stub 后 2层训练步仍 0 NaN,grad_norm=4.287e-2(与 stub 的 4.251e-2 一致 → 证明 stub 是忠实 no-op,且真 routing-replay 在非 rollout 训练步里就是 clean pass-through)。stub 仍保留在 `_miles_stub/` 作为无 miles 包时的 fallback。
- ⚠ loss 是 driver 的 `o2.pow(2).mean()`(占位 loss,验证梯度流),不是真任务 loss。
- ⚠ 这是 megatron-layer **训练步**(fwd+bwd);完整 RL on-policy loop 还需 rollout(sglang inference on NPU)+ 经验生成 + weight-sync —— 独立的大 phase。
- 减层基础(owner 确认"就按减层来");full 43层 = 分布式 TP/PP 工程,非本验证目标。

## Patches(可提 PR 的 V4-megatron-NPU 适配)

`_v4_megatron_fixes/`:
- `01_dsv4_attn_return_tuple.patch` → miles_plugins dsv4 模型
- `02_sparse_attn_softmax_sink_nan.patch` → miles_plugins dsv4 模型(**最有价值,真数值 bug**)
- `03_tilelang_bfloat16_alias.patch` → tilelang-mlir-ascend
- `04_tilelang_passconfig_keys.patch` → tilelang-mlir-ascend

PR 目标:patch 01/02 → miles(radixark fork);patch 03/04 → tile-ai/tilelang-mlir-ascend。先攒,完整 e2e(真 RL loop)后批量提 [[project_v4_upstream_pr_batch_after_e2e]]。
