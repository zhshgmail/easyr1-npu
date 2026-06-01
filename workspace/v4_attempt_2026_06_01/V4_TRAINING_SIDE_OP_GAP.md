# V4 训练侧算子 gap inventory(miles + DeepSeek-V4-Flash on NPU)

从 synth-delta RL loop PASS → **真训练 e2e** 的缺口。这是 held 批量 PR 前剩下的真活。
来源:`miles-v4-extracted/miles/miles_plugins/models/deepseek_v4/`(真训练侧 V4 实现)+
`mbridge/deepseekv4.py`(config bridge,确认参数清单)。

## 训练侧参数清单(来自 mbridge,确认 V4 训练侧 module 拓扑)
- **Compressor**: `compressor.{ape, wkv, wgate, norm}`(self_attn 级 + indexer 内各一份)
- **C4 Indexer**: `indexer.{wq_b, weights_proj, compressor.*}`,n_heads=64 head_dim=128 topk=512
- **Hash-coding(hc)**: `hc_attn_{fn,base,scale}` / `hc_ffn_{fn,base,scale}` / `hc_head_{fn,base,scale}`;
  hc_mult=4, sinkhorn_iters=20, eps=1e-6 → **这就是 hc_split_sinkhorn 的来源**
- **o_lora**: `wo_a/wo_b`,o_lora_rank=1024,o_groups=8
- **attn_sink**,swiglu_limit(clamp),window_size=128,n_hash_layers=3

## 算子 × 后端 × NPU 状态(关键 gap 表)
| 算子 | 文件 | 后端 | NPU 状态 |
|---|---|---|---|
| **hash-coding sinkhorn** | `ops/kernel/sinkhorn.py` | **TileLang** `@tilelang.jit` | ❌ CUDA-TileLang → **= hc_split_sinkhorn**,正用 a5_ops `/ascendc-op-gen` 生成 AscendC kernel(进行中) |
| **act_quant** | `ops/kernel/act_quant.py` | **TileLang** | ❌ 需 NPU path(fp8 act-quant) |
| **C4 indexer fwd** | `ops/kernel/tilelang_indexer_fwd.py` | **TileLang** `@T.prim_func` | ❌ |
| **C4 indexer bwd** | `ops/kernel/tilelang_indexer_bwd.py` | **TileLang** | ❌(训练需要梯度) |
| **sparse MLA fwd** | `ops/kernel/tilelang_sparse_mla_fwd.py` | **TileLang** | ❌ FA-class,合法 IL-chain 目标 |
| **sparse MLA bwd** | `ops/kernel/tilelang_sparse_mla_bwd.py` | **TileLang** | ❌ FA-class |
| Compressor | `ops/compressor.py` | ❌ **不是纯 torch** | forward 经 `qat.fp8_simulate_qat` → `.kernel.act_quant`(**TileLang**)。import 即拉 tilelang,forward 真调 act_quant |
| HyperConnection | `ops/hyper_connection.py` | ❌ **不是纯 torch** | forward **真调 `hc_split_sinkhorn`**(hyper_connection.py:55,`.kernel.sinkhorn` TileLang)。这就是 #311 op-gen 的算子 |
| rope | `ops/rope.py` | ⚠️ 需 `miles_megatron_plugins` | 计算是 torch,但 import 依赖 miles megatron 插件包(未装) |
| utils | `ops/utils.py` | ✅ **纯 torch,import OK**(实测) | rotate_activation 等 helper,唯一无外部 kernel 依赖 |
| attention_core(组装) | `ops/attention_core.py` | 调 sparse_mla TileLang + autograd.Function | ❌ 依赖 sparse_mla |

## 缺口结论(2026-06-01 实测纠正 — 推翻了之前"3 个纯 torch module 大概率直接跑"的假设)
**实测**(verl-8.5.2 容器 + editable megatron-core 0.16.0rc0 + mindspeed 0.16.0):
- 只有 `ops/utils.py` 是纯 torch、import OK。
- `compressor` / `hyper_connection` **import 时就拉 tilelang**(经 qat→act_quant / kernel→sinkhorn),
  且 **forward 真调** act_quant / hc_split_sinkhorn —— **不是装饰性 import,是运行时硬依赖**。
- `rope` import 依赖 `miles_megatron_plugins`(miles 的 megatron 插件包,未装)。

**关键结论:训练侧 module 无法拆成"纯 torch 先跑 / tilelang 后补"。** `hyper_connection.forward`
没有 hc_split_sinkhorn 的 NPU path 就跑不了;`compressor.forward` 没有 act_quant 的 NPU path 就跑不了。
所以 **hc_split_sinkhorn AscendC kernel(#311)在训练侧 e2e 的真关键路径上** —— 不是 nice-to-have。
tilelang 是比预想更硬的前置(挡的不止 6 个 kernel,还挡 torch module 的 import + forward)。

训练侧 e2e = **6 个 TileLang kernel 要 NPU path**(sinkhorn / act_quant / indexer-fwd /
indexer-bwd / sparse_mla-fwd / sparse_mla-bwd),且这些是 module forward 的硬依赖,不可绕。

- **sinkhorn**:正在 a5_ops `/ascendc-op-gen` 生成 AscendC kernel(在 e2e 关键路径上,不是 side quest)
- **sparse_mla fwd/bwd**:FA-class,是 TileLang-IL chain(designer→translator)的合法目标
- **indexer fwd/bwd + act_quant**:vector/quant 类,走 kw AscendC 路径或 native torch_npu
- 训练侧比推理侧多了 **bwd 梯度** kernel(indexer_bwd / sparse_mla_bwd)—— 这是推理 PASS
  之外的纯训练增量

## 与推理侧(sglang)gap 的关系
推理侧 14-gap(已 PASS,native op 已替换)是 forward-only;训练侧多出 backward + 真 sinkhorn/
indexer/sparse-mla 的 compute kernel。两侧共享 Compressor/Indexer/hc 的 module 拓扑,但训练侧
是真 TileLang kernel(CUDA target),推理侧我们已用 torch fallback + native op 跑通 forward。

## 下一步(到真 e2e)
1. ✅ sinkhorn → a5_ops AscendC kernel(进行中,#311)
2. 验证 3 个纯 torch module(compressor/hc/rope)在 NPU 直接跑
3. sparse_mla fwd/bwd → 评估 tilelang-mlir-ascend backend vs AscendC FA-class IL chain
4. indexer fwd/bwd + act_quant → AscendC kw 路径
5. 全部 NPU path 就绪后,接真训练 delta 替换 RL loop 的 synth-delta → 真 e2e → 批量 PR
