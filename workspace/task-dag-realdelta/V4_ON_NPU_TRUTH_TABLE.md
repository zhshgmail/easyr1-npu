# V4-on-NPU 真实状态一页表（2026-06-02，逐项 verified/未验证标注）

> 目的：之前我边查边口头讲,造成训练/推理、op-gen/CANN/tilelang、sglang 通没通 一连串前后不一致。
> 这页只列**我实际验证过的**(标 ✅ verified + 证据),和**没验证/只是声称的**(标 ⚠/❓),不含口头推断。

## 一、训练侧（actor / megatron）

| 项 | 状态 | 算子来源 | fwd | bwd | 证据 |
|---|---|---|---|---|---|
| megatron 2-layer 训练步 | ✅ verified 跑通 | 见下 | ✅ | ✅ 0 NaN，grad_norm=4.29e-2，1051/1051 finite | 本 session 实跑 `_v4_real_2layer_fixed.py`，`miles_v4_npu` 容器 |
| attention | ✅ | **纯 torch `sparse_attn_torch`**（attention_core.py:7，einsum/exp/sum） | torch 算子 | **PyTorch autograd 自动反向**（非 op-gen、非 tilelang） | 源码已读；softmax-sink NaN 在此路径修复 |
| attention（备选路径） | 存在未走 | `DeepSeekV4SparseAttention(torch.autograd.Function)`（:113）显式 fwd+bwd / CANN-native NSA | — | 显式 backward(:125) / `npu_nsa_select_attention` 自带 bwd state | 源码已读，未在 2-layer 步走它 |
| 5 个核心 op（NSA/MLA/indexer/rms_norm） | ✅ CANN-native 可用 | torch_npu `npu_nsa_select_attention` 等 | ✅ | NSA +bwd state；indexer `npu_sparse_lightning_indexer_grad_*` 带 grad | 记忆 project_v4_ops_cann_native_mapping（A3 RAN） |

**训练侧结论**：fwd+bwd 都通,**反向来自 autograd（纯 torch）或 CANN-native NSA 自带 bwd**。**无 op-gen、无 tilelang。** op-gen 是早期错误路径,已弃,不在训练路径。

## 二、推理侧（rollout / sglang）

| 项 | 状态 | 算子来源 | 证据/卡点 |
|---|---|---|---|
| sglang V4 5-step RL 闭环 | ✅ **上次(2026-06-01) verified PASS** | sglang 的 tilelang kernel | `v4_RL_LOOP_PASS_log_2026_06_01.txt`：rollout+update_weights×5 闭环。**但仅 attention-only weight-sync（5 tensor），有限闭环 PoC** |
| sglang V4 server（本 session 从零重搭） | 🔴 **这次撞墙** | sglang tilelang kernel（attention/compressor/indexer，**无 CANN-native 退路**，已查证） | FP8→SWA→tilelang-import→ACL(format_cast fallback 解了)→tilelang DSL 版本错配(proxy 修了2个/reduce 卡 TVM buffer-alias) |
| **为何上次通这次不通** | ❓ **未核实** | — | 上次 6-01 的 sglang/tilelang 环境 vs 这次从零重搭 不同；最可能上次没真编 V4 tilelang attention kernel 走了别的路。**这个我必须查清才能下结论** |

**推理侧结论**：sglang V4 上次闭环 PASS 过（有限版）;这次从零重搭撞 tilelang 0.1.8-DSL-vs-fork 版本错配。**"通没通"取决于环境状态,我尚未核实两次差异 → 在核实前不该断言 sglang"blocked"。**

## 三、tilelang 这摊到底是什么

- miles 有 6 个 `@tilelang.jit` **训练** kernel；sglang 有自己的 **推理** tilelang kernel。**两侧都有 tilelang 算子。**
- 训练侧我**用 CANN-native 替换掉了** tilelang（没真跑 miles 的 tilelang 训练 kernel）。
- 推理侧我**试图真跑** sglang 的 tilelang kernel → 撞版本错配。
- **统一真问题 = 让 tilelang-ascend 编 miles/sglang 用的 mainline 0.1.8 DSL**（proxy/reduce/或 port-0.1.8）。**与训练/推理无关** —— 哪侧真跑 tilelang 都要这个。

## 四、诚实的未决项（不许声称为已完成）

1. ❓ **sglang 上次(6-01)PASS 用的什么环境,为何这次重搭撞墙** —— 未核实，最高优先。
2. ❓ **megatron-generate 作为零-tilelang rollout 的可行性** —— 未验证（训练侧 fwd 通,但 generate/解码循环没跑过）。
3. ⚠ tilelang 0.1.8 DSL backport：proxy 2 个 verified；reduce 卡 TVM buffer-alias def-use；port-0.1.8 = 1212-commit TVM drift 的中型工程。
4. ⚠ 2-layer 训练步是**减层 PoC + 占位 loss**（`o.pow(2).mean()`），不是真任务 loss / 真 RL。

## 五、我把话讲乱的地方（对自己诚实）
- 制造了无谓的"训练 vs 推理"区分（实为"哪侧用 CANN-native 替换 vs 哪侧试跑 tilelang"）。
- 一路说 sglang"blocked",忽略了 6-01 它 PASS 过 → 前后不一致。
- op-gen/CANN/tilelang 混讲。
→ 以后：先核实再讲；状态走这页表,不口头推断。
