# 升级演练（drill）当前状态

**本文档回答**：
- 我想用 `transformers 5` / `vllm 0.18` / `CANN 8.5.1` / 新 image 跑 EasyR1，可以吗？
- `image-upgrade-drill` 这个 skill 能自动跑起来吗？

**短答**：
- **生产不推荐现在切**。v1（CANN 8.5.0）是已验证的发布路径，`ascend-port` 分支已 backward-compat 兼容新 image，但没在新 image 上跑完整 smoke 梯子
- **skill 有效性在验证中**，还没拿到"一个 agent 按 skill 从 0 走完 7 步"的干净数据点

---

## 1. 升级演练是什么、不是什么

**是什么**：一次**结构化演练** —— 2026-04-19 把 v1 `ascend-port` 往前推到 v2 image（`verl-8.5.2-a3-*`，里面是 CANN 8.5.1 + torch_npu 2.9 + transformers 5.3.0.dev0 + vllm_ascend 0.17）跑一遍，目的是：
1. **找出**一次跨大版本升级实际撞了多少 API break
2. **量化**升级成本（预测 vs 实际）
3. **沉淀**一个可复用的 [`image-upgrade-drill`](../skills/image-upgrade-drill/SKILL.md) skill，下次新 CANN / 新 transformers 出了照着跑
4. **cherry-pick** backward-compat fix 进 `ascend-port`，让**主发布分支**同时兼容两套 image

**不是什么**：
- ❌ **不是 v1 → v2 发布迁移**。v2 演练用的 image + transformers 5 不是我们推荐给用户的生产栈
- ❌ **不是对 "新 deps 在 NPU 上完全可用" 的最终签字**。只验证了 drill smoke（2-step + 20-step），没跑完整 V1.1→V2.2 梯子
- ❌ **skill 复现验证没完成**（见 §3）

---

## 2. drill 的具体结果（2026-04-19）

**Drill 分支**：`zhshgmail/EasyR1:ascend-port-transformers-upgrade`，head `2fd9337`

**Drill image**：`quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`（24 GB）

### 精确版本表 — 实测 vs 推理兼容

**本次 drill 实测运行的版本**（在 8.5.2 image 上跑通 2-step + 20-step smoke 的版本）：

| 组件 | v1 生产（实测） | v2 drill（实测） |
|---|---|---|
| CANN | 8.5.0 | **8.5.1** |
| torch_npu | 2.8.0 | **2.9.0** |
| transformers | 4.57.6 | **5.3.0.dev0** |
| vllm_ascend | 0.13.1.dev18+g2e5f72f92 | **0.17.0rc2.dev109+g54879467c** |
| vllm（框架） | 0.13.0+empty | （0.17 的 empty placeholder） |
| triton_ascend | 3.2.0 | 3.2.0 |
| triton（upstream） | — | 3.6.0（image 额外带的） |

**注意**：`ascend-port` 分支**源码**里针对 vllm **0.18** 的 backward-compat fix（`ecce71d`）**是为 0.18 的 `SamplingParams.eos_token_id` 只读 property 设计的**。但我们**实测运行**的是 vllm_ascend **0.17**（上表）。对 0.18 的兼容性来自：

1. 从 0.17 → 0.18 的 `SamplingParams` 源码 diff 看，eos_token_id 的 property 化在 0.17 就引入了（drill 实测中 bug 真的触发了）
2. 我们的 fix 用 `isinstance(cls_attr, property) and cls_attr.fset is None` **泛化**到任何未来只读 property

**因此对 vllm 0.18 的支持属于"推理兼容"（inferred compat），不是实测验证**。如果你要用 vllm 0.18，建议自己跑 V1.4 smoke 验证。

**推理兼容**（源码 backward-compat 写法覆盖，未实测）：
- transformers >= 5.0（任何 5.x、6.x，只要 `no_init_weights` 还在 `modeling_utils` 或 `initialization` 之一）
- vllm >= 0.18（只要 `SamplingParams` 保持 `@property` 风格只读字段）
- 任何介于 drill image 和未来 image 之间的 CANN 小版本

**通过了**：
- ✅ drill V2.2 2-step smoke（原 drill 报告）：`entropy_loss` step1 = **1.434** on 8.5.2 image（V2.2 config：padding_free=True, ulysses=2, 4-chip）
- ✅ drill V2.2 20-step smoke（原 drill 报告）：全 20 步稳定（entropy_loss ∈ [1.31, 1.83]，grad_norm max ~3.2，no HCCL / vector core 错误）
- 🟡 **2026-04-22 V1.4 smoke 单 rung 手动 PASS（on both images）**：
  - **v1 image（8.5.0）**：step1 entropy_loss = **0.991**（exact match V1.4 baseline），step2 = **1.263**（exact match）
  - **v2 drill image（8.5.2）**：step1 entropy_loss = **1.275**，step2 = **0.895**（grad_norm 2.07，非 all-reward-tied）
  - 这**没有**证明"P1 端到端闭环"。只证明了 V1.4 这一根 rung 能在两个 image 上跑出合理数值。V1.1 / V1.3 / V1.5 / V2.1 / V2.2 都**没跑过**；skill chain 也**没被冷启动驱动**。真正的"端到端闭环"要 second actor 仅凭 SKILLS-GUIDE + 仓库产物复现出同样结果
  - 见 `knowledge/npu-patterns.md#npu-ops-010` 和 memory `end_to_end_vs_described.md`

**代价**：
- 2 个代码级 backward-compat fix（`no_init_weights` import try/except + `SamplingParams.eos_token_id` 只读 property 跳过）—— **已 cherry-pick 到 `ascend-port`**
- 3 个基础设施坑 → 新加的 stable ID：NPU-OPS-006（docker daemon proxy）、NPU-OPS-007（image 没 pip.conf）、NPU-OPS-008（huaweicloud mirror 空）
- 1 个额外 bug 定位：NPU-BUG-004（triton 3.6 与 triton-ascend 3.2 共存冲突）

**代码变更**：总共 **4 LOC**，跨 2 个文件。

**仍然存在的问题**：
- **NPU-BUG-003 在 CANN 8.5.1 上更糟**：inductor 路径不再 crash 而是返回 silently corrupted 的 `grad_norm`（基线 1.49 → corrupt 88973），step 2 再 crash。仍然必须 `use_torch_compile=false`
- **没跑完整 V1.1 → V2.2 smoke 梯子**，只跑了 2-step + 20-step drill smoke
- **没做长训练收敛验证**（500+ step）

完整细节：[`transformers-upgrade-drill.md`](transformers-upgrade-drill.md)

---

## 3. skill 有效性验证 —— **尚未完成**

2026-04-19 为了验证 [`image-upgrade-drill`](../skills/image-upgrade-drill/SKILL.md) skill 真的能让**一个完全没有上下文的 agent 从 0 走完 7 步**，spawn 了一个 isolated agent 做复现。

**进展**：
- ✅ Step 1-3（infra 预检、drill 分支 `ascend-port-transformers-upgrade-reproduce`、build image `easyr1-npu-852:drill-reproduce`）走完
- ✅ 第一个 API break `no_init_weights` 撞到，log 在 A3 `/tmp/z00637938/reproduce/logs/v22_reproduce_20260419_163542.log`
- 🟥 **然后 silent 卡住 8+ 小时**

**判断**：harness 级 limitation（context 满 / 超时），不是 skill 内容的问题。但**没有"干净的一次"说 skill 能端到端跑通**。

留存：drill 分支 + `easyr1-npu-852:drill-reproduce` image 还在 A3 上，供下次分段复现用。

---

## 4. 什么时候你该切到 v2？

**现在不应该**，除非：

- 你有 CANN 8.5.1 的强依赖（某个新 op / bug fix 只在 8.5.1 有）
- 你愿意**自己跑完整 V1.1 → V2.2 smoke 梯子**来验证 v2
- 你明确理解：skill 的"自动化复现"现在是 best-effort 状态

**应该**：继续用 v1（`ascend-port` 分支在 8.5.0 image 上）。`ascend-port` 上的 backward-compat cherry-pick 是**无害的** —— 在 8.5.0 跑也正常（⚠️ 见 §5 caveat）。

---

## 5. 已知 caveat

**2026-04-22 更新**：`1f716ea` + `ecce71d` 两个 backward-compat cherry-pick 在 8.5.0 image 上已跑过 **V1.4 smoke**（step1 0.991 exact match）。但**完整 smoke ladder（V1.1/V1.3/V1.5/V2.1/V2.2）没跑过**，skill chain 也没端到端驱动过。所以"理论上 backward-compat"这句话目前**只在 V1.4 这一根 rung 上被验证**，不是在完整 workload 上。

如果你担心：
- 把 `ascend-port` revert 到 `6f8197f`（V2.2 smoke 那个 commit）用，跳过两个 cherry-pick
- 或帮我们跑一次 V1.4 回归，数值匹配就算确认

这是 [`HANDOVER.md`](HANDOVER.md) §6.2 标的 P1，~30 分钟工作。

---

## 6. 新 CANN / 新 transformers 出来了要不要升级？

推荐流程：**用 [`skills/image-upgrade-drill/`](../skills/image-upgrade-drill/SKILL.md)** 走 7 步演练。

产物：
1. 一份带数字的 drill report（预测 vs 实际成本、LOC 变更、bug probe 结果）
2. backward-compat fix 的 commit 序列，cherry-pick 进 `ascend-port`
3. 新增的 stable ID 加到 `npu-patterns.md`
4. PASS / BLOCKED 决策依据

不要手动跳过演练直接切 —— 这是 `image-upgrade-drill` SKILL.md 里"When not to use" 的反例。

---

## 7. 相关文档

- **v2 drill 完整报告**：[`transformers-upgrade-drill.md`](transformers-upgrade-drill.md)
- **drill skill 说明**：[`skills/image-upgrade-drill/SKILL.md`](../skills/image-upgrade-drill/SKILL.md)
- **v1 生产路径**：[`PORT-GUIDE.md`](PORT-GUIDE.md)
- **从 0 重做移植 / skill 使用**：[`SKILLS-GUIDE.md`](SKILLS-GUIDE.md)
- **HANDOVER 里 drill 相关 open items**：[`HANDOVER.md`](HANDOVER.md) §6.1、§6.2
