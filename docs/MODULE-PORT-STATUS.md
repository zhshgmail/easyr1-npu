# MODULE-PORT-STATUS — 模块化 NPU 移植总览

**项目目的**（user 2026-04-23T21:47 重申）：手动把一个大颗粒软件（EasyR1 + 其依赖链）完整移植到 NPU，每个模块各自跑通 V1.3 语义 PASS + V1.4 training PASS；把积累的知识沉淀到 KB；最后 codify 成 skills 自动化。**不是**为了快速完成 task，是要真跑通端到端。

本文件是所有模块的**唯一权威总览**：每个上游模块一行，指向它的 session workspace / skill 定义 / trace branch / 当前状态。会被 auto-compact 保留（repo 内）。

## 模块状态表

| 模块 | 目标版本 | Skill | Session workspace | Trace branch | V1.3 | V1.4 | 交付物 | 备注 |
|---|---|---|---|---|---|---|---|---|
| **transformers (upgrade)** | 5.3.0.dev0 → 5.6.0 | [`src/experts/transformers-upgrade-expert/`](../src/experts/transformers-upgrade-expert/) | [`workspace/transformers-upgrade-trans-upg-e2e-20260422-2200/`](../../workspace/transformers-upgrade-trans-upg-e2e-20260422-2200/) | `ascend-upg-trans-upg-e2e-20260422-2200` on `zhshgmail/transformers` | ✅ PASS | ✅ PASS entropy_loss=1.310 in band [1.21, 1.34] | v2 image `easyr1-npu-trans56:trans-day0-wetrun-20260423-0109` | 2026-04-22 完成 |
| **transformers (Day-0 probe 5.6.0)** | 5.6.0 | [`src/experts/transformers-day0-expert/`](../src/experts/transformers-day0-expert/) | [`workspace/transformers-day0-trans-day0-wetrun-20260423-0109/`](../../workspace/transformers-day0-trans-day0-wetrun-20260423-0109/) | 同 upgrade branch | ✅ PASS | ✅ PASS | 同上 | Day-0 + upgrade merged artifact |
| **torch / torch_npu (Day-0)** | torch 2.11.0+cpu + torch_npu 2.11.0rc1 | [`src/experts/torch-day0-expert/`](../src/experts/torch-day0-expert/) | [`workspace/torch-day0-{analysis,manual,deploy}-20260423-*`](../../workspace/) | (Fix C commits in `ascend-day0-torch211-20260423` on `zhshgmail/vllm-ascend`) | ✅ PASS | ✅ PASS entropy_loss=1.275 **exact match** baseline | Fix C image `easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423` | 需要 Fix C `vllm_ascend_C.so` rebuild 才过 V1.4 |
| **vllm-ascend (Day-0 for torch 2.11)** | patched 0.17.0rc2.dev109 | [`src/experts/vllm-ascend/day0-expert/`](../src/experts/vllm-ascend/day0-expert/) | [`workspace/vllm-ascend-day0-{analysis,deploy}-20260423-*`](../../workspace/) | `ascend-day0-torch211-20260423` (4 commits for torch 2.11 Fix B+/C) | ✅ PASS | ✅ PASS | 同 torch Fix C image | torch 2.11 上 V1.3+V1.4 都过 |
| **vllm (Day-0 probe 0.20.0)** | 0.20.0 | [`src/experts/vllm-day0-expert/`](../src/experts/vllm-day0-expert/) | [`workspace/vllm-day0-vllm0200-20260423-1623/`](../../workspace/vllm-day0-vllm0200-20260423-1623/) | 同 `ascend-day0-torch211-20260423` (iter 1-18 新增 13+ commits) | ✅ bit-exact PASS (iter 18) | ⏳ V1.4 running (with VLLM_BATCH_INVARIANT=0) | iter 18 image `easyr1-npu-vllm0200:vllm-day0-vllm0200-20260423-1623-iter18` | **正在 wet-run**; PROGRESS.md 实时更新 |
| **EasyR1 (consumer port)** | main tip (dd71bbd) | [`src/experts/easyr1-expert/`](../src/experts/easyr1-expert/) | [`workspace/easyr1-port-*`](../../workspace/) | `ascend-port` on `zhshgmail/EasyR1` (16 commits) | ✅ PASS | ✅ PASS | v1 image `easyr1-npu:ascend-port` | 最外层 consumer，不 Day-0 |

## 活跃 session（正在跑的，**非** auto-compact 可抹去）

| Session | Status | PROGRESS.md | 接手入口 |
|---|---|---|---|
| `vllm-day0-vllm0200-20260423-1623` | V1.3 PASS; V1.4 smoke running | [`workspace/.../PROGRESS.md`](../../workspace/vllm-day0-vllm0200-20260423-1623/PROGRESS.md) | 读 PROGRESS → 看 iter 18 最后段 |

## 每个模块的 4 份基本文件约定

下次开模块 port session 时按这四份来产出，格式统一，新人一眼接手：

1. **`analysis.md`** — Phase 1 feasibility：目标版本 vs 当前 NPU 生态 cursor 差多少，drift 在哪，是不是真 Day-0
2. **`PROGRESS.md`** — ground-truth iteration log：每 iter 改了什么、iter 结果、下 iter 计划。每个 wet-run 发现新问题都 append
3. **`blocker-report.md`** / **`PR_MATERIAL.md`** — 交给上游 maintainer 的手工可读的 fix 说明（file-level diff + rationale + reproducer）
4. **`ONBOARDING.md`** — 给下游消费者（下一层 expert / EasyR1 porter）看的 "怎么装上这个 image 继续用"

**未来**每跑一次 port session 时（不论是 Day-0 还是 upgrade），产出都要落在上面的 4 份里，并在本表里加一行。

## 下一步（优先级顺序）

1. **当前**：V1.4 smoke on iter 18 vllm 0.20 image（VLLM_BATCH_INVARIANT=0），验证 training path
2. V1.4 过后：把 vllm 0.20 所有 iter 1-18 的 14 个 file-level fix 整理成给 vllm-ascend 维护者的 `PR_MATERIAL.md`
3. 等 V1.4 + PR_MATERIAL 都完成后，考虑 codify 进 `vllm-day0-expert` skill（需要现在手工过一遍的路径被 state_machine 捕获）
4. 后续可做：triton-ascend Day-0 probe（如果遇到 triton kernel 层的 bug），CANN Day-0 probe
