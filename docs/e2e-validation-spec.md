# End-to-end 验证契约 — 判定 skills 是否真的能"冷启动复现"

**目的**：给"cold-drive skill chain → working EasyR1 port on A3" 这件事定一个**我自己不能 fudge 的判定标准**。

**触发的教训**：2026-04-22 之前，我把"手工跑通 V1.4 + 写文档"讲成"P1 end-to-end closed"。这个文档是修这个错的后续工程 — 先把判定口径立死，再跑 agent，避免我再用暧昧语言包装失败。

---

## 1. 验证对象

**"Skill chain end-to-end 能用"** 的操作定义：

> 一个**完全不知道本项目历史**的 agent，只能读本仓 README / PORT-GUIDE / SKILLS-GUIDE / skill md / script，外加一条一句话任务，在 A3 上产出 working EasyR1 port，所有 smoke ladder rung 都绿。

**不是**：
- ❌ "我（熟悉项目的人）用这些 skills 做了一次工作，所以 skills 可用"
- ❌ "skills 的文档写得很详细"
- ❌ "skill scripts 各自单元测试 OK"
- ❌ "V1.4 smoke 过了"

---

## 2. Input 契约（agent 能看到什么）

**Agent 能读的**（冷启动需要的）：

- `README.md`
- `docs/PORT-GUIDE.md`（路径 1 — "跑起来"）
- `docs/SKILLS-GUIDE.md`（路径 2 — "从 0 重做"）
- `docs/DOCS-CONVENTION.md`（所有文档的 place-of-record map）
- 所有 `skills/*/SKILL.md`
- 所有 `scripts/*`（包括 smoke 脚本）
- `knowledge/npu-patterns.md`（坑目录 — 合理的快捷通道）
- `knowledge/upstream-refs.md`、`knowledge/smoke-ladder-convention.md`、`knowledge/images/*.md`
- A3 host ssh access（port 443）
- 公开的 github 仓（zhshgmail/EasyR1、hiyouga/EasyR1）

**Agent 不能读的**（这些会**让它作弊**，因为内容是"我曾经做过了什么"）：

- `docs/HANDOVER.md`（§6 是 transit state，含过去具体命令 / 结果）
- `docs/porting-journal.md`（我手工干活的流水账，看了等于抄答案）
- `docs/transformers-upgrade-drill.md`（drill 过程答案）
- `docs/UPGRADE-DRILL-STATUS.md`（drill 结果 + 数值）
- `docs/P2-WORKFLOW.md`（过去场景的设计，可能有硬编码信息）
- `docs/skill-dry-run-2026-04-20.md`（之前作弊性质的 dry-run 记录）
- `docs/DELIVERABLE.md` / `docs/codex-*.md`（历史结论）
- `docs/design.md` / `docs/dep-matrix.md`（过度细节）
- 历史 commit message（`git log` 里能看到过去手动 fix 的细节）

> 第三类文档（HANDOVER / journal / drill）**就是** agent 作弊会拿的东西。禁读的理由不是"它们不对"，而是"看了就等于给了它我已经踩过的坑 + 具体 fix，就不叫 cold-drive 了"。

**给 agent 的一句话任务**：

> 给定本仓和 A3 host，把 EasyR1 master (commit `dd71bbd` on `hiyouga/EasyR1`) 移植到 NPU 并在 A3 上跑通 smoke ladder。使用仓内的 skills 和 scripts，不要绕过它们。完成后报告：跑过的所有命令 + 每根 smoke rung 的结果 + 每次你卡住时的 workaround（这些是 skills 的缺口）。

---

## 3. PASS 标准（全部满足才算 PASS）

### 3.1 每根 rung 的技术标准

| Rung | 标准 |
|---|---|
| V1.1 | `torch_npu.npu.device_count() > 0`；`verl.utils.device.is_npu_available()` 返回 True |
| V1.3 | vllm_ascend 跑通一次 rollout（≥3 prompts 生成不乱码的 text） |
| V1.4 | 2-chip GRPO 2-step 跑完，step1 entropy_loss 在 [0.94, 1.04]（0.991 ± 5%），step2 在合理区间，checkpoint 存成功 |
| V1.5 | 4-chip HCCL 通，2-step 跑完 |
| V2.1 | `padding_free=True`，step 1 entropy_loss 在 V1.4 附近（±5%） |
| V2.2 | `padding_free=True` + `ulysses_size=2`，4-chip 跑完 |

**所有 rung 都 pass 才算 ladder 过。V1.4 pass + 其他没跑 = FAIL。**

### 3.2 "Cold-drive" 约束（流程标准）

还要**全部**满足：

- **Agent 从未 ssh 到 A3 去手工改 script 或手工调命令**（只通过 skills / scripts 调）
- **Agent 从未读禁读的文档**
- **Agent 从未让我（session 主 agent）手工帮它 debug**
- **Agent 遇到的每个"skill 不够用"的地方都记在它的报告里**

## 4. FAIL 的分类

失败不是 binary。agent 报告后分三类：

1. **Infra blocker**（A3 host 不可用 / chip 被抢 / 网络挂）：重试
2. **Skill gap**（skill 缺了步 / 文档漏了信息 / script 有 bug）：**这是我们的错，要修**。记 NPU-OPS-XX 或 skill issue，改，再跑
3. **Agent incompetence**（skill/doc 其实够但 agent 没领会）：**可能也是 skill 问题**（表述不清），至少换 agent 再跑一次

连续 2 次 FAIL 在第 2 类 → 说明 skills 没 ready to ship，诚实记录在 `docs/e2e-validation-results.md`。

## 5. PASS 后才能做的声明

只有当所有 rung 的技术标准 + cold-drive 约束都满足，才能：

- 在 README / PORT-GUIDE 说 "skill 链路已端到端验证"
- 在对客户的材料里说 "customers can reproduce using our skills"
- 关闭 task #33
- 以此为基础做 P2 或 vllm 升级规划

**PASS 前**：用精确口径 "V1.x smoke manually verified; skill-chain cold-drive 验证 pending"

## 6. 结果归档

- Agent 的完整 log → `/tmp/e2e-agent-log-<timestamp>.txt`（仓外）
- 结果汇总 → `docs/e2e-validation-results.md`（新文件）含：
  - 用的 prompt 原文
  - Agent 走的每一步命令
  - 每根 rung 的数值（或 error）
  - 识别的 skill gaps
  - 修复动作
  - 是否 PASS（严格按 §3 标准）
- 每次失败都记录，不删

---

## 7. 不在本次 scope 的

- **P2 场景**（D ≥ 1 需要 NPU 生态适配）—— 当前 D=0，P1 都没验过就跑不了 P2
- **vllm 多上游升级场景** —— #32 on hold
- **多个 RL 框架的迁移性**（OpenRLHF / TRL）—— skill 宣称可复用，但先证明 EasyR1 这一个
- **性能 / 长训练 / 多节点** —— smoke ladder 不包括这些

---

## 8. 我和 agent 的分工

- **我做的**：准备 A3 环境（chip 空、checkpoint 清）、spawn agent、看 agent log、判定、修 gap、再 spawn
- **我不做的**：agent 跑过程中替它调命令、替它改 script、替它挑选做什么 rung、在 agent 报告里包装 FAIL 成 PASS
