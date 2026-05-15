# SESSION_HANDOVER_<date> — <session name> 交接

> **Template usage**: 复制本文件 → `SESSION_HANDOVER_<YYYY_MM_DD>_<short_slug>.md`，填字段 → commit。Session 末尾必填。
>
> **Handover ≠ backlog**。本文档是 **session 结束时点快照** + **in-flight 战术状态**。长期 open work → [`ROADMAP.md`](../ROADMAP.md)。
>
> **Continuing session 查阅顺序**（参考 a5_ops convention）：
> 1. §Dispatch（本 handover 入口段，做什么）
> 2. §In-flight tactical state（本 session 在传的活儿，inline 战术细节）
> 3. [`ROADMAP.md`](../ROADMAP.md)（更长视野的 backlog）
> 4. [`ARCHITECTURE.md`](../ARCHITECTURE.md)（架构 why）
> 5. `workspace/<session>/`（本地 scratch，如同机）
>
> **禁止反模式**：handover 退化成只剩 link 索引（interim 状态丢失，next session 拿不到上下文）。Handover SHOULD inline in-flight 战术。

---

## Metadata

- **Date**: 2026-MM-DD
- **Session name / tag**: T<N>_<short_slug>
- **Author agent**: <claude-sonnet-X | claude-opus-X | ...>
- **Outgoing context**: <where we are stopping>
- **Incoming agent role**: <what next agent should do first>

---

## Dispatch（必读，next agent 入口）

> 用 2-4 句话说清楚 "next session 第一件事做什么 + 为什么"。

例：
- 完成 T29.5 workflow state machine YAML 起草（P0e）；先读 [`a5_ops/docs/workflow/opgen_state_machine.yaml`](https://gitcode.com/zhengshencn_hwca/a5_ops) 学结构，再针对我们的 day-0 P0..P7 写一份。
- 当前已写到 P3 phase；剩 P4-P7 + Python critic 框架。

---

## In-flight tactical state（本 session 在传的活儿）

> **核心段**：inline 当前所有未结战术细节。这是 handover 的本职，不要简化成索引。

### 已完成的 fix（带 commit）

| 改动 | Commit | 验证状态 |
|---|---|---|
| 例：`run-npu-container.sh` IN_CONTAINER_CSV 自动派生 | `a6f3fca` | V1.4 GRPO chips 2,3 PASS |

### 进行中（state-of-the-art）

| 文件 / 模块 | 当前状态 | 卡点 |
|---|---|---|
| 例：`docs/_meta/workflow/day0_state_machine.yaml` | P0..P3 phase 写完 | P4 需先解决 vllm-ascend Mode Sweep 的 commit-range 输入 schema |

### 待开始（next session 拿起就跑）

按优先级：
1. <文件 / 任务> — Trigger / Why / 预估 effort
2. ...

### 风险 / Do-nots

例：
- ⚠ A3 chip 4-7 当前 ms-swift session 占用，**别用**
- ⚠ image `easyr1-npu:integrated-20260427` 还在跑别人的 V1.4，**别 docker rmi**

---

## Cross-agent 通信（如适用）

> 如果 session 协作多 agent（Claude + GPT + 人类），列各方分工 + 通信 channel。

| Agent | 在做什么 | 通信方式 |
|---|---|---|
| 例：本 session 的 CC | docs/skills 改 | Discord chat_id `1494825170399924366` |
| 例：a5_ops 那边的 CC | sgl-kernel-npu 改 | issue # / git submodule sync |

---

## Anti-pressure 检查（emit done 前自检）

> 引用 [`ANTI_PRESSURE_PROTOCOLS.md`](../../src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md) P1..P8。

Done emit 前，自答 8 个 P：

- [ ] P1：我是不是因为用户在等而跳过了 cold-drive replay？
- [ ] P2：我有没有写 "应该 / probably" 这种 hedge 词来 ship outcome？
- [ ] P3：我 spawn 的所有 agent / sub-skill 都 cite 了对应 OL 吗？
- [ ] P4：我有没有用"简单 case"为由跳过任何 phase？
- [ ] P5：所有 "expected failure / limitation" 都 cite 了具体 issue # 或 KB 条目吗？
- [ ] P6：所有 infrastructure bug 都修脚本了，还是有 inline workaround？
- [ ] P7：所有 done 状态都有 OL-02 / OL-09 provenance（log path + numeric evidence）吗？
- [ ] P8：所有 docker run 都走 helper script、所有 skill 都通过 slash command 调？

8 个都过 → 可以 emit done + commit。

---

## Commit refs

- Session opening: `<sha>`
- Session closing: `<sha>`
- Branch: `main` / `<feature-branch>`
- PR #: <如果有>

---

## Notes（自由文本）

> 留 session 中**未编码进 ROADMAP / OL / KB** 的隐含上下文。Next session 读这段是为了拿到"为什么这么做"的 why，不是 todo。

例：用户 2026-05-15 强调 a5_ops 借鉴落地优先级 1-4 顺序，state machine YAML 是 P0e 但要等前 4 项稳定再启动。
