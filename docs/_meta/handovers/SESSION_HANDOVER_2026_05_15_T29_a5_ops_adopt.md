# SESSION_HANDOVER_2026_05_15 — T29 a5_ops 架构借鉴落地

## Metadata

- **Date**: 2026-05-15
- **Session name / tag**: T29_a5_ops_adopt
- **Author agent**: claude-opus-4-7 (1M ctx)
- **Outgoing context**: T29.1-T29.4 完成（ROADMAP + OL catalog + ANTI_PRESSURE + handover 模板），entry-point 文档（README / CLAUDE.md / DOCS-CONVENTION）已串联；T29.5 state machine YAML 未启动
- **Incoming agent role**: 启动 T29.5（P0e）— 写 day-0 workflow state machine YAML + Python critic

---

## Dispatch

T29 a5_ops 借鉴 4 项落地全部 commit；本 session 在 T29.5（workflow state machine YAML + Python critic）启动前停止，**优先级最高的 P0e**留给 next session。

**Next session 第一步**：Read [`ROADMAP.md` §2 P0e](../ROADMAP.md)，然后看 a5_ops 的 [`docs/workflow/opgen_state_machine.yaml`](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/workflow/opgen_state_machine.yaml) 和 [`src/scripts/workflow/workflow_critic.py`](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/src/scripts/workflow/workflow_critic.py)（1750 行）学结构，再针对我们的 day-0 P0..P7 写一份。

---

## In-flight tactical state

### 已完成的 fix（带 commit）

| 改动 | Commit | 验证状态 |
|---|---|---|
| `docs/_meta/ROADMAP.md` 新建 | （本 session 末尾） | 文件存在 + 7 段（SI / P0 / next-wave / fork 跟踪 / decision gates / DEBT / completed） |
| `src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md` 新建 | 同上 | 27 个 OL（universal 18 + expert-specific 9）+ grep keywords + when-to-load |
| `src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md` 新建 | 同上 | P1..P8 + incident anchor（用本仓 T22-T28 真实事故）+ 决策点 cite 映射 |
| `docs/_meta/handovers/SESSION_HANDOVER_TEMPLATE.md` 新建 + 首个填好的 handover（本文件） | 同上 | 模板 + 第一个填好的真 handover（eating own dog food）|
| README / CLAUDE.md / DOCS-CONVENTION wiring | 同上 | 4 处入口加链；接手 checklist 改成 ROADMAP-first 顺序 |

### 进行中（state-of-the-art）

| 文件 / 模块 | 当前状态 | 卡点 |
|---|---|---|
| —— | 本 session 4 sub-task 全部结束；T29.5 未启动 | 见下方 "待开始" |

### 待开始（next session 拿起就跑）

按 ROADMAP §2 顺序：

1. **T29.5 / P0e** workflow state machine YAML + Python critic
   - 起点：a5_ops `docs/workflow/opgen_state_machine.yaml` + `src/scripts/workflow/workflow_critic.py`
   - 目标：`docs/_meta/workflow/day0_state_machine.yaml` 描述 day-0 P0..P7 phase + invariants；`src/scripts/workflow/day0_critic.py` 在 hook 里强制
   - 预估 effort：4-6 小时（大头是 Python critic 写 + 调试）
   - 优先级：高（a5_ops 借鉴 5 项里回报最大的一项）

2. **DEBT-1** A3-side repo stale 检测（30 min）
3. **DEBT-2** NPU 容器 bind set 抽象（1 h）
4. **DEBT-3** install-skills.sh 版本管理（1 h）

5. SI-2 / next-wave NPU 适配启动（要等用户给方向）

### 风险 / Do-nots

- ⚠ A3 host 当前 `quay.io/ascend/verl:verl-8.5.0-a3-...` 47h 前刚 pull 过；client side 已有 ms-swift / llamafactory / roll-npu / cann9.0.0-a3 等新镜像，**不要 docker rmi 任何这些**（其他 user 在用）
- ⚠ 全 16 chip 空闲，但 487 GB free disk 已 86% used；新 image build 前先 `docker system df` 看空间
- ⚠ a5_ops clone 在 `/home/z00637938/workspace/a5_ops`，**不在 easyr1-npu/upstream/**——别 push 到我们的 fork

---

## Cross-agent 通信

仅本 session（CC，1M context）。Discord chat_id `1494825170399924366`（用户 zzcn2422）。

---

## Anti-pressure 检查

- [x] P1：cold-drive 不适用（本 session 是 doc + KB 起草，没 NPU smoke）
- [x] P2：所有 commit 都有具体内容，无 "应该 / probably" hedge
- [x] P3：本 session 没 spawn sub-agent
- [x] P4：每个 doc 都按 a5_ops 完整结构写，无简化跳过
- [x] P5：OPERATIONAL_KNOWLEDGE 没把任何空 OL 标记成 "expected"
- [x] P6：CLAUDE.md / README / DOCS-CONVENTION 都更新了，无 inline workaround
- [x] P7：4 sub-task 都有具体 artifact + commit；emit T29 done 前确认 ROADMAP §2 P0a-d 全部 ✓
- [x] P8：所有 file 改动通过 Edit / Write，无 nohup / raw docker bypass

8/8 → 可以 emit done + commit。

---

## Commit refs

- Session opening: `0c86e0e` (T27+T28 之后)
- Session closing: （本 commit）
- Branch: `main`

---

## Notes

- 用户 2026-05-15 强调按 priority 1-4 顺序做（不要跳到最重的 state machine YAML）；P0e 留下次再说。
- a5_ops 的 `workflow_critic.py` 是 PreToolUse hook（Agent / Edit / Bash / WebFetch matcher），我们的 hook 当前只有 SessionEnd + PostToolUse on `git commit`，要扩展到 PreToolUse 是大改。
- a5_ops 的 ROADMAP 风格里 P0 用字母（P0a/P0b/P0c）而不是数字，我们沿用了——避免跨 session 编号冲突。
- 决策：state machine 不用 jsonschema，跟 a5_ops 一样用 PyYAML + 手写 invariant check（更灵活，hook timeout < 30s）。
