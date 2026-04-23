# 文档组织 convention

**给本仓贡献者（人 + agent）的规则**。所有文档改动遵守这些 convention，下一个 session 才能快速接手而不需要重新计划。

---

## 0. 最高原则

**README 是入口 + 索引，不是内容仓库。**

- 所有**重要文档从 README 出发，通过链接直接或间接可达**
- README 本身**只负责导航** —— 内容写在各自专门文档里
- 违反了就失效了：如果 README 膨胀到承载具体决策 / 状态 / 知识条目，下次维护会失控

"间接可达" = 2 跳以内到达任何重要文档（README → 某个索引 doc → 目标 doc）。

---

## 1. 每类信息放哪里 —— place-of-record map

| 信息类型 | 唯一归属 | 从 README 怎么到达 |
|---|---|---|
| **NPU 坑 stable ID**（bug / code pattern / env / ops 规则） | `knowledge/npu-patterns.md` 单文件，按 ID schema 追加 | README → "仓库布局" → 点 knowledge/ |
| **目标 image 的依赖快照** | `knowledge/images/<slug>.md`（每个 image 一份） | README → knowledge/images/ |
| **upstream ref 对应版本** | `knowledge/upstream-refs.md` | README → knowledge/ |
| **smoke 梯子命名约定 + 基准数值** | `knowledge/smoke-ladder-convention.md` + `scripts/smoke/README.md` | README → scripts/smoke/ |
| **EasyR1 依赖链分级** | `docs/easyr1-dep-chain-audit.md` | README → PORT-GUIDE.md §依赖 → audit |
| **NPU 适配 task 清单（档 1/2/3）** | `docs/npu-adaptation-tasks.md` | README 路径 4 → 直接链接 |
| **P2 端到端 workflow**（D ≥ 1 时怎么闭环） | `docs/P2-WORKFLOW.md` | README 路径 4 + SKILLS-GUIDE Step 1.5 引用 |
| **"怎么跑起来" 手册** | `docs/PORT-GUIDE.md` | README 路径 1 |
| **"怎么重做移植" 手册** | `docs/SKILLS-GUIDE.md` | README 路径 2 |
| **升级演练当前状态（v2/新 image）** | `docs/UPGRADE-DRILL-STATUS.md` | README 路径 3 |
| **v2 drill 完整实证报告** | `docs/transformers-upgrade-drill.md` | UPGRADE-DRILL-STATUS → transformers-upgrade-drill |
| **Skill 验证实验记录**（dry-run / 回归结果） | `docs/skill-dry-run-<date>.md` 每次独立文件 | SKILLS-GUIDE + HANDOVER 引用 |
| **移植时间线 / 日记** | `docs/porting-journal.md` 单文件，按日期追加 | HANDOVER → porting-journal |
| **原始需求 + 任务拆解** | `docs/design.md` | HANDOVER → design |
| **正式 sign-off 报告** | `docs/DELIVERABLE.md` | README / HANDOVER 引用 |
| **当前状态 + 未结工作 + 交接清单** | `docs/HANDOVER.md` | README 的仓库布局 + 相关文档段 |
| **每个 skill 的权威说明** | `skills/<name>/SKILL.md` (legacy pre-2026-04-23) **或** `src/experts/<name>/{SKILL.md,agent.md,state_machine.yaml,references/ALWAYS_LOADED_RULES.md}` (新 expert 架构) | SKILLS-GUIDE §1 表引用；新架构 → HANDOVER §7 expert 清单 |
| **新 expert 架构设计 + Day-0 reframing** | `docs/design/SKILLS_ARCH_TARGET.md` (V3.0 + Day-0 段) | HANDOVER §7 + README 路径 3.5 |
| **跨 expert 共享层（OL rules、hook 模板、state_machine skeleton）** | `src/experts/_shared/` + `_shared/README.md` | `_shared/README.md` 本身 + HANDOVER §7.1 |
| **项目指令（给 Claude Code 用）** | `CLAUDE.md` | README 的仓库布局 |
| **本 convention（给贡献者用）** | `docs/DOCS-CONVENTION.md`（本文件） | README → 维护小节 |
| **用户侧 0-交互 skill 使用示例**（针对具体版本 / 具体场景） | `docs/examples/<skill>-<trigger>.md`（每个示例独立文件；zero-interaction reproduce 路径 + 反作弊 verify 步骤） | README → PORT-GUIDE / SKILLS-GUIDE → examples 目录 |

**不要**：
- 把同一信息写在多个地方（会 drift；以上表里每一条都是 **single source of truth**）
- 把内容塞进 README（违反最高原则）
- 新建文档而不挂进 README 的某条链路里（孤岛文档）

---

## 2. 什么时候更新什么

### 规则：每次实质性变更必须更新至少一个文档

| 发生了什么 | 必须更新 | 可能还要更新 |
|---|---|---|
| 发现新 NPU 坑 | `knowledge/npu-patterns.md` 追加 stable ID | 若跨档，加 `npu-adaptation-tasks.md` 任务 |
| 跑通 / 跑挂一个 smoke | `porting-journal.md` 按日期追加 + `scripts/smoke/README.md` 基准表（若基准变了） | HANDOVER 里的 smoke 状态表 |
| 完成一个 cherry-pick / patch | `porting-journal.md` + HANDOVER 的 open items 标记 | 涉及 backward-compat → UPGRADE-DRILL-STATUS |
| 新建 / 修改 skill | 对应 `skills/<name>/SKILL.md` + SKILLS-GUIDE §1 表 + HANDOVER 里的 skill 状态 | skills-design.md（如果是架构级变动） |
| 拿到新 base image | `knowledge/images/<slug>.md`（跑 npu-image-inspect skill）+ `upstream-refs.md` | 可能触发 drill，写 `docs/<target>-upgrade-drill.md` |
| 识别出一个 NPU 适配 gap（档 1/2/3） | `docs/npu-adaptation-tasks.md` 追加 task + 状态 | README 路径 4 如果 scope 变了 |
| 完成一个 adaptation task | 同上（标完成 + 结果） | `npu-patterns.md` 如果留下 stable pattern |
| kb 库的事实发生变化（版本 / 路径 / host 状态） | 对应 knowledge 文件 | HANDOVER |
| 计划 / 方向调整 | HANDOVER §11（候选工作）+ 相关 guide 文档 | design.md 如果是原始需求变了 |
| 文档 convention 本身要改 | **本文件** + HANDOVER 里的 convention 引用 | CLAUDE.md 的 "Working preferences" |

### 绝不能**只**改 README

如果一个状态 / 决策 / 知识要写进文档，先写到上表对应的权威文件，然后（如果需要）在 README 里加一句索引。

---

## 3. 稳定内容 vs Transit 内容

两类文档的角色**严格分开**：

### 稳定内容（stable convention / 设计）

不随 session 状态变动的东西：
- **仓库 / 文件夹 / 文件结构说明** → **仅** 放在 `README.md` 的"仓库布局"段。改布局时改 README，**不要在 HANDOVER 等状态类文档里重复布局树**（会 drift）
- **文档归属 + 更新规则** → `docs/DOCS-CONVENTION.md`（本文件）
- **项目 working preferences**（commit message 风格、语言约定、Discord 同步规则）→ `CLAUDE.md`
- **四条用户路径** → `README.md`
- **每类文档的完整使用说明** → 各自的手册文件（PORT-GUIDE / SKILLS-GUIDE / UPGRADE-DRILL-STATUS 等）

### Transit 内容（session-to-session 变化）

状态快照、当前进度、未结工作 —— 随时变：
- **当前 A3 host 状态 + docker image 版本 + 占用情况** → `HANDOVER.md`
- **当前分支 head commit / 各分支状态** → `HANDOVER.md`
- **open items / P0 / P1 工作清单** → `HANDOVER.md` §6 + `npu-adaptation-tasks.md`
- **下一步候选工作** → `HANDOVER.md` §11
- **当前 session 的 TaskCreate list** → 运行时工具（不进 git）

### 规则

- `HANDOVER.md` **不重复 README 布局**。只在 §1 指向 README + DOCS-CONVENTION 就够
- `HANDOVER.md` **不承载 convention / 规则**。它是状态快照
- 改布局时改 README；改 convention 时改 DOCS-CONVENTION；改状态时改 HANDOVER。三者互不重叠
- 任何开 session 的第一件事 → 读 `README.md` + `HANDOVER.md` + `DOCS-CONVENTION.md`（3 篇）
- HANDOVER 本身要**实时更新**，完成一件事立刻更新对应条目（不要攒到 session 末尾）

---

## 4. Language convention

- **项目文档（README / design / HANDOVER / PORT-GUIDE / SKILLS-GUIDE / 任何 `docs/` 和 `knowledge/` 下的 md）**：默认中文
- **代码注释**：英文
- **Commit message**：英文，且**不要**加 `Co-Authored-By: Claude` / `🤖 Generated with Claude Code`（项目 + 用户 global rule）
- **SKILL.md frontmatter（`name` / `description` / `type`）**：英文（机器读取面稳定）
- **SKILL.md 正文**：英文（skill 是跨项目的抽象，英文更通用）

---

## 5. Cross-reference 规则

- 引用另一个 md 用相对路径：`[锚文本](../knowledge/npu-patterns.md)` 或 `[锚文本](DESIGN.md)`
- 引用同目录文件：`[锚文本](OTHER.md)`（不写 `./`）
- 引用章节：`[锚文本](PORT-GUIDE.md#5-从-0-在一台-a3-host-上跑通-v14-smoke)` 用 anchor
- 所有引用的外部 URL 都要验证可达性，失效了立即去掉或换

---

## 6. 文件命名

- `docs/` 里的主手册用大写简写：`README.md`、`PORT-GUIDE.md`、`SKILLS-GUIDE.md`、`HANDOVER.md`、`DELIVERABLE.md`、`UPGRADE-DRILL-STATUS.md`、`DOCS-CONVENTION.md`
- 归档 / 分析类小写：`porting-journal.md`、`codex-review-*.md`、`transformers-upgrade-drill.md`、`skill-dry-run-<date>.md`
- `knowledge/` 里全部小写 kebab-case：`npu-patterns.md`、`upstream-refs.md`、`smoke-ladder-convention.md`
- `skills/<skill-name>/SKILL.md` 是 skill 的固定入口文件名，名字大写

---

## 7. 给 agent 接手的 checklist

接手本项目第一件事（按顺序）：
1. 读 `README.md`（整篇）—— 理解目标 + 四条用户路径 + scope 三档
2. 读 `docs/HANDOVER.md`（整篇）—— 当前状态 + 未结工作
3. 读本文件（`DOCS-CONVENTION.md`）—— 理解文档怎么组织、哪里该写什么
4. 按任务需要，遵循 place-of-record map 找对应权威文档
5. 开工前 TaskCreate 记录计划；每完成一步立即 TaskUpdate + 更新对应权威文档
6. 向 Discord 同步 milestone + 等输入的点（见 CLAUDE.md / memory `discord_cadence.md`）

---

## 8. 本 convention 的版本

- v0.1（2026-04-20 创建）：首次成文，原因是 README 开始膨胀 + agent 每 session 重新计划文档归属

更新时在此段追加 changelog，不要 inline 修改之前的条目（便于追溯）。
