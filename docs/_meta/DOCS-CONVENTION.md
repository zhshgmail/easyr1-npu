# 文档组织 convention

**给本仓贡献者（人 + agent）的规则**。文档改动遵守这些 convention，避免 drift / 重复 / 孤岛。

---

## 0. 最高原则

**README 是入口 + 索引，不是内容仓库。**

- 重要文档从 README 出发 ≤ 2 跳可达
- README 只导航；内容写在专门文档
- 内容塞进 README 即违反

---

## 1. 每类信息放哪里 — place-of-record map

| 信息类型 | 唯一归属 |
|---|---|
| 项目入口 + 当前状态总览 | [`README.md`](../../README.md) |
| 客户一页 quickstart | [`ONBOARDING.md`](../../ONBOARDING.md) |
| 客户跑 EasyR1 (v1) | [`docs/easyr1/PORT-GUIDE.md`](../easyr1/PORT-GUIDE.md) |
| 客户跑 EasyR1 (v2 integrated) | [`docs/easyr1/PORT-GUIDE-v2-integrated.md`](../easyr1/PORT-GUIDE-v2-integrated.md) |
| 整体架构 + 流程（mermaid） | [`docs/_meta/ARCHITECTURE.md`](ARCHITECTURE.md) |
| 4 fork + EasyR1 fork 的权威分支表 | [`docs/_meta/UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md) |
| Slash command 使用 | [`docs/_meta/SKILLS-USAGE.md`](SKILLS-USAGE.md) |
| 术语表 | [`docs/_meta/GLOSSARY.md`](GLOSSARY.md) |
| 上游维护者面 PORT GUIDE | `docs/{vllm-ascend,torch-npu}/PORTING-GUIDE.md` |
| 上游 PR 资料 | `docs/transformers/PR_MATERIAL_*.md`、各 fork 分支根目录的 `PR_MATERIAL.md` |
| NPU 坑 stable ID（CP/BUG/ENV/OPS） | [`knowledge/npu-patterns.md`](../../knowledge/npu-patterns.md) |
| 镜像里的依赖快照 | `knowledge/images/<slug>.md`（每 image 一份） |
| 上游 ref 对应表 | [`knowledge/upstream-refs.md`](../../knowledge/upstream-refs.md) |
| Smoke 梯子约定 + 基准 | [`knowledge/smoke-ladder-convention.md`](../../knowledge/smoke-ladder-convention.md) |
| 跨层移植教训 | [`docs/_meta/kb/porting_lessons/`](kb/porting_lessons/) |
| Self-critic 11 模板 | [`docs/_meta/kb/challenge_patterns/`](kb/challenge_patterns/) |
| 项目级 Claude Code 指令 | [`CLAUDE.md`](../../CLAUDE.md) |
| 本 convention | 本文件 |
| Skill 权威说明 | `src/skills/<area>/<expert>/SKILL.md` |
| Skill 共享 OL 规则 / pattern / workflow | `src/skills/_shared/` |

**禁止**：

- 同一信息出现在多份文档
- 把内容塞进 README
- 新建文档不挂入 README 或某索引 doc

---

## 2. 什么时候更新什么

| 发生了什么 | 必须更新 |
|---|---|
| 发现新 NPU 坑 | `knowledge/npu-patterns.md` 追加 stable ID |
| 跑通 / 跑挂一个 smoke | 对应 skill 的 KB_INDEX.md case 段 |
| 新建 / 修改 skill | 对应 `SKILL.md` + 在 SKILLS-USAGE 表里更新命令名 |
| 拿到新 base image | `knowledge/images/<slug>.md`（用 `/npu-image-inspect`）+ `upstream-refs.md` |
| 上游 fork 分支前进 | `docs/_meta/UPSTREAM_FORKS.md` |
| Architecture / 端到端流程变了 | `docs/_meta/ARCHITECTURE.md` |
| 出现新跨层教训 | `docs/_meta/kb/porting_lessons/<layer>-NNN-<slug>.md` 新建 + index 索引 |

---

## 3. 客户文档 vs 内部文档

**客户文档**（README / ONBOARDING / PORT-GUIDE / SKILLS-USAGE / PORTING-GUIDE / PR_MATERIAL）：

- 只放**当前已验证准确的事实**
- 删除：session 标签（T22.x）、worklog 措辞、内部 dev 流程、过期 baseline 数值
- 链接的目标必须存在于 git tree 当前 HEAD

**内部文档**（KB / lessons / SKILL.md / CLAUDE.md）：

- 可以保留 session 标签和迭代历史（用于追溯）
- 但跨 session 留下来的 lesson 必须按 `docs/_meta/kb/porting_lessons/_schema.md` 写
- 客户不读这些；agent 接手前才读

任何 session 内的工作流程文档（HANDOVER / WORKLOG / 周报）→ **不进 customer-facing 树，超过 1 周的归档**到 `docs/_archive/obsolete_<date>/`。

---

## 4. Language convention

- **项目文档**（README / ARCHITECTURE / port guides / 任何 `docs/` + `knowledge/`）：默认中文
- **代码注释**：英文
- **Commit message**：英文，**不要** Claude 相关后缀（用户 global rule）
- **SKILL.md frontmatter** (`name` / `description` / `type`)：英文
- **SKILL.md 正文**：英文（skill 是跨项目抽象）

---

## 5. Cross-reference 规则

- 相对路径：`[文本](../knowledge/npu-patterns.md)`
- 同目录：`[文本](OTHER.md)` 不写 `./`
- 章节锚点：`[文本](PORT-GUIDE.md#5-跑通-v14-smoke)`
- 外部 URL 验证可达；失效立即去掉或换

---

## 6. 文件命名

- 主手册大写连字符：`README.md`、`ONBOARDING.md`、`PORT-GUIDE.md`、`PORTING-GUIDE.md`、`PR_MATERIAL_*.md`、`ARCHITECTURE.md`、`UPSTREAM_FORKS.md`、`DOCS-CONVENTION.md`、`GLOSSARY.md`、`SKILLS-USAGE.md`
- KB / 知识文件 kebab-case：`npu-patterns.md`、`upstream-refs.md`、`smoke-ladder-convention.md`、`cross-layer-007-walk-through-is-not-real-run.md`
- Skill 入口：`src/skills/<area>/<expert>/SKILL.md`

---

## 7. 给 agent 接手的 checklist

按顺序：

1. 读 [`README.md`](../../README.md)
2. 读 [`docs/_meta/ARCHITECTURE.md`](ARCHITECTURE.md)（含 mermaid）
3. 读本文件（DOCS-CONVENTION）
4. 读 [`docs/_meta/UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md)
5. 按任务需要查对应权威文档
6. 开工前 TaskCreate；完成一步即 TaskUpdate + 更新对应权威文档
7. Milestone / 等输入 → Discord 同步（chat_id `1494825170399924366`）

---

## 8. 本 convention changelog

- v0.1（2026-04-20）：首次成文
- v0.2（2026-04-28）：T26 文档整理；移除已归档文档的引用；place-of-record 表精简到当前活跃文档；客户文档 vs 内部文档分类
