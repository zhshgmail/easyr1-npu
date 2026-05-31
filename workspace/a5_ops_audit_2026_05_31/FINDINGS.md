# a5_ops preflight + KB 学习 — 2026-05-31

> 来源:用户 Discord 2026-05-31 09:20 「完善文档和 preflight 体验,看看 a5_ops 怎么做的」
> 范围:`/home/z00637938/workspace/a5/a5_ops/`(pull 到 `3b3e3898`)
> 对比对象:`easyr1-npu`(本仓)

## 数据对比

| 指标 | a5_ops | easyr1-npu | 比例 |
|---|---|---|---|
| KB references 文件总数 | 662 | 40(27 porting_lessons + 13 challenge_patterns) | 16.6× |
| OL 条目(operational knowledge) | 195 | 0(分散在 27 cookbook 里)| — |
| EC 条目(error corrections) | 0 文件(已合并到 OL) | 0 | — |
| Patterns 文件 | 14 个 domain | 0 | — |
| Skills 总数 | 30 | 16 | 1.9× |
| Preflight 脚本 | 多个(3+ Phase O Gate + 1 hook 安装 + 1 ref sanity) | **0** | ∞ |

a5_ops 的 KB 是 4 年项目沉淀,easyr1-npu 是 6 周。完全照搬不现实,但 **结构** 和 **机制** 值得借鉴的几个点很清晰。

---

## 1. Preflight — a5_ops 怎么做

### 1.1 三层 preflight

**Layer A: `/aog-preflight` skill**(用户主动触发,首次和换环境时)
- 文件:`src/skills/aog-preflight/SKILL.md`
- 做的事:
  1. 读 / 创建 `workspace/.ascendc_env`(多 TARGET 配置 a3/a5/a2)
  2. SSH 通畅、容器 running、NPU 可见、CANN 路径有 bisheng、SOC 匹配
  3. Benchmark 框架在 docker 里能找到
  4. **engineering baseline 8 个 assertion**(从 `docs/baseline/environment_baseline.yaml` 读)— 库大小 + 符号数 + 必需文件 + bisheng macro 支持等
  5. 输出 ASCII table verdict,active TARGET 不绿就**拒绝继续**

**Layer B: Phase O0 entry-gate**(orchestrator 触发,在 spawn worker 前)
- 文件:`src/scripts/orchestrator/phase_o0.py`
- 检查 6 个 KB 文件 + 2 个 deploy script 必须 reachable + sane
- 失败 → `BLOCKED` verdict,refuse to spawn agents
- 这一层是**程序内的 sanity check**,不是 shell 调用

**Layer C: Phase O2.6 reference-sanity**(spawn worker 前的最后保险)
- 文件:`src/scripts/preflight_reference_sanity.py`
- 在 isolated subprocess 里跑 reference model 的 `forward()`,**不带 kernel**
- 每个 case 独立进程:NPU stream poisoning 不会跨 case 污染
- ≥ 100% PASS → CLEAN;90%-99% → WARN;<90% → ABORT
- 防 OL-87 那种「benchmark 数据生成器吐了 OOB index → reference 自己挂 → 把 worker 后续 failure 误判成 kernel bug」

### 1.2 关键设计原则

| 原则 | 怎么体现 |
|---|---|
| **fail loud and early** | preflight ABORT 直接 exit 2,refuse 进入 orchestrator |
| **isolated probe per case** | per-case subprocess,防 stream poisoning cascade |
| **不 paper over** | `feedback_no_paper_over_env_problems.md` discipline:worker 没权改 baseline,只能 preflight + 用户 |
| **verdict 是 contract** | `CLEAN` / `WARN_EDGE_CASES` / `ABORT_HARNESS_BROKEN`,downstream 用 verdict 字符串决策 |
| **JSON 输出 +人类摘要** | `preflight_reference.json` 给 agent,ASCII table 给人 |
| **首次配置交互 + 之后读文件** | `.ascendc_env` 缓存,非 `--reconfig` 不重问 |
| **Mandatory + Optional 分级** | SSH/container/NPU/CANN/SOC 是 MANDATORY;Codex/OpenCode 是 OPTIONAL |

### 1.3 我们差什么

**我们 0 个 preflight script**。当前用户跑 `/npu-adapt-assist` 体验:
- 没检查 KB dir 在不在
- 没检查 Python 版本
- 没检查 stdlib 模块都 OK
- 没检查输入格式合法
- 没检查 cwd 是否在 repo root

新用户 clone 仓库 + 装 skill,第一次 `/<skill>` 失败可能要 debug 半小时才找到「KB dir 不在 site-packages 路径上」这种小事。

---

## 2. KB 组织 — a5_ops 怎么做

### 2.1 5 层架构

```
src/skills/references/
├── KB_INDEX.md                    # 搜索入口表
├── shared/                        # 跨 skill 通用
│   ├── ALWAYS_LOADED_RULES.md     # 无条件加载
│   ├── ANTI_PRESSURE_PROTOCOLS.md # P1-P8 LLM 漂移模式
│   ├── exploration/
│   └── retrospectives/
├── target/                        # per-target language/runtime
│   ├── ascendc/
│   │   ├── ASCENDC_LANGUAGE_REFERENCE.md
│   │   ├── ASCENDC_API_CATALOG.md
│   │   ├── OPERATIONAL_KNOWLEDGE.md  # 195 OL 条目
│   │   ├── ERROR_CORRECTIONS.md      # EC-1..EC-67
│   │   ├── ROOFLINE_MODEL.md
│   │   ├── SIMT_VS_SIMD_DECISION.md
│   │   └── patterns/
│   │       ├── PATTERN_INDEX.md      # 路由表
│   │       └── domains/              # 14 个 domain 文件
│   │           ├── precision.md
│   │           ├── platform_compat.md
│   │           └── ...
│   ├── tilelang/
│   └── triton/
├── hardware/                      # per-chip 硬件 spec
│   ├── source/
│   ├── target/
│   └── probe_findings/
├── plugin-scope/                  # 跨 op 的 plugin / port 知识
│   └── port_a3/
└── workbench_imports/             # 从外部 import 的辅助数据
```

### 2.2 三种条目类型

**OL(Operational Lessons,195 个)**:跨 op 通用的 process rule 或 universal trap
- 例:`OL-80 API existence must be grepped from catalog, never invented`
- 都在 `OPERATIONAL_KNOWLEDGE.md` 一份大文件里(便于 grep)

**EC(Error Corrections)**:编译 / 运行时 error → fix recipe
- 例:`EC-22 overlap-tail vec pipeline race`
- 一文件 list,grep by error keyword

**F-P / P-P(Patterns)**:正面 / 反面 algorithm patterns
- 例:`F-P1 bf16 precision handling`,`P-P9 SIMT vs SIMD decision framework`
- 在 `patterns/PATTERN_INDEX.md` 路由表,domain 文件存细节
- 每个 domain 文件有 frontmatter `chip_scope: all|a5|a3|...`(选择性加载)

### 2.3 关键设计原则

| 原则 | 怎么体现 |
|---|---|
| **入口 grep,实体加载** | KB_INDEX.md 是 keyword 表,worker grep keyword 后才 load 实体文件 |
| **alias 多名字** | 同一概念多种表述同时挂(双缓冲 / DoubleBuffer / pipeline overlap)防 worker 想不到关键词 |
| **chip_scope frontmatter** | per-target 文件标 `chip_scope: a5 only`,选择性激活 |
| **MANDATORY vs ON-DEMAND** | ALWAYS_LOADED_RULES.md 强制读,其他按需 grep |
| **数字 ID 单调递增** | OL-1..OL-196 永远不重用、不重写 ID,旧 ID 死了就 deprecated 不删除 |
| **每条带 applies_to/verified_on/unverified_on tags** | KB-merge gate 强制要求 scope tag |
| **新 OL 走 kb_manager** | finalize 阶段强制 kb_manager 处理 `knowledge_update.md`,没 KB merge 不允许 done |

### 2.4 我们差什么

| 维度 | a5_ops | easyr1-npu | 差距分析 |
|---|---|---|---|
| **入口索引** | KB_INDEX.md 表格 + keyword alias | `index.md` linear list,无 alias | **小差距,加 alias 列即可** |
| **scope tag** | `applies_to: a3` / `verified_on: a5` 强制 | 没有 | **中等,值得加** |
| **ID 命名单调** | OL-1..OL-196 不重用 | porting-lessons 用 `<layer>-<NNN>` 也不重用 ✓ | **我们做了** |
| **per-skill ALWAYS_LOADED** | 每条 skill 强制读 ANTI_PRESSURE + ALWAYS_LOADED | 0 | **新 skill 没装这个礼仪** |
| **patterns vs lessons 分层** | 正面 patterns / 反面 lessons / runtime ECs 三类分开 | 都塞 porting_lessons | **大差距,但短期不痛** |
| **KB-merge gate** | finalize 不让通过未走 kb_manager 的工作 | 无 | **流程层,先放着** |

---

## 3. 我们应该借鉴什么(短期 ROI 排序)

### 高 ROI(本 session 落地)

1. **为 `/npu-adapt-assist` 加 preflight**
   - 当前 retrieve.py 没任何 prereq 检查,失败信息散落在 stack trace 里
   - 加 `scripts/preflight.sh`:KB dir 在不在 / python3 版本 / 必需文件存在 / 输入有效
   - 失败给 actionable verdict(CLEAN / WARN / ABORT)+ 修复 hint
   - 顺便给 `cold_drive_validate.sh` 加 preflight check

2. **KB index 加 alias / keywords 列**
   - a5_ops 的 KB_INDEX 每行有 "Keywords/Aliases" 列
   - 我们的 `porting_lessons/index.md` 是 linear bullet,grep 友好度低
   - 加一张 keyword grep 表(同时保留现有 bullet 索引)
   - retrieve.py 已经在用 frontmatter trigger/symptom 做 keyword match,可以输出建议关键词

3. **新 skill 引入 ANTI_PRESSURE 引用**
   - 我们有 `_shared/references/ANTI_PRESSURE_PROTOCOLS.md`(P1-P8)
   - `/npu-adapt-assist` SKILL.md 没引用
   - 应该在 SKILL.md 顶部加一行「mandatory: read ANTI_PRESSURE before any action」

### 中 ROI(下次 session 或 follow-up)

4. **统一 ONBOARDING preflight**
   - a5_ops 用 `/aog-preflight` 做 first-run 引导(交互式 + 配置文件 + 验证 8 项)
   - 我们当前没有等价物;`ONBOARDING.md` 是文档,不是 skill
   - 做一个 `/easyr1-preflight` skill:检查 A3 ssh、container、CANN、image SHA、git submodule

5. **scope tag 加到 KB frontmatter**
   - 我们 27 条 cookbook 都没标 `applies_to: <upstream-version>` / `verified_on: <image-tag>`
   - 加上之后 retrieve.py 能基于 version 排序 / 过滤
   - 暂不重写已有 27 条,只在 schema 里加可选字段,新条目用

### 低 ROI(后续考虑)

6. **patterns vs lessons 分层**
   - a5_ops 的 patterns 是 algorithm patterns(写代码时学的)
   - 我们的 lessons 都是 bug-driven(踩坑后写的)
   - 短期没有正向 algorithm pattern 要积累,先不分层

7. **KB-merge gate / 强制 kb_manager**
   - a5_ops 这套是大规模团队保持 KB 不退化的机制
   - 我们 1 人维护,不痛

---

## 4. 直接 actionable 改动 list(本 session)

| # | 改动 | 文件 | 预计 |
|---|---|---|---|
| 1 | 写 `scripts/preflight.sh` for `/npu-adapt-assist` | `src/skills/npu-adapt-assist/scripts/preflight.sh` | 1 文件 |
| 2 | 修改 `retrieve.py` 入口调用 preflight | `src/skills/npu-adapt-assist/scripts/retrieve.py` | small edit |
| 3 | KB index 加 keyword 表 | `docs/_meta/kb/porting_lessons/index.md` | top section |
| 4 | SKILL.md 顶部加 ANTI_PRESSURE 引用 | `src/skills/npu-adapt-assist/SKILL.md` | 5 行 |
| 5 | `_schema.md` 加可选 `applies_to` / `verified_on` 字段 | `docs/_meta/kb/porting_lessons/_schema.md` | small edit |
| 6 | Cold-drive 重跑确认改动后 retrieval 仍 3/3 PASS | — | run |
| 7 | README 加一行说明 preflight 入口 + KB index 增强 | `README.md` | 1 行 |

不做的:不照搬 a5_ops 的 ALWAYS_LOADED 强制 / phase O0 entry gate(我们没有 orchestrator 架构 / KB-merge gate(规模不痛)/ patterns 分层(没正向 patterns 累积))。

---

## 5. 给用户的一句话总结

a5_ops 的 preflight 是 **三层 fail-loud-early gate**:用户层 `/aog-preflight` 做配置 + 8 项 baseline 验证;orchestrator 层 Phase O0 在 spawn agent 前查 KB 完整性;运行层 Phase O2.6 在 isolated subprocess 里 sanity check reference model。KB 是 **5 层 + keyword 索引 + scope tag**:`shared / target / hardware / plugin-scope / workbench_imports`,每条都有 `applies_to` / `verified_on` 让 retrieval 可以按 version 过滤,前置 `KB_INDEX.md` 是 grep-friendly 关键词表。**借鉴本 session 做 4 件事**:`/npu-adapt-assist` 加 preflight script + retrieval 入口先跑 preflight + KB index 加 keyword 表 + SKILL.md 顶部引用 ANTI_PRESSURE。
