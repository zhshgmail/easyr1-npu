# a5_ops 设计借鉴笔记（research-only）

> 写于 2026-05-28。研究对象:`/home/z00637938/workspace/a5/a5_ops/`(主分支 `768284f0`)。目的:为 easyr1-npu 抽取可借鉴的工作模型,不复制 a5_ops 的重型 codegen pipeline。

---

## §1 a5_ops 四件套 — 最小工作模型

| 件 | 一句话 | 关键文件 |
|---|---|---|
| **Deterministic Pipeline** | 算子生成走死板的 8 阶段状态机 O1→O6,每阶段有 required_actions / required_artifacts / invariants / prohibited_actions | `docs/workflow/opgen_state_machine.yaml`(canonical YAML)+ `docs/workflow/WORKFLOW_CRITIC_DESIGN.md`(prose 伴生) |
| **Mode-aware FSM** | 子状态机 `phase_o4_states`:6 个活跃态(await_worker/probe/optimizer/fused_optimizer/researcher/det_analyzer)+ 2 终态,iter cap + 入口 artifact 由 YAML 声明;**初始态按 MODE 路由**(benchmark/opgen-cuda/opgen-pytorch→await_worker;optimize→await_optimizer)。`state_machine.py next` 是唯一被授权 append `state_transitions.jsonl` 的接口 | YAML `phase_o4_states` + `phase_o4_initial_state_by_mode` + `src/scripts/workflow/state_machine.py` + DEBT-045 |
| **Feedback Loop / KB** | 每个 op 收尾产出 `knowledge_update.md` + `failures_ledger.md`,由 `/ascendc-knowledge-maintain`(Mode 1 update / Mode 2 scan / Mode 4 --learn)以"max 10 change / max 3 edit per file / never delete-only archive"安全策略并入 KB(OL/EC/PB/Pattern 四类,ID 不可复用)。会话级:`/session-retrospective` 写 retro,跨会话 DEBT-031 聚合,再 codify 为 `/self-critic` 检查项 | `src/skills/ascendc-knowledge-maintain/SKILL.md` + `src/skills/session-retrospective/SKILL.md` + `src/skills/references/{OPERATIONAL_KNOWLEDGE,ERROR_CORRECTIONS,PLATFORM_BUGS,patterns}.md` |
| **Safety Net** | 两层:**workflow_critic**(hook 读 YAML,PreToolUse/PostToolUse 拒绝违反 G1-G7 / phase invariant 的工具调用,exit 2)+ **per-agent Stop hook**(`check_worker.sh` 校验 pybind 纯度 / static check / verification.json schema / Phase-A mtime ordering 等)+ **self-critic skill**(C1-C16,人调用或里程碑前调用,捕 reward hacking / 状态机绕过 / 单 agent 数据点直接进 KB 等) | `src/scripts/workflow/workflow_critic.py` + `src/hooks/v3/*.sh` + `src/skills/self-critic/SKILL.md` |

---

## §2 a5_ops 的「重型确定性 pipeline」长什么样、能 work 的前提

形态:每个算子被压成同一条 8 阶段轨道(parse_args → det_policy → benchmark_sync → reference_provider → progress_init → agent_loop → post_verify → archive_commit),agent 间通过 **artifact 文件**(analysis.md / probe_report.md / optimization_directive.md / verification.json / PROGRESS.md)而非消息传递,任何 phase 跳跃 / 违规 edit / vocab 污染 / 未独立复测都被 critic exit-2 阻断。Worker(kernel-worker)产 5 文件 → orchestrator 检查 artifact + 重测 perf → archive 到 `output/npukernelbench/`,KB 增量从 `knowledge_update.md` 流回 `references/`。

**能 work 的前提**:
1. **任务高度同构**:每个 op = "读 CUDA/PyTorch ref → 写 AscendC kernel → verify 50 case → 比 perf",同一模板能套 30+ 算子(L1 31 个,L2 30 个)。
2. **artifact 是契约**:每阶段产物 schema 固定(verification.json 字段、analysis.md 必填章节),才允许 hook 做 grep 级机械校验。
3. **YAML 是 single source of truth**:SKILL.md 是 prose 伴生,YAML/Python 分歧由 pre-commit drift hook 阻断(用户多次纠正 "YAML wins")。
4. **有可复测的基准**:每个 op 都跑同一份 NPUKernelBench benchmark,perf ratio / precision 50/50 是一个数字标尺。
5. **失败有结构**:每次踩坑能归入 EC/PB/OL/Pattern 之一,KB ID 单增不复用。

---

## §3 easyr1-npu 当前差距

| 件 | 已有 | 缺 |
|---|---|---|
| **Pipeline** | `docs/_meta/ROADMAP.md` + `workspace/T32_tilelang_rescue/ROADMAP.md` 有 task 表 + DAG + 状态字段(TODO/IN-PROGRESS/BLOCKED/DONE/PARKED);各 skill SKILL.md 内部有 phase 划分(A/B/C/D);`src/skills/orchestrators/npu-port/` 是个串接 orchestrator | 没有跨 skill 的机器可读 phase YAML,没有 phase 跳跃 hook 校验;每个 skill 自己写 phase,无统一 critic |
| **FSM** | task 字段是状态,但没有 transition 表,没有 mode 路由(easyr1-port、torch-npu-day0、vllm-ascend-day0 各自维护状态),也没有 `state_transitions.jsonl` 类的可审计 transition 日志 | 缺 mode-aware initial-state(day0/upgrade/integrated 模式入口不同,但目前没声明 spec) |
| **Feedback Loop / KB** | `docs/_meta/kb/porting_lessons/`(13 条 lesson,有 `_schema.md` + `index.md`)+ `docs/_meta/kb/challenge_patterns/`(12 条)+ `_shared/references/OPERATIONAL_KNOWLEDGE.md`(OL-01..OL-27 grep 索引)+ `_shared/references/ANTI_PRESSURE_PROTOCOLS.md`(P1..P8) | **没有"postmortem → KB → 下次 skill prompt"的闭环机制**;新 OL/lesson 靠人手工写;没有 a5_ops `knowledge_update.md` 那种 per-task 强制产出契约;没有 `/ascendc-knowledge-maintain` 那种 dedup + generalization + archive 安全策略;没有 retro skill 标准化产物 |
| **Safety Net** | `porting-self-challenge` skill(10 问)、`ANTI_PRESSURE_PROTOCOLS` P1-P8、ROADMAP 写得很硬 | **没有 hook**,critic 检查靠 LLM 自觉 cite 协议;没有 per-skill Stop hook 校验 artifact schema;没有 self-critic 那种"承认错误后又重犯"的 substring 检查;没有 `pre-commit drift hook` 阻 SKILL/YAML/spec 三方漂移 |

具体引用:
- 已有 ROADMAP DAG: `/home/z00637938/workspace/easyr1-npu/repo/workspace/T32_tilelang_rescue/ROADMAP.md`(50+ task 表 + DAG)
- 已有 KB: `/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/kb/porting_lessons/index.md`
- 已有 anti-pressure: `/home/z00637938/workspace/easyr1-npu/repo/src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md`
- 缺的关键文件:没有 `docs/_meta/workflow/*.yaml`、没有 `src/scripts/workflow/critic.py`、没有 `src/hooks/`、没有 `src/skills/self-critic/`

---

## §4 「重型 pipeline 适合 easyr1-npu 吗」

**结论:不全套搬,只搬其中 4 个机制。**

理由:a5_ops 跑的是 codegen "同一模板套 60 算子",任务同构 → 重型 FSM + artifact 契约能均摊设计成本。easyr1-npu 是"上游 drift + 集成栈跑通",每个 day-0 / port 任务的 bug 形态都不同(triton-ascend 是 LLVM 版本 skew、vllm-ascend 是 plugin init order、torch_npu 是 ABI 改动),这种变化型工作的 phase invariant 写不动死板的 grep 黑白名单。**但** mode-aware FSM(day0 / upgrade / integrated 三模式入口不同)、Stop-hook 校验 outcome artifact schema、postmortem→KB 闭环、self-critic 这四个机制是任务无关的,值得搬。

---

## §5 可借鉴的 5 件具体事项

1. **machine-readable workflow spec + pre-commit drift hook**
   a5_ops 在 `docs/workflow/opgen_state_machine.yaml` 用 YAML 声明 phase / invariant / prohibited / required_artifact,SKILL.md 是 prose 伴生,YAML wins。在 `/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/workflow/`(新建)给 `npu-port` / `easyr1-port` / `torch-npu-day0` / `vllm-ascend-day0` 各写一份 YAML phase 表 + 一个 pre-commit hook 阻"SKILL.md 改了 YAML 没改"的漂移。解决痛点:目前 ROADMAP 状态字段是人工维护,phase 转换无机器校验,踩了 P1(user-watching pressure)就跳过 verify。

2. **per-skill outcome artifact schema + Stop hook**
   a5_ops `src/hooks/v3/check_worker.sh` 在 worker exit 时强校验 verification.json key、pybind 纯度黑白名单、analysis.md 必填章节。在 `/home/z00637938/workspace/easyr1-npu/repo/src/hooks/` 给每个 day-0 skill 加一个 Stop hook,校验它声明的 outcome A/B/C 必须配 `log_path` + numeric evidence(OL-02 已经在 KB 但没机制兜底)。直接解决 cross-layer-007(walk-through is not real run)的复发。

3. **`/session-retrospective` + cross-session aggregate → 新 self-critic 条目**
   a5_ops 流水线:每会话末 `/session-retrospective` 写定型 retro → `DEBT-031` 跨 session 聚合 → 出 `/self-critic` 新 check 条目(C7-C16 全是这样来的)。easyr1-npu 已经有 ANTI_PRESSURE_PROTOCOLS P1-P8 + porting-self-challenge 10 问,但**这两份是"出厂硬编码"**,没有"踩坑→retro→自动加条"的回路。在 `/home/z00637938/workspace/easyr1-npu/repo/src/skills/_shared/` 加 `session-retrospective` skill,用 a5_ops 的 skeleton(verbatim quotes / pattern-hit scorecard / 能做但没做 + root cause),输出到 `docs/_meta/kb/retrospectives/`,每月人手 review 后增 P9/P10。

4. **KB update 安全策略(max-10/max-3/never-delete/never-rename-ID)+ generalization check**
   a5_ops `/ascendc-knowledge-maintain` Mode 1 的"先 generalization check 再 dedup 合并"对 KB 质量起决定作用(防止 OL 写成 case-specific 而失去复用价值)。easyr1-npu `OPERATIONAL_KNOWLEDGE.md` 是手工加,没有 dedup/generalization 流程。把 a5_ops Mode 1 的 7 步骤(读 update.md → check failures_ledger → generalization 评估 → dedup → 写报告)做成 easyr1-npu 的 `/kb-maintain` skill,触发条件:每个 day-0 / port skill 完成时强制产出 `knowledge_update.md`(SKILL.md frontmatter 加 artifact 声明)。

5. **state_transitions.jsonl 风格的可审计 transition 日志**
   a5_ops 把 phase O4 的每次 agent 切换 append 到 `workspace/{op}/state_transitions.jsonl`,critic 用它做"当前打算 spawn 的 agent 是不是和日志 tail 的 to_state 匹配"校验(C15)。easyr1-npu 的 ROADMAP 状态字段是覆盖式写(只看最后一次),没有 transition 历史 → 出 bug 后无法回放"为什么从 IN-PROGRESS 直接跳 DONE"。在 `workspace/T32_tilelang_rescue/`(以及未来 npu-port 任务的每个 session workspace)加 `state_transitions.jsonl`,每次 task 状态改写时 append 一行 `{ts, task_id, from, to, evidence_path}`,让 challenge skill 能机械检查 "claim DONE 时是否有 evidence_path"。

---

## §6 `ascendc-op-gen` / kernel-generator 这个算子生成工具我需要吗

**结论:近期(2026-Q2-Q3 内的 T32 rescue + day-0 持续推进)不需要;远期(miles tilelang 算子 R-KA-16 上游修完后想批量补 sparse_mla 变体 / 新增 DSv4 衍生算子)可能需要,但门槛高。**

理由:
- `ascendc-op-gen` 生成的是 **AscendC SIMT/SIMD 算子源码**(`.h` + `.cpp` + `pybind11.cpp` + `tiling.h` + `model_new_ascendc.py`),输入是 CUDA `.cu` 或 PyTorch `model.py`。T32 当前痛点是 **tilelang DSL 在 MLIR 后端编译时 bishengir HIVM pass 把 iter_args 塌错**(R-KA-16, AscendNPU-IR #251)—— 这是**编译器层 bug**,不是缺一个 AscendC 算子。当前 4 个 tilelang 算子 (`lighting_indexer_fwd/bwd`, `sparse_mla_fwd/bwd`) 都是 tilelang DSL 写的,如果改成 AscendC 是一次性大重写而非 "用 codegen 工具生成"。
- a5_ops 的 op-gen 工具链假定基线是 NPUKernelBench 那 60 算子,有 `INPUT_CASES` 标注、有 reference Model 类,easyr1-npu 这边没有这种基准集。
- 远期若要给 miles 加新 sparse 变体(比如 sparse_mla_v2 with 不同 topk pattern),且 R-KA-16 已修,且想绕开 tilelang DSL 直接出 AscendC kernel,那时 op-gen 工具的 "schema-driven case_gen + 自动 verify 50 case + KB 累积 EC/PB" 的产能优势才显现。但 cold-start 这个工具到能跑通需要先建好 case_gen schema + benchmark harness + KB 索引,本身是 2-3 周的投资。

短期更值得借鉴的不是工具本身,而是它**产生 KB 的方式**(§5.4)。

---

## §7 分阶段引入计划(1-day / 1-week / 1-month)

### 1-day(可立刻做,零依赖)

- 在 `repo/docs/_meta/workflow/` 新建一个目录,把现有 `easyr1-port` / `npu-port` 两个 orchestrator skill 的 phase 序列**手抄成 phase_a/b/c/d 列表 + required_artifact** 写进 YAML(不接 critic,只是落契约)。这一步即使没机器校验,也能让"我现在在 phase B 但还没写 outcome A/B/C 文件"暴露给 LLM 自己。
- 在 `workspace/T32_tilelang_rescue/` 加 `state_transitions.jsonl`,从今天起每次 ROADMAP 状态改写都 append 一行。零工具,只是 `echo >> file`,但可以回放。
- 把 §5.1 / §5.5 写进 `docs/_meta/ROADMAP.md` §6 backlog,确保不丢。

### 1-week(单 skill 验证)

- 在 `repo/src/skills/_shared/` 加一个 **`session-retrospective` skill**(直接借 a5_ops `src/skills/session-retrospective/SKILL.md` 的 skeleton + 表格 schema),输出到 `repo/docs/_meta/kb/retrospectives/YYYY-MM-DD_<slug>.md`。先手工调用一周,看产物质量。
- 给一个 **小目标 skill**(比如 `torch-npu-day0`)加 Stop hook:`repo/src/hooks/check_torch_npu_day0_outcome.sh`,只校验"如果 outcome=A 必须有 log_path + import 成功的 stdout 行",exit 2 拒绝不带证据的 PASS 声明。在该 skill 的 SKILL.md frontmatter 加 hook 配置。这步是把 §5.2 拿到一个 skill 上验证。
- 一周末 review 这个 skill 在 hook 下的产出,看 false positive / 摩擦是否可接受。

### 1-month(全栈引入)

- §5.1 完整化:为每个 day-0 / port skill 写 YAML,加 pre-commit drift hook + `repo/src/scripts/workflow/critic.py` 雏形(只读 YAML 做 phase artifact 存在性校验,不必复制 a5_ops 全部 G1-G7)。
- §5.4 完整化:加 `/kb-maintain` skill,每个 day-0 / port skill 完成时强制产 `knowledge_update.md`,由 `/kb-maintain` 并入 `OPERATIONAL_KNOWLEDGE.md` / `porting_lessons/`,执行 max-10 / max-3 / never-delete 安全策略。
- §5.3 + retro 数据积累后,从 retro 文件里聚合出 P9-P12 候选,加入 `ANTI_PRESSURE_PROTOCOLS.md`,并把 `porting-self-challenge` 的 10 问扩成 15 问。
- 月末:用一个真实的"上游 drift 到达 → 触发 day-0 → 产 outcome → KB 沉淀"全链路 cold-drive 验证整套机制是否实际拦住了一次"P1 想跳过 verify"。
