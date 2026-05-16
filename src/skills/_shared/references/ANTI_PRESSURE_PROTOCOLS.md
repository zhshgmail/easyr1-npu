# Anti-Pressure Protocols（cross-skill 无条件加载）

> **所有 day-0 / port skill 必须在 Phase A 加载本文件，并在决策点重新引用相关 Px。**
>
> 这里收录的不是技术规则，而是 **LLM agent 在压力下自我合理化绕过规则的失效模式**。这些不是 easyr1-npu 特有——跨项目通用，因为压力来源是模型架构层面的。
>
> 每个 P 项带 **incident anchor**（本仓过去具体事故）+ **detection signal**（如何在自己的 thought stream 里识别"我现在正在 P-X 模式"）。**抽象警告会被忽略，具体事故锚点能触发识别**。
>
> 借鉴自 [a5_ops/src/skills/references/shared/ANTI_PRESSURE_PROTOCOLS.md](https://gitcode.com/zhengshencn_hwca/a5_ops)（2026-05-15 cloned + ported）。
>
> 最后更新：2026-05-15（T29.3 首次成文）。

---

## 加载契约

- **Session start**：每个 day-0 / port skill 的 "Required reading" 列表里必须含本文件。一次性加载，便宜。
- **Decision-point checkpoint**：高 leverage 时刻**必须重新引用相关 Px**。仅在 session start 加载不够——长 session 会被 compress 掉，压力来临时已经看不见。

**High-leverage 时刻清单**（skill 必须在这些点显式 cite Px）：

| 时刻 | 引用 |
|---|---|
| 准备 spawn agent / sub-skill | P3 + P8 |
| 准备 emit outcome A / B / C-patch / C-report | P1 + P5 + P7 |
| 准备 emit "outcome 改不动，是上游 limitation" | P5 |
| 准备跳过 cold-drive replay / 跳过 on-A3 import smoke | P1 |
| 准备 nohup / `&` / 直接 Agent 跳过 Skill | P8 |
| 准备 manual workaround 而非 fix root cause | P6 |
| Server 起来但 inference 失败时声称 "PASS"（T28 反例）| P7 |

---

## P1：User-watching pressure（用户在看 → 跳过慢步骤）

**Urge**：用户在等结果，我要快速产生可见进展。

**Rule**：cold-drive replay + on-A3 import smoke + 独立 perf re-run **不是可选的**。它们是 agent **唯一**能判断"工作是真的"的信号。跳过一次 = 同一任务后面要重做，代价是省下时间的 10×。

**Incident anchor**：
- 2026-04-27 T22 session：claim "Skill chain end-to-end PASS" 基于 walk-through agent reports；T25 cold-drive 替换 4 个 fresh agent 实际 run，**3/4 outcome 都和原 claim 不一致**（vllm-ascend shim 3 真 fail + transformers 5.4 有 2 处 additive drift + helper script ASCEND_RT_VISIBLE_DEVICES bug）。原 T22 session 是被 P1 拉过去的。

**Detection signal**：
- 我在写 "PASS" 但还没 cite `log_path: /tmp/<owner>/...` + 一个具体数值
- 我在跑 single happy-path 而不是 cold-drive replay
- "用户在等，先报一个结果"——这个念头本身就是 P1

**Fix**：cite `OL-09` + `cross-layer-007`；继续跑到能 cite log path + numeric evidence 为止。

---

## P2：Context-filling pressure（不知道下一步 → 填充 plausible code）

**Urge**：context window 还有空间，我要继续 produce token / 写 code 表示在工作。

**Rule**：当 phase 没明确指令时，**stop and ask** > **fill plausible**。Plausible code 一旦走入下游会留 fake artifact，比 honest "我卡在 X" 难纠 10×。

**Incident anchor**：
- 2026-04-27 T22.4 row 3 retry：第一次 base image 选错（`vllm-ascend:releases-v0.13.0-a3`，vllm 0.13 pre-introduction of `SpecDecodeBaseProposer`），shim 3 import 失败但报"3/3 PASS"——是 P2 让 agent 把 import error 解读成"看起来差不多 work 了"。

**Detection signal**：
- 我在描述某个 artifact 但说不出 path + sha
- 我在用 "应该 / 大概 / probably / look like" 这种 hedge 词来 ship outcome

**Fix**：cite OL-02 + OL-09；先回填 evidence 再继续。

---

## P3：Spawn-agent pressure（context 没准备好就 spawn）

**Urge**：让 sub-agent / sub-skill 处理，主线 context 不用承担。

**Rule**：spawn 前**必须**verify：(a) brief 文件存在 + 包含 task-specific 信息；(b) 必读文件已在 brief 里 enumerate；(c) success criterion 是 mechanical（grep / file-exists / numeric），不是 LLM-judgment。

**Incident anchor**：
- 2026-04-25 T18：spawn `vllm-day0-worker` 时 brief 没写 OL-08 edit scope，worker 跑去改 community vllm 代码（直接违反 day-0 G1 invariant）。是 P3 让 agent 觉得 "worker 自己会读 OL"——worker 在 sub-context 里**看不见** main session 的 OL。

**Detection signal**：
- 我准备 `Agent(subagent_type=...)` 但还没读对应 SKILL.md / ALWAYS_LOADED 文件
- "让 explore agent 看下"——但没指定 search keyword

**Fix**：cite OL-08 + 把 success criterion 写进 brief 的最后一行。

---

## P4：Simple-op assumption（"这是简单 case，跳过 phase"）

**Urge**：这次只改一行 / 这次只是 byte-compare，phase 走完整套 overkill。

**Rule**：phase 没有"简单 case 可以跳过"的例外。byte-compare 也要 cite SHA；single-line cap loosen 也要跑 on-A3 forward smoke。"简单"是 agent 给自己的 prepared exit。

**Incident anchor**：
- 2026-04-27 T22.4 row 2：transformers 5.4 outcome A 第一次只 byte-compare + 文档化，没跑 on-A3 forward smoke；T21 真跑发现 npu_flash_attn_with_kvcache placeholder + flash_attention_4 keys 都是新加的，outcome 修正成 A-with-note。

**Detection signal**：
- "这个不需要 phase X 因为..."——这个"因为"几乎总是合理化
- "single line change 不可能出错"

**Fix**：phase 全跑；如果某 phase 真不适用，加 phase-skip exception 文档（不是默认跳）。

---

## P5：Failure discomfort（写 "expected failure" / "structural limitation"）

**Urge**：失败不舒服，把它包装成 "expected"，session 就能 advance。

**Rule**：失败必须分类（OL-10）。是 EC（error correction） / NPU-BUG / unsupported feature 还是 real bug？分类清楚才能 cite 出处。**禁止** "this is just the way it is" / "structural ceiling" / "V220 limitation"——这些都是 P5 的标记词。

**Incident anchor**：
- 2026-04-28 T28：sglang server 起来但 inference 第一次调用就 `RuntimeError: 0 active drivers`；最初 inner monologue 想写 "这是 sglang main vs sgl-kernel-npu 配对的 expected gap，outcome A 维持"——P5。**实际**：cold-drive 的 outcome 就是 **C-report**（真 bug，发 issue），不是 A。

**Detection signal**：
- 我在写 "as expected / 不影响 outcome / 已知 limitation" 但没 cite 具体 issue # / PR # / KB 条目
- 我把 partial fail 包装成 partial pass

**Fix**：cite outcome ladder OL-16；如果是 C-report，写真的 reproducer + 关联上游 issue 链接。

---

## P6：Infrastructure friction（manual workaround 而非 fix script）

**Urge**：基础设施挡路，我 inline 改一下绕过去就行。

**Rule**：infrastructure friction（脚本 bug、helper 没传对参数、image 路径错）**必须修脚本**而不是 inline workaround。inline workaround 让下一个 cold-drive 重新撞同一个坑。

**Incident anchor**：
- 2026-04-28 T25.5：`run-npu-container.sh` 默认 `--chips 0,1`，T25 切 `--chips 4,5` 后 Ray 报 0 GPUs。当时 inner monologue 想 "我手动 export 一下 ASCEND_RT_VISIBLE_DEVICES=0,1 就行"——P6。**实际**：helper script bug，必须改 script 加 `IN_CONTAINER_CSV` 自动派生（commit `a6f3fca`）。

**Detection signal**：
- 我在用 `export FOO=...; bash ...` 替代修 script
- "这次先手动一下"

**Fix**：cite DEBT-N（或新加 DEBT 行到 ROADMAP §6）；修 script 后 commit。

---

## P7：Closure desire（want to emit done）

**Urge**：session 跑了很久了，想要一个 done 状态。

**Rule**：done = 所有 phase artifact 齐全 + outcome ladder 在 A / A-with-note / C-patch / C-report 之一。任一 phase 还在 in-progress 就**不能** emit done。

**Incident anchor**：
- 2026-04-27 T22.7：第一次 V1.4 GRPO smoke 起来后看到 `reward_score=0.0126`，inner monologue 想 emit "L5 PASS done"——但还没 cite checkpoint path、还没 verify val_response_length。被 P7 推。补完 cite 才 emit done。

**Detection signal**：
- 我跳过 `OL-02 provenance` 想直接 ship outcome
- "看起来够了"

**Fix**：cite OL-02 + OL-09；逐个 artifact 验证完才 emit done。

---

## P8：Tool-path-of-least-resistance（绕过 skill 直接 Agent / nohup）

**Urge**：调用 skill 太麻烦，直接 spawn agent 或 nohup 跑 raw command 快多了。

**Rule**：skill 是**唯一** entry point。skill 才会 load 对应的 OL / KB / brief / state machine spec。绕过 skill 等于绕过所有 hook，进入了"我自己 freestyle"模式。

**Incident anchor**：
- 2026-04-25 T18 多个 session：用 raw `docker run` 启 NPU 容器而不调 `run-npu-container.sh`，每次都漏 bind 一些必需路径（`/etc/ascend_install.info` / `/etc/ascend_driver.conf` / `/etc/ascend_filelist.info`），错出 NPU-OPS-009 / 011 系列 dcmi -8020。

**Detection signal**：
- 我在写 `docker run --device ...` 直接而不是 `bash run-npu-container.sh`
- 我在 `Agent(subagent_type=Explore, ...)` 而不是 invoke 对应 skill

**Fix**：cite OL-04 / OL-13 / OL-14；改用 helper script + 对应 skill。

---

## P9：Infrastructure-friction paper-over（env 异常 → 自己绕过，不向上托管）

> ⚠ **与 P6 的区别**：P6 是"local 脚本卡住 → 手动绕过而非 fix 脚本" (day-0 域内的 friction)。
> P9 是 **out-of-domain env 异常**：NPU 驱动错码 / CANN install desync / lib size or symbol 不匹配 / docker exec timeout / SSH hang / proxy 429 / install-tree 路径漂移。这些信号代表**环境基线 (baseline) 违反**，不是 worker 该决策的事——worker 一旦"决定自己继续"，专门的 env-diagnose path 永远收不到信号；真正的问题在多层错误传导后变成大问题。

**Urge**: V1.4 smoke 半数失败 / `aclrtSetDevice` 报 507033 / `libascendcl.so` size 不对 / image pull timeout —— "再试一下"、"换个 NPU"、"replace .so"、"绕开 --pkg"、"backup 然后手动 cp" 都很诱人，因为它们比"停下来上抛 + 让 user/skill 决定"快。但**这正是 vibe-coding 的反例**：harness engineering 项目里，env 异常 = 工程级 signal，必须按基线 (engineering baseline) 处理，不是"看着办"。

**Rule** — 机械分类后双分支处理：

### Mechanical classifier table (借鉴 a5_ops 2026-05-15 攻防演练 + 移植到 NPU 上下文)

| 症状（worker 实际看到的）| 类别 | 处理路径 |
|---|---|---|
| `aclrtSetDevice` 偶发 507033（首次出现）/ dcmi -8020 偶发 | **transient** | retry ≤ 3 with exp backoff；retry 计数**在 orchestrator 暴露**（不藏在 worker 内 loop）；用尽 → 上抛终态 `INFRA_TRANSIENT_RETRY_EXHAUSTED` |
| HuggingFace proxy 短暂 429 / 短暂 connect timeout | **transient** | 同上 |
| HCCL init fail（独立 process 内首次出现）| **transient** | 同上 |
| `/etc/ascend_install.info` 不存在 / `which npu-smi` 失败 | **baseline-violated** | **永远不进 phase O1+ work**；必须 graceful exit `INFRA_BASELINE_VIOLATED`；worker **绝对禁止**"探一下 baseline 然后决定怎么绕过" |
| `npu-smi info` 报 driver != 25.5.x | **baseline-violated** | 同上 |
| Docker proxy 持续 > 5 min 挂 / pull 长 timeout | **baseline-violated** | 同上 |
| A3 上 `repo/` 不是 git clone（NPU-OPS-014）| **baseline-violated** | 同上 |
| `libascendcl.so` size != 已知 baseline | **baseline-violated** | 同上 |
| Helper script 自己 bug（如 `--chips` 错传，T25.5）| **our-script-bug** | 修 script + commit，**不是** workaround；登记 DEBT |
| Skill 自己默认参数错（如 T25.5 `ASCEND_RT_VISIBLE_DEVICES`）| **our-script-bug** | 同上 |

### 禁止行为白名单

Worker / probe / 任何 day-0 agent 收到 env-class signal 必须立刻上抛，**不在同一 spawn 内做以下 paper-over**：

- 手动 cp / replace .so / replace lib
- 重启 NPU 后再试
- 换个 NPU chip 试到通为止
- bypass `--pkg` / `--no-verify` / `--force`
- 在 PROGRESS.md / handover 写 "环境问题，先跳过"
- 修改 host 上 `/etc/ascend_*` 配置
- 跳过 `repo/src/scripts/run-npu-container.sh` 直接 docker run（绕开 bind 检查）

**Incident anchor**：

- 2026-04-28 T25.5：`run-npu-container.sh` 默认 `--chips 0,1` 巧合下 happy path 通过 10 个 commit；T25 切 `--chips 4,5` 后 Ray 报 0 GPUs。**正确类别 = our-script-bug**（不是 transient，不是 baseline-violated）。Fix: 修 helper script 自动 derive `IN_CONTAINER_CSV`（commit `a6f3fca`）。当时差点被解读为 "换 chip 试试看" → P6 路径，那会进一步掩盖根本问题。
- 2026-04-28 T25.5：A3 上 `repo/` 是 stale v0 layout，path `repo/src/scripts/run-npu-container.sh` 不存在。**正确类别 = baseline-violated**（NPU-OPS-014）。差点被解读为 "在 A3 上 mkdir + cp 文件就行" → P6 inline workaround，会让下一次 cold-drive 再撞同一个坑。

**Detection signal**：

- 我开始写 `mv /etc/ascend_install.info.bak /etc/ascend_install.info`
- 我在试第 4 次 retry 而 retry_count 没上报
- 我写 `# 临时改一下 .so，先跑通`
- 我决定 "换 NPU 7 试试" 而 NPU 0-6 状态没明确

**Fix**：cite OL-04 / OL-13 / OL-14 / OL-15 + 按 classifier table 分类后选 transient retry / baseline-exit / script-fix 路径之一；**禁止 freestyle**。

---

## 决策点 cite 映射

| 决策动作 | 必 cite |
|---|---|
| spawn agent / sub-skill | P3 + P8 |
| emit outcome `A` | P1 + P7 |
| emit outcome `A_WITH_NOTE` | P4 + P5 |
| emit outcome `B` | P5 + P7 |
| emit outcome `C_PATCH` | P5 + P7 |
| emit outcome `C_REPORT` | P5（最重要——确认不是 P5 把真 bug 包装成 expected limitation） |
| skip cold-drive replay | P1 |
| nohup / & / 直接 Agent | P8 |
| inline workaround 替代修 script | P6 |
| "expected failure" / "limitation" 措辞 | P5 |
| env-class 异常（NPU 错码 / CANN install desync / .so 不匹配 / proxy timeout）| **P9**（先 mechanical classifier 表分类，再选 retry / baseline-exit / script-fix 路径） |
| 准备 manual cp / replace .so / bypass --pkg / 改 host 配置 | P9（这些都是 baseline-violated 类的 paper-over，禁止） |

---

## 见也

- [`OPERATIONAL_KNOWLEDGE.md`](OPERATIONAL_KNOWLEDGE.md) — OL-XX 总索引
- [`ALWAYS_LOADED_UNIVERSAL.md`](ALWAYS_LOADED_UNIVERSAL.md) — universal OL 详情
- [`../../../docs/_meta/kb/porting_lessons/`](../../../docs/_meta/kb/porting_lessons/) — cross-layer + per-upstream lessons（incident 完整记录）
- [`../../../docs/_meta/kb/challenge_patterns/`](../../../docs/_meta/kb/challenge_patterns/) — 11 条 self-critic 模板
- [`../../../docs/_meta/ROADMAP.md`](../../../docs/_meta/ROADMAP.md) — DEBT-N 登记
