# NPU 移植 Skills 目标架构 + 演进路径 (V3.0 + Day-0 reframe)

> 状态：2026-04-23，V3.0 主体 + 顶部 Day-0 reframe 追加。
>
> **核心修正 (V3.0)**：先证明最基本的情况 (1 expert 完成 D=0 移植) 能 cold-drive 工作，再谈扩展。DAG / 多 expert 架构**挪到附录**。
>
> 过往版本：
> - V1.0（commit `89b2530`）：照抄 a5_ops kernel-pipeline，被证明不适用
> - V2.0（commit `db801cd`）：过早的 DAG + per-dep expert 泛化
> - V3.0（本版）：Stage 0 只做 1 expert；DAG 挪附录

---

## Day-0 Reframing (2026-04-23, user directive)

**原 V3.0 + Stage 2 experts 的隐含假设（错误）**：
> "dep 升级 = 消费者 reqs 换新版本号 → NPU image 已经 ship 好了对应
> 版本 → 我们只需 apply shim 兼容 API rename"

这假设**社区版本总是已经在 NPU 上 ready**。**错**。

**真实问题（user 2026-04-23T00:44-45）**：
> "开源社区有了的版本，npu 上跑不起来，因为配套软件没有。这才是我们的
> skills 需要解决的问题。easyr1 的所有配套软件都就位是最简单的场景。
> 我们要解决 day 0 支持新版本社区软件的问题。"

换言之：
- 上游社区发布 `transformers 5.4` / `vllm 0.20` / `torch 2.11`
- NPU 生态里 `torch_npu` 或 `vllm_ascend` 或 `transformers.integrations.npu_flash_attention` **还没跟上**
- EasyR1 consumer 想用新版
- **这时候才是真正难的场景**，也是 skills 必须解决的
- "shim-only" 的 expert 在这种场景里**没意义**——因为目标 NPU image 里根本没有对应 vllm-ascend / 配套 wheel

**按难度重排（user 观察 2026-04-23T00:44）**：
1. **pytorch 最容易**：`torch_npu` 跟社区 torch 版本差异很小，NPU 团队跟得紧
2. **vllm 中等**：vllm-ascend 通常慢 vllm 1-2 minor，可以通过分支 cherry-pick
3. **transformers 最难**：社区迭代最快，API drift 多，NPU FA / device_map 集成点多

### 今天 Stage 2 experts 的真实定位

重新看三个 upgrade expert 的实际能力：

| Expert | 现在做的事 | 能否处理 Day-0 缺 NPU 配套 |
|---|---|---|
| `transformers-upgrade-expert` | 消费已 ship 的 v2 base image；apply EC-02/03 shim | ❌ 不能；v2 image 是假设存在的。若 transformers 5.4 来了且没 matching NPU image，本 expert 无工具可用 |
| `vllm-upgrade-expert` | apply CP-002/004/EC-03 shim；复用已 ship 的 vllm-ascend | ❌ 不能；若 vllm 0.20 来了 vllm-ascend 还是 0.17，shim 不足以补全缺失的 NPU kernel 调用 |
| `torch-npu-upgrade-expert` | 给新 image 加 Dockerfile NPU-BUG-001/004 workaround | ⚠️ 部分能；torch 一般 NPU 跟得紧，所以本 expert 的 "添加 workaround + 验证 import 通" 就可能够用。但如果 torch 真跑到 2.11 而 torch_npu 还卡 2.9，也不行 |

**结论**：**三个 Stage 2 expert 都是 "shim 消费者"，不是 "ecosystem driver"**。
它们只证明了 skills harness 能跑起来，没证明 skills 能解决 Day-0 真问题。

### Day-0 expert 真正需要的能力（Stage 3 重新设计）

以 `transformers-upgrade-expert` 为例，如果 transformers 5.4 发布而 NPU 还没跟上，真正的 expert 需要做的：

1. **诊断层**：
   - pip freeze 对比：告诉我哪个 NPU 生态包滞后（vllm-ascend / transformers.integrations.npu_flash_attention / torch_npu）
   - API diff：新 transformers 的 public API 变化 + NPU FA 适配层哪些 hook 点没跟上
   - 找出"社区有而 NPU 版本没支持的"功能集（例如新的 attention impl / 新的模型结构）

2. **决策层**：
   - 是否能用已 ship 的 torch_npu 跑新 transformers？（短期 workaround 路径）
   - 是否需要本地 patch vllm-ascend / transformers NPU FA？（中期 forward-port）
   - 是否需要等上游 NPU 团队？（长期 block）
   - 哪些 consumer feature 可以临时关闭？（降级路径）

3. **实施层（如果 forward-port）**：
   - Checkout vllm-ascend / transformers 的 NPU 适配分支，cherry-pick 到新上游 tip
   - Build 本地 wheel / image
   - 验证 import + V1.3 rollout + V1.4 training 全通

4. **Exit 路径**：
   - 成功 → emit forward-port 分支 + patched image + baseline numerics（可能不在已知 band，需要 fresh baseline 建立 protocol）
   - 失败 → 精准 blocker 报告（"NPU 端没人适配 X 这个 op；consumer 要么等 NPU 团队，要么放弃 transformers 5.4 这个新 feature"）

**这**才是 skills 对用户有价值的地方。

### 这对今天产出的影响

- 昨天-今天跑的 /npu-port 端到端 E2E（fixture transformers>=5 → v2 image 已 ship → 通）证明的是 **harness 管道能跑通**，**不证明** Day-0 场景能处理
- 三个 Stage 2 upgrade expert 保留有价值，但 **scope 需要改名 / 分层**：
  - 现在的它们：`<dep>-shim-adapt-expert`（消费已 ship 的 NPU 版本，apply consumer-side shim）—— 已完成
  - 真正需要的 Stage 3：`<dep>-day0-forward-port-expert`（当 NPU 生态没跟上时，诊断 + 决策 + 可能的 forward-port）—— **未做**
- Transformers 难度最高，Stage 3 从 transformers day-0 入手合理
- Pytorch 难度最低，Stage 3 做 pytorch day-0 主要是建立 pattern 模板
- Vllm 难度中等，有 vllm-ascend pre-release 可参考，Stage 3 第二优先级

### 下一步实际需要做什么

1. **重命名今天三个 Stage 2 experts**：加明确后缀或在 README 第一行说清 "assumes NPU build exists"
2. **新设计 Stage 3 experts** 专门解决 Day-0：`transformers-day0-expert`、`vllm-day0-expert`、`torch-day0-expert`
3. **dep-analysis-expert 路由规则扩展**：区分 "NPU 已 ship 的版本" vs "NPU 未 ship 的版本"，分别路由到 shim-adapt 或 day0 expert
4. **orchestrator** 的 task-plan 输出增加 scenario 维度：P1 (D=0) / P2a (shim-adapt 能搞定) / P2b (需要 day0 forward-port)

---

## 0. 一句话定位

**当前阶段的任务**：做 1 个 easyr1-expert，让它能在 A3 上 cold-drive 完成 EasyR1 master（D=0 场景）→ V1.1-V2.2 smoke 全绿。

Stage 0 PASS = 基线。之后才有资格谈扩展。

---

## 1. 基本事实 (before 设计)

### 1.1 EasyR1 master 今天 D=0

`docs/easyr1-dep-chain-audit.md` 已证明：EasyR1 master 20 个 runtime deps 在 v1 (CANN 8.5.0) image 上**没有** D 类 blocker（需要新 NPU 适配的依赖）。所有适配都是**EasyR1 自己源码的改动**：

- 35 处 CUDA-only callsite 替换（NPU-CP-001）
- Ray NPU resource 注册 shim（NPU-CP-003）
- Flash attention 走 `transformers.integrations.npu_flash_attention`（NPU-CP-007）
- vllm 0.13 的 2 处 API rename 适配（NPU-CP-002 / CP-004）
- transformers 5.x / vllm 0.18 的 backward-compat（已 cherry-pick 进 ascend-port）
- 写 `Dockerfile.npu`（fix triton-ascend NPU-BUG-001 + deps install）
- 写 smoke 脚本

→ **技术上 1 个 expert 做完所有这些事情就够**。我们现在就是要证明：能不能**做一个 expert 把上面这些全自动化**。

### 1.2 Skills 现状（NPU-OPS-010 之后的诚实基线）

- 8 个独立 skill md（npu-image-inspect / dep-gap-detect / npu-code-path-sweep / npu-container-runner / upstream-branch-hygiene / ray-npu-shim / image-upgrade-drill / codex-review）
- 各自单独功能可用，但**没有 pipeline 串联**
- SKILLS-GUIDE.md 是**人工 9 步 playbook**，没有 agent / hook 强制执行
- Round 1 cold-drive 用 `zhshgmail/EasyR1:ascend-port` 作弊成功
- Round 2 tightened prompt 让 agent 从 `hiyouga/EasyR1:main` 出发：产出 4 commits 的 port code **但**：
  1. `verl/workers/fsdp_workers.py` 有 SyntaxError（未被 skill 拦住）
  2. Agent 停在 "docker image built"，没 ssh 到 A3 跑 smoke
  3. 读了 denylist 里的 HANDOVER.md

**→ 单 skill 可用；skill chain 自动驱动 agent 到"端到端验证"不可用。这正是 Stage 0 要填的 gap。**

### 1.3 DAG / 多 expert 情况还没出现

今天没有 D≥1 场景实际发生。一旦未来 EasyR1 新 commit 真要求某个 NPU 还没适配的依赖，我们**才**需要拆 expert（第一个会是 blocker 那个，比如 torch-npu-expert 或 transformers-expert）。现在**不预先设计**。

---

## 2. Stage 0 设计

### 2.1 范围：1 个 expert

**easyr1-expert**：负责 EasyR1 → NPU 移植的全流程。

```
easyr1-expert/
├── SKILL.md              # /easyr1-port 入口
├── agent.md              # easyr1-port-worker 的 agent 定义（Phase A/B/C/D + Stop hook）
├── state_machine.yaml    # 内部工作流（phases + invariants）
├── references/           # EasyR1-port 专属 KB
│   ├── ALWAYS_LOADED_RULES.md
│   ├── KB_INDEX.md
│   ├── CODE_PATH_PATTERNS.md    # NPU-CP-001..007 具体怎么 apply
│   ├── ERROR_CORRECTIONS.md     # traceback → root cause → fix
│   ├── PLATFORM_BUGS.md         # NPU-BUG-001..004
│   ├── SMOKE_BASELINE.md        # V1.1-V2.2 per-image 期望数值
│   └── patterns/
│       ├── device_dispatch.md
│       ├── ray_integration.md
│       ├── attention_backend.md
│       ├── vllm_compat.md
│       ├── transformers_compat.md
│       └── dockerfile.md
├── scripts/
│   ├── static_check.py          # py_compile + dry-import (from V1.0)
│   ├── deploy_to_a3.sh          # tar → scp → docker cp
│   ├── smoke_validate.sh        # 跑 smoke + grep entropy_loss + assert
│   └── code_path_sweep.sh       # (从 easyr1-npu/scripts/code-path-sweep.sh 搬)
├── tested_combinations.yaml     # 已验 tuple: v1 image + ascend-port ecce71d
└── hooks/
    ├── check_port_worker.sh     # Stop: static_check PASS + PROGRESS 签名
    └── ...
```

### 2.2 Agent 内部 phases

（一个 agent 包揽所有 phase，无 orchestrator；worker internal fix loop）

| Phase | Action | Artifact |
|---|---|---|
| A. Analyze | 读 KB (ALWAYS_LOADED + KB_INDEX) + run code-path-sweep → analysis.md | `analysis.md` |
| B. Code gen | 按 CODE_PATH_PATTERNS apply device dispatch / ray shim / flash attn swap / vllm compat；写 Dockerfile.npu + smoke scripts | git commits on branch |
| C. Build + static | **static_check.py must PASS** (Stop hook 强制 G2) → build docker image | image tag |
| D. Deploy + smoke + fix loop | ssh A3 → deploy → run V1.1 → if fail, read log + match ERROR_CORRECTIONS + apply fix + rebuild → up to N iters → V1.4 → continue | smoke logs + verification.json |

**Fix loop in D**: if smoke rung N fails, agent reads log, greps ERROR_CORRECTIONS.md, applies fix, rebuilds, retries. Up to M iters per rung (e.g. 3). Stuck → exit with handoff (未来 Stage 1 会有 smoke-probe agent，Stage 0 只是手工 escalate)。

### 2.3 Hook 强制

Stage 0 需要的最小 hook set：

1. **Stop hook on easyr1-port-worker**：
   - Static check pass (G2)
   - PROGRESS.md 有签名
   - 声称 smoke PASS 必须有 `/tmp/.../easyr1-logs/*.log` 文件存在 + 文件里有 entropy_loss 数值 (G3)
2. **PreToolUse on Edit**：
   - Orchestrator 不能 Edit consumer code paths (G1 — 此 stage 我们只有 1 个 expert 包办一切，这条先 lenient)

Phase 完了的 Acceptance check：
- 一个 cold-drive Explore agent，只给 repo + A3 ssh access + 一句任务 "把 EasyR1 master 移植到 A3 跑 V1.4"，能**自己完成 V1.1 V1.3 V1.4**（agent context 里），**没有我手动帮忙**。

### 2.4 KB 组织（per-expert，Stage 0 就 1 个）

V2.0 说 "per-expert KB"，Stage 0 只有 1 个 expert，所以所有 KB 在这一个目录里。未来拆 expert 时按 dep 拆出去。

关键 KB 文件（Stage 0 必须写）：

- `ALWAYS_LOADED_RULES.md`（最多 10 条硬规则；一 phase 就读）
- `KB_INDEX.md`（Keywords/Aliases 索引）
- `ERROR_CORRECTIONS.md`（从 porting-journal 整理出的 traceback → fix，至少 10 条）
- `CODE_PATH_PATTERNS.md`（NPU-CP-001..007 从 npu-patterns.md 拆出）
- `PLATFORM_BUGS.md`（NPU-BUG-001..004）
- `SMOKE_BASELINE.md`（V1.1-V2.2 per-image 数值）
- `patterns/*.md`（按领域）

---

## 3. Acceptance (Stage 0 PASS = 基线达成)

**T0.1 — static check 拦住 round 2 的 SyntaxError**
- 跑 `scripts/static_check.py --files <round2-agent's fsdp_workers.py>` → 必须 exit 1 + 指出行号

**T0.2 — Stop hook 拦住"声称 PASS 无 log 证据"**
- 人为构造一个假 PROGRESS.md 说 "V1.4 PASS" 但没有 log 文件 → Stop hook exit 2 reject

**T0.3 — cold-drive round 3 产 port + 自己跑 V1.4 smoke 在 agent 内**
- Fresh Explore agent，严格 denylist（HANDOVER / journal / drill / 现有 ascend-port 分支 git log 禁读）
- Task: "把 EasyR1 master 移植到 NPU 跑通 V1.4 smoke"
- Agent 必须在它自己 session 内完成：产 port code → static_check PASS → build image → ssh A3 → V1.1 PASS → V1.4 step1 entropy_loss ∈ [0.94, 1.04]
- **我不介入 debug**
- PASS = Stage 0 基线达成，可以展开 Stage 1

**T0.4 — V1.5 / V2.1 / V2.2 bonus**
- 同上 agent 继续走完 V1.5 HCCL + V2.1 padding_free + V2.2 ulysses=2
- 不强制，Stage 0 最小 PASS 是 T0.3

---

## 4. 从当前 → Stage 0 (分步)

### 4.1 现有 "资产"

（V1.0 / V2.0 冗余都移除）

- `src/scripts/static_check.py` （V1 写的，保留）
- `docs/workflow/port_state_machine.yaml` （V1 draft，保留作为 state machine 参考；实际 YAML 归 easyr1-expert）
- `knowledge/npu-patterns.md`（24 stable IDs，要拆到 easyr1-expert references/）
- 现有 8 个 skill md（concept 不丢；部分内容迁到 easyr1-expert/{scripts, references, state_machine}；部分作为未来 multi-expert 时拆出去的种子）

### 4.2 执行步骤（each 是一个 commit）

**S1. rewrite design doc (= this commit)**  
Stage 0 only, DAG 挪附录

**S2. create `src/experts/easyr1-expert/` 骨架**  
空目录 + README.md 说明定位 + 占位 SKILL.md + agent.md

**S3. state_machine.yaml 草稿**  
8 phases (P0..P7) + invariants G1-G3（不是 V2.0 的 G1-G6；stage 0 只需要 3 条 hard）

**S4. 第一批 KB 文件**  
- ALWAYS_LOADED_RULES.md (≤10 条核心规则)
- ERROR_CORRECTIONS.md (从 porting-journal 挖 10 条)
- CODE_PATH_PATTERNS.md (从 npu-patterns.md NPU-CP-001..007 重写成 actionable pattern)
- SMOKE_BASELINE.md (V1.1-V2.2 per-image 数值表)

**S5. Scripts + Hooks**  
- `deploy_to_a3.sh` + `smoke_validate.sh`
- `check_port_worker.sh` Stop hook

**S6. agent.md 完整化**  
Phase A/B/C/D 具体 brief，含 fix loop，含 Stop hook reference

**S7. Round 3 cold-drive test**  
T0.3 acceptance test. 按结果：
- PASS → Stage 0 达成，更新 HANDOVER + porting-journal，等用户决定 Stage 1
- FAIL → 诊断 gap，补 KB / hook / agent md，round 4

### 4.3 预估时间

S1: 30 min（此 commit）  
S2-S3: 1 h  
S4: 2-3 h（KB 内容最花时间）  
S5: 1 h  
S6: 1 h  
S7 + iterate: 不定（首次 cold-drive 可能 1-3 轮）

**→ Stage 0 首次 T0.3 尝试：半天**。如果 round 3 直接 PASS 那 Stage 0 完成；FAIL 就再迭代。

---

## 5. Stage 1+ （等 Stage 0 PASS 再规划）

**明确保留**：不提前设计。

当真遇到 D≥1 场景（e.g. EasyR1 新依赖 NPU 没适配 / vllm 升级带动 torch_npu 升级），才开拆第一个 dep expert。

**附录 A** 记录了**可能的**扩展方向（DAG + multi-expert），供未来参考。

---

## 6. 已知债（进入 Stage 0 前坦白）

- Round 2 ascend-port-e2e-round2 分支有 SyntaxError：我**不手工 fix**（fix 了就作弊）。round 3 由新 agent 重新产。
- A3 那个 huaweicloud-mirror 卡死的 build：Dockerfile 里 aliyun+timeout 要写进 easyr1-expert 的 dockerfile.md pattern。
- HANDOVER / journal / drill 是 reproduction-kit 的污染源。Stage 0 的 prompt 层面 denylist；未来可能要物理隔离（repo 分 reproduction-kit 和 history 两个子目录），但这个不是 Stage 0 必需。
- `src/scripts/static_check.py` 在 V1 写了，但 hook wire 要 S5 做。

---

## 附录 A：多 expert / DAG 扩展（future-work placeholder）

**当以下条件真的发生才考虑**：
1. EasyR1 新 commit 触发某个 dep 需要 NPU 适配（dep-gap-detect 返回 D≥1）
2. 我们要 serve 非 EasyR1 场景（e.g. "帮我把 pytorch 新版跑到 NPU"），此时 easyr1-expert 的代码跟需求无关，需要独立的 torch-npu-expert

**届时的设计候选**：
- 从 easyr1-expert 拆出 `torch-npu-expert`（或 `transformers-expert` 等，取决于哪个是第一次真正独立需要的）
- 加入一个最小 orchestrator 来协调 2 个 expert
- Contract schema 定义 3 个 action（declare_dependencies / declare_constraints / apply）

**不承诺**：
- 不预先写 6 个 expert
- 不预先写 DAG solver
- 不预先定义跨 expert contract

**原则**：跟随实际需求拆；任何扩展前，先问"有证据说现在的结构不够"吗？

（V2.0 的设计稿 commit `db801cd` 保留在 git 历史，是未来真需要时的参考骨架，但不是**承诺**。）
