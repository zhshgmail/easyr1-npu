# Skills design — 可复用多 repo 移植系统

> **⚠️ SUPERSEDED (2026-04-23)**：本文档描述 V0.3（2026-04-19）的"7 shipped + 3 deferred skills"
> flat 目录结构，**已被 Stage 0/1/2/3 expert 架构取代**。新设计见
> [`design/SKILLS_ARCH_TARGET.md`](design/SKILLS_ARCH_TARGET.md) 和 [`HANDOVER.md`](HANDOVER.md) §7。
>
> 本文件**不再维护**，保留作为历史记录 —— 当时的 skills 设计 rationale + 状态。
> `repo/skills/` 目录里的老 skill 文件也保留（见 HANDOVER §7.3 的 legacy 说明），但
> 实际起作用的是 `src/experts/<expert-name>/` 和 `src/orchestrators/npu-port/`。

> 状态：V0.3，2026-04-19。参考 `~/workspace/a5/a5_ops/docs/_meta/design-subdocs/SKILLS_DESIGN.md`（V3.1）的模式。本次移植 (EasyR1 → A3) 的经验作为该系统的第一个实例。更新历史：V0.1 是规划蓝图；V0.2 (2026-04-18) 是规划-vs-落地状态对齐；V0.3 (2026-04-19) 在 transformers-upgrade drill 后新增 `image-upgrade-drill` skill + 反映 catalog 从 19→23 条的增长。

## Status table

| Skill / artifact | Status | Shipped as |
|---|---|---|
| `inspect-ascend-image` | **shipped** (D2) | `skills/npu-image-inspect/` + `scripts/inspect-ascend-image.sh` |
| `dep-diff` | **deferred** | done manually in `docs/easyr1/dep-matrix.md`; automation low-priority (only triggered on new image/upstream) |
| `version-align` | **deferred** | done manually in `knowledge/upstream-refs.md`; each upstream's README has its own format, not uniform enough to script cheaply |
| `code-path-sweep` | **shipped** (D3) | `skills/npu-code-path-sweep/` + `scripts/code-path-sweep.sh` |
| `container-run` | **shipped** (D1) | `skills/npu-container-runner/` + `scripts/run-npu-container.sh` |
| `upstream-branch-hygiene` | **shipped** (pre-D) | `skills/upstream-branch-hygiene/` |
| `npu-smoke-test` | **partially shipped** | the smoke scripts themselves ARE the harness (`smoke_v11_device.py`, `smoke_v13_rollout.py`, `examples/qwen2_0_5b_math_grpo_npu_smoke*.sh`); and the **ladder convention** is now documented at `knowledge/smoke-ladder-convention.md` for reuse by future ports. No single `SKILL.md` wrapper — the convention doc + the concrete scripts suffice. |
| `codex-review` | **shipped** (pre-D) | `skills/codex-review/` |
| `gap-plan` | **deferred** | done as a hand-written `docs/easyr1/npu-gap-plan.md`; automation over-engineered for a 1-day port |
| `ray-npu-shim` | **shipped (unplanned addition, D4)** | `skills/ray-npu-shim/` + `ray_npu_shim.py` — extracted from the EasyR1 port when NPU-CP-003 + NPU-BUG-002 + NPU-ENV-002 kept appearing together |
| `image-upgrade-drill` | **shipped (unplanned addition, D5)** | `skills/image-upgrade-drill/SKILL.md` — emerged from the 2026-04-19 transformers upgrade drill. A playbook-type skill (no script) that operationalizes PORT-SUMMARY §5-6 for major base-image / framework bumps. First instance: `docs/transformers/transformers-upgrade-drill.md`. |
| `kernel-delegate-a5` | **v2+ (speculative)** | only needed when we decide v1 torch fallback is too slow; route to `a5_ops/` |

**Summary**: 7 shipped, 3 deferred (manual version suffices for v1), 1 partial, 1 speculative. Two skills (`ray-npu-shim`, `image-upgrade-drill`) were **emergent** — they weren't in V0.1's plan, both surfaced when the same workflow repeated and we realized a future session would re-derive it. That's the trigger criterion for promoting something to a skill: "would a new session from cold cache learn this faster by reading a SKILL.md than by scanning commits?"

---

## 1. 这个系统要解决什么问题

从一个已有 ML/RL 框架（EasyR1、veRL、OpenRLHF、TRL、其他），把它从 **GPU (CUDA/NCCL/flash-attn/vllm)** 移植到 **Ascend NPU (910C A3, CANN 8.5+, torch_npu, HCCL, vllm-ascend, triton-ascend)** 上。

> **scope 的第一性原则**：目标是"让 EasyR1 在 A3 上跑"。遇到 NPU 生态 gap 必须**识别 + 驱动解决**，不用"不在 scope" 绕过。"驱动"≠"自己写 commit"，但每一条 gap 都要 track 到完成。
>
> **档 1 — 本仓直接做**：
> - EasyR1 / 其他 RL 框架 **自己源码的改动**（device 路由、版本 compat shim、Ray 集成、Dockerfile 等）
> - Python 层 shim / fork 桥接 CUDA-only 包
> - 向 vllm-ascend / triton-ascend / torch_npu 的 **Python 层** 提 issue 或 PR
> - 识别 **NPU 适配 gap**，记到 `docs/easyr1/npu-adaptation-tasks.md`（待建）
>
> **档 2 — 委托给姐妹项目 / 独立仓（本仓识别 + track，commit 在别的仓）**：
> - A3 kernel 精度验证 → [`ascend-fused-accuracy-probe`](https://gitcode.com/zhengshencn_hwca/ascend-fused-accuracy-probe)
> - A5 kernel 生成 → [`a5_ops`](https://gitcode.com/zhengshencn_hwca/a5_ops)；A3 有类似独立仓
> - torch_npu C++ ATen op 实现 → Ascend PyTorch 团队或专门 kernel 项目
>
> **档 3 — 只能提需求给 Ascend 团队**（本仓没权限改的底层）：
> - CANN runtime 框架 bug（ACL / HCCL C 层、driver）—— 提 issue + 写 workaround + 记 stable ID + 等修
>
> **真正不碰**：
> - RL 算法本身的改动（reward shaping、loss 公式、模型结构）。我们只做"让它在 NPU 上跑"，不碰算法层

---

## 2. 和 a5_ops 的关系 —— 维度不同

a5_ops 是 **单 kernel 生成**：一个算子一条 pipeline，5-50 个算子并行，每个都走 `analyzer → generator → builder → verifier → optimizer`。

**我们是跨多 repo 的移植**。一次完整移植要同时处理：

| 层 | 涉及 repo（examples） | 每层的工作类型 |
|---|---|---|
| 顶层应用 | EasyR1 / OpenRLHF / TRL | device 路由、attention backend 选择、flash-attn 替代、CUDA API 替换 |
| 推理引擎 | vllm-ascend / sglang-ascend | 本身就是 NPU 专用项目，但可能缺特定模型覆盖 |
| 训练引擎 | veRL / TransformerEngine / Megatron | 这一层一般已经被厂商适配过，但 version pin 常常需要重对齐 |
| 核心库 | torch-npu / transformers | 主要是**版本匹配**和**镜像里的装配完整性** |
| 底层算子 | torch-npu ops / triton-ascend / CANN | 缺少时才要下到 kernel 生成（委托 a5_ops）或用 torch 兜底 |
| 基础设施 | docker image / CANN / driver / HCCL | 镜像选定、设备映射、网络拓扑 |

**模式差异**：a5_ops 重**纵向深挖**（一个 op 的精度、性能、算法），我们重**横向协调**（多个 repo 的版本对齐、依赖闭合、代码路径扫描）。

**取舍结论**：不套 a5_ops 的 4-agent pipeline。我们要**多个独立的 skill**，每个处理一类 cross-cutting 关注点，通过**持久化知识库（dep-matrix / upstream-refs / porting-journal）串起来**。

---

## 3. 系统轮廓

```
┌────────────────────────────────────────────────────────────────────┐
│  User: /port-ml-stack <source-repo> [--target ascend-a3]            │
└────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  orchestrator (slim): pick starting point from KB, schedule skills  │
│  via state machine, enforce gates, commit+push, Discord update      │
└──────────────────────┬─────────────────────────────────────────────┘
         ┌─────────────┼─────────────┬─────────────┬─────────────┐
         ▼             ▼             ▼             ▼             ▼
   ┌─────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ inspect │   │ dep-diff │  │ version- │  │  code-   │  │ container│
   │  image  │   │          │  │  align   │  │  sweep   │  │   run    │
   └────┬────┘   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
        │             │             │             │             │
        └─────────────┴──────┬──────┴─────────────┴─────────────┘
                             ▼
                  ┌────────────────────────┐
                  │  KB (source-of-truth)  │
                  │  dep-matrix.md         │
                  │  upstream-refs.md      │
                  │  images/*.md           │
                  │  porting-journal.md    │
                  │  gap-plan.md           │
                  └────────────┬───────────┘
                               ▼
                  ┌────────────────────────┐
                  │  review / optimize     │
                  │  codex-review (exists) │
                  │  opcode-review (TBD)   │
                  │  perf-tune    (TBD)    │
                  └────────────────────────┘
```

每个 skill 在下面 §4 一条。所有 skill 都把**结构化 artifacts** 写进 `repo/`，形成持久化 KB。

---

## 4. Skills 清单

> 本节把每个 skill 按 `name` / `input` / `output artifact` / `Stop hook gate` 列出。所有 artifact 都 commit 到 `zhshgmail/easyr1-npu`，跨 session 可见。

### 4.1 `inspect-ascend-image`

- **做什么**：给一个 docker image ref，提取 pip freeze、站点包布局、apt 包、env、entrypoint、CANN 版本、容器启动需要的设备/挂载。
- **已有实现痕迹**：`repo/knowledge/images/verl-8.5.0-a3.md`、`verl-8.5.2-a3.md` 就是这个 skill 的手动输出。
- **Input**：image tag、目标 knowledge 文件路径。
- **Output**：`repo/knowledge/images/<tag>.md`，schema 参考现有两份。
- **Gate**：生成的 md 必须有 "Runtime env / Core ML stack / Secondary deps / Matching upstream refs / Totals / Open questions" 六节，且至少一个版本号（验证不是占位模板）。

### 4.2 `dep-diff`

- **做什么**：给两个 pip 声明源（requirements.txt / pyproject / pip freeze），产生 per-package 差异矩阵 + gap 分类 (V/P/R/A/D/—)。
- **Input**：两个 dep 源，额外 image inventory 文件（可选）。
- **Output**：`repo/docs/easyr1/dep-matrix.md`（增量 patch 或首次生成）。
- **Gate**：gap 分类列必须填满，有"Hidden imports"和"Code-path blockers"两个小节占位（即使为空也写出来）。

### 4.3 `version-align`

- **做什么**：给一个目标 image（比如 verl-8.5.0-a3）和一批 upstream repo（torch-npu、vllm-ascend、…），解析每个 repo 的 README compat 表 / tag list，输出**匹配该 image 的 ref 表**。
- **已有实现痕迹**：`repo/knowledge/upstream-refs.md`。
- **Input**：image inventory md、upstream repo 列表。
- **Output**：`repo/knowledge/upstream-refs.md`。
- **Gate**：每个 upstream 都要给 ref + evidence（引用 README 某行或 tag 存在）。
- **为什么是独立 skill**：a5_ops 不需要这个（只有一个 CANN 版本）；我们跨 repo，这是高频犯错点（之前我用 master 做 review，codex 给出过时答案）。

### 4.4 `code-path-sweep`

- **做什么**：给一个 source repo + 一套已知 GPU-only 模式（torch.cuda.*、dist backend=nccl、init_device_mesh("cuda", ...)、attn_implementation="flash_attention_2"、device="cuda"、flash_attn.*、liger_kernel.*、bert_padding），grep 出所有 call site，分类（可直接替换 / 需要 shim / 需要 vendoring），产出 sweep 报告。
- **Input**：repo 路径、模式清单（可扩展）。
- **Output**：`repo/docs/code-path-sweep-<repo>.md`（表：文件/行号/模式/建议替换）。
- **Gate**：至少列出 10 个 patterns 的扫描结果（即使某个 0 命中也要显示）。
- **目前状态**：已经手动做过一次（见 `dep-matrix.md` §Code-path blockers），但没 skill 化。

### 4.5 `container-run`

- **做什么**：给 image tag + chips + 挂载目录，产出一个能直接执行的 docker run 命令，带正确的 Ascend 设备映射、HCCL 所需的 /etc/hccn.conf、netrc/ipc、用户 scratch 路径的 bind mount。
- **已有实现痕迹**：`repo/scripts/run-npu-container.sh`。
- **Input**：image、chips list、scratch dirs。
- **Output**：要么直接 exec，要么 dump 一份可再用的命令文件。
- **Gate**：执行前 check chip 是否被别人占用（`npu-smi info -t proc-mem -i <chip>`），有占用就拒绝。

### 4.6 `upstream-branch-hygiene`

- **做什么**：强制"本地改 → push personal → 远端 pull"规则。任何改动 upstream repo（EasyR1、transformers、vllm-ascend、verl）都必须：
  - 先 `git fetch origin`
  - 创建或 checkout `<personal>/<task-branch>` 分支
  - 本地 edit → test → commit
  - `git push personal <branch>`
  - 远端（A3）pull 相应 branch
- **Input**：repo dir、change description。
- **Output**：commit SHA、push 结果、A3 pull 结果。
- **Gate**：远端不允许直接 edit — A3 上的 git worktree 如果检测到 unstaged changes 就报错。
- **为什么独立**：这是这次移植最重要的 **operational discipline**，犯一次就可能丢一天工作（容器删、服务器清）。

### 4.7 `npu-smoke-test`

- **做什么**：一组阶梯式 smoke tests（V1.1 device accessor → V1.2 model load → V1.3 rollout → V1.4 training step → V1.5 multi-card），每级跑过才能进下一级。
- **已有实现痕迹**：`repo/scripts/smoke_v11_device.py`、`smoke_v13_rollout.py`。
- **Input**：image、test level。
- **Output**：`repo/logs/smoke-<level>-<date>.log` + `repo/docs/easyr1/porting-journal.md` 追加条目。
- **Gate**：每级有固定的 PASS/FAIL 断言；FAIL 就返回失败模式供 `precision-probe`-类 skill 分析。

### 4.8 `codex-review`（已存在）

- 现状：`repo/skills/codex-review/SKILL.md` 已经实现。
- 与本系统的关系：作为**通用二次评审**手段，在任何 skill 产生 artifact 后可选调用。尤其适合 `dep-diff` 结果（检查 gap 分类是否合理）、`code-path-sweep` 结果（检查是否有遗漏 pattern）。

### 4.9 `gap-plan`（暂用 `npu-gap-plan.md` 承担）

- 现状：`repo/docs/easyr1/npu-gap-plan.md` 手写。后续可演化为 skill：根据 dep-matrix + code-path-sweep + smoke 结果，自动生成/更新 gap 分类（v1 scope / v2 defer / perf follow-up / blocker）。

### 4.10 `kernel-delegate-a5` (将来)

- **做什么**：当 `code-path-sweep` 或 `npu-smoke-test` 发现我们需要一个 NPU 上没有的 kernel（例如 FA varlen、特定 dtype 的 CE），把它作为算子规格转给 a5_ops 的 `/ascendc-op-gen-v3`。
- **Input**：PyTorch 参考实现或 CUDA 源。
- **Output**：a5_ops 产出的 AscendC kernel + pybind binding。
- **Gate**：只在已经确认 torch 兜底精度不够或 perf 差距大到不可接受时触发。**初版我们不做**——用 torch 算子兜底是明确允许的。

---

## 5. 持久化 KB

与 a5_ops 的 knowledge files 类比：

| a5_ops 位置 | 我们的位置 | 作用 |
|---|---|---|
| `ALWAYS_LOADED_RULES.md` | `repo/knowledge/upstream-refs.md` + `memory/*.md` | 每次都要遵守的硬规则 |
| `PATTERN_INDEX.md` | `repo/docs/easyr1/dep-matrix.md` §Code-path blockers | 反复出现的模式+处理方式 |
| `ERROR_CORRECTIONS.md` | `repo/docs/easyr1/porting-journal.md` | 踩过的坑+解法（按日期+编号） |
| `PLATFORM_BUGS.md` | `repo/docs/easyr1/porting-journal.md` 对应 entry + `memory/a3_server.md` | 平台级已知 bug（如 triton-ascend 安装缺失） |

编号化已完成：see `knowledge/npu-patterns.md` (V2, 2026-04-18). **16 stable IDs** across 4 categories (CP × 6, BUG × 2, ENV × 4, OPS × 4), uniform schema `Symptom / Root cause / Fix / Commit ref / Generalizable rule`. Commit messages cite IDs (`git log --grep=NPU-CP-003`); skill outputs reference them (`scripts/code-path-sweep.sh` uses canonical IDs). `NPU-CP-001` is the *family* of CUDA-named APIs (namespace, tensor method, device strings, visibility env, nccl backend) — all share the `verl/utils/device.py` helper fix.

---

## 6. State machine（简化版）

不搞 a5_ops 那种精度-性能循环状态机。我们的流程更像 **装配线**：

```
fresh repo 
  → inspect-ascend-image × {images}
  → dep-diff (source vs image pip freeze)
  → version-align (lock upstream refs)
  → code-path-sweep (identify GPU-only sites)
  → plan: gap-plan (classify each gap V/P/R/A/D)
  → for each gap in v1 scope:
       fix locally → upstream-branch-hygiene push
  → build image layer
  → container-run + npu-smoke-test (step up V1.1 → V1.5)
  → iterate: any failure → back to sweep or plan
  → declare v1 done
```

和 a5_ops 最大差异：**没有精度回归循环**。NPU vs GPU 不保证 bit-close（除非对齐确定性）。Smoke 只看 "跑得通 + 输出合理"，精度对比是 v2+ 话题。

---

## 6.5 Mandatory pre-design step: read the reference port

Before designing any non-trivial port work, grep the adjacent ported system + the upstream library for the same sub-problem. Rationale in `knowledge/npu-patterns.md::NPU-OPS-005`.

Concrete for this project: we have **veRL** (the NPU-ported parent of EasyR1) and **transformers.integrations.npu_flash_attention** (upstream library's own NPU adapter). Before writing any adapter / shim / from-scratch code for an NPU-specific concern, **always** check:

1. `upstream/verl/` — does veRL solve this? If yes, in which module? Copy its pattern.
2. Upstream library's `integrations/` / `backends/` / `*_utils.py` directories on the tag matching our target image — does it ship the adapter?
3. Only after both come up dry, design from scratch.

Concrete 2026-04-18 incident that seeded this rule: estimated v2 (NPU flash-attn varlen) at 2 days. User asked "how does veRL do it?" — 4-line import swap from `transformers.integrations.npu_flash_attention`. Total real cost: 1-2 hours. See NPU-OPS-005.

Generalization: **this project is a port. Every port has a reference. Reading the reference is always cheaper than re-deriving it.** This rule applies to:
- EasyR1 → veRL (same framework family)
- Any RL framework port → veRL (veRL is the most-NPU-ported RL stack)
- Any ML framework port → transformers, accelerate, vllm-ascend, torch_npu (all ship integration helpers under `integrations/` / `backends/`)
- Any ops port → a5_ops (has the AscendC / CANN patterns)

If a piece of work needs more than a paragraph of explanation, invest the 5-10 minutes to grep the reference first.

## 7. Next steps (this session)

1. 把已经手写的 knowledge / journal / scripts 归位到 §4 定义的 skill 输出契约上。缺少的 section schema 补齐。
2. ~~给 pattern / error correction 加 stable ID~~ DONE (2026-04-18) — 16 IDs in `knowledge/npu-patterns.md`.
3. Skill-化：挑 2 个高价值 skill 先写成 `repo/skills/<name>/SKILL.md` + SKILL 内 runbook — candidates:
   - `inspect-ascend-image` (已有很好的模板)
   - `upstream-branch-hygiene` (operational 最高价值)
4. 后续会话根据需要再抽，不一口气全做。a5_ops 的经验：**skill 早出比 skill 完美重要**。
