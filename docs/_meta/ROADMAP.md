# ROADMAP — easyr1-npu

> **唯一权威 backlog**。所有 open work（strategic / P0 / next-wave / technical debt）只活在本文件。
>
> **禁止 ad-hoc backlog 文件**：禁止建 `TODO.md` / `BACKLOG.md` / `FOLLOW_UPS.md` / `OPEN_ISSUES.md` 等；handover / design doc / SKILL.md 末尾**禁止**加 "Open tasks" 段当 backlog 用。
>
> **维护规则**：
> 1. 每条 `P0xxx` / `DEBT-N` / `SI-N` 必须在引入它的 commit 里 link 回本文件
> 2. 完成后**立即删除**对应行（commit history 即审计）；ROADMAP 只反映 _open_ work
> 3. Strategic initiatives 用 design doc 承载，§1 只列阶段 + 链接
> 4. Technical debt 当场登记到 §6（**禁止 "以后" / "someday"**；必填 Trigger）
> 5. CC `TaskList` 是 in-conversation tracker，不和 ROADMAP 自动同步
>
> **Session 启动检查**：任何 "what's pending / 下一步 / TODO / 技术债" 类请求 → **第一步永远是 Read 本文件**，不要先翻 handover / design doc。
>
> 最后更新：2026-05-15。

---

## §1 — Strategic Initiatives

跨季度方向。每项链到 design doc，本段仅列阶段 + status。

| SI-N | 主题 | Status | Design doc |
|---|---|---|---|
| **SI-1** | a5_ops 架构借鉴落地（T29） | **进行中**（2026-05-15）：ROADMAP + OL catalog + ANTI_PRESSURE + handover 模板 → state machine YAML | （T29 task chain 直接驱动；无独立 design doc，本文件 §2 跟踪） |
| **SI-2** | Day-0 skill 扩展到下一波 NPU 上游软件（deepspeed / liger / bitsandbytes / mindspeed / HCCL） | **未启动**（候选清单 + 优先级见 [`NPU_ADAPTATION_GAP.md` §5](NPU_ADAPTATION_GAP.md)） | 见 NPU_ADAPTATION_GAP.md |

---

## §2 — P0 Backlog（pipeline integrity / 当前 sprint）

进行中的 P0xxx，按出现顺序。

| P0 | 主题 | Trigger / Why | Status |
|---|---|---|---|
| **P0a** | T29.1 ROADMAP.md 建（本文件） | a5_ops 借鉴 priority #1，单一 backlog 入口 | **进行中**（2026-05-15 commit） |
| **P0b** | T29.2 `src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md` | 跨 skill 通用规则散在 NPU_ADAPTATION_GAP / SKILL.md / KB_INDEX，没有集中处 | 待启动 |
| **P0c** | T29.3 `src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md` | a5_ops P1..P8 模式 + 决策点 cite 要求 | 待启动 |
| **P0d** | T29.4 `docs/_meta/handovers/SESSION_HANDOVER_TEMPLATE.md` | 当前 handover 都归档，新 session 无衔接模板 | 待启动 |
| **P0e** | T29.5 workflow state machine YAML + critic（最重活，回报最大） | a5_ops `workflow_critic.py` 模式；day-0 P0..P7 phase 转换的机械 critic | 待启动（先完成 P0a-d） |
| **P0f** | T30.D 测试网+安全网+反馈环设计文档（v2 post-codex-review）| a5_ops 三件套借鉴；2026-05-15 user direction："设计 → codex review → 实现" | **完成**（v2，commit 待 land）|
| **P0g** | T30.R codex-review T30 设计稿 | 外部独立 review；6 项 recommendation 已 incorporate 进 v2 | **完成**（2026-05-15）|
| **P0h.0** | claim_manifest.yaml schema + validate_claim_manifest.py | v2 codex R1 推荐：结构化 claim 替代 markdown regex；scanner+gate 共享单一 schema | 待启动（schema first，所有后续 task 依赖此）|
| **P0h.1** | sanity suite + claim_manifest tests + 5 doc tests | a5_ops `run_sanity_suite.sh` 模式；< 2s；包括 claim_manifest schema roundtrip + good/bad fixtures | 待启动（依赖 P0h.0）|
| **P0i** | scan_outcome_claims.py（schema-level validation）| 校验 manifest 字段、mode/artifact 组合、self_challenge 状态 | 可与 P0j / P0l 并行（依赖 P0h.0 schema）|
| **P0j** | finalize_day0_check.py + 4 mode-derived gates | `CLAIM_EVIDENCE_PRESENT / REQUIRED_ARTIFACTS_PRESENT / EXTERNAL_PUBLICATION_VERIFIED / VALIDATION_ARTIFACT_VERIFIED`；codex R3 cut from 6→4 | 可与 P0i / P0l 并行（依赖 P0h.0）|
| **P0l** | postmortem 模板 + T25.5 范例 + v1-mode-coupling 自反思范例 | a5_ops `docs/postmortem/` 模式；eat own dog food 写本设计 v1 自己的 mode-coupling postmortem | 可与 P0i / P0j 并行 |
| **P0k** | regression snapshot YAML + diff_snapshot.py | 共享 claim_manifest 部分字段；mode-aware；触发 next session cold-drive 时 diff | 待启动（依赖 P0i+P0j 落地，schema 稳定）|
| **P0m** | T31.2 design v3 — incorporate adversarial-audit lessons | a5_ops 2026-05-15 攻防演练记录 4 项洞察：M2 anti-cycle / M1 gate fitness / P9 infra paper-over / M5 tool-use signature | **进行中**（2026-05-15）|
| **P0n** | T31.3 codex review v3 | 第二轮 adversarial review，确认 v2→v3 修订没引入新洞 | 待启动（依赖 P0m）|
| **P0o** | P0i scanner 加 anti-cycle verifier check (M2 only, NOT M5) | v4：dynamic forbidden-path set（按当前 manifest evidence[].paths derive，不 hardcode）；repo-root 路径解析显式；exec-bit check | 待启动（依赖 P0h.0 schema） |
| **P0p** | P0h.1 bad fixtures with crafted fraud (M1) | v4：拆 `schema_bad/` + `gate_bad/` 双层；4 个 gate 各至少 1 fixture；coverage 门 sanity-suite-time enforced | 待启动（依赖 P0h.0） |
| **P0q** | ANTI_PRESSURE_PROTOCOLS.md 加 P9 (infra paper-over) + mechanical classifier table | v4：classifier table（症状 → 类别），3 类（transient / baseline-violated / our-script-bug）；禁止行为白名单 | 待启动（独立任务，可与 P0o/P0p 并行）|
| **P0r** | M3 scope to `/npu-port` orchestrator | v4：standalone day-0 skill 不适用；`/npu-port` 真启用 sub-skill spawn 时落 aggregate-X cap 设计；现在写 design doc 占位 | 待启动（独立 design 任务，可与 P0o/P0p/P0q 并行）|

---

## §3 — Next-Wave NPU 适配（档 C → day-0 候选）

按"客户/团队会真触发的概率 × 阻断程度"排。详见 [`NPU_ADAPTATION_GAP.md`](NPU_ADAPTATION_GAP.md)。

| 候选 | 优先级 | Trigger（什么条件触发开 skill） |
|---|---|---|
| **deepspeed-NPU 集成验证** | 高 | 客户切到 ZeRO-3 / pipeline parallel 时是 blocker；NPU 上游 ds-ascend 项目存在可以接 |
| **liger-kernel NPU 实现** | 高（长期） | GPU 训练性能优势核心；当前完全 gap，需要 kernel 团队配合 |
| **bitsandbytes-NPU / QLoRA** | 高 | QLoRA 是中下游客户刚需；NPU 完全不支持是商业卡点 |
| **mindspeed 复活决策** | 中 | 8.5.2 镜像移除 mindspeed 是对的？MoE 大模型场景需要回它 |
| **torchvision NPU op 补齐** | 中（VLM 路线触发） | `nms` / `roi_align` 已确认 CPU fallback；做 detection head 的 RL 是 blocker |
| **HCCL 多机压力测试** | 中（规模触发） | 单节点 2-chip OK；4-chip / 8-chip / 多机扩展时压测必查 |
| **peft NPU 路径完整 smoke** | 中 | LoRA SFT / RLHF 触发；T26 测试因 peft 0.19 API 变化未跑通 |

---

## §4 — 已交付 fork 分支跟踪 / 回归 monitor

各 fork 分支跟踪上游版本前进的触发条件。详见 [`UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md)。

| 上游 | 分支 | 回归触发条件 |
|---|---|---|
| vllm-ascend | `ascend-port/vllm-main` | 社区 vllm tip 前进 → 跑 `/vllm-ascend-day0` sweep 检查新 F1-F8 漂移 |
| torch-npu | `ascend-port/torch-2.12-rc3` | torch 出新 RC（如 2.13-rc1）→ 跑 `/torch-npu-day0` Mode B |
| transformers | `ascend-port/transformers-v5.4` | transformers 出新 patch → `/transformers-day0` byte-compare |
| triton-ascend | `ascend-port/triton-v3.6.0` | bishengir-compile 出 LLVM 22 build → 重启源码端到端 smoke |
| sglang + sgl-kernel-npu | n/a（不开 fork） | 任一 axis 前进 → `/sglang-npu-day0` 3-axis 验证 |
| EasyR1 (consumer) | `ascend-port-integrated-20260427` | base image 升级 / 4 fork 任一前进 → `/integrated-overlay-build` 重 build |

---

## §5 — Decision gates（未结决策点）

需要外部输入或 evidence 才能推进的决策。

| 决策 | 等待什么 |
|---|---|
| triton-ascend 源码端到端是否能解锁 | 等 Huawei CI ship 一个 LLVM-22 built `bishengir-compile`（详见 [`triton-ascend KB_INDEX.md` case 1](../../src/skills/triton-ascend/port-expert/references/KB_INDEX.md)） |
| sglang main + sgl-kernel-npu 版本配对回归 | 等 `sgl-project/sgl-kernel-npu` ship `2026.04.15.rc4` 之后的 image rebuild（详见 [`sglang KB_INDEX.md` case 1](../../src/skills/sglang/port-expert/references/KB_INDEX.md)） |
| mindspeed 是否复活 | 等客户用例确认是否需要 MoE / pipeline parallel 大模型场景 |

---

## §6 — Technical Debt

格式：`DEBT-N` + 简述 + Why / What fix / Trigger / Effort。**禁止 "以后" / "someday"**。

| DEBT-N | Why / Description | What fix | Trigger | Effort |
|---|---|---|---|---|
| **DEBT-1** | A3 host 的 `repo/` 之前是早期 v0 layout 的非 git 拷贝（T25.5 发现，NPU-OPS-014）；当前虽已重 clone，但没有自动检测机制防止 future drift | 在 `run-npu-container.sh` 启动时加 `git rev-parse HEAD` 检查 + 对比 origin/main，警告 stale | T29 整理过程中 / 下次出现 A3-side script not found 错误 | 30 min |
| **DEBT-2** | 每条 day-0 SKILL.md 都重复了 NPU 容器 bind set 的描述（NPU-OPS-009/011/012/013/014），有漂移风险 | 抽出 `src/skills/_shared/npu-container-runner/SKILL.md` 作 single source，所有 day-0 SKILL 改为引用 | 任一 NPU-OPS-XXX 更新时 | 1 h |
| **DEBT-3** | install-skills.sh 没做 hooks 版本管理（不像 a5_ops 的 `_owner` + manifest_sha256 stamp），他人安装时可能 silently conflict | 加 `.claude/.easyr1_hooks_version` 含 manifest_sha256；preflight 检查 mismatch 时 backup+overwrite | 任一 hook 配置变更时 | 1 h |
| **DEBT-4** | day-0 SKILL.md 和 _shared/upstream-day0-workflow.md 的 P0..P7 描述目前是 prose-only，没有 machine-readable spec；workflow critic 无法执行 | T29.5：建 `docs/_meta/workflow/day0_state_machine.yaml` + Python critic | T29.4 完成后 | 4-6 h |
| **DEBT-5** | T30 设计 v2 接受 known gap：bypass-control 不完全闭合（agent 在 P8 压力下 nohup / raw docker 绕开 scanner+gate）；只靠 3 重 enforcement，完全防需 P0e workflow critic PreToolUse hook | P0e (T29.5) 上线后，把 scanner+gate 调用作 PreToolUse Bash hook 强制拦截 | P0e 完成时 | 2-3 h |
| **DEBT-6** | a5_ops M5 tool-use signature 我们当前无 hook 能抓（worker `Read finalize_day0_check.py` + `Bash test gate` 这种 reward-hacking-by-engineering 签名）；v4 经 codex review 删了 self_challenge.notes 字符串扫描（security theater），完全延后 | P0e workflow critic 上线后用 PreToolUse hook 捕 tool-use log；写 OL-28 + 加 gate 拦截 | P0e 完成 + 工具链有 tool-use logging 时 | 1-2 h |
| **DEBT-7** | WORKER-GATE-ENUMERATION meta-attack a5_ops 自己也 STILL OPEN；我们 4 个 gate 暴露在源代码里，worker 同样可逆向 engineer | 设计 research：选 1 个对策落地——(a) 随机化 gate 顺序、(b) 部分 gate 移到 hook 层不可读、(c) 通过 tool-use log 检测 worker 读 gate 源码 | 当 day-0 真 spawn sub-agent 时 / 见首个 reward-hacking 事故时 | 4-6 h（research）|
| **DEBT-8** | Cross-backend joint regression：当前只 codex review，无第二个 LLM backend（gpt-4o / gemini）做盲点 detect；a5_ops 用 A5 (Opus) + DS (V4) 跨 backend 抓到 A5 漏的攻击 | 第二个 backend 独立 review T30 设计 + cold-drive；可考虑用 codex 之外的 gemini / gpt-4o | T30 实现 P0o/p/q 完成时 / 见首个跨 backend 盲点事故 | 1-2 h |

---

## §7 — Completed（最近 30 天，作 commit-history 索引）

完成的工作直接删除对应行；本段保留**索引**便于查找 commit。

- **T26**（2026-04-28，commit `f2d0156`）：docs 整理 — 24 文件归档 + 8 客户文档精简 + ARCHITECTURE.md (5 mermaid)
- **T26.5**（2026-04-28，commit `2cb0cf2`）：NPU_ADAPTATION_GAP.md codify
- **T27+T28**（2026-04-28，commit `0c86e0e`）：`/sglang-npu-day0` skill + 3-axis 版本验证 + 官方 image cold-drive 发现 sglang main↔kernel-npu 配对 bug（C-report）
- **T25**（2026-04-27/28，commits `c8cde08` + `a6f3fca`）：端到端冷启动重演 6 sub-task + helper-script ASCEND_RT_VISIBLE_DEVICES bug fix + NPU-OPS-014 新增

完整审计：`git log --oneline --since=2026-04-15`。
