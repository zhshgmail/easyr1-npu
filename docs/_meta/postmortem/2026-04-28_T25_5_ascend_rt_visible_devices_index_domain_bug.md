# Postmortem: `run-npu-container.sh` host phy-id vs in-container index 索引域 bug — 2026-04-28

**Date**: 2026-04-28
**Severity**: MEDIUM — shipped 10 commits in happy-path mode；外部用户用 `--chips != 0,1` 立刻 Ray 报 0 GPUs
**Cost**: 不主动 cold-drive 复现就抓不到；T25.5 cold-drive 时切 `--chips 4,5` 立刻爆雷；fix + post-fix V1.4 GRPO re-validate 约 90 分钟
**Reporter**: T25.5 自发现（cold-drive replay 用 `--chips 4,5` 触发）
**Status**: fix landed commit `a6f3fca`（T25.5 chip2,3 V1.4 GRPO PASS）

## Summary

`repo/src/scripts/run-npu-container.sh` 默认 `--chips 0,1`，代码里把 `${CHIPS}` 直接传给 docker 的 `-e ASCEND_RT_VISIBLE_DEVICES`。但**容器内** ASCEND_RT_VISIBLE_DEVICES 期望的是**容器内 device index**（按 `--device /dev/davinciN` 出现顺序从 0 开始重新编号），不是 host phy-id。`--chips 0,1` 巧合下 host phy-id 0,1 和容器内 index 0,1 同号——所以 T22.7 happy path 通过；T25.5 用 `--chips 4,5` 后 host phy-id 4,5 不等于容器内 0,1，CANN 看不到任何 device，Ray 报 `Total available GPUs 0 is less than total desired GPUs 2`。

10 个 commit 都跑在 `--chips 0,1` 上，没人测过非零 chip 值，bug 静默累积。如果外部用户拿默认配置上 NPU 0 被占的机器，就立刻撞到。

## What happened (timeline)

- 2026-04-22~27: T22.x sessions 多次用 `--chips 0,1` 跑 V1.4 GRPO smoke，全部 PASS；helper script 看似稳定
- 2026-04-27 T22.7: V1.4 GRPO 在 `--chips 0,1 / image easyr1-npu:integrated-20260427` 上 PASS（commit `5d5d756`）；标 "L5 PASS"
- 2026-04-28 T25.5 cold-drive replay 启动：user 要求 fresh agent 按 SKILL.md 重跑；A3 上 NPU 0 被别人占用
- 2026-04-28 11:13Z: 切 `--chips 4,5` 启动 helper script；Ray 立刻报 `Total available GPUs 0 is less than total desired GPUs 2`
- 2026-04-28 11:14Z: 复现 `--chips 2,3` 同样报错；排除"chip 4,5 物理故障"假设
- 2026-04-28 11:16Z: 翻 KB → 命中 NPU-OPS-012 "`ASCEND_RT_VISIBLE_DEVICES` is in-container chip index, not host phy-id"；翻 helper script 代码发现确实在传 host phy-id
- 2026-04-28 11:18Z: fix → 加 `IN_CONTAINER_CSV` 自动派生 `0,1,...,N-1`；scp 到 A3
- 2026-04-28 11:19Z: 重跑 `--chips 2,3 + fix`；V1.4 GRPO 2 steps + post-train val PASS（reward_score=0.013 与 T22.7 baseline 0.0126 在 ±10% 内）
- 2026-04-28 11:20Z: commit `a6f3fca` + 新增 NPU-OPS-014（A3 stale repo）+ ROADMAP DEBT-1 (auto-detect A3 stale repo)

## Root cause

**Architectural**:

1. **Happy-path-only test history**: helper script 自 day 1 就只在 `--chips 0,1` 跑；没有"非零 chip 值的回归测试"。CI / sanity suite 不存在（直到 T31）。
2. **KB 知识与代码脱节**: NPU-OPS-012 已经记下 ASCEND_RT_VISIBLE_DEVICES 的容器内 index 语义——但 helper script 代码本身不知道这条规则；KB 是 prose 文档，agent 写代码时**没人强制读 KB 来 review helper script**。
3. **Helper script 缺 sanity check**: 没有 "传进来的 host phy-id 是否需要重映射" 的 explicit translation step；变量直接传到 docker `-e`。

**Immediate cause**: `-e ASCEND_RT_VISIBLE_DEVICES="${CHIPS}"` 这一行没有把 host phy-id 翻译成容器内 index。

## How it slipped past every layer

| Layer | What it was supposed to do | What actually happened |
|---|---|---|
| Helper script 内部逻辑 | 把 `--chips` 参数语义化 | 直接当字符串传给 docker；没区分 host vs container 索引域 |
| KB (NPU-OPS-012) | 当时还没写；NPU-OPS-009/011 只覆盖 bind set | NPU-OPS-012 在 T25.5 之后才写（自发现后倒回去 codify） |
| 代码审查 / agent review | 没有 mechanical critic 扫"传 chip 值给 docker env"模式 | 0 review；script 写完直接 ship |
| 测试 | 没 unit test；没 sanity suite | T31 之前完全没 mechanical test |
| 文档 | SKILL.md 例子全用 `--chips 0,1` | 客户复制例子也用 0,1，不会暴露 |
| 真用户 cold-drive | 唯一能抓的层 | T25.5 是第一次真冷启 + 切非零 chip — bug 立刻暴露 |

**6 层都没接住**。Defense in depth 失败的标准模式：所有上游都信"helper script 自己应该把语义弄对"。

## Fix (landed `a6f3fca`)

- **Code (helper script)**: `run-npu-container.sh` 加 `IN_CONTAINER_CSV` 派生：
  ```bash
  IN_CONTAINER_IDX=()
  for ((i=0; i<${#CHIP_ARR[@]}; i++)); do
    IN_CONTAINER_IDX+=("$i")
  done
  IN_CONTAINER_CSV=$(IFS=,; echo "${IN_CONTAINER_IDX[*]}")
  ```
  然后 `-e ASCEND_RT_VISIBLE_DEVICES="${IN_CONTAINER_CSV}"`（不是 `${CHIPS}`）

- **KB (knowledge/npu-patterns.md NPU-OPS-012)**: 之前已写；T25.5 后倒回去补**正确的 fix recipe**（之前 KB 只说"问题是什么"没说"helper script 怎么改"）

- **KB (knowledge/npu-patterns.md NPU-OPS-014)**: 新增 — A3 上 `repo/` 可能是早期 v0 layout 的 stale 非 git 拷贝；NPU-OPS-009 修了 bind set，但路径漂移本身是独立问题

- **Validation**: V1.4 GRPO smoke 用 `--chips 2,3 + fixed helper` PASS；reward_score 与 T22.7 baseline 在 ±10% 内

- **DEBT 登记**: ROADMAP §6 DEBT-1（auto-detect A3 stale repo）

## What we couldn't fix yet

- **没 sanity suite 跑非零 chip case** → T31 P0h.1 解决（但当时不存在）
- **没 mechanical scanner 扫 "helper script 传 chip 值给 docker env" 这类模式** → 现在仍没；可能值得未来加 OL-N + scanner rule（但优先级低，因为类似 bug 不常见）
- **A3 上 repo stale 检测** → DEBT-1，未实现

## Lessons (永久记录)

1. **Helper script 处理 chip / device / index 类参数必须显式做语义转换** —— "传字符串就完了" 是反模式。Codified as `OL-13` in OPERATIONAL_KNOWLEDGE.md (cross-skill OL-XX catalog).

2. **任何用 `--chips 0,1` 默认参数测试的 helper / skill 都属于"happy-path lucky 测试"** —— 必须用非零值（如 `--chips 4,5`）回归至少一次才能 ship。Codified as bad fixture coverage rule in T30/T31 P0h.1 (sanity suite covers diverse fixture inputs).

3. **KB 知识与代码脱节是 bug 滋生场** —— NPU-OPS-012 当时已"知道"语义，但 helper script 当时不"知道"。需要 mechanical mechanism 让 helper script 受 KB 约束（如 hooks 扫 SKILL.md 引用的 OL；或 sanity suite 测 helper 的语义）。Codified as M5 (agent tool-use signature) in adversarial-audit-design v3 — though full enforcement requires DEBT-6 (P0e workflow critic).

4. **Cold-drive replay 不是可选** —— T22.7 happy path 通过 ≠ skill 真在 work；T25.5 才是真证据。Codified as cross-layer-007 (walk-through-is-not-real-run) + ANTI_PRESSURE P1.

5. **P9 mechanical classifier 必要性** —— 当时 inner monologue 一度想 "换 NPU 7 试试看" 或 "手动 export 一下"；这两条都是 P6（inline workaround）/ P9（infrastructure paper-over）的反例。如果 P9 当时存在，分类时会直接 routing 到 "our-script-bug → 修 script + commit"，省 10 分钟。Codified as P9 in ANTI_PRESSURE_PROTOCOLS.md (added T31).

---

## 见也

- KB 条目: `NPU-OPS-012`（ASCEND_RT_VISIBLE_DEVICES 容器内 index 语义）、`NPU-OPS-014`（A3 stale repo layout）
- OL 条目: `OL-13`（容器内 index vs host phy-id）
- ROADMAP: `DEBT-1`（A3 stale repo auto-detect）
- ANTI_PRESSURE: `P1`（user-watching pressure 跳过 cold-drive），`P6`（inline workaround），`P9`（infra paper-over）
- Cross-layer lesson: `docs/_meta/kb/porting_lessons/cross-layer-007-walk-through-is-not-real-run.md`
- Fix commits: `a6f3fca`（helper script）
