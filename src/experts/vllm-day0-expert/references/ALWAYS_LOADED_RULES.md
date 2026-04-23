# vllm-day0-worker 专属规则

> 读 `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md` 后再读本文。

## OL-03 denylist

**禁读**：
- 通用：`docs/HANDOVER.md` / `docs/porting-journal.md` / `docs/P2-WORKFLOW.md`
  / `docs/DELIVERABLE.md` / `docs/codex-*.md` / `docs/design.md` /
  `docs/dep-matrix.md` / `docs/PORT-SUMMARY.md` / `docs/easyr1-dep-chain-audit.md`
  / `docs/handoff-2026-04-19.md` / `docs/skill-dry-run-2026-04-20.md` /
  `docs/UPGRADE-DRILL-STATUS.md` / `docs/transformers-upgrade-drill.md`
- `upstream/EasyR1` 上 `ascend-port*` / `round3-*` / `round4-*` /
  `ascend-port-e2e-*` 分支
- 其他 expert 的 workspace，特别是 `workspace/vllm-upgrade-*/` —— Stage 2
  shim-adapt sibling，其 **已 ship 匹配 vllm-ascend** 的假设对 Day-0 场景
  不适用，读它的结论会误导
- `workspace/easyr1-port-*`、`workspace/transformers-upgrade-*`、
  `workspace/transformers-day0-*`、`workspace/torch-npu-upgrade-*`、
  `workspace/dep-analysis-*`、`workspace/npu-port-*`

**允许读**：
- 本 expert 自己的 `references/**`
- 消费者 repo 源码（UPSTREAM_REF）
- **上游 vllm + vllm-ascend 源码、release notes、CHANGELOG**（WebFetch
  / git fetch / pip download） —— Day-0 证据的核心源头
- `vllm_utils.py` / `vllm_rollout_spmd.py` / `fsdp_vllm.py` 在 UPSTREAM_REF
  里的当前状态 —— 确定是否已有 shim
- `docs/design/SKILLS_ARCH_TARGET.md` 的 **Day-0 reframing 段**

## OL-08 edit scope

**可写**：
- `$WORKSPACE/**`
- `upstream/<consumer>/Dockerfile.overlay-vllm<MM>*` —— overlay Dockerfile
- `upstream/<consumer>/requirements*.txt` —— 只松 vllm 上界（fixture 路径），
  不碰其它 dep
- **如 outcome B**，以下三个 vllm-adjacent shim 文件（和
  vllm-upgrade-expert 同 scope）：
  - `verl/utils/vllm_utils.py` — CP-002 / VLLMHijack
  - `verl/workers/rollout/vllm_rollout_spmd.py` — EC-03 SamplingParams
  - `verl/workers/sharding_manager/fsdp_vllm.py` — CP-004 TP group
- **如 outcome C-patch（2026-04-23 新增）**：
  - `upstream/vllm-ascend/**/*.py` 在 `ascend-day0-<SESSION_TAG>` 分支上 ——
    产出 vllm-ascend upstream patch 是 Day-0 skill 的正当交付物（skills
    的受众就是 vllm-ascend 维护者）
  - 同样适用于其它华为开源适配层（`triton-ascend/**`、`torch-npu/**`） ——
    但 vllm-day0 session 通常只碰 vllm-ascend
  - **不改**社区 vllm 本身（社区的决定我们无权替 rep）；不改非华为开源的
    任何上游
  - patch 要走 git branch，不直接 mutate image 内 `.py`；overlay image 通过
    `pip install git+...@<branch>` 把 patch 装进去

**禁写**：
- 任何其它 `verl/**/*.py`（是 easyr1-expert 的 domain）
- `Dockerfile.npu` / `Dockerfile.npu-852` / `Dockerfile.npu-torch-*` /
  `Dockerfile.overlay-trans*` —— 其它 expert 的 domain
- 社区 vllm (`upstream/vllm/**`) 本身
- `src/experts/**` 自身

## Outcome 矩阵（2026-04-23 细化）

| Outcome | 含义 | 该做什么 |
|---|---|---|
| A | 直接 pip overlay + V1.3 PASS（无需改任何 upstream） | 写 PROGRESS，存 overlay image |
| B | V1.3 FAIL on consumer-side API drift，3 shim 文件能修 | 改 shim + V1.3 PASS |
| C-patch | V1.3 FAIL because vllm-ascend（或其它华为开源）need code change | 在 `upstream/vllm-ascend/` 开 `ascend-day0-<SESSION_TAG>` 分支改，overlay 装 patch 后 V1.3 PASS，打包 PR-ready diff |
| C-report | Fix 需要社区 vllm 侧做（我们无权改）；或 fix 超出 skill 领域范围 | 写 blocker-report 列最小复现 + suggested fix 交给 vllm-ascend/community owner |

**目标是 A 或 C-patch 且 PASS。** C-report 只在真没法自己解的时候用。

## Target 选择的 pre-probe（必做）

在 `pip download vllm==<TARGET>` 之前，先 probe：
1. `cd upstream/vllm-ascend && git fetch origin --tags && git log origin/main -S '<关键 symbol>'` —— 查 vllm-ascend main 有没有已经适配过该版本的 breaking change
2. 如果 **vllm-ascend main 已 handle** → 换**更新 target**（社区更新的版本），回 step 1；否则当前 session 没 skill 价值
3. 找社区最新 tag + `git log <ascend-cursor>..v<TARGET> --oneline` 看 delta 深度；delta < 10 commit 的可能不值得做 Day-0（换更新 target）

记这步的发现到 PROGRESS.md 的 Phase A。

PreToolUse hook 拦违规 Edit。

## 退出签名

`Worker signed: vllm-day0-worker <ISO-8601-UTC>`
