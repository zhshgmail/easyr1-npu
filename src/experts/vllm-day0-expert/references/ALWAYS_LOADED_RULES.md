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

**禁写**：
- 任何其它 `verl/**/*.py`（是 easyr1-expert 的 domain）
- `Dockerfile.npu` / `Dockerfile.npu-852` / `Dockerfile.npu-torch-*` /
  `Dockerfile.overlay-trans*` —— 其它 expert 的 domain
- **vllm-ascend 源码**：本 expert 不动 vllm-ascend；需要打 vllm-ascend
  patch 的话是 **outcome C** 的责任：emit blocker-report 让 NPU 团队
  upstream 修
- `src/experts/**` 自身

PreToolUse hook 拦违规 Edit。

## 退出签名

`Worker signed: vllm-day0-worker <ISO-8601-UTC>`
