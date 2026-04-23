# vllm-upgrade-worker 专属规则

> 第一步：读 `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`（cross-expert OL）
> 再读本文件（OL-03 denylist + OL-08 edit scope），然后 KB_INDEX。

## OL-03 (vllm-upgrade-worker denylist)

**禁读**：
- 通用 denylist: `docs/_meta/HANDOVER.md` / `docs/easyr1/porting-journal.md` /
  `docs/_archive/P2-WORKFLOW.md` / `docs/easyr1/DELIVERABLE.md` / `docs/_archive/codex-*.md` /
  `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md` / `docs/_meta/design.md` / `docs/easyr1/dep-matrix.md` /
  `docs/easyr1/PORT-SUMMARY.md` / `docs/easyr1/easyr1-dep-chain-audit.md` /
  `docs/_archive/handoff-2026-04-19.md` / `docs/_archive/skill-dry-run-2026-04-20.md` /
  `docs/transformers/UPGRADE-DRILL-STATUS.md`
- `upstream/EasyR1` 中 `ascend-port*` / `ascend-port-e2e-*` /
  `ascend-port-round3-*` / `ascend-port-round4-*` / `ascend-port-e2e-2026*`
  分支（easyr1-expert 的答案域）
- 其他 expert 的 workspace 目录：`workspace/easyr1-port-*/`、
  `workspace/transformers-upgrade-*/`、`workspace/dep-analysis-*/`、
  `workspace/npu-port-*/`

**允许读**（跟 transformers-upgrade-expert 类似的 prior-art 放行原则）：
- `ascend-port-transformers-upgrade` / `ascend-port-transformers-upgrade-reproduce`
  分支里 **仅涉及 vllm 的 commit**：`d213f01` (EC-03 SamplingParams
  read-only)、`2d8ee2c` (CP-004 TP group rename) 是直接 vllm-shim 先例，
  本 expert 读它们属于 prior art，不是 easyr1 答案
- 本 expert 自己的 `references/**` KB
- vllm / vllm-ascend / compressed_tensors 等上游官方 CHANGELOG / release
  notes（WebFetch）
- 目标 image 内的 pip-freeze / 头文件检查

读禁读 = round 作废。

## OL-08 (vllm-upgrade-worker 可 edit 路径)

**可写**：
- `$WORKSPACE/**`
- `upstream/<consumer>/Dockerfile.npu-vllm-*`（如果需要专门 Dockerfile；
  一般不需要，vllm-ascend 随 base image 走）
- `upstream/<consumer>/requirements*.txt`（只改 vllm 行；不碰别的 dep
  pin，那是其它 expert 的地盘）
- **以下 vllm-adjacent shim 文件**（和 EasyR1 master 对得上的位置）：
  - `verl/utils/vllm_utils.py` — CP-002 LoRAModel import + VLLMHijack
  - `verl/workers/rollout/vllm_rollout_spmd.py` — EC-03 SamplingParams
    read-only setattr 处理 + 新 SamplingParams 字段的兼容
  - `verl/workers/sharding_manager/fsdp_vllm.py` — CP-004 TP group
    hasattr 检测 + 新 vllm TP API 的兼容

**禁写**：
- 任何不在上面列出的 `verl/**/*.py`（那是 easyr1-expert 的域，本 expert
  只碰 vllm 粘合层）
- `Dockerfile.npu`（base Dockerfile，是 transformers-upgrade-expert 或
  easyr1-expert 的域）
- `src/experts/**` 自身（本 expert 不自改）
- `docs/` / `knowledge/` 中除 `knowledge/images/<target-slug>.md`
  之外的任何文件

PreToolUse hook 拦违规 Edit。

## 退出签名

`Worker signed: vllm-upgrade-worker <ISO-8601-UTC>`
