# torch-npu-upgrade-worker 专属规则

> 第一步：读 `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`（cross-expert OL）
> 再读本文件，然后 KB_INDEX。

## OL-03 (torch-npu-upgrade-worker denylist)

**禁读**：
- 通用 denylist：`docs/_meta/HANDOVER.md` / `docs/easyr1/porting-journal.md` /
  `docs/_archive/P2-WORKFLOW.md` / `docs/easyr1/DELIVERABLE.md` / `docs/_archive/codex-*.md` /
  `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md` / `docs/_meta/design.md` /
  `docs/easyr1/dep-matrix.md` / `docs/easyr1/PORT-SUMMARY.md` /
  `docs/easyr1/easyr1-dep-chain-audit.md` / `docs/_archive/handoff-2026-04-19.md` /
  `docs/_archive/skill-dry-run-2026-04-20.md` / `docs/transformers/UPGRADE-DRILL-STATUS.md`
- `upstream/EasyR1` 中 `ascend-port*` / `ascend-port-e2e-*` /
  `ascend-port-round{3,4}-*` 分支（easyr1-expert 的答案域）
- 其他 expert 的 workspace：`workspace/easyr1-port-*/`、
  `workspace/transformers-upgrade-*/`、`workspace/vllm-upgrade-*/`、
  `workspace/dep-analysis-*/`、`workspace/npu-port-*/`

**允许读**（此 expert 的 prior-art 白名单）：
- `ascend-port-transformers-upgrade` 分支里 **和 torch-stack 相关的
  commit**：`15f9450` (NPU-BUG-004 amd/nvidia prune)、`a18d1f8`
  (NPU-BUG-004 importlib fix)、`cd16649` (Dockerfile.npu triton-ascend
  force-reinstall, NPU-BUG-001) — 这些是 torch-stack 处理的直接 prior art
- 本 expert 自己的 `references/**`
- upstream torch / torch_npu / triton-ascend / vllm-ascend CHANGELOG
  (WebFetch)
- 目标 image 内的 pip-freeze / `/usr/local/python*/lib/python*/site-packages/triton/backends/` 目录检查

读禁读 = round 作废。

## OL-08 (torch-npu-upgrade-worker 可 edit 路径)

**可写**：
- `$WORKSPACE/**`
- `upstream/<consumer>/Dockerfile.npu-torch-*` — 专用 Dockerfile（通常
  以 target image 版本号命名，如 `Dockerfile.npu-torch-2.9`）
- `upstream/<consumer>/requirements*.txt` — 仅限改 torch / torch_npu /
  triton-ascend / torchdata 行；不碰 transformers / vllm 等其它 dep pin
- **极少数情况** 需要改一个 canonical smoke config 文件加
  `use_torch_compile=false`（若 NPU-BUG-003 在新 stack 还复现）：
  - `upstream/<consumer>/examples/qwen2_0_5b_math_grpo_npu_smoke.sh`
  - 改时 **必须在 PROGRESS.md 里注明理由 + cite 重现命令**

**禁写**：
- 任何 `verl/**/*.py`（consumer source code 是 easyr1-expert 的域。
  如果 torch API 改动了需要 consumer 适配，那属于 NPU-CP-001 扩展，
  应该走 orchestrator 向 easyr1-expert 开 follow-up task，不是本 expert）
- `verl/utils/vllm_utils.py` / `vllm_rollout_spmd.py` /
  `fsdp_vllm.py` — 是 vllm-upgrade-expert 的域
- `Dockerfile.npu` / `Dockerfile.npu-852` — 是 transformers-upgrade-expert
  的域
- `Dockerfile.npu-vllm-*` — 是 vllm-upgrade-expert 的域
- `src/skills/**` 自身

PreToolUse hook 拦违规 Edit。

## 退出签名

`Worker signed: torch-npu-upgrade-worker <ISO-8601-UTC>`
