# transformers-day0-worker 专属规则

> 读 `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md` 后再读本文。

## OL-03 denylist

**禁读**：
- 通用：`docs/_meta/HANDOVER.md` / `docs/easyr1/porting-journal.md` / `docs/_archive/P2-WORKFLOW.md`
  / `docs/easyr1/DELIVERABLE.md` / `docs/_archive/codex-*.md` / `docs/_meta/design.md` /
  `docs/easyr1/dep-matrix.md` / `docs/easyr1/PORT-SUMMARY.md` / `docs/easyr1/easyr1-dep-chain-audit.md`
  / `docs/_archive/handoff-2026-04-19.md` / `docs/_archive/skill-dry-run-2026-04-20.md` /
  `docs/transformers/UPGRADE-DRILL-STATUS.md` / `docs/transformers/transformers-upgrade-drill.md`
- `upstream/EasyR1` 上 `ascend-port*` / `ascend-port-round3-*` /
  `ascend-port-round4-*` / `ascend-port-e2e-*` 分支（easyr1-expert 的答案域）
- 其他 expert 的 workspace：`workspace/easyr1-port-*/`、
  `workspace/transformers-upgrade-*/`（Stage 2 shim-adapt expert；其结论对
  Day-0 场景不适用，reading 会误导），`workspace/vllm-upgrade-*/`、
  `workspace/torch-npu-upgrade-*/`、`workspace/dep-analysis-*/`、
  `workspace/npu-port-*/`

**允许读**：
- 本 expert 自己的 `references/**`
- 消费者 repo 源码（UPSTREAM_REF）
- **上游 transformers 源码 + 历史 release notes**（GitHub API / WebFetch） —
  本 expert 的核心参考，必读
- NPU image 里的 `transformers.integrations.npu_flash_attention` 源码 —
  Day-0 判断 sig/契约变化的关键
- `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md` 的 **Day-0 reframing 段**（本 expert
  存在的理由）—— 允许读

读禁读 = round 作废。

## OL-08 edit scope

**可写**：
- `$WORKSPACE/**`
- `upstream/<consumer>/Dockerfile.overlay-trans<version>*` —— 叠加式
  Dockerfile（`FROM base-image` + pip install 新 transformers）
- `upstream/<consumer>/requirements*.txt` —— 只放松 transformers 版本上
  界的 fixture 路径允许；不碰其它 dep pin
- **如果 outcome B（forward-port NPU FA 适配）**：允许编辑
  `npu_flash_attention.py` 副本（通常放在 `$WORKSPACE/patches/` 下，
  build 时 COPY 进 image 覆盖 base 里那份），**不要**直接改 image 内
  文件系统中的原文件
- **EC-02 shim（if UPSTREAM_REF=master 场景）**：
  `verl/workers/fsdp_workers.py` 中 `no_init_weights` 的 try/except
  import。master 分支不带这个 shim；如果 UPSTREAM_REF 是纯 master
  而不是 baseline-working port ref，必须 apply。参照
  `easyr1/port-expert/references/ERROR_CORRECTIONS.md §EC-02` 的 try/except
  template。**这是 2026-04-23 vllm-day0 wet-run 暴露的通用教训**：day0
  expert 在 fixture=master 场景下需要 apply 基础 shim，即使 target 版本
  没引入新的 API drift
- **其它 `verl/**/*.py` 不碰** —— consumer 代码的大规模 port 是
  easyr1-expert 的域；如果 transformers 新版真的改了 consumer 必须 port
  的 API 超出 EC-02 范围，本 expert 的 outcome 是 C，交接回 orchestrator
- **如 outcome C-patch（2026-04-23 新增）**：允许编辑以下华为开源适配层
  在 `ascend-day0-<SESSION_TAG>` 分支上的 `.py`：
  - `upstream/transformers/src/transformers/integrations/npu_*.py`
    （NPU-specific integration 文件，Ascend 团队贡献的）
  - 不改社区 transformers 本身（非 npu_* 部分属于社区 maintainer 的决定）
  - patch 要走 git branch + Dockerfile COPY overlay；不直接 mutate image
    内 .py。产出物是 PR-ready diff for transformers upstream's NPU
    integration owner
- **Smoke harness 允许新建**（if UPSTREAM_REF=master 场景）：
  `upstream/<consumer>/scripts/smoke_v11_device.py`,
  `upstream/<consumer>/scripts/smoke_v13_rollout.py`,
  `upstream/<consumer>/examples/qwen2_0_5b_math_grpo_npu_smoke.sh`.
  Master 没这些脚本；写 minimal 版本参照
  `references/patterns/domains/smoke-harness-minimal.md`（待补），
  不依赖 `verl.utils.device`（那是 CP-001 产物，master 没有）

PreToolUse hook 拦违规 Edit。

## 退出签名

`Worker signed: transformers-day0-worker <ISO-8601-UTC>`
