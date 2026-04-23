# transformers-day0-worker 专属规则

> 读 `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md` 后再读本文。

## OL-03 denylist

**禁读**：
- 通用：`docs/HANDOVER.md` / `docs/porting-journal.md` / `docs/P2-WORKFLOW.md`
  / `docs/DELIVERABLE.md` / `docs/codex-*.md` / `docs/design.md` /
  `docs/dep-matrix.md` / `docs/PORT-SUMMARY.md` / `docs/easyr1-dep-chain-audit.md`
  / `docs/handoff-2026-04-19.md` / `docs/skill-dry-run-2026-04-20.md` /
  `docs/UPGRADE-DRILL-STATUS.md` / `docs/transformers-upgrade-drill.md`
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
- `docs/design/SKILLS_ARCH_TARGET.md` 的 **Day-0 reframing 段**（本 expert
  存在的理由）—— 允许读

读禁读 = round 作废。

## OL-08 edit scope

**可写**：
- `$WORKSPACE/**`
- `upstream/<consumer>/Dockerfile.overlay-trans<version>*` —— 叠加式
  Dockerfile（`FROM base-image` + pip install 新 transformers）
- `upstream/<consumer>/requirements*.txt` —— 只放松 transformers 版本上
  界的 fixture 路径允许；不碰其它 dep pin
- **如果 outcome B（forward-port）**：允许编辑 `npu_flash_attention.py`
  副本（通常放在 `$WORKSPACE/patches/` 下，build 时 COPY 进 image 覆盖
  base 里那份），**不要**直接改 image 内文件系统中的原文件
- **绝不**碰 `verl/**/*.py` —— 本 expert 不做 consumer 代码 port（那是
  easyr1-expert / transformers-upgrade-expert 的域；如果 transformers
  新版真的改了 consumer 必须 port 的 API，本 expert 的 outcome 是 C，
  交接回 orchestrator）

PreToolUse hook 拦违规 Edit。

## 退出签名

`Worker signed: transformers-day0-worker <ISO-8601-UTC>`
