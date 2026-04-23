# transformers-upgrade-worker 专属规则（补充到 _shared 通用 OL 之上）

> **Worker 必须在 Phase A 写任何代码前**读完：
>  1. 本文件（expert-specific）
>  2. `../../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`（cross-expert OL）
>
> 本文件只列**本 expert 特有**的 OL-03 denylist 和 OL-08 edit scope。
> 通用规则（OL-01/02/04/04b/05/05b/06/07/09/10）见 `_shared/…/ALWAYS_LOADED_UNIVERSAL.md`。

---

## OL-03 (transformers-upgrade-worker 专用 denylist)

在 agent context 里**禁读**：

- `docs/_meta/HANDOVER.md` / `docs/easyr1/porting-journal.md` / `docs/_archive/P2-WORKFLOW.md` /
  `docs/easyr1/DELIVERABLE.md` / `docs/_archive/codex-*.md` / `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md` /
  `docs/_meta/design.md` / `docs/easyr1/dep-matrix.md` / `docs/easyr1/PORT-SUMMARY.md` /
  `docs/easyr1/easyr1-dep-chain-audit.md` / `docs/_archive/handoff-2026-04-19.md` /
  `docs/_archive/skill-dry-run-2026-04-20.md`
- `docs/transformers/UPGRADE-DRILL-STATUS.md`（同 topic 的 status 历史，会 leak 结论）
- **`upstream/EasyR1` 中 `ascend-port`、`ascend-port-e2e-round*`、
  `ascend-port-round3-*`、`ascend-port-round4-*` 分支的任何内容**
  （这些是 easyr1-expert 的答案域，本 expert 不看）
- **`src/experts/easyr1-expert/workspace/easyr1-port-round*-*/`**（round3/4
  cold-drive workspace）

**本 expert 允许读**（和 easyr1-expert denylist 的关键区别）：

- `upstream/EasyR1` 中 `ascend-port-transformers-upgrade` / 
  `ascend-port-transformers-upgrade-reproduce` 分支 — 这是本 expert 的
  **先行 drill**，commit history 就是本 expert 的 prior art，可以参考
- `docs/transformers/transformers-upgrade-drill.md` — 本 expert 的 drill report，是
  reference material，不是 cheat sheet

原因：denylist 的目的是"让 cold-drive 每次都面对 upstream 原始状态"。对
easyr1-expert 来说，`ascend-port*` 是答案；对本 expert 来说，**本 expert
的 drill 分支就是 prior art，属于 KB 的历史沉淀**，应该读。

读禁读文件 = 作弊 = round 作废。

## OL-08 (transformers-upgrade-worker 允许 edit 的路径)

本 expert 可以写/改：

- ✅ `upstream/<consumer>/Dockerfile.npu-<target-version>*` — 新 Dockerfile
- ✅ `upstream/<consumer>/requirements*.txt` — 如果 target image 的 pip-freeze
  需要调
- ✅ **极少量 backcompat-shim `.py`**，仅限以下已知文件（从 drill 实证总结）：
  - `verl/workers/fsdp_workers.py`（`no_init_weights` import 位置变了）
  - `verl/workers/rollout/vllm_rollout_spmd.py`（SamplingParams read-only property）
  - 其它如果新 transformers/vllm API 又 rename，**必须先 grep ERROR_CORRECTIONS.md
    看有没有 EC 对应**，再小改；不许在未列入 EC 的 `.py` 文件上批量改动
- ✅ 自己的 `workspace/transformers-upgrade-{SESSION_TAG}/`
- ✅ 自己新建的 `examples/qwen2_0_5b_math_grpo_npu_validate_<target>.sh`
  （validation smoke 脚本，复用 V1.4 config + target-image-band 的 baseline 值）

**禁止 edit**：

- ❌ `upstream/<consumer>/verl/**` 除上面列出的 2 个 shim 文件 — 这是
  easyr1-expert 的 domain，本 expert 不碰
- ❌ `upstream/<consumer>/Dockerfile.npu`（不带 `-<version>` 后缀的）— 是
  source image 的 Dockerfile，不要覆写
- ❌ `src/experts/**` 自身代码 — 本 expert 不自改
- ❌ `docs/` / `knowledge/`（除 P1 要求写的 `knowledge/images/<target-slug>.md`）
- ❌ 其他 host / 其他用户的目录

PreToolUse hook 会拦违规 Edit。

---

## 加载顺序（和 universal 规则配合）

1. **Phase A 开始**：先读 `_shared/references/ALWAYS_LOADED_UNIVERSAL.md`
   再读本文件，然后读 `KB_INDEX.md`
2. **Phase B 前**：按需加载 `patterns/domains/*.md`（本 expert 的 pattern
   比 easyr1-expert 少，主要是 `image-diff-shim.md` / `dockerfile-target.md`）
3. **Phase D fail 时**：defer-load `ERROR_CORRECTIONS.md` 把 traceback
   match 到 EC-NN；查 `PLATFORM_BUGS.md` NPU-BUG-NN

## 退出协议（和 universal 一致）

签名：`Worker signed: transformers-upgrade-worker <ISO-8601-UTC>`
