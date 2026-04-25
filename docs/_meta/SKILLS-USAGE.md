# Skills 使用 — 给上游 maintainer 的 5 行入门

```bash
# 1. 装
git clone https://github.com/zhshgmail/easyr1-npu.git && cd easyr1-npu
./src/scripts/install-skills.sh

# 2. 在 Claude Code 里输你那条 skill：
#    /vllm-ascend-day0 --target-delta vllm==0.20.0 --base-image <你的 torch-day0 image>
#    /torch-npu-day0 --target-torch-version 2.12-rc3
#    /transformers-day0 --target-version 5.4
#    /triton-ascend-port --target-triton-version v3.6.0

# 3. skill 跑完后看产物：fork 分支上的 diff + PR_MATERIAL.md（你拿去给自己 repo 开 PR）
```

成果索引 → [`STATUS.md`](../../STATUS.md) §"已完成的 cold-drive"

---

<details>
<summary><b>为什么这 4 个 skill / 它们是给谁的 / skill 跟人怎么分工</b></summary>

**目标读者**：vllm-ascend / torch-npu / triton-ascend / transformers (NPU 集成部分) 的维护者。EasyR1 用户走 [`docs/easyr1/PORT-GUIDE.md`](../easyr1/PORT-GUIDE.md)，不是这里。

**Skill 是给 [Claude Code](https://docs.anthropic.com/claude-code) 的 slash-command**（`~/.claude/skills/<name>/SKILL.md`）。`install-skills.sh` 把本仓的 skill 软链到那里。

**skill 跟你的分工**：
- 你给目标版本（`--target-delta`）。skill 跑：扫漂移 → 按 KB family 写 shim → 在 personal fork 上落 commit → `/drift-port-validate` 验证 → 写 `PR_MATERIAL.md`。
- skill **不会替你开 PR 到你自己的真 upstream**。它在 `gitcode.com/zhengshencn_hwca/<upstream>` 这个 demo fork 上做交付，你拿 `PR_MATERIAL.md` 去你自己仓开 PR。

</details>

<details>
<summary><b>各 skill 的权威 SKILL.md / 知识库 / scanner</b></summary>

| Upstream | SKILL | KB | Scanner |
|---|---|---|---|
| vllm-ascend | [`src/skills/vllm-ascend/port-expert/SKILL.md`](../../src/skills/vllm-ascend/port-expert/SKILL.md) | [`references/KB_INDEX.md`](../../src/skills/vllm-ascend/port-expert/references/KB_INDEX.md) | `kb_drive_test.py`、`sweep.sh` |
| torch-npu | [`SKILL.md`](../../src/skills/torch-npu/port-expert/SKILL.md) | [`KB_INDEX.md`](../../src/skills/torch-npu/port-expert/references/KB_INDEX.md) | 同上 |
| transformers | [`SKILL.md`](../../src/skills/transformers/port-expert/SKILL.md) | [`KB_INDEX.md`](../../src/skills/transformers/port-expert/references/KB_INDEX.md) | 同上 |
| triton-ascend | [`SKILL.md`](../../src/skills/triton-ascend/port-expert/SKILL.md) | [`KB_INDEX.md`](../../src/skills/triton-ascend/port-expert/references/KB_INDEX.md) | git merge / py_compile（fork-merge 模型） |

每条 skill 的 SKILL.md 顶部就是它自己的 usage / argument-hint，比这页全。

</details>

<details>
<summary><b>FAQ</b></summary>

- **Web 模式 Claude Code 也行吗？** 行。Skill install 在跑 `claude` 的那台机器上做。
- **Scanner 不进 skill 也能跑吗？** 行。`python3 src/skills/_shared/scanners/<name>.py --help`。
- **Skill 会动我自己 repo 吗？** 不会。只动 demo fork（`zhengshencn_hwca/<upstream>`）。
- **耗 token？** 一次 day-0 跑 100k–500k tokens（取决于 drift 数）。

</details>
