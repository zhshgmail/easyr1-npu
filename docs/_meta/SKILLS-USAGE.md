# 升级工具链使用说明

本仓为 4 个 NPU 上游（vllm-ascend / torch-npu / transformers / triton-ascend）以及一条集成 overlay build 流程，提供 [Claude Code](https://docs.anthropic.com/claude-code) slash command 形式的工具链。安装后即可在 Claude Code 中以一条命令调用。

## 安装

```bash
git clone https://github.com/zhshgmail/easyr1-npu.git
cd easyr1-npu
./src/scripts/install-skills.sh
```

`install-skills.sh` 会把 `repo/src/skills/**/SKILL.md` 用 symlink 挂到 `~/.claude/skills/<name>/`。装完在 Claude Code 中输入 `/help` 即可看到工具链已注册。

## 调用

| 工具链 | 命令 | 必填参数 |
|---|---|---|
| vllm-ascend day-0 升级 | `/vllm-ascend-day0` | `--target-delta`（如 `vllm==0.21.0`）<br>`--base-image`（已部署的 torch-day0 base image 标签） |
| torch-npu day-0 升级 | `/torch-npu-day0` | `--target-torch-version`（如 `2.12.0-rc3`）<br>`--target-torch-npu-version`（如 `2.12.0rc1`）<br>`--base-image` |
| transformers day-0 升级 | `/transformers-day0` | `--target-transformers-version`（如 `5.6.2`，注意：upstream tag 是 `v5.6.2` 含前缀 `v`） |
| triton-ascend port | `/triton-ascend-port` | `--target-triton-version`（如 `v3.6.0`） |
| sglang NPU 3-axis 版本验证 | `/sglang-npu-day0` | `--target-sglang-tag`（如 `main` / `v0.5.10.post1`）<br>`--target-kernel-npu-tag`（如 `2026.04.15.rc4`）<br>`--target-cann-version`（如 `8.5.0`）<br>`--device-type a3\|a2`（默认 `a3`） |
| 整合 overlay build + V1.4 e2e | `/integrated-overlay-build` | `--base-image`<br>`--easyr1-branch`<br>`--output-tag`（如 `easyr1-npu:integrated-<DATE>`） |

工具链运行流程：扫描漂移（commit range 或 byte-compare）→ 按 KB 模板生成修复 → 推到对应上游的演示性 fork 分支 → 跑验证脚本 → 生成给上游维护者用的修复资料包（`PR_MATERIAL.md`）。

## 工具链产出

每条工具链跑完产出：

- 一个演示性 fork 分支，分支命名约定 `ascend-port/<target-version-slug>`（详见 [`UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md)）
- 分支根目录的 `PR_MATERIAL.md`：diff 摘要 + commit 列表 + reproducer + 给上游维护者的简短说明

## 单独使用扫描脚本

每条 day-0 / port skill 都有可独立运行的扫描脚本：

```bash
# vllm-ascend Mode Sweep（v0.20.0..origin/main 漂移扫描）
bash src/skills/vllm-ascend/port-expert/scripts/sweep.sh \
  --commit-range v0.20.0..origin/main \
  --vllm-path <community-vllm-clone> \
  --vllm-ascend-path <vllm-ascend-clone>

# torch-npu Mode B（社区 pytorch 版本对扫描）
bash src/skills/torch-npu/port-expert/scripts/sweep.sh \
  --baseline v2.11.0 --target v2.12.0-rc3 \
  --pt-repo <community-pytorch-clone> \
  --torch-npu-path <torch-npu-clone>
```

每条 SKILL.md 里都列了 prereq + 完整命令例子。

## Hooks（critic 自动触发）

本仓配置了 [Claude Code hooks](https://code.claude.com/docs/en/hooks.md)，以下时机自动触发 `/porting-self-challenge` 自查：

- **SessionEnd**：每次 Claude Code 会话结束，按 `docs/_meta/kb/challenge_patterns/` 11 条问题做一次自查，必要时把新教训写回 `docs/_meta/kb/porting_lessons/`。
- **PostToolUse `git commit`**：每次提交后由 `.claude/hooks/critic-on-significant-commit.sh` 检查 commit message；若是 feat/fix/perf 或动了 SKILL/KB/port-expert，给 session 留 `[CRITIC-HOOK]` 提示，提醒在对外宣布完成前先跑 `/porting-self-challenge`。

配置在 `.claude/settings.json`（仓内项目级，团队共享）。如要 opt-out，删除或改对应条目。

## 责任边界

工具链产出落在演示性 fork 分支（[`UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md) 表）。**正式 PR 由对应上游仓的维护者基于这些分支提交到他们自己的官方仓库**——本仓不替代该流程。
