# 版本升级工具链使用说明

本仓为以下 4 个 NPU 上游仓库分别提供一条版本升级工具链：vllm-ascend、torch-npu、transformers、triton-ascend。每条工具链以 [Claude Code](https://docs.anthropic.com/claude-code) 的 slash command 形式发布，安装后即可在 Claude Code 中以一条命令调用。

## 安装

```bash
git clone https://github.com/zhshgmail/easyr1-npu.git
cd easyr1-npu
./src/scripts/install-skills.sh
```

安装完成后，可以在 Claude Code 中输入 `/help` 看到工具链已注册。

## 调用方式

在 Claude Code 中输入对应命令并补全参数：

| 工具链 | 命令 | 必填参数 |
|---|---|---|
| vllm-ascend 版本升级 | `/vllm-ascend-day0` | `--target-delta`（目标 vllm 版本，如 `vllm==0.20.0`）<br>`--base-image`（已部署的 torch-day0 base image 标签） |
| torch-npu 版本升级 | `/torch-npu-day0` | `--target-torch-version`（如 `2.12-rc3`） |
| transformers 版本升级 | `/transformers-day0` | `--target-version`（如 `5.4`） |
| triton-ascend 版本升级 | `/triton-ascend-port` | `--target-triton-version`（如 `v3.6.0`） |

工具链运行后会执行以下流程：扫描漂移、按 KB 模板生成修复、推送到对应上游的演示分支、跑验证脚本、生成给上游维护者用的修复资料包。

## 工具链产出

每条工具链跑完会产出：

- 一个演示分支，落在 `gitcode.com/zhengshencn_hwca/<对应上游仓>`
- 一份修复资料包（`PR_MATERIAL.md`），含 diff 摘要、commit 列表、给上游维护者的简短说明

最近一次运行的产出汇总在主 [`README.md`](../../README.md) 的"已完成"段。

## 单独使用扫描脚本

如果不通过工具链调用，扫描脚本本身也可以单独运行：

```bash
python3 src/skills/_shared/scanners/<scanner-name>.py --help
```

## 责任边界

工具链产出的修改落在 `gitcode.com/zhengshencn_hwca/<上游仓>` 这一组演示性分支。**正式 PR 由对应上游仓的维护者基于这些分支提交到他们自己的官方仓库。**
