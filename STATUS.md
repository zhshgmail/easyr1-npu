# 项目状态

> 一眼看懂"做了什么 / 在做什么 / 还没做"。详情链点进去。
> 最后更新 2026-04-25。

## 这周做完的（可对外展示）

| # | 成果 | 物 |
|---|---|---|
| 1 | EasyR1 在 A3 上跑通（v1，文本 RL，已与 GPU 基线对齐） | [`zhshgmail/EasyR1` 分支 `ascend-port`](https://github.com/zhshgmail/EasyR1/tree/ascend-port)（20 commit） |
| 2 | 4 个 NPU 上游 port-expert skill 骨架 + 对应 KB | `src/skills/{vllm-ascend,torch-npu,transformers,triton-ascend}/port-expert/` |
| 3 | vllm-ascend day-0 cold-drive 成功（target = community vllm `main`） | [fork branch `vllm-main_cold_20260425`](https://gitcode.com/zhengshencn_hwca/vllm-ascend/tree/vllm-main_cold_20260425)，13/13 PASS |
| 4 | torch-npu day-0 cold-drive 成功（target = `torch 2.12-rc3`） | [fork branch `torch-2.12-rc3_cold_20260425`](https://gitcode.com/zhengshencn_hwca/pytorch/tree/torch-2.12-rc3_cold_20260425)，6/6 PASS |
| 5 | transformers day-0 cold-drive 成功（target = `v5.4`，分类为无漂移） | 独立复核确认 outcome A |
| 6 | triton-ascend v3.5.0 → community v3.6.0 手动迁移（fork-merge 模型） | [fork branch `v3.6.0_manual_porting`](https://gitcode.com/zhengshencn_hwca/triton-ascend/tree/v3.6.0_manual_porting)，9 处具体漂移定位+修复，C++ build PASS, `import triton` PASS |
| 7 | KB / 教训沉淀 | `knowledge/npu-patterns.md`（+1 = NPU-OPS-011）；`docs/_meta/kb/porting_lessons/`（+1 = cross-layer-005） |
| 8 | 给上游 maintainer 的 skill 使用入门 | [`docs/_meta/SKILLS-USAGE.md`](docs/_meta/SKILLS-USAGE.md) |

## 在做的

- triton-ascend v3.6.0 端到端 NPU smoke：`bishengir-compile` 当前 CANN 8.5.x 二进制不认 `--link-aicore-bitcode`（这个 flag 定义在 [`AscendNPU-IR`](https://gitcode.com/Ascend/AscendNPU-IR) submodule 的 `Options.td:585`），需从源码 build。LLVM 22 + 79 处 Huawei patch 已 apply，build at `-j 2`（共享主机限制）进行中。pass 标准：`python3 smoke_triton_vector_add.py` 输出 `PASS`。

## 还没做

- 4 个 cold-drive 的 `PR_MATERIAL.md` 生成（diff 摘要 + commit hash + 给上游的简短解释，让 upstream maintainer 能直接拿去开 PR）
- 客户可演示的"输入参数 → fork 分支"的 5 分钟 reproducer 命令链（每个 skill 一条）
- EasyR1 v2（VLM、video）

## 项目目标

EasyR1 (master, 2026-04) → Ascend 910C (A3)，并沉淀**可复用的 GPU→NPU 上游迁移 skill 体系**。详见 [`docs/_meta/design.md`](docs/_meta/design.md)。

## 入口索引

- 跑 EasyR1 → [`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md)
- 上游 maintainer 用 skill → [`docs/_meta/SKILLS-USAGE.md`](docs/_meta/SKILLS-USAGE.md)
- 全部用户路径选择 → [`README.md`](README.md)
- 各 skill 的 KB → `src/skills/<X>/port-expert/references/KB_INDEX.md`
- 跨层教训 → [`docs/_meta/kb/porting_lessons/`](docs/_meta/kb/porting_lessons/)
- NPU 操作模式 → [`knowledge/npu-patterns.md`](knowledge/npu-patterns.md)
