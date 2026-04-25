# 项目状态 / Status

> **唯一权威的"做了什么 / 正在做什么 / 还没做什么"主页。**
> 对外给客户、对内交接、对自己当看板都看这页，不再到处翻。
> 任何成果先更新这里，再考虑别处。
>
> 仓库主入口（README）按用户路径组织；这页按时间线和工作项组织。
>
> 最后更新：2026-04-25

---

## 项目目标（一句话）

把 EasyR1 (master, 2026-04) 移植到 Ascend 910C (A3) NPU 上，并沉淀**可复用的 GPU→NPU 移植 skill 体系**（不只是 EasyR1，而是上游升级、版本漂移识别、NPU 适配整套）。

详细见 [`docs/_meta/design.md`](docs/_meta/design.md)。

---

## 已完成（cumulative，可对外展示）

### 1. EasyR1 在 A3 上跑通

- **代码改动落在客户可见的 fork**：[`zhshgmail/EasyR1`](https://github.com/zhshgmail/EasyR1) `ascend-port` 分支，20 个 commit。
- **运行入口**：[`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md)
- **依赖 image**：`quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`
- 状态：rollout + RL 训练在 V1.4 entropy_loss 上对齐 GPU 基线（exact match）

### 2. 4 个 NPU 上游 port-expert skill 骨架（可复用核心交付物）

| Skill | 作用 | 入口文件 | KB |
|---|---|---|---|
| vllm-ascend day-0 | 给一个新 vllm 版本，扫漂移、写 shim、验证 | `src/skills/vllm-ascend/port-expert/SKILL.md` | `references/KB_INDEX.md` |
| torch-npu day-0 | 给一个新 torch 版本，扫漂移、写 shim、验证 | `src/skills/torch-npu/port-expert/SKILL.md` | 同上 |
| transformers day-0 | 给一个新 transformers 版本，扫漂移、写 shim、验证 | `src/skills/transformers/port-expert/SKILL.md` | 同上 |
| triton-ascend port | 给一个新 community triton tag，做 fork-merge 升级 | `src/skills/triton-ascend/port-expert/SKILL.md` | 同上 |

**漂移分类法**：F1–F8 + F2-path-move（详见 `knowledge/npu-patterns.md`）。
**扫描器工具**：`kb_drive_test.py`、`check_f4.py`、`check_f7_f8.py`、`extract_imports.py`、`check_drift.py`、`check_sig_drift.py`、`sweep.sh`（在 `src/skills/_shared/scanners/`）。

### 3. 真实 cold-start 端到端验证（4 个 skill 各跑了一遍真上游）

| Skill | 目标版本 | Fork 分支 | 验证 | 状态 |
|---|---|---|---|---|
| vllm-ascend | community vllm `main` | [`zhengshencn_hwca/vllm-ascend` `vllm-main_cold_20260425`](https://gitcode.com/zhengshencn_hwca/vllm-ascend/tree/vllm-main_cold_20260425) | 13/13 PASS | ✅ Done |
| torch-npu | torch `2.12-rc3` | [`zhengshencn_hwca/pytorch` `torch-2.12-rc3_cold_20260425`](https://gitcode.com/zhengshencn_hwca/pytorch/tree/torch-2.12-rc3_cold_20260425) | 6/6 PASS | ✅ Done |
| transformers | community `v5.4` | classified outcome A（无漂移，无需 patch） | 独立复核确认 | ✅ Done |
| triton-ascend | community `v3.6.0` | [`zhengshencn_hwca/triton-ascend` `v3.6.0_manual_porting`](https://gitcode.com/zhengshencn_hwca/triton-ascend/tree/v3.6.0_manual_porting) | merge + 9 处漂移修复 + import OK | ⏳ 端到端 NPU smoke 进行中 |

每个 cold-drive 的 fork 分支命名遵循 `<target-version>_auto_porting`（或 `_manual_porting`）约定，可作为给上游的 PR 草稿。

### 4. KB / 教训沉淀（决定下次怎么不再犯）

- **NPU 操作模式**：`knowledge/npu-patterns.md`（25 条 → 26 条；本周新增 `NPU-OPS-011`：禁止 rm+recreate 试错容器配置；先 `docker inspect` 工作中的 sibling）。
- **跨层教训库（critic）**：`docs/_meta/kb/porting_lessons/`（5 条；本周新增 `cross-layer-005`：开源依赖调研充分前不要过早断言"out of scope"）。
- **triton-ascend 9 处漂移**：每条记录 file:line + community 改动 + 修复 recipe，下次升 v3.7.0 表格匹配，见 `src/skills/triton-ascend/port-expert/references/KB_INDEX.md`。

---

## 正在进行（in flight）

### triton-ascend v3.6.0 端到端 NPU smoke（T6+T9）

- **当前状态**：build 阶段。`bishengir-compile` 不在 CANN 8.5.x release 二进制里（CANN 8.5.2 自带的版本不认 `--link-aicore-bitcode` flag），需从 AscendNPU-IR submodule（`Options.td:585` 定义这个 flag）源码构建。
- **路线**：A3 上 LLVM 22 (cd708029) + torch-mlir (155680c0) tarball 已下载并 untar；Huawei 79 处 patch 准备 apply（容器内有 `patch` 工具，前一次 host 没装 `patch`）；apply 完后跑 `build-tools/build.sh -j 2`（A3 共享主机，限 2 线程）。
- **预计**：LLVM build at -j2 ≈ 2-4 小时；ccache 后续 rebuild 快。
- **Smoke pass 标准**：A3 容器内 `python3 smoke_triton_vector_add.py` 输出 `PASS, max abs err < 1e-6`。

### EasyR1 → 新 transformers 5.x / vllm-ascend 升级链路（drill）

- 演练状态见 [`docs/transformers/UPGRADE-DRILL-STATUS.md`](docs/transformers/UPGRADE-DRILL-STATUS.md)
- 目前实测 path 4（image-upgrade-drill 7 步）已跑通早期阶段。

---

## 还没做（not started / queued）

| 工作 | 优先级 | 备注 |
|---|---|---|
| 把 4 个 cold-drive 的真实 fork commit 整理成"上游 PR 草稿包"（PR_MATERIAL.md + diff + reproducer 命令）| 高 | 客户可见成果直接交付 |
| 给客户 < 5 分钟可重放的 reproducer 命令链（每个 skill 一条） | 高 | "把项目演示成可演示的"必经之路 |
| EasyR1 v2 / 新版（含 VLM、video） | 中 | 当前 v1 文本对齐通过 |
| `liger-kernel` 通过 triton-ascend 在 NPU 上启用 | 低 | 性能调优类，等 v1 端到端打通后再说 |

---

## 路线图（下 1-2 周）

1. 完成 triton-ascend v3.6.0 端到端 smoke，把这套案例（v3.5.0 → v3.6.0 manual porting，含 9 处漂移 + bishengir 自建）合并写入 `KB_INDEX.md` 并更新本页 case registry。
2. 给 4 个 port-expert skill 各写一个**客户可演示的 5 分钟 reproducer**（"输入：fork URL / 目标版本；输出：合并的分支 + 验证报告"）。
3. 给项目交付物做**一次完整 demo 流程**：从一份 EasyR1 master 到 A3 上跑起来，所有依赖都通过 skill 自动驱动。

---

## 文件指南（按使命读）

| 想看 | 读哪份 |
|---|---|
| **本周状态 / 进展 / 路线图（这页）** | `STATUS.md`（本文件） |
| 用户路径选择（5 类用户） | `README.md` |
| 完整设计 / 任务分解 / 依赖矩阵 | `docs/_meta/design.md`、`docs/_meta/MODULE-PORT-STATUS.md`、`docs/dep-matrix.md` |
| 怎么把 EasyR1 跑到 A3 | `docs/easyr1/PORT-GUIDE.md` |
| 怎么用 4 个 port-expert skill 做新版本移植 | `docs/_meta/SKILLS-GUIDE.md` |
| NPU 操作模式 / OPS 教训 | `knowledge/npu-patterns.md` |
| 跨层 critic 教训 | `docs/_meta/kb/porting_lessons/` |
| 各 skill 知识库（漂移目录） | `src/skills/<X>/port-expert/references/KB_INDEX.md` |
