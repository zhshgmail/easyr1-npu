# easyr1-npu

把 EasyR1（`hiyouga/EasyR1`）适配到 Ascend 910C (A3) NPU，并沉淀一套针对 NPU 上游（vllm-ascend / torch-npu / transformers / triton-ascend）的可复用版本升级工具链。

最后更新 2026-04-26。

---

## 当前进度

### 已完成

- **EasyR1 v1 在 A3 上完成端到端运行**：文本 RL 流程，rollout + 训练，与 GPU 基线在 entropy_loss 指标上对齐。
  代码：[`zhshgmail/EasyR1` 分支 `ascend-port`](https://github.com/zhshgmail/EasyR1/tree/ascend-port)（20 commit）。
  使用方法：[`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md)。

- **vllm-ascend 版本升级工具链**：可处理社区 vllm 升级时的 API 漂移识别、修复模板、修复后验证。已对社区 vllm `main` 版本完整跑通一遍。
  产出分支：[`zhengshencn_hwca/vllm-ascend` 分支 `vllm-main_cold_20260425`](https://gitcode.com/zhengshencn_hwca/vllm-ascend/tree/vllm-main_cold_20260425)。

- **torch-npu 版本升级工具链**：处理 torch 私有模块路径变化、签名变化、新增字段/方法的扫描和修复。已对 `torch 2.12-rc3` 完整跑通一遍。
  产出分支：[`zhengshencn_hwca/pytorch` 分支 `torch-2.12-rc3_cold_20260425`](https://gitcode.com/zhengshencn_hwca/pytorch/tree/torch-2.12-rc3_cold_20260425)。

- **transformers 版本升级工具链**：检查 `ALL_ATTENTION_FUNCTIONS` 等 NPU 关键集成点的字节级变化。已对 `v5.4` 完成判定（无需修改）。

- **triton-ascend 版本升级工具链（fork-merge 模型）**：`v3.5.0 → v3.6.0` 已完成代码合并 + 9 处 LLVM/Python 漂移定位修复 + C++ 构建通过 + Python 导入通过。
  产出分支：[`zhengshencn_hwca/triton-ascend` 分支 `v3.6.0_manual_porting`](https://gitcode.com/zhengshencn_hwca/triton-ascend/tree/v3.6.0_manual_porting)。

- **triton-ascend NPU 端到端 smoke**：在 A3 容器内用 vendor `triton_ascend-3.2.0` wheel + image 自带 CANN 8.5.2 `bishengir-compile`，跑 `@triton.jit vector_add` on NPU，输出与 torch 参考实现 exact match（max abs err 0.000e+00）。验证脚本：[`src/scripts/smoke_triton_vector_add.py`](src/scripts/smoke_triton_vector_add.py)。

### 进行中

- **triton-ascend main 分支端到端验证**：v3.5.0 → v3.6.0 源码合并通过、构建通过、Python 导入通过；但 main 分支的 libtriton.so 用 LLVM 22 编译，emit 的 MLIR 文本含新语法（`bufferization.to_tensor : memref<…> to tensor<…>`），当前公开的 bishengir-compile（CANN 8.5.x 自带的 0.1.0 / AscendNPU-IR 各分支自建）均基于 LLVM 19，无法解析。需等 Huawei 内部 CI ship 一个 LLVM 22 编译的 bishengir-compile 才能跑通。代码层面修复已经全部到位。

### 待启动

- 为 4 个版本升级工具链各生成一份给上游 PR 用的资料包（diff 摘要、commit 序列、变更说明），方便对应上游仓库的维护者直接采用。
- 客户可重放的 5 分钟示范命令链：每个工具链一条。
- EasyR1 v2（VLM、video 场景）。

---

## 目录

按你要做的事情选：

| 你要做 | 入口 |
|---|---|
| 在 A3 上运行 EasyR1 | [`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md) |
| 安装并使用版本升级工具链（slash command 形式） | [`docs/_meta/SKILLS-USAGE.md`](docs/_meta/SKILLS-USAGE.md) |
| 升级 EasyR1 + 新依赖（image 升级演练） | [`docs/transformers/UPGRADE-DRILL-STATUS.md`](docs/transformers/UPGRADE-DRILL-STATUS.md) |
| 查 vllm-ascend 升级流程详解 | [`docs/vllm-ascend/PORTING-GUIDE.md`](docs/vllm-ascend/PORTING-GUIDE.md) |
| 查 torch-npu 升级流程详解 | [`docs/torch-npu/PORTING-GUIDE.md`](docs/torch-npu/PORTING-GUIDE.md) |
| 查 transformers 升级流程详解 | [`src/skills/transformers/port-expert/SKILL.md`](src/skills/transformers/port-expert/SKILL.md) |
| 查 triton-ascend 升级流程详解 | [`src/skills/triton-ascend/port-expert/SKILL.md`](src/skills/triton-ascend/port-expert/SKILL.md) |
| 查项目设计与任务分解 | [`docs/_meta/design.md`](docs/_meta/design.md) |
| 查工具链各自的 KB（漂移目录与修复模板） | `src/skills/<工具链名>/port-expert/references/KB_INDEX.md` |
| 查 NPU 操作与适配模式总集 | [`knowledge/npu-patterns.md`](knowledge/npu-patterns.md) |
| 查跨层经验沉淀（教训库） | [`docs/_meta/kb/porting_lessons/`](docs/_meta/kb/porting_lessons/) |

---

## 责任边界

工具链产出的修改落在 `gitcode.com/zhengshencn_hwca/<上游仓>` 这一组演示性分支，作为可见的修复样例与验证凭据。**正式 PR 由对应上游仓的维护者基于这些分支提交到他们自己的官方仓库**——本仓不替代该流程。
