# easyr1-npu

把 EasyR1（`hiyouga/EasyR1`）适配到 Ascend 910C (A3) NPU，并沉淀一套针对 NPU 上游（vllm-ascend / torch-npu / transformers / triton-ascend）的可复用版本升级工具链。

最后更新 2026-04-28。

> **新用户从这里开始 → [`ONBOARDING.md`](ONBOARDING.md)（一页 quickstart，两条已验证路径）**
>
> 各 NPU 上游的 personal fork URL + 当前 `ascend-port/<target>` 分支 + PR_MATERIAL 链接，全部统一记录在 [`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md)。下面进度行里的分支链接如与 ledger 不一致，以 ledger 为准。

---

## 当前进度

### 已完成

- **EasyR1 v1 在 A3 上完成端到端运行**：文本 RL 流程，rollout + 训练，与 GPU 基线在 entropy_loss 指标上对齐。
  代码：[`zhshgmail/EasyR1` 分支 `ascend-port`](https://github.com/zhshgmail/EasyR1/tree/ascend-port)（20 commit）。
  使用方法：[`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md)。

- **vllm-ascend 版本升级工具链**：可处理社区 vllm 升级时的 API 漂移识别、修复模板、修复后验证。已对社区 vllm `main` 版本完整跑通一遍。
  产出分支：[`zhshgmail/vllm-ascend` `ascend-port/vllm-main`](https://github.com/zhshgmail/vllm-ascend/tree/ascend-port/vllm-main)（含 `PR_MATERIAL.md`）。

- **torch-npu 版本升级工具链**：处理 torch 私有模块路径变化、签名变化、新增字段/方法的扫描和修复。已对 `torch 2.12-rc3` 完整跑通一遍。
  产出分支：[`zhengshencn_hwca/pytorch` `ascend-port/torch-2.12-rc3`](https://gitcode.com/zhengshencn_hwca/pytorch/tree/ascend-port/torch-2.12-rc3)（含 `PR_MATERIAL.md`）。

- **transformers 版本升级工具链**：检查 `ALL_ATTENTION_FUNCTIONS` 等 NPU 关键集成点的字节级变化。已对 `v5.4` 完成判定（outcome A，无需修改）。Marker 分支 [`zhshgmail/transformers` `ascend-port/transformers-v5.4`](https://github.com/zhshgmail/transformers/tree/ascend-port/transformers-v5.4)；hand-off 文档在 [`docs/transformers/PR_MATERIAL_v5.4_outcome_A.md`](docs/transformers/PR_MATERIAL_v5.4_outcome_A.md)。

- **triton-ascend 版本升级工具链（fork-merge 模型）**：`v3.5.0 → v3.6.0` 已完成代码合并 + 9 处 LLVM/Python 漂移定位修复 + C++ 构建通过 + Python 导入通过。
  产出分支：[`zhengshencn_hwca/triton-ascend` `ascend-port/triton-v3.6.0`](https://gitcode.com/zhengshencn_hwca/triton-ascend/tree/ascend-port/triton-v3.6.0)（含 `PR_MATERIAL.md`）。

- **triton-ascend NPU 端到端 smoke**：在 A3 容器内用 vendor `triton_ascend-3.2.0` wheel + image 自带 CANN 8.5.2 `bishengir-compile`，跑 `@triton.jit vector_add` on NPU，输出与 torch 参考实现 exact match（max abs err 0.000e+00）。验证脚本：[`src/scripts/smoke_triton_vector_add.py`](src/scripts/smoke_triton_vector_add.py)。

- **EasyR1 v2 (整合 overlay) e2e** (2026-04-28)：在 `easyr1-npu:integrated-20260427` 镜像（vllm 0.20.0 + torch 2.11 + 4 个 ascend-port 上游 overlay + EasyR1 master 去掉 transformers 上限）上跑 V1.4 GSM8K-style GRPO smoke 通过——2 GRPO step + post-train val，10 分钟，无异常，checkpoint 保存。详见 [`docs/easyr1/PORT-GUIDE-v2-integrated.md`](docs/easyr1/PORT-GUIDE-v2-integrated.md)。

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
| **看每条工具链当前在哪个 fork / 分支 / PR_MATERIAL 文档** | [`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md) |
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

工具链产出的修改落在演示性 fork 分支（每条上游对应的 personal fork 上的 `ascend-port/<target-version-slug>` 分支，权威表见 [`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md)）。每条分支根目录含 `PR_MATERIAL.md`，是给上游 maintainer 的 PR 包（diff、commit 序列、reproducer、已知边界）。**正式 PR 由对应上游仓的维护者基于这些分支 + `PR_MATERIAL.md` 提交到他们自己的官方仓库**——本仓不替代该流程。
