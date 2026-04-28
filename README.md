# easyr1-npu

把 [`hiyouga/EasyR1`](https://github.com/hiyouga/EasyR1) 适配到 Ascend 910C (A3) NPU，并沉淀一套针对 4 个 NPU 上游（vllm-ascend / torch-npu / transformers / triton-ascend）的可复用版本升级工具链。

最后更新：2026-04-28。

> **想直接在 A3 上跑 EasyR1 → [`ONBOARDING.md`](ONBOARDING.md)**（一页 quickstart，两条已验证路径）
>
> **想看整体架构与流程 → [`docs/_meta/ARCHITECTURE.md`](docs/_meta/ARCHITECTURE.md)**（含 mermaid 图）
>
> **想看每条上游 fork 当前在哪个分支 → [`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md)**

---

## 当前状态

| 上游 | Outcome | 验证级别 | 产出分支 |
|---|---|---|---|
| vllm-ascend | C-patch（3 shim：`shared_fused_moe` / `default_moe_runner` / `spec_decode_base_proposer`） | on-A3 import 3/3 PASS | [`ascend-port/vllm-main`](https://github.com/zhshgmail/vllm-ascend/tree/ascend-port/vllm-main) |
| torch-npu | A（v2.12-rc3）+ 13 个 defensive shim | on-A3 import PASS | [`ascend-port/torch-2.12-rc3`](https://gitcode.com/zhengshencn_hwca/pytorch/tree/ascend-port/torch-2.12-rc3) |
| transformers | A-with-note（v5.4.0 / v5.6.2 byte-compat） | byte-compare PASS | [`ascend-port/transformers-v5.4`](https://github.com/zhshgmail/transformers/tree/ascend-port/transformers-v5.4) |
| triton-ascend | C-patch（v3.6.0 源码合并 + 9 drift 修） | vendor 3.2.0 wheel 6/6 NPU smoke PASS；源码端到端 BLOCKED on bishengir LLVM-22 公开 release | [`ascend-port/triton-v3.6.0`](https://gitcode.com/zhengshencn_hwca/triton-ascend/tree/ascend-port/triton-v3.6.0) |
| EasyR1 (consumer) | A（`transformers<5.0.0` cap 解除，单 commit） | **V1.4 GRPO smoke PASS**（2 步 + post-train val，2 次独立运行） | [`ascend-port-integrated-20260427`](https://github.com/zhshgmail/EasyR1/tree/ascend-port-integrated-20260427) |

**集成 image**：`easyr1-npu:integrated-20260427`（28.2 GB，SHA `044ba0b76183`，A3 host 上）。

---

## 目录

按角色选入口：

| 你是谁 / 你要做什么 | 入口 |
|---|---|
| 想在 A3 上把 EasyR1 跑起来（一页 quickstart） | [`ONBOARDING.md`](ONBOARDING.md) |
| 想看 v1 path（verl-8.5.0 base + ascend-port 分支） | [`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md) |
| 想看 v2 path（integrated overlay image） | [`docs/easyr1/PORT-GUIDE-v2-integrated.md`](docs/easyr1/PORT-GUIDE-v2-integrated.md) |
| 想看整体架构、组件分工、端到端流程 | [`docs/_meta/ARCHITECTURE.md`](docs/_meta/ARCHITECTURE.md) |
| 想用 slash command 触发版本升级工具链 | [`docs/_meta/SKILLS-USAGE.md`](docs/_meta/SKILLS-USAGE.md) |
| 想看每条工具链当前在哪个 fork / 分支 | [`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md) |
| 想查 NPU 操作模式与已知 bug（29 stable IDs） | [`knowledge/npu-patterns.md`](knowledge/npu-patterns.md) |
| 想查跨层移植教训（lessons learned） | [`docs/_meta/kb/porting_lessons/`](docs/_meta/kb/porting_lessons/) |
| 你是 vllm-ascend 维护者，想看升级流程详解 | [`docs/vllm-ascend/PORTING-GUIDE.md`](docs/vllm-ascend/PORTING-GUIDE.md) |
| 你是 torch-npu 维护者 | [`docs/torch-npu/PORTING-GUIDE.md`](docs/torch-npu/PORTING-GUIDE.md) |
| 你是 transformers 维护者，想看 PR 资料 | [`docs/transformers/PR_MATERIAL_v5.4_outcome_A.md`](docs/transformers/PR_MATERIAL_v5.4_outcome_A.md) |
| 想查项目术语 | [`docs/_meta/GLOSSARY.md`](docs/_meta/GLOSSARY.md) |

---

## 责任边界

工具链产出的修改落在演示性 fork 分支（每条上游对应的 personal fork 上的 `ascend-port/<target-version-slug>` 分支，权威表见 [`UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md)）。每条分支根目录含 `PR_MATERIAL.md`，是给上游 maintainer 的 PR 包（diff、commit 序列、reproducer、已知边界）。

**正式 PR 由对应上游仓的维护者基于这些分支提交到他们自己的官方仓库**——本仓不替代该流程。
