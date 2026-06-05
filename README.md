# easyr1-npu

把 [`hiyouga/EasyR1`](https://github.com/hiyouga/EasyR1) 适配到 Ascend 910C (A3) NPU，并沉淀一套针对 4 个 NPU 上游（vllm-ascend / torch-npu / transformers / triton-ascend）的可复用版本升级工具链。

最后更新：2026-06-04（tilelang-on-A3 能力边界 + miles 训练精度厘清,详见报告 §三.3 末 + §八）。前次(2026-06-02):DeepSeek-V4-Flash 在 A3 NPU,减层基线:推理侧真 V4 model class `generate()` + 推理-权重同步-再推理循环闭合;训练侧真 DSV4 config 减层 1 层完整训练迭代 + 2 层 fwd+bwd;**两层共享权重下训练→推理参数流动经判别实验验证为训练特异性**;op-gen AscendC 算子(act_quant)已在 NPU 上从 pytorch 真实调用(逐位等价,**独立测试,未接入训练层** —— 接入被 torch_npu 缺 fp8 支持阻塞)。统一移植报告已重写为正式中文文书(见下)。新增 7 条 V4 cookbook(共 32 条)+ `/task-dag-planner` skill。

> **2026-06-04 新增结论**(实测,详见 [`docs/_meta/DSV4_NPU_PORTING_REPORT.md`](docs/_meta/DSV4_NPU_PORTING_REPORT.md) §三.3 + §八):
> 1. **tilelang-on-A3 能力图**:vector(sinkhorn)+ gemm(matmul/sparse_mla)在 NPU 上均**能编 + 真 NPU run + 数值对**(API codegen 路径);perf:vector batched **0.17–1.15×** torch、gemm **0.13×** torch(未调优)——功能可行但未达 torch/CANN 速度。
> 2. **fp8 = A3(V220)硬件墙**:开源软件栈(tilelang dtype + MLIR type)本 session 已打通并验证,但 9.1.0 bishengir 仍报 `hardware doesn't support fp8`——A3 跑 DSv4(原生 FP8)走 `fp8_cast_bf16` dequant→bf16,真 fp8 需 A5。
> 3. **miles 训练精度**:训练真高精(bf16+fp32 累加),量化仅 rollout 侧(FP8-rollout-RL 范式),训练侧 fp8 是 env-gated QAT 假量化且默认关 → A3 上整链 bf16 可跑训练;无 PTQ。
> 4. **CANN 9.1.0-beta.1 复用基座**:torch_npu ABI ✅,但 bishengir gemm 回归 ⚠️ 且不修 tilelang in-process PassManager segfault(issue #100)→ A3 gemm 暂留 8.5,9.1.0 价值在 A5/fp8。
> 5. tilelang PassManager segfault 已提 [issue #100](https://github.com/tile-ai/tilelang-mlir-ascend/issues/100)。)

> **想看 DeepSeek-V4-Flash NPU 移植报告（重点：SGLang 推理 + Megatron/miles 训练两侧的坑 / 解法 / walkaround-vs-production 分类）→ [`docs/_meta/DSV4_NPU_PORTING_REPORT.md`](docs/_meta/DSV4_NPU_PORTING_REPORT.md)**
>
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
| sglang | A（NPU 在上游 first-class，3-axis 版本验证：sglang main + sgl-kernel-npu 2026.04.15.rc4 + CANN 8.5.0） | import smoke PASS（`is_npu()=True`、Engine、sgl_kernel_npu、deep_ep）；端到端推理验证用 `quay.io/ascend/sglang:main-cann8.5.0-a3` | 上游主线 [sgl-project/sglang](https://github.com/sgl-project/sglang) + [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu)；本仓不 fork |
| EasyR1 (consumer) | A（`transformers<5.0.0` cap 解除，单 commit） | **V1.4 GRPO smoke PASS**（2 步 + post-train val，2 次独立运行） | [`ascend-port-integrated-20260427`](https://github.com/zhshgmail/EasyR1/tree/ascend-port-integrated-20260427) |

**集成 image**：`easyr1-npu:integrated-20260427`（28.2 GB，SHA `044ba0b76183`，A3 host 上）。

### 子项目(独立 sub-projects)

每个独立子项目在 [`output/<slug>/`](output/) 下,标配 `PROJECT.json` + `README.md` + `docs/{REPORT, REPRODUCE, kb_index}.md` + `artifacts/`(借鉴 [a5_ops audit](workspace/a5_ops_audit_2026_05_31/FINDINGS.md))。索引在 [`output/README.md`](output/README.md);schema 在 [`output/_project_schema/PROJECT.schema.json`](output/_project_schema/PROJECT.schema.json);新建模板在 [`output/_project_template/`](output/_project_template/)。

| Slug | Kind | Status | 一句话 |
|---|---|---|---|
| [`miles-dsv4-flash-poc`](output/miles-dsv4-flash-poc/) | poc | active | miles + DeepSeek-V4-Flash 在 A3 NPU 上 RL 后训练 PoC;5 PR + 2 Issue + KB cookbook + `/npu-adapt-assist` skill。**统一移植报告（推理+训练两侧、walkaround-vs-production）见 [`docs/_meta/DSV4_NPU_PORTING_REPORT.md`](docs/_meta/DSV4_NPU_PORTING_REPORT.md)**;训练侧已达减层基线（1 层完整训练迭代 + 2 层 fwd+bwd + 两层共享权重下训练→推理参数流动验证,见报告 §四 + KB `miles-002/003`、`cross-layer-012/013`)。**2026-06-04 补:tilelang-on-A3 能力图 + fp8=A3 硬件墙 + miles 训练精度厘清 + CANN 9.1.0 验证(报告 §三.3 + §八)** |

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
| 想用通用编排 skill 自动分解任务（分析→子任务→依赖 DAG→分阶段执行+对抗验证） | `/task-dag-planner <目标>`，详见 [`src/skills/orchestrators/task-dag-planner/SKILL.md`](src/skills/orchestrators/task-dag-planner/SKILL.md) |
| 想看每条工具链当前在哪个 fork / 分支 | [`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md) |
| 想看下一波 NPU 适配候选清单（镜像里还没 NPU 适配的主要软件） | [`docs/_meta/NPU_ADAPTATION_GAP.md`](docs/_meta/NPU_ADAPTATION_GAP.md) |
| 想看 open work / 技术债 / 下一步做什么 | [`docs/_meta/ROADMAP.md`](docs/_meta/ROADMAP.md)（**唯一权威 backlog**） |
| 想接手项目（continuing agent / 新 session） | [`docs/_meta/handovers/`](docs/_meta/handovers/) + [`ROADMAP.md`](docs/_meta/ROADMAP.md) + [`ARCHITECTURE.md`](docs/_meta/ARCHITECTURE.md) |
| 想查 NPU 操作模式与已知 bug（29 stable IDs） | [`knowledge/npu-patterns.md`](knowledge/npu-patterns.md) |
| 想查跨层移植教训（lessons learned，32 条 NPU 适配 cookbook） | [`docs/_meta/kb/porting_lessons/`](docs/_meta/kb/porting_lessons/)（顶部有 keyword grep 表） |
| 想查 tilelang-on-Ascend 知识库（env / bug 分类 / cold-drive runbook / fp8=硬件墙 / #100） | [`workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md`](workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md)（§13 cold-drive 2026-06-03） |
| 想复现 fp8 补丁后的 tilelang 构建（Dockerfile + 精确补丁 + 自检） | [`workspace/_fp8_dockerfile_2026_06_05/`](workspace/_fp8_dockerfile_2026_06_05/)（`Dockerfile.fp8patch` + 2 个 fp8 diff，补丁已 `git apply --check` 验证 clean） |
| 想看 DeepSeek-V4-Flash NPU 移植统一报告（推理+训练两侧坑/解法/walkaround-vs-production） | [`docs/_meta/DSV4_NPU_PORTING_REPORT.md`](docs/_meta/DSV4_NPU_PORTING_REPORT.md) |
| 想根据 error trace 自动找匹配的 cookbook | `/npu-adapt-assist <paste-trace>`，详见 [`src/skills/npu-adapt-assist/`](src/skills/npu-adapt-assist/README.md)（启动会自动跑 preflight） |
| 想看 miles + DSv4-Flash PoC（一句话 + 状态表 + 上游 PR 列表） | [`output/miles-dsv4-flash-poc/`](output/miles-dsv4-flash-poc/) |
| 想看 miles PoC 完整报告（问题分类 + 解决方案 + empirical evidence） | [`output/miles-dsv4-flash-poc/docs/REPORT.md`](output/miles-dsv4-flash-poc/docs/REPORT.md) |
| 想从 0 复现 miles PoC | [`output/miles-dsv4-flash-poc/docs/REPRODUCE.md`](output/miles-dsv4-flash-poc/docs/REPRODUCE.md) |
| 想看每个独立 sub-project 在哪 + 怎么新建 | [`output/README.md`](output/README.md) |
| 你是 vllm-ascend 维护者，想看升级流程详解 | [`docs/vllm-ascend/PORTING-GUIDE.md`](docs/vllm-ascend/PORTING-GUIDE.md) |
| 你是 torch-npu 维护者 | [`docs/torch-npu/PORTING-GUIDE.md`](docs/torch-npu/PORTING-GUIDE.md) |
| 你是 transformers 维护者，想看 PR 资料 | [`docs/transformers/PR_MATERIAL_v5.4_outcome_A.md`](docs/transformers/PR_MATERIAL_v5.4_outcome_A.md) |
| 想查项目术语 | [`docs/_meta/GLOSSARY.md`](docs/_meta/GLOSSARY.md) |

---

## 责任边界

工具链产出的修改落在演示性 fork 分支（每条上游对应的 personal fork 上的 `ascend-port/<target-version-slug>` 分支，权威表见 [`UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md)）。每条分支根目录含 `PR_MATERIAL.md`，是给上游 maintainer 的 PR 包（diff、commit 序列、reproducer、已知边界）。

**正式 PR 由对应上游仓的维护者基于这些分支提交到他们自己的官方仓库**——本仓不替代该流程。
