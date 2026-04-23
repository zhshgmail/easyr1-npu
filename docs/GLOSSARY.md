# GLOSSARY — easyr1-npu 项目术语表

**用途**：项目里 Fix A/B/B+/C、Level 1-4、V1.x、outcome A/B/C-patch/C-report、session-tag 这些术语散在 17+ 份文档里，没人能从一个地方把它们看齐。本文件是**单一权威定义**。任何文档使用这些术语时**必须链到本文件**（或等价地解释）。如果发现冲突，以本文件为准，别的文档更新向本文件对齐。

**维护约定**：新术语第一次出现时先进本文件再用；已过期术语标 `⚠️ deprecated`，不删除；时间节点附 ISO 日期，避免 "上次"/"最近" 这种相对描述。

---

## 1. Day-0 skill 的 outcome 分类

Day-0 skill（`vllm-day0-expert` / `torch-day0-expert` / `vllm-ascend-day0-expert` / `transformers-day0-expert`）每跑完一个 session，产出**一个**下列 outcome：

| Outcome | 含义 | 下游拿到什么 | 案例 |
|---|---|---|---|
| **A** | 上游 community 新版在 NPU 上**直接能跑**，不需要改 Huawei-owned 上游的任何代码 | overlay image tag + ONBOARDING（只说怎么装，不说怎么 patch） | torch 2.11 + torch_npu 2.11.0rc1 组合本身就能载入，只要 overlay 对齐 |
| **B** | 上游新版跑不通，但**设一个 env var 或加一行 monkey-patch** 就能绕过；不需要改源码 | overlay image + ONBOARDING 说明需要设的 env | `VLLM_BATCH_INVARIANT=1` 绕过 custom op |
| **C-patch** | 需要改 **Huawei-owned** 上游的源码（`vllm-ascend` / `torch_npu` / `triton-ascend` / `transformers` NPU integrations）；我们有权限直接出 patch | overlay image + `PR_MATERIAL.md` + 参考 `.py.patched`，交给该上游的 maintainer 落到他们自己 tree | vllm-ascend `utils.py` 加 torch-ABI guard；CMakeLists.txt 放宽 torch version pin |
| **C-report** | 需要改 **community** 上游的源码（`vllm` / `pytorch` / community `transformers` 非 NPU 部分）；我们**没权限**直接出 patch | 给 community 上游提 issue + 最小 reproducer；session 在 report 阶段就结束 | vllm 的 dispatcher change 引入的 bug |

**核心分水岭**：C-patch vs C-report 的唯一区别是**我们这些 skill 的受众 owner 是不是能直接改那个 repo**。`vllm-ascend` 的 Ascend 团队 owner 能改 `vllm-ascend`，所以是 C-patch；`vllm` 的社区 owner 不归 Ascend，所以是 C-report。见 `src/experts/README.md` §"Enabled Huawei-owned targets for C-patch"。

---

## 2. Fix-level 阶梯（C-patch 内部的复杂度分层）

`C-patch` 内部，补丁的**侵入性**分 4 个 Level，Day-0 session **优先级从低到高**选：

| Level | 名字 | 改什么 | 例子 | session 内能做？ |
|---|---|---|---|---|
| **Level 1** | env-var | 只设一个环境变量或 one-line monkey-patch | `VLLM_BATCH_INVARIANT=1` | 能（其实这是 outcome B） |
| **Level 2** | plugin-entry-guard | 在 Huawei plugin 包的 `__init__.py` 或 `utils.py` 里加 guard 函数（根据条件自动设 env var、自动跳过某路径） | `vllm_ascend/__init__.py` 里 import 时自动 `os.environ["VLLM_BATCH_INVARIANT"]="1"` | 能 |
| **Level 3** | per-call-site python patch | 修改 Huawei plugin 里具体 python 文件的 forward / call site，通常加 version-gated branch | `vllm_ascend/ops/linear.py` 的 `linear_batch_invariant` 加 3D-reshape；vllm 0.20 drift 的 13 处 file-level 改动 | 能 |
| **Level 4** | C++ rebuild | 改 CMakeLists.txt / C++ 源码 → 重 build `vllm_ascend_C.so` 等 native 产物，需容器里 `python3 setup.py build_ext --inplace` | `vllm_ascend/CMakeLists.txt:26` 把 `VERSION_EQUAL "2.9.0"` 放宽成接受 `2.11.x`，重编 `vllm_ascend_C.so` 产出 472KB `.so` | 能，但属**长期 tech debt** —— 产出物（新 `.so`）要 COPY 到 overlay image |

**优先级原则**：Level 从低到高尝试，**能用 Level N 解决就不升 N+1**。Level 1-3 是纯 python 改动，Level 4 涉及 C++ rebuild 和 ABI。

---

## 3. "Fix A / Fix B+ / Fix C" 这批速记（torch 2.11 Day-0 session 专用）

这是 **2026-04-23 torch 2.11 porting session** 内部用的速记，不是通用术语。严格讲它们是上面 Level-ladder 的**具体实例**。各自只在 torch 2.11 Day-0 上下文有意义，其他 Day-0 session（比如 vllm 0.20）**不应该用这套标签**。

| 标签 | 对应 Level | 干了什么 | 产物 |
|---|---|---|---|
| **Fix A** | — | 「社区 torch 2.11 + torch_npu 2.11.0rc1 能装上，但 vllm-ascend 的 `vllm_ascend_C.so` 是按 torch 2.9 ABI 编的，call 时 SIGSEGV」—— 这是**问题定义**，不是修复 | 问题 reproducer 而已 |
| **Fix B** | Level 2 | vllm-ascend `utils.py` 加 `_torch_abi_safe_for_custom_ops()` guard，ABI mismatch 时短路 `enable_custom_op()` | 部分绕开 custom op，还需要设 env var |
| **Fix B+** | Level 2 + 增强 | 除了 Fix B 的 guard，还在 `vllm_ascend/__init__.py` import 时**自动** `os.environ["VLLM_BATCH_INVARIANT"]="1"`（在 vllm cache env var 之前）。V1.3 推理路径 PASS。V1.4 训练路径 FAIL on `aten::linear_backward CPU fallback` | overlay image `easyr1-npu-torch211-vllmascend-fixb:ascend-day0-torch211-20260423` |
| **Fix B+ / Option 1** | Level 3 | `ops/linear.py` 的 `linear_batch_invariant` wrapper 加 3D→2D reshape，让 batch-invariant 路径也能接受 FSDP 3D 输入。绕过 2D assert，但仍 FAIL on CPU fallback（因为 batch-invariant 路径的 autograd backward dispatch 走 CPU） | 中间态 |
| **Fix C** | Level 4 | **放弃 batch-invariant fallback**。CMakeLists.txt 放宽 torch version 要求 → 在 Fix B+ 容器里 rebuild `vllm_ascend_C.so` 对齐 torch 2.11 ABI。native custom op 重新能用，不再走 batch-invariant fallback，autograd backward 走 PrivateUse1 NPU 正常路径 | overlay image `easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423`，V1.3 + V1.4 都 PASS，V1.4 entropy_loss=1.275 exactly matches v2 baseline |

**Fix B+ vs Fix C 核心差异**：
- Fix B+ = python-only，绕开 broken custom op 走 batch-invariant fallback → 推理 OK、训练 FAIL
- Fix C = 重编 `.so` 让 custom op 原生能用 → 推理 + 训练都 OK

---

## 4. Smoke rung（V1.x / V2.x）

EasyR1 port 验证的不同**深度**。数字越大，覆盖越深。一次 port 不必都跑，但每次 Day-0 / upgrade session 至少跑到其对应的 rung。

| Rung | 名字 | 覆盖什么 | 时长估算 | chip 数 |
|---|---|---|---|---|
| **V1.1** | hello-import | container 能起，能 `import vllm`, `import torch_npu`, `import vllm_ascend` | <1min | 1 |
| **V1.3** | rollout smoke | 运行 Qwen2-0.5B 推理（rollout 路径），3 个 prompt 各出非空输出，marker `V1.3 ROLLOUT SMOKE PASSED` | 2-5min | 1 |
| **V1.4** | GRPO training smoke | Qwen2-0.5B + math12k dataset 跑 2 step GRPO 训练 + checkpoint，FSDP 2D + optim state 完整写出 | 8-15min | 2（HCCL） |
| **V1.5** | 4-chip training | 同 V1.4 recipe scaled to 4 chips（2 A3 cards），`world_size=4` HCCL cross-card | 4-10min | 4（跨卡 HCCL） |
| **V2.2** | full GRPO training | 完整 reward + multiple epochs 训练收敛，不是只 2 step smoke | 小时级 | ≥4 |

**Day-0 session 的最小 G2（gate 2）**：V1.3 PASS 算"推理能跑"；V1.4 PASS 才算"训练能跑"。V1.3 PASS 但 V1.4 FAIL 是 known 状态（如 torch 2.11 Fix B+），session 标记为"部分 outcome"。

**V1.3 PASS 的严谨定义**：
- 3 个 prompt 全部产出**非空**输出（formal criterion）
- 输出**分布和 baseline 对齐**（semantic criterion，更严格）

目前 smoke harness 只检 formal criterion（marker 匹配 + 输出非空）。semantic criterion 需要和 baseline image 对比 logits 或 token sequence。2026-04-23 vllm 0.20 session 发现：formal V1.3 PASS 但 semantic FAIL（输出是噪声 token 而非正常英文），**这是 smoke harness 的盲点**，待补。

---

## 5. Session tag 命名规范

每次 Day-0 / upgrade / probe session 产出的 workspace / image / branch 都以 session-tag 命名，格式：

```
<expert-name>-<short-purpose>-<YYYYMMDD>-<HHMM>
```

例：
- `torch-day0-manual-20260423-0537` = 2026-04-23 05:37 UTC 开始的 `torch-day0-expert` manual-port session
- `vllm-day0-vllm0200-20260423-1623` = 2026-04-23 16:23 UTC 开始的 `vllm-day0-expert` 跑 vllm 0.20 target 的 session

**session-tag 用在**：
- workspace：`workspace/<session-tag>/`
- overlay image tag：`<image-family>:<session-tag>` 或 `<image-family>:<short-tag>`
- trace branch name：`ascend-day0-<delta>-<YYYYMMDD>`

---

## 6. Upstream 范围 + 编辑权限矩阵

Day-0 expert 的 C-patch **允许直接改**（Huawei-owned）：
- `github.com/vllm-project/vllm-ascend`
- `gitcode.com/Ascend/pytorch`（`torch_npu`）
- `gitcode.com/Ascend/triton-ascend`
- `github.com/huggingface/transformers`（**仅** `src/transformers/integrations/npu_*` 下的文件）

C-report-only（我们不改，只报）：
- `github.com/vllm-project/vllm`（community vllm core）
- `github.com/pytorch/pytorch`（community PyTorch）
- `github.com/huggingface/transformers`（非 NPU 集成部分）

参见 memory `day0_patch_scope.md`。

---

## 7. 术语链条 cheat-sheet

把一个 Day-0 session 结果**完整描述**的正确模板：

> **Outcome** = C-patch
> **Fix level** = Level 4（C++ rebuild + CMakeLists 修改）
> **Rung** = V1.3 + V1.4 都 PASS（semantic + formal 两层都过）
> **Session tag** = `vllm-ascend-day0-analysis-20260423-0636`
> **Deliverable** = `workspace/<session-tag>/PR_MATERIAL.md` + overlay image `<tag>`，交给 vllm-ascend maintainer

**不要再说** "Fix C session PASS"——这种话没有 Outcome / Rung 上下文，听者猜不出是推理还是训练，是 formal 还是 semantic。

---

## 8. 历史 / 过时术语（⚠️ deprecated — 避免新使用）

| 过时术语 | 替换用 | 为什么过时 |
|---|---|---|
| "main2main skill / main2main scope" | （直接说是 vllm-ascend 团队的 main-branch 同步工作） | 自造词，memory `no_invented_scope_boundaries.md` 里说明过是用来合理化停工的 excuse |
| "Day-0 skill boundary hit at layer N" | （直接说具体哪个 drift layer 卡住 + 为什么） | 同上，是发明的边界 |
| "outcome C" 单独（不带 -patch / -report 后缀） | `C-patch` 或 `C-report` | 2026-04-23 before 把 C 拆成两种：Huawei-owned vs community |
