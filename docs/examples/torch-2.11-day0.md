# Example: Day-0 NPU support for PyTorch 2.11 (zero-interaction)

> **Status**: 实测 2026-04-23，PyTorch 2.11.0（community 2026-04 发布，
> CANN 生态 pair 只到 pytorch 2.10.0）通过 `/torch-day0` skill 叠一层到
> v2 NPU base image 上，6/6 runtime smoke 全部 PASS，torch_npu 2.11.0rc1
> + CANN 8.5.1 端到端可用。下游 vllm-ascend 段用 `/vllm-ascend-day0`
> skill 打 Fix B+ patch，V1.3 Qwen2-0.5B rollout 生成文字 PASS。全程零
> 人为介入。

## 场景

你是 Ascend pytorch / vllm-ascend 维护团队（或依赖它们的 RL 框架
用户）：
- community 刚发 PyTorch 2.11.0，你们的 stable torch_npu 只有 2.9.0
- PyPI 上有 `torch-npu==2.11.0rc1`（pre-release），但没经过完整验证
- 当前 NPU base image ship torch 2.9 + torch_npu 2.9 + CANN 8.5.1
- 想知道：**2.11.0rc1 在 NPU 上是不是能用？能用的话以什么组合？**

这正是 `/torch-day0` skill 解决的 Day-0 场景。之后 `/vllm-ascend-day0`
接力把 vllm-ascend 在新 torch 层面的 C++ ABI drift 修掉。

## 零交互调用

### 步骤 1：Pre-flight（用户手动跑一次）

```bash
# chip precheck (OL-05)
ssh -p 443 root@115.190.166.102 'npu-smi info -t proc-mem -i 0 2>&1 | grep "Process id" | head -1'
# empty = idle, OK to proceed

# base image present
ssh -p 443 root@115.190.166.102 'docker images easyr1-npu-852:trans-upg-e2e-20260422-2200 --format "{{.Tag}}"'
# should print: trans-upg-e2e-20260422-2200

# disk headroom (overlay ~3GB)
ssh -p 443 root@115.190.166.102 'df -h / | tail -1'
```

### 步骤 2：调用 torch-day0 skill（零交互核心）

```
/torch-day0 \
    --target-torch-version 2.11.0 \
    --target-torch-npu-version 2.11.0rc1 \
    --base-image easyr1-npu-852:trans-upg-e2e-20260422-2200
```

Skill 自动：
1. `pre-probe` 确认 Ascend/pytorch main 没 stable release 2.11，rc wheel
   在 PyPI，CANN 8.5.1 和 README-paired 8.5.0 差一个 patch 可接受
2. 分析 PyTorch 2.10 → 2.11 delta：8 个新 op，6 个 CUDA/ROCm only，2 个
   CompositeExplicitAutograd（NPU 自动覆盖），DispatchKey.h 1-line
   noexcept，无 ABI blocker
3. 建 overlay Dockerfile：`FROM base` + `pip install --no-deps torch==2.11.0+cpu
   torch_npu==2.11.0rc1`（**关键**：build-time 不 `import torch`，避开
   PyTorch 2.11 的 `_import_device_backends()` auto-load 陷阱）
4. `docker build` + 运行 6-step runtime smoke（pip metadata / import
   torch / import torch_npu / device count / NPU matmul / API presence）
5. 全 PASS 后生成 Phase 2.5 deploy artifacts（`Dockerfile.overlay-torch211`,
   `smoke_torch211.sh`, `deploy_torch211.sh`, `ONBOARDING.md`）
6. 返回 overlay image tag `easyr1-npu-torch211:torch-day0-<SESSION>`

**预期输出**：outcome `A`（torch layer 可用）。下游 vllm-ascend 的 C++
扩展 ABI drift 在 known-broken 段里说明 → 下一步 skill。

### 步骤 3：调用 vllm-ascend-day0（下一层）

```
/vllm-ascend-day0 \
    --target-delta torch-2.11 \
    --base-image easyr1-npu-torch211:torch-day0-<SESSION>
```

Skill 自动：
1. 读 step 2 的 ONBOARDING.md 的 known-broken 段，定位问题：
   `torch.ops._C_ascend.npu_add_rms_norm_bias` SIGSEGV
2. Minimal reproducer（`isolate_segfault_v3.py`）确认根因是 C++ ABI drift
   (`.so` loads + `TORCH_LIBRARY_IMPL` 注册 OK + op call SIGSEGV)
3. 枚举 call sites：layernorm.py:73 (guard-gated)、sampler.py:139
   (guard-gated)。全 guard-gated → Fix B+ 可行（env-var 层面 bypass）
4. 在 `upstream/vllm-ascend/` 上开 `ascend-day0-torch211-<SESSION>` 分支，
   按 fix-level 升级顺序打 patch（直到 smoke PASS）：
   - **Level 1-2 (Fix B+)**：utils.py 加 `_torch_abi_safe_for_custom_ops()`
     guard + `__init__.py` 在 plugin entry 早设 `VLLM_BATCH_INVARIANT=1`。
     跑 V1.3，通常 PASS；跑 V1.4（训练）会遇到 2D assert → 加 Level 3。
   - **Level 3** (per-call-site python patch)：`linear_batch_invariant`
     wrapper 接 3D reshape。V1.4 推进到 "Update policy" step，之后仍
     FAIL on `linear_backward CPU fallback` → 升到 Level 4。
   - **Level 4 (Fix C, C++ rebuild)**：CMakeLists.txt 放宽 torch
     version hard-pin（`VERSION_EQUAL "2.9.0"` → accept 2.11.x），
     在 Fix B+ overlay container 里跑
     `python3 setup.py build_ext --inplace`，提取 472KB 的新
     `vllm_ascend_C.cpython-*.so`。COPY 到 overlay 构造出
     Fix C image。
5. 建 overlay：Fix B+ 和 Fix C 可以分别独立 overlay 层（4 个
   file-level edits，session-local trace branch 上 4 commits）。
   Fix C image 是完整 production-ready 的最终产物。
6. V1.3 smoke（rollout）→ PASS on Fix B+ or Fix C
7. V1.4 smoke（training）with `VLLM_BATCH_INVARIANT=0`（强制 native
   custom op path）→ **only PASS on Fix C image**。期望 entropy_loss
   exactly matches v2 baseline band [1.21, 1.34]（2026-04-23 实测 1.275）
8. 返回 patched overlay image tag + PR material（4 个 file-level edits 的
   diff 全集，给 vllm-ascend maintainer 落到他们自己 tree 里）

**预期输出**：outcome `C-patch` with Fix C rebuild，V1.3 + V1.4 都 PASS，
`PR_MATERIAL.md` 就绪可交给 vllm-ascend maintainer，V1.4 entropy_loss
和 v2 baseline exact match。

## 人工 G2 验证（可选，但推荐）

Skill 返回后，用户自己验证两个 overlay image 确实 PASS，不信 skill 的
自我报告：

```bash
# (a) torch layer 6-step runtime smoke
ssh -p 443 root@115.190.166.102 "bash /tmp/torch-day0-<SESSION>/smoke_torch211.sh"
# 最后一行: ALL SMOKE STEPS PASSED

# (b) vllm-ascend V1.3 rollout（Fix B+ 或 Fix C 都 PASS）
bash repo/src/experts/vllm-day0-expert/scripts/smoke_validate.sh \
    --rung V1.3 \
    --image-tag easyr1-npu-torch211-fixc:ascend-day0-torch211-<SESSION> \
    --image-family v2 --chips 0
# 结束应显示 ✅ V1.3 PASS (marker 'V1.3 ROLLOUT SMOKE PASSED' found)

# (c) **V1.4 training — 只有 Fix C image PASS**，需要强制 native custom op
#     并在 ascend-port branch 上跑（需要 CP-001 device dispatch）
ssh -p 443 root@115.190.166.102 "cd /home/z00637938/workspace/easyr1-npu/upstream/EasyR1 && git checkout ascend-port"
# temp-edit smoke script to set VLLM_BATCH_INVARIANT=0（session 末尾需 revert）
ssh -p 443 root@115.190.166.102 "sed -i 's|^set -eux|set -eux\nexport VLLM_BATCH_INVARIANT=0|' /home/z00637938/workspace/easyr1-npu/upstream/EasyR1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh"
bash repo/src/experts/vllm-day0-expert/scripts/smoke_validate.sh \
    --rung V1.4 \
    --image-tag easyr1-npu-torch211-fixc:ascend-day0-torch211-<SESSION> \
    --image-family v2 --chips 0,1 --timeout-min 15
# 结束应显示 ✅ V1.4 PASS: entropy_loss=<N> in band [1.21, 1.34]
# 2026-04-23 实测 entropy_loss=1.275 (exact v2 baseline match)
ssh -p 443 root@115.190.166.102 "cd /home/z00637938/workspace/easyr1-npu/upstream/EasyR1 && git checkout -- examples/qwen2_0_5b_math_grpo_npu_smoke.sh"
```

## 为什么要做这件事（为什么不用 transformers-upgrade-expert）

`transformers-upgrade-expert` 和 `torch-npu-upgrade-expert` 都假设 NPU
已经 ship 了带新版本的 base image。**torch 2.11 stable 在 Ascend
那里还没有**——rc 都才 2026-03-24 放出来。shim-adapt 模式不适用。

Day-0 skill 就是 fill 这个 gap：消费者不必等 NPU team 做新版 image +
验证 + 发布整个 cycle，而是**今天就能在 NPU 上跑**（虽然走 batch-invariant
fallback 的 slightly-slower 路径）。skill 同时把 Fix B+ patch 打包成
PR-ready diff 交给 vllm-ascend upstream。

## 交付物（Day-0 session 结束时存在的东西）

- `workspace/torch-day0-{analysis,manual,deploy}-<SESSION>/`（session 产出，
  不 git track）
- `workspace/vllm-ascend-day0-{analysis,deploy}-<SESSION>/`
- A3 上的两个 overlay image（保留用于下游 RL 框架 FROM 使用）：
  - `easyr1-npu-torch211:torch-day0-<SESSION>` (torch layer)
  - `easyr1-npu-torch211-vllmascend-fixb:ascend-day0-torch211-<SESSION>` (Fix B+ intermediate)
  - `easyr1-npu-torch211-fixc:ascend-day0-torch211-<SESSION>` (Fix C, production-ready)
- Personal fork 上的 PR-ready 分支：
  - `zhshgmail/vllm-ascend/ascend-day0-torch211-<SESSION>`（4 commits：
    utils.py guard + __init__.py early-set + linear reshape + CMakeLists.txt widen）
- `PR_MATERIAL.md` 文件（给上游 maintainer 提 PR 时粘贴）

## KB 沉淀

Skill 跑完自动把新发现的 Day-0 gotcha 写进各自 `references/KB_INDEX.md`
(下次 cold-drive 会读)：
- `torch-day0-expert`: `_import_device_backends()` 建 trap、rc wheel
  pin 严格、CANN 一个 patch 宽容
- `vllm-ascend/day0-expert`: C++ ABI drift 三步诊断、fix-level 选择
  顺序、editable install 技巧

## 2026-04-23 首次实测参数 + 结果

- 基础镜像: `easyr1-npu-852:trans-upg-e2e-20260422-2200`（CANN 8.5.1 +
  Python 3.11.14 + torch 2.9 + torch_npu 2.9）
- Session tags:
  - `torch-day0-manual-20260423-0537`（Phase 1/2 torch layer）
  - `vllm-ascend-day0-analysis-20260423-0636` + `deploy-20260423-0655`
    （Phase 5 patch + Fix C rebuild + wet-run）
- 产出 image 都在 A3 保留（3 个 overlay 层）
- Patch branch: `ascend-day0-torch211-20260423` on `zhshgmail/vllm-ascend`
  （**4 commits**）
  1. `7c2078e7` utils.py `_torch_abi_safe_for_custom_ops()` guard
  2. `caa55fed` __init__.py early VLLM_BATCH_INVARIANT=1 set
  3. `87b507ed` linear_batch_invariant 3D-reshape wrapper
  4. `ab26a534` CMakeLists.txt widen torch version check to accept 2.11.x
- **V1.3 rollout**（inference）：PASS on 所有 Fix B+/Fix C image
  - `"Hello, my name is"` → `" Sarah and I am a 20"`
  - `"The capital of France is"` → `" ______.\nA. Paris\nB."`
  - `"2 + 2 equals"` → `" 4. 2 + 2"`
- **V1.4 training**（GRPO 2-step Qwen2-0.5B + math12k, 2 chips）：
  - Fix B+ only：FAIL（linear_batch_invariant 2D assert 或
    linear_backward CPU fallback）
  - **Fix C 鏡像 with VLLM_BATCH_INVARIANT=0**：**PASS**
    `entropy_loss=1.275` 在 band `[1.21, 1.34]` 中，**exact match v2
    历史 baseline**
  - 证明 native custom op + native NPU backward pass 路径完整跑通
