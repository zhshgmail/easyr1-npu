# fp8 补丁可复现 Dockerfile

把 DSv4 fp8 open-stack 工作中对 tilelang-mlir-ascend 打的补丁，固化为**可复现的 Dockerfile 构建脚本**（不交付镜像，交付 Dockerfile + 补丁 + 自检）。

## 文件

| 文件 | 作用 |
|---|---|
| `Dockerfile.fp8patch` | 主构建脚本：`FROM quay.io/ascend/cann:8.5.0-a3` → clone fork 到 pinned commit → `git apply` 两个 fp8 补丁 → 跑 fork 既有 build（AscendNPU-IR + TVM + tilelang）→ CPU-only 自检 |
| `data_type.h.fp8.diff` | 补丁 1：`3rdparty/tvm` 的 `String2DLDataType` 加 `float8_e4m3fn/e4m3/e5m2 + float4_e2m1fn/e2m1` 解析 |
| `codegen_npuir_api.cc.fp8.diff` | 补丁 2：`DTypetoMLIRType` 加 fp8→MLIR `getFloat8E4M3FNType()/getFloat8E5M2Type()`（**仅 fp8 hunk**，已剔除非 load-bearing 的 nd2nz 改动） |
| `verify_fp8_patch.py` | 构建期自检（CPU-only，无需 NPU）：`float8_e4m3fn/e5m2` 能解析 + tilelang 前端接受 fp8 dtype |
| `codegen_npuir_api.cc.FULL.diff` | 参考：workspace 里 api.cc 的完整 dirty diff（含 nd2nz；**不进补丁**，仅留档说明取舍） |
| `fork_docker_Dockerfile.ref` | 参考：fork 自带 `docker/Dockerfile` 原文（本 Dockerfile 的 Layer 1–5/7 据此忠实派生） |

## Pinned 基线 commit

- tilelang-mlir-ascend: `a19acd548a066f519869acbe7b60b36e00cbbcc3`
- 3rdparty/tvm: `c2921fdaf795b1103d21abc962e83a209c7258d7`

## 构建

```bash
# 中国大陆（用 tuna 镜像拉 llvm 子模块，同 fork docker/Dockerfile）
docker build -f Dockerfile.fp8patch -t tilelang-fp8patch:a3 .
# 其它地区
docker build --build-arg REGION=other -f Dockerfile.fp8patch -t tilelang-fp8patch:a3 .
```

⚠️ **冷构建会编译 LLVM/MLIR + AscendNPU-IR + TVM + tilelang，多小时、约 30G**（峰值中间产物可达 40–60G）。这重头是 **fork 既有 build 步骤**，非补丁引入。

## 已做的验证

- ✅ **两个 fp8 补丁对 pinned base commit `git apply --check` = APPLIES CLEAN**（2026-06-04，A3）。
- ✅ Dockerfile 的 AscendNPU-IR build 调用（`build.sh --apply-patches --bishengir-publish=off`）与 fork `docker/Dockerfile` **逐字一致**。
- ✅ 补两个忠实性缺口（相对"照搬 fork"）：Layer 3 加 `requirements-build.txt`（直跑 cmake 而非 `install_npuir.sh`，否则缺 build 依赖）；Layer 8 `PYTHONPATH` 加 bishengir `python_packages/{mlir_core,bishengir}`（NPUIR 运行期必需，install_npuir.sh Step 11 同款）。
- ⏳ 全量冷构建：未跑（磁盘紧张 + 重头是 fork 既有步骤；待 owner 决定是否值得花数小时）。

## 重要前提（A3 fp8 = 硬件墙）

此补丁打通的是**开源软件层**（dtype 字符串 → MLIR float8 type）。**A3（Ascend V220）硬件无 fp8 单元**，bishengir 仍会报 `Current hardware doesn't support fp8 type`。补丁价值在 A5（arch35）+ 让 fp8 dtype 走通编译器前端，**不会让 fp8 在 A3 上跑起来**。详见报告 §三.3 + §八 + KB §13.1。
