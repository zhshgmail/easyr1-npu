# tilelang on NPU — 全算子精度 & 性能对照大表（vs torch_npu = CANN）

> **基线说明**：表中 "torch_npu" = PyTorch 算子跑在 NPU 上，torch_npu 派发到 **CANN aclnn 实现**。所以 **tilelang-vs-torch_npu ≡ tilelang-vs-CANN**。不是 CPU。
> **诚实标注**：🟢=本 session 真测（编译+真 NPU run+数值对比，有 log）；🟡=仅编译验证（compile pass / All check passed，未单独测 perf）；⚪=静态 API/idiom 分类（未 cold-drive）；🔴=被缺口阻塞（dtype/backend）。
> 全部基于 `_v4_runlogs/RESULT_*.log`。日期 2026-06-03。

## A. 推理侧 tilelang 算子（sglang/fork deepseek_v4 inference，6 个）
| # | kernel | 类 | 编译 | 精度 (vs torch_npu/CANN) | 性能 (× torch_npu) | 状态 | 阻塞/备注 |
|---|---|---|---|---|---|---|---|
| 1 | hc_split_sinkhorn | vector | ✅ | 🟢 max_abs_diff **8.9e-8**（全 shape band N=1..1024）| 🟢 1-iter拆 0.06–0.45×; **3-iter batch 0.17–1.15×**（N≤1024 持平/反超）| 🟢 **端到端真测** | #100 segfault → multi-launch workaround；batch 后小-中 N 追平 CANN |
| 2 | act_quant (FP8块量化) | quant | 🔴 | — | — | 🔴 dtype-gap | fork TVM 无 FP8 dtype（e4m3/e5m2 全 FAIL）。int8/fp16 OK |
| 3 | fp4_quant | quant | 🔴 | — | — | 🔴 dtype-gap | fork TVM 无 FP4 dtype |
| 4 | fp8_gemm | gemm | 🔴 | — | — | 🔴 dtype-gap | 同 FP8（GemmWarpPolicy 可删，gemm 本身能编）|
| 5 | sparse_attn | attention | 🟡(idiom) | — | — | ⚪ idiom-port | 删 policy+transpose_B 换 npuir_transpose+1D grid 后过 parser；单 kernel 撞 #100，需拆 |
| 6 | fp4_gemm | gemm | 🔴 | — | — | 🔴 dtype-gap | 同 FP4 |

## B. 训练侧 tilelang 算子（miles V4 deepseek_v4/ops/kernel，7 个）
| # | kernel | 类 | 编译 | 精度 (vs torch_npu/CANN) | 性能 (× torch_npu) | 状态 | 阻塞/备注 |
|---|---|---|---|---|---|---|---|
| 7 | sinkhorn | vector | ✅ | 🟢 同 #1（同算子）| 🟢 同 #1 | 🟢 | 同推理 sinkhorn |
| 8 | act_quant | quant | 🔴 | — | — | 🔴 dtype-gap | FP8 dtype（同 #2）|
| 9 | tilelang_indexer_fwd | gemm+vector | 🟡 | — | — | ⚪ idiom-port | gemm + transpose_B；#100 风险，未 cold-drive |
| 10 | tilelang_indexer_bwd | gemm | 🟡 | — | — | ⚪ idiom-port | gemm×3，反向，未 cold-drive |
| 11 | tilelang_sparse_mla_fwd | gemm+attention | 🟢 | 🟢 **All check passed!** rtol=5e-3（example 自带 ref）| 🟡 未单独测（attention shape 复杂）| 🟢 编译+数值（API 路径）| 我的 _api.cc nd2nz 修复后 expert 路径过 |
| 12 | tilelang_sparse_mla_bwd | gemm | ⚪ | — | — | ⚪ idiom-port | gemm×5，最复杂，未 cold-drive |
| 13 | (matmul, gemm 类基准代表) | gemm | ✅ | 🟢 max_abs **6.25e-2**（fp16 容差内）| 🟢 **0.13×**（M1024×N512×K2048）| 🟢 **端到端真测** | basic-tiling vs CANN 手调 gemm；非 V4 算子，作为 gemm 类性能代表 |

## C. 净结论（精度 + 性能，vs CANN）
**精度**：所有真测的（sinkhorn 8.9e-8 / matmul fp16-容差 / sparse_mla_fwd rtol5e-3）**精度都 PASS** —— tilelang 在 NPU 上数值正确。功能不是问题。

**性能（= vs CANN）**：
- **vector（sinkhorn）**：naive 拆法 0.06–0.45×；**batching 后 0.17–1.15×**，小-中 batch(N≤1024) **追平/反超 CANN**，大 batch(4096) 仍 0.17×（global 往返 bound，等 #100 真修）。
- **gemm（matmul）**：**0.13×**（慢 ~8×）—— CANN 的 gemm 是华为手调 Cube，tilelang example 是朴素 tiling，没调优。
- → **tilelang 现在普遍比 CANN 慢**（正常：在跟手调实现比）；vector 小-中 batch 能打平。**tilelang 的真实价值不在"比 CANN 快"，而在：① CANN 没覆盖的算子（fp8/fp4 quant、mhc/compressor）② 跨算子 fusion ③ 与 miles 训练侧对齐。**

**真测覆盖诚实边界**：13 个里只有 **3 个端到端真测了精度+性能**（sinkhorn、matmul、sparse_mla_fwd-精度）；4 个 FP8/FP4 类此前标 dtype-gap；6 个仅 idiom 分类/未 cold-drive。

## D. FP8/FP4 阻塞的更深定位（2026-06-03 codegen 实测，修正 "dtype-gap" 说法）
之前标 "fork TVM 无 FP8 dtype" **不准**。逐层实测推进后：
- ✅ TVM 已有 fp8/fp4 enum；缺 PyTorch 命名解析（`float8_e4m3`）→ **已修**（String2DLDataType + substr off-by-one）
- ✅ codegen dtype→MLIR-type 缺 fp8 → **已修**（DTypetoMLIRType → getFloat8E4M3FNType/E5M2）
- ✅ tilelang 现在生成**合法 fp8 MLIR**（npuir dump 验 `f8E4M3FN`）
- ❌ **真·墙 = bishengir-compile（CANN 8.5.0 闭源后端）不处理 fp8 npuir**（"Failed to run BiShengIR pipeline"）。同 bf16 #1199 类。**FP4 更早撞墙**（fork MLIR 无 Float4 type builder）
- → **#2/3/4/6/8 的真实阻塞 = bishengir 后端 fp8/fp4 缺失，不是 dtype-parse**（那层已打通）。CANN 闭源后端，可提 issue（#1199 族）
- **★ build-target 诚实修正**：runtime 加载 `libtilelang_module.so`（非 libtilelang.so）；早先 matmul/sparse_mla 的 "All passed" 在 stale codegen 上跑 → 它们本就在 stock API codegen 能跑，**我的 nd2nz 修复对它们非 load-bearing**（不贪功）。

## E. FP8 真根因 = A3 硬件不支持（2026-06-03，CANN 9.1.0-beta.1 实测，颠覆 §D 的"软件可修"）
拉了 `quay.io/ascend/cann:9.1.0-beta.1-a3-ubuntu22.04-py3.12-devel`，用其 **bishengir 1.1.0**（2026-05-09）跑之前 8.5.0 失败的同一 fp8 npuir：
- 8.5.0：`store op element type [无 float8]`（verifier 白名单遗漏）
- **9.1.0：`'hivm.hir.store' op Current hardware doesn't support fp8 type`** ← verifier 已认 fp8（白名单修复在 9.1.0），但**硬件能力检查拒绝：A3(V220) 不支持 fp8**。
- → **fp8 在 A3 是硬件地板，不是软件/CANN/verifier 可修。** act_quant/fp8_gemm/fp4(#2/3/4/6/8) 的 fp8 路径**在 A3 上根本跑不了**，与 CANN 版本、与我打通的 open 层无关。§D 的"软件可修"是症状层，**真根因是 A3 无 fp8 硬件**。
- **A5 可能支持**：9.1.0 镜像带 `bishengir-compile-a5`/`hivmc-a5`（A5 专用）→ fp8 tilelang 大概率是 **A5 目标**。
- **bf16**：简单 ub-vector bf16 vmul/vexp 在 8.5.0 和 9.1.0 都编过（"bf16 墙"是 flash_attn 上下文特定，非通用 bf16）。
- **#100 PassManager segfault**：tilelang fork in-process bug（`tladapter/utils.py:103`），**CANN-independent** → 升 CANN 不解，需 fork 修 / multi-launch workaround。
- **CANN 升级的真实价值**：不是"解锁 fp8"，是**把 fp8 真根因从"软件可修"钉死为"A3 硬件不支持、需 A5"** —— 实测才挖出来的，避免了错误结论。
