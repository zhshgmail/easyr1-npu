# V4 推理侧 tilelang kernel 逐一 idiom-port PoC 扫描（owner 指令：逐一测推理 kernel，再计算 kernel）

> 方法:沿用 sinkhorn 已验证的范式 —— ① 用 NPU idiom 写/改(alloc_shared/alloc_ub + is_npu=True + 块向量 intrinsic)② 编 BUILT OK ③ 真 NPU run 对 torch ref 验数值 ④ 撞 bug 就分类(idiom / dtype / 缺 API / 后端 crash)+ 找 workaround。
> 源:fork `examples/deepseek_v4/inference/kernel.py`(注意:此文件按更新版 tilelang 写,部分 API fork 不支持)。

## fork API 可用性(实测 `hasattr(tilelang.language, ...)`，2026-06-03)
| API | fork 有? | 备注 |
|---|---|---|
| gemm | ✅ | 但 GemmWarpPolicy 缺 |
| GemmWarpPolicy | ❌ | gemm/attention kernel 引用它 → 需补或改写不用 policy |
| use_swizzle | ✅ | |
| alloc_ub / alloc_shared | ✅ | NPU idiom 核心 |
| vmul/vexp/vbrc/vmax/vmin | ✅ | |
| vlog | ❌ | 若 kernel 用到需绕(exp/log 组合或 npuir_*) |
| reduce_sum/reduce_max | ✅ | |
| npuir_transpose / npuir_rsqrt | ✅ | |
| clamp / if_then_else / sigmoid | ✅ | |
| float8_e4m3 (dtype) | ❌ | act_quant/fp8_gemm 输出 dtype,fork TVM 不识别 → dtype gap |
| FP4 dtype | ❌(待验) | fp4_* kernel |

## 推理侧 tilelang kernel 清单 + 预分类(扫描进度表)
| kernel | 源 | 状态 | 预判缺口 |
|---|---|---|---|
| 1. hc_split_sinkhorn_kernel | kernel.py:435 | ✅ **DONE (shape-swept)** | idiom-port + bishengir 结构-segfault;workaround=multi-launch;**NUMERIC PASS 真 NPU,全 shape band(N=1/7/256/512/1009/1024)max_abs_diff ≤1.5e-7**;case-gen→tilelang→NPU→verify 管线已打通 |
| 2. act_quant_kernel (FP8 块量化) | kernel.py:77 | 🔶 **BLOCKED-dtype** | **实测(in-kernel cast+copy):`int8`/`float16` BUILT OK,但 `float8_e4m3`/`e4m3fn`/`e5m2` 全 FAIL** → fork TVM 不支持 FP8 dtype。缺口 = fork TVM datatype 注册缺 FP8(C++/TVM 层,同 bf16 #1199 族),**不是 idiom**。act_quant 的功能本身能在 fp16/int8 上 PoC,但 V4 要的 FP8 输出被 dtype gap 挡。 |
| 3. fp4_quant_kernel | kernel.py:161 | 🔶 **BLOCKED-dtype** | **实测确认:FP4 全变体 FAIL**(`float4_e2m1fn`/`e2m1`/`float4_e2m1fn_x2`/`int4` 全 FAIL;uint8 storage OK)→ fork TVM 无 FP4 datatype。同 FP8 dtype-gap 族。 |
| 4. fp8_gemm_kernel | kernel.py:236 | 🔶 TODO(blocker=FP8 dtype) | **GemmWarpPolicy 不是硬坑**:实测 `T.gemm(a,b,c)` 不带 policy fp16 → BUILT OK,fork gemm 签名根本无 policy 参数 → 删掉 policy 即可(idiom 小改)。真坑 = FP8 dtype(同 #2)。 |
| 5. sparse_attn_kernel | kernel.py:340 | 🔷 TODO(idiom-port,无 wall) | **gap 全是 idiom,无 dtype/闭源 wall**:① 删 `policy=T.GemmWarpPolicy.FullRow`(fork gemm 无 policy 参数)② `T.gemm(...,transpose_B=True)` → fork `npuir_dot` 不收 transpose_B → 先 `npuir_transpose(kv)` 再 gemm ③ alloc_fragment→alloc_shared/ub、threads=256→is_npu=True。`T.Pipelined/clear/fill/infinity` fork 都有 ✅。bf16 q/kv → 注意 bf16 那条闭源 vmul 墙(#1199),但本 kernel 的 exp/softmax 在 fp32 acc 上,bf16 只在 gemm 输入/输出 copy,大概率不撞。**最可能整条 port 成的 kernel,无 dtype-gap。** ④ **实测:NPU kernel 必须单 block 维** —— `T.Kernel(m,b,is_npu=True)` 报 "NPU kernel must have exactly one block dimension";改 `T.Kernel(m*b,is_npu=True)` + 内部 `bx=pid%m; by=pid//m`。**port 进展(`port_sparse_attn.py`)**:idiom 全改对(过 parser/结构检查),但**撞 in-process PassManager segfault(同 #100 类)** —— attention(2 gemm + transpose + online-softmax)的结构也触发那个后端 crash。workaround(拆小 kernel)对 attention 更难拆。→ **sparse_attn:idiom-port 可行(无 dtype/API wall),但卡 #100 同一个后端 segfault;是 #100 的又一个受害 kernel,等后端修或想 attention 专属拆法。** |
| 6. fp4_gemm_kernel | kernel.py:505 | 🔶 **BLOCKED-dtype** | GemmWarpPolicy 可删;**FP4 dtype 已确认 FAIL**(同 #3)→ FP4 dtype gap |

## 推理侧扫描完成（6/6 分类钉死，2026-06-03）
| kernel | 结论 |
|---|---|
| #1 sinkhorn | ✅ DONE — NUMERIC PASS 全 shape(idiom-port + multi-launch 绕 #100) |
| #2 act_quant | 🔶 FP8 dtype gap |
| #3 fp4_quant | 🔶 FP4 dtype gap |
| #4 fp8_gemm | 🔶 FP8 dtype gap(+ GemmWarpPolicy 可删) |
| #5 sparse_attn | 🔷 idiom-port 可行,卡 #100 后端 segfault |
| #6 fp4_gemm | 🔶 FP4 dtype gap(+ GemmWarpPolicy 可删) |

**两个共性 blocker 解锁全部:**
1. **FP8/FP4 dtype-support(fork TVM datatype 注册)** → 解锁 #2/#3/#4/#6(4/6 个)。同 bf16 #1199 族。**最高杠杆**:补一个 dtype-support 解锁一大半。
   - **责任层钉准(实测 traceback)**:报错在 **`3rdparty/tvm`(fork 自带 TVM)** 的 `tvm::runtime::DataType` 从字符串 `"float8_e4m3"` 构造时 —— fork 的 bundled TVM datatype parser 不识别 FP8/FP4 字符串。mainline tilelang/TVM 支持(kernel.py 就用 float8_e4m3),所以是 **fork TVM pin 旧 / 没把 FP8-FP4 datatype 注册带过来**,不是 fundamental missing。责任仓 = `tile-ai/tilelang-mlir-ascend`(它 vendor+build 这个 TVM)。
   - **提 issue 决策**:与"fork TVM 旧"是同一根因,可能和现有认知重叠 → 不急着单提第 3 个 issue(避免 issue-spam),先报 owner 批量决定。若提:目标 tile-ai/tilelang-mlir-ascend,标题"bundled TVM datatype parser 不支持 FP8/FP4(float8_e4m3/e5m2/float4_e2m1*),挡住 V4 推理 quant/gemm kernel"。
2. **issue #100 后端 PassManager segfault** → 阻 #1(已 multi-launch 绕过)#5(attention,难拆)。
3. **GemmWarpPolicy / transpose_B / 2D-grid** = 纯 idiom 小改(已验证 fork 等价写法),不阻塞。

→ **推理侧 tilelang-port 的真实工作量 = 2 个上游缺口(FP8/FP4 dtype + #100 segfault)+ 逐 kernel idiom 重写。** sinkhorn 已证全链路可行(编译+真跑+数值对全 shape)。

## 扫描策略(从易到难，每个走 sinkhorn 同范式)
1. ~~act_quant_kernel~~ **已测 → BLOCKED-dtype**:fork TVM 不支持任何 FP8 dtype(e4m3/e4m3fn/e5m2 全 FAIL;int8/fp16 OK)。缺口 = 补 fork TVM 的 FP8 datatype 注册(C++/TVM 层,同 bf16 #1199 族),不是 idiom。→ 可单开 fp8-dtype-support issue/PR;或 act_quant 走 int8 量化路径 PoC。
2. fp8_gemm / sparse_attn:查 `T.gemm` 不带 GemmWarpPolicy 的默认行为(fork 有 gemm,可能 policy 是可选/有默认);若必须 policy → 补 fork 的 GemmWarpPolicy 或改写。
3. fp4_*:FP4 dtype 支持。
4. 每个 BUILT OK 后,真 NPU run 对 torch ref 验数值(判别器:随机 look-alike 不会数值对)。

## case-gen 驱动的验证管线（owner 指令：用 a5_ops case-gen 做 dtype/shape 排列组合）
- **管线已打通并验证**(`sweep_sinkhorn_shapes.py`,log `RESULT_sinkhorn_shapesweep_*.log`):每个 kernel × shape band → tilelang 编 → 真 NPU run → 对 torch ref(a5_ops `model.py` 的 forward)验数值。
- **sinkhorn 全 shape PASS**:N ∈ {1(degenerate), 7(prime), 256/512(core), 1009(prime-large), 1024(tile-boundary)},max_abs_diff ≤ 1.5e-7,全 PASS。
- a5_ops 资产复用:`workspace/hc_split_sinkhorn/model.py`(torch truth)、`src/scripts/reference_provider/case_gen.py`(`generate_cases`,shape_plan + distribution + per-tensor dtype 三轴)。
- 下一步对每个 kernel:定 SCHEMA(张量 shape/dtype 空间)→ generate_cases 出 dtype×shape case → 编+跑+验 → 产覆盖矩阵(kernel × dtype × shape → 编过?/数值对?/撞哪类墙)。
- dtype 轴对 FP8/FP4 kernel 直接暴露 dtype-gap(#2/#4/#6);shape 轴对 #100 segfault 暴露触发边界。

## 共性结论(已从 sinkhorn 得出，适用全扫描)
- NPU idiom(alloc_shared/ub + is_npu=True + 块 intrinsic)是对的写法;GPU idiom(fragment/threads/scalar 循环)会 lower 进 zero-scope 路由不了。见 [[tilelang_npu_idiom_vs_gpu_idiom]]。
- bishengir/tilelang in-process PassManager 有结构敏感 segfault;拆小 kernel(multi-launch)可绕。见 [[bishengir_col_reduce_segfault]]。
- 这些 kernel 源文件按更新版 tilelang 写,fork 缺 GemmWarpPolicy/float8_e4m3/vlog 等 → 部分是"补 fork API / 换 dtype"而非纯 idiom。

---

# 训练侧 tilelang 算子扫描（miles V4，2026-06-03）

> 源:`_miles_dsv4_preserved/miles_plugins/models/deepseek_v4/ops/kernel/*.py`(miles V4 训练算子,全 `@tilelang.jit`/CUDA 目标)。
> 注:训练侧运行层已定走 CANN-native(报告 §4.2);tilelang 化是"对齐/退路"场景。本扫描建与推理侧同样的覆盖矩阵 + idiom/dtype 分类。
> 方法:静态读每 kernel 的 API/idiom 用法(GemmWarpPolicy/T.gemm/transpose_B/float8/alloc_fragment/Pipelined),套用推理侧已实测的 fork 等价规则分类。

## 训练侧 kernel 覆盖矩阵(API/idiom 静态分析 + 推理侧实测规则)
| kernel | GemmWarpPolicy | gemm/transpose_B | float8 | 预分类 |
|---|---|---|---|---|
| act_quant | — | — | **e4m3 ×2** | 🔶 **FP8 dtype gap**(同推理 act_quant;fork TVM 无 FP8) |
| sinkhorn | — | — | — | 🔷 idiom-port(同推理 sinkhorn,已证可行 + multi-launch 绕后端) |
| tilelang_indexer_fwd | ✅(删) | gemm + transpose_B(换 npuir_transpose) | — | 🔷 idiom-port;gemm 有 → 注意 #100 后端 segfault 风险 |
| tilelang_indexer_bwd | — | gemm×3 + transpose_B×3 | — | 🔷 idiom-port(gemm-heavy,反向) |
| tilelang_sparse_mla_fwd | ✅×2(删) | gemm + transpose_B | — | 🟢 **idiom-port 已有 working 先例**:报告 §4.0 记录 fork 自带 `examples/sparse_mla_fwd.py` 在 NPU 上 `All check passed!` rtol=5e-3 → 这条 attention/gemm 在对的 idiom 下确实能编能跑 |
| tilelang_sparse_mla_bwd | ✅×5(删) | gemm×5 + transpose_B×2 | — | 🔷 idiom-port(最复杂,5 gemm,反向);#100 风险最高 |

## 训练侧净结论(与推理侧一致 + 一个关键正面证据)
- **同两个共性 blocker**:① FP8 dtype(act_quant)② #100 后端 segfault(gemm/attention-heavy 的 fwd/bwd 风险)。`GemmWarpPolicy`/`transpose_B` 仍是纯 idiom 小改。
- **关键正面证据(比推理侧更强)**:`sparse_mla_fwd` 这条 **gemm+attention kernel 在 fork 上有 working 先例**(§4.0,`examples/sparse_mla_fwd.py` `All check passed!` 真 NPU)→ **证明 gemm/attention 类 tilelang kernel 在对的 NPU idiom 下确实能编能跑数值对**,不是只有 sinkhorn 这种纯 vector 能成。这是"tilelang-port 路通"在 gemm/attention 上的硬证据(之前只在 sinkhorn vector 上验过)。
- **训练侧 vs 推理侧差异**:训练侧 gemm/attention 占比高(indexer/mla fwd+bwd 全是 gemm),#100 后端 segfault 风险更集中在这些大 kernel 上;但 sparse_mla_fwd 先例说明对的 idiom + 合适规模能过。
- **诚实边界**:训练侧本扫描是**静态 API/idiom 分类 + 复用 sparse_mla_fwd 既有实测**,没对每个 kernel 重新 cold-drive 编译(act_quant 的 FP8 与推理侧同根、sparse_mla_fwd 已有 §4.0 实测;indexer_bwd/mla_bwd 未单独重编验)。要完全钉死需逐个 cold-drive,工作量大;当前给出的是证据支撑的分类,不是 6/6 全新实测。

---

# fork-native NPU op-class 编译基线（capability floor，2026-06-03）

> 目的:建立 fork NPU 后端"哪些 op 类在对的 idiom 下确实能编"的正面基线,作为所有 tilelang-port 分类的地板。这样"撞墙"的 kernel 能明确归因到具体 gap(dtype/structure),而不是"后端整体不行"。log `RESULT_fork_opclass_baseline_*.log`(确定性,本 session 实编)。

| op class | 编译 | 覆盖的原语 |
|---|---|---|
| elementwise(1D,ub + T.Parallel) | ✅ BUILT OK | alloc_ub + vadd-class |
| rms_norm(reduce + rsqrt + 广播 mul) | ✅ BUILT OK | reduce_sum + npuir_rsqrt/mul/div/add |
| gemm fp16 | ✅ BUILT OK | T.gemm(无 policy) fp16→fp32 |
| gemm int8 | ✅ BUILT OK | T.gemm int8→int32 |
| exp elementwise | ✅ BUILT OK | T.exp on ub |

**基线结论**:fork NPU 后端的 **vector / reduce / matmul(fp16+int8) / 超越函数(exp)** 这些 op 类在对的 NPU idiom 下**全部 BUILT OK**。→ 前面所有"撞墙"的 kernel(sinkhorn/sparse_attn 的 #100 segfault、act_quant 的 FP8)**不是后端整体不行,是两个具体 gap**:① FP8/FP4 dtype 注册缺 ② 大/特定结构 kernel 的 PassManager segfault(#100)。**fork 的 NPU 算子能力地板是健全的** —— 这强化"tilelang-port 路通、差异是有限 gap"的结论。
