# DeepSeek-V4-Flash 昇腾 A3 NPU 移植报告

**版本**:2026-06-04 ·  **目标平台**:Ascend 910C(A3,SOC `Ascend910_9382`,CANN 8.5.2;9.1.0-beta.1 复用基座已验证,见 §三.3) ·  **基线**:减层(1–2 层)

> **2026-04 增补**:新增 §三.3 末「tilelang-port PoC + 性能 + fp8 修正 + CANN 9.1.0-beta.1 验证」+ §八「miles DSv4 训练精度与 fp8 范式」。核心结论:(1) tilelang-on-A3 vector+gemm 均能编+真 NPU run+数值对(API 路径);(2) **fp8 = A3 硬件墙**;(3) miles 训练 bf16 高精、量化仅 rollout 侧;(4) CANN 9.1.0-beta.1 torch_npu ABI ✅ 但 bishengir gemm 回归 ⚠️;(5) issue #100 已提。
>
> **2026-06-05 增补(基于最新 miles 重新基线)**:新增 §九。M1/M2/M4 完成(均独立 agent 验证)。**核心转向:最新 `radixark/miles` main 的 DSv4 plugin 纯 GPU 无 NPU;torch_npu CANN-native 覆盖全部核心算子 fwd+bwd → NPU 运行层走 CANN-native,不逐个 tilelang re-port**(miles-001 cookbook 已降级 FALLBACK)。**M3(§9.4):3 个核心算子(sparse-MLA fwd+bwd / compress fwd / indexer fwd)A3 真机验证 PASS**(reduced+watchdog,HBM 未扰);attn_sink 适配是唯一真工程点(native 签名无 attn_sink)。未达 PR-bar 的产品化待 owner 节奏。

> **适用范围**:本报告覆盖 **DeepSeek-V4-Flash**(4096 隐藏维 / 43 层 / 64 头 / 256 专家)。同系列的 **V4-Pro**(7168 / 61 / 128 / 384,更大)尚未做对应移植,差异与计划见附录 A。

---

## 一、摘要

本报告记录 DeepSeek-V4-Flash 在昇腾 A3 NPU 上的移植工作,覆盖**推理(SGLang)**与**训练(Megatron + miles)**两条链路。结论分两个层面陈述,严格区分**已实测**与**待完成**:

- **推理链路**:SGLang 主线的 `DeepseekV4ForCausalLM` 模型类在 A3 NPU(bfloat16)上完成 `generate()` 端到端执行,并闭合一个推理-权重同步-再推理的循环。模型类本身在上游已具备;移植工作集中在补齐昇腾后端(`AscendAttnBackend` / sgl-kernel-npu)缺失的 V4 专用算子接口。
- **训练链路**:DeepSeek-V4-Flash 真实配置(4096 隐藏维、64 注意力头、256 专家 MoE)减层后,单层完成完整训练迭代(前向+反向+优化器更新),两层完成前向+反向。五个核心算子经 CANN 原生算子实测打通。
- **训练→推理参数流动**:在两层减层、共享权重的设定下,经判别实验确认 megatron 训练得到的注意力权重传入 SGLang 推理引擎后,引擎输出发生**训练特异性**改变(与等幅随机扰动可区分)。

全模型(43 层)、数值正确性、真实 RL 奖励训练尚未完成,原因与边界见**第五节**。

> 术语:本报告中"减层权重 / 减层 checkpoint"(代码与文件名中简写为 *fab*,即 fabricate)指**按 DeepSeek-V4-Flash 真实架构与 config 字段构造、但层数缩小、权重随机初始化**的合成 checkpoint。其用途是验证链路可运行性(模型能否加载、前反向/推理能否跑通),不验证数值或文本质量——随机权重的输出本就是乱码。

---

## 二、移植对象

DeepSeek-V4-Flash 真实配置(来源 `workspace/T32_tilelang_rescue/v4_real_truth/v4_real_config.json`):

| 维度 | 取值 |
|---|---|
| 隐藏维 / 层数 / 注意力头 | 4096 / 43 / 64 |
| MLA | `q_lora_rank=1024`、`kv_lora_rank=512`、`qk_rope_head_dim=64`、`v_head_dim=128` |
| 输出侧 LoRA | `o_lora_rank=1024`、`o_groups=8` |
| Hash-coding 路由 | `hc_mult=4`、`hc_sinkhorn_iters=20` |
| C4 indexer | `index_n_heads=64`、`index_topk=512` |
| MoE | 256 专家、`moe_intermediate_size=2048`、每 token 6 专家、1 共享专家 |
| RoPE(yarn) | `factor=16`、`original_max_position_embeddings=65536`、`beta_fast=32`、`beta_slow=1` |

相对 V2/V3,V4 的新增结构决定了昇腾适配的主要难点:**hash-coding sinkhorn 路由**、**C4 多级压缩 KV(c4/c128)**、**lightning indexer(稀疏 topk)**、**o_lora 分组输出投影**、**compressor(NSA 风格压缩注意力)**。

---

## 三、推理链路(SGLang)

SGLang 主线已包含 `deepseek_v4.py` 与 `EntryClass=[DeepseekV4ForCausalLM]`,模型定义完整。移植的全部工作量在于昇腾后端缺失的 V4 专用接口与算子。

### 3.1 已验证结果

- **`generate()` 端到端执行**:真实 `DeepseekV4ForCausalLM` 配合减层权重,bfloat16,A3 NPU,0.9 秒返回非空输出。
- **推理循环闭合**:推理 → 通过 `update_weights_from_tensor`(仅注意力权重)同步权重 → 再推理,权重同步可观测地改变了引擎输出。

### 3.2 适配项分类(按上游归属)

14 个适配点按**责任上游**分组。状态:**production** = 数值实测等价、可直接提交;**设计选择** = 故意保留、非妥协;**临时方案** = 仅满足短序列演示,生产须上游真实现。

**A. `sgl-project/sglang`(主仓 jit_kernel/dsv4 —— 算子级,4 个 production + 1 设计选择)**

| 适配点 | 处理方式 | 状态 |
|---|---|---|
| `jit_kernel/dsv4` `fused_q_norm_rope` 的 RMSNorm 部分 → `npu_rms_norm` | 原生算子替换,逐位等价(0.000e+00) | **production** |
| `jit_kernel/dsv4` `silu_and_mul_clamp` → `npu_clipped_swiglu` | 原生算子替换,相对误差 ≤0.77%,clamp 内逐位等价,端到端验证 | **production** |
| `jit_kernel/dsv4` `linear_bf16_fp32` 的 `torch.mm(out_dtype=fp32)` 昇腾不支持 → 去 kwarg + `.float()` | 等价 | **production** |
| `AscendAttnBackend.forward_extend/decode` 不接收 `compress_ratio` → `**kwargs` 吸收 | 向后兼容 | **production** |
| `jit_kernel/dsv4` RoPE 复数乘部分保留 fp32 torch | `npu_apply_rotary_pos_emb` 为 rotate-half 约定,与 V4 interleaved 复数乘不符(误差 4.22),fp32 更优 | **设计选择(非妥协)** |

**B. `sgl-project/sglang`(主仓 AscendAttnBackend —— V4 后端接口缺失,7 个临时方案)**

| 适配点 | 处理方式 | 状态 |
|---|---|---|
| `_maybe_upgrade_forward_metadata` 缺失 | 空实现占位 | 临时方案 |
| `forward_c4_indexer` 缺失 | 空实现占位 | 临时方案 |
| `forward_core_compressor` 缺失 | 空实现占位 | 临时方案 |
| `store_cache` / `forward_compress` / `init_forward_metadata_indexer` 缺失 | 空实现占位 | 临时方案 |
| `DeepSeekV4TokenToKVPool.get_key_buffer` 未实现 | V4 dense 短路 | 临时方案 |
| `forward_decode` cache 读取不匹配 | 同上短路 | 临时方案 |
| `fused_k_norm_rope_flashmla` 写 FP8 打包字节进 paged KV cache | 演示跳过 scatter(短序列足够) | 临时方案 |

**C. `sgl-project/sgl-kernel-npu`(kernel 库 —— 真实 NPU kernel,1 个临时方案)**

| 适配点 | 处理方式 | 状态 |
|---|---|---|
| `mhc.hc_split_sinkhorn` 依赖 tilelang/CUDA,昇腾无 | torch 组合实现 | 临时方案(生产须 sgl-kernel-npu 提供真 kernel;另见训练侧 op-gen sinkhorn) |

> 注:B 组的 7 个 AscendAttnBackend 接口,其"真实现"最终也落到 `sgl-kernel-npu` 提供底层 NPU kernel + `sglang` 主仓接好接口的协作——空实现占位在 `sglang` 主仓,底层算子在 `sgl-kernel-npu`。

**D. `Ascend/pytorch`(torch_npu / CANN —— 平台算子缺口,1 个临时方案)**

| 适配点 | 处理方式 | 状态 |
|---|---|---|
| NPU `aclnnIndex` 不支持 complex64 | 取实部后再 view-as-complex | 临时方案(提 torch_npu/CANN complex64 indexing issue) |

**空实现占位为何能在演示中工作、代价是什么**:上述 `forward_c4_indexer` / `forward_core_compressor` / `store_cache` 等接口负责 V4 的多级 KV cache 压缩与稀疏 indexer 选择。在 1 层、序列极短(2 token)的演示中,cache 尚未被填满,跳过这些接口不影响前向能否走通。代价是**功能性而非性能**:跳过后该层注意力退化为不压缩、不稀疏选择,长序列会显存溢出或数值错误,且输出在数值上并非真实 V4(仅形状正确、能跑完)。将单个空实现升级为 torch 组合的 PoC 级实现约需半天到一天;涉及 KV cache 写入路径(FP8 打包 scatter)的约需 2–3 天;生产级(sgl-kernel-npu 真实 AscendC 实现)为上游工作,单项周级。

### 3.3 推理侧 tilelang-port PoC(2026-06-03):把 V4 推理 tilelang 算子搬到 NPU 的可行性与边界

> 背景:V4 推理侧若不走 torch-fallback / CANN-native,而要恢复高性能,需让 sglang/fork 的 `@tilelang.jit` 算子在 tilelang-mlir-ascend(NPU MLIR 后端)上编译。本节是对"这条路差异到底多大"的实测回答。全程 log 在 `_v4_runlogs/`,详见 `workspace/task-dag-realdelta/TILELANG_PORT_POC_2026-06-03.md` + `V4_INFERENCE_TILELANG_KERNEL_SWEEP.md`。

**tilelang-port 适用场景(何时该走这条路):**
- CANN 无 native 覆盖、torch-fallback 性能不够的推理算子(如 mhc sinkhorn、compressor)——这些没有 wired-in 的 CANN 路径(§3.2),tilelang 是恢复性能的唯一编译路。
- 与 miles 训练侧算子对齐(训练侧若也走 tilelang 而非 CANN)。
- V4-Pro / 其他模型 / 未来算子,CANN 不一定有 native —— tilelang-ascend 是退路(同 §4.0 结论)。

**核心实测结论:差异 = 有限的"GPU-idiom → NPU-idiom"kernel 重写 + 一个可绕的后端 bug,不是闭源死墙。**
- **关键发现:卡点是 kernel 写法 idiom 错配,不是 fork codegen 缺口、不是闭源墙。** sglang/fork 的 V4 推理 kernel 源码用 GPU/CUDA idiom(`alloc_fragment` / `threads=` / scalar `T.Parallel` 写 / `GemmWarpPolicy` / `transpose_B=`),fork(v0.1.1.030)会 lower 进路由不了的 `zero`/`cbuf` scope。**NPU-correct idiom = `alloc_ub`/`alloc_shared` + `T.Kernel(n, is_npu=True)`(单 block 维) + 块向量 intrinsic(`T.vadd/vsub/vmul/vdiv/vexp/vbrc/reduce_max/reduce_sum/npuir_transpose`) + Python `range()` 展开**。参照 fork-native example(`vectorization_in_parallel.py` / `norm/example_rms_norm.py` / `flash_attn_npuir.py`),不是 fork ship 的 `deepseek_v4/inference/kernel.py`(后者按更新版 tilelang 写,fork 缺 `GemmWarpPolicy`/`float8_e4m3`/`vlog`)。

**做的修改(本 session,kernel-source 层 + 工具):**
- 重写 `hc_split_sinkhorn` 为 NPU idiom → **编译 BUILT OK + 真 NPU run + 对 torch ref 数值 PASS**(`max_abs_diff` 8.9e-8;全 shape band N=1/7/256/512/1009/1024 均 ≤1.5e-7)。
- 针对后端 segfault 的 **multi-launch 拆解 workaround**(softmax-preamble kernel + 单轮迭代 kernel,host 循环 19 次 = `sinkhorn_iters=20` 等价)—— 已验证 BUILT OK + 数值对。
- patch `jit_npu.py` 加 `TILELANG_DUMP_NPUIR`(dump 中间 npuir,定位崩点用,诊断工具非生产改动)。
- 复用 a5_ops `case_gen` 打通 case-gen→tilelang→NPU→verify 管线(shape/dtype 排列组合覆盖矩阵)。

**逐 kernel 扫描结果(6 个推理 tilelang kernel,覆盖矩阵):**
| kernel | 状态 | 缺口分类 |
|---|---|---|
| hc_split_sinkhorn | ✅ DONE(shape-swept NUMERIC PASS) | idiom-port + 后端 segfault(multi-launch 绕过) |
| act_quant_kernel | 🔶 blocked | **FP8 dtype** fork TVM 不支持(e4m3/e4m3fn/e5m2 全 FAIL;int8/fp16 OK) |
| fp8_gemm_kernel | 🔶 blocked | FP8 dtype(GemmWarpPolicy 可删、gemm 无 policy 能编) |
| fp4_gemm_kernel | ⬜ TODO | FP4 dtype(预计同 FP8) |
| fp4_quant_kernel | ⬜ TODO | FP4 dtype |
| sparse_attn_kernel | 🔷 idiom-port 可行但卡后端 | idiom 全可改(删 policy、`transpose_B` 换 `npuir_transpose`、1D grid),但撞 **issue #100 同一个 PassManager segfault** |

**共性缺口(决定工作量):**
1. **后端 segfault(issue #100)**:tilelang in-process MLIR PassManager 在较大 vector/attention kernel 上结构敏感 crash;`faulthandler` 定位 `tladapter/utils.py:103 Pipeline.run`。已提 issue #100 含最小复现 crash/OK 配对 + workaround。**sinkhorn 与 sparse_attn 都受影响** → 这是 tilelang-port 推理路的头号 blocker。
2. **FP8/FP4 dtype 支持缺失**:fork TVM datatype 注册无 FP8/FP4(同 bf16 #1199 族),挡住 3/6 个 kernel(act_quant / fp8_gemm / fp4_gemm)。补这一个 dtype-support 即解锁一半。
   - **★ 关键修正(2026-06-04,深挖到底层后):FP8 在 A3 上是硬件墙,不是软件可补的 dtype 缺口。** 本 session 已把开源软件栈这一层打通(在 workspace tilelang 的 `codegen_npuir_api.cc::DTypetoMLIRType` 加 fp8→`getFloat8E4M3FNType()/getFloat8E5M2Type()` 映射 + `data_type.h::String2DLDataType` 解析 `float8_e4m3fn`/`float8_e5m2`,均已重编 `libtilelang_module.so` 生效),但 dtype 通了之后撞到真正的墙:**A3(V220)硬件没有 fp8 单元**。8.5.0 bishengir 0.1.0 报 `'hivm.hir.store' op ... should have element type [allow-list 含 bf16 但不含 float8]`(HIVM op verifier 的 type allow-list 漏了 float8,非"硬件"报错而是编译期 allow-list 缺失);升到 **CANN 9.1.0-beta.1(bishengir 1.1.0)** 后,verifier allow-list **已含 fp8**(2026-01-20 上游加入),但 fp8 store 仍直接报 **`'hivm.hir.store' op Current hardware doesn't support fp8 type`** —— 这次是芯片层 reject,不是编译器允许列表问题(机制已在源码 `HIVMDMAOps.cpp:248` confirmed;⚠️该 bishengir-compile run 的原始 stdout 未单独落盘,字符串凭当时观察;待 A3 空闲可重跑一行复证)。→ **act_quant / fp8_gemm / fp4_gemm 这 3 个 kernel 在 A3 上无法靠补软件解锁;fp8 是 A5(arch35)目标。** A3 跑 DSv4(原生 FP8 权重)走 `fp8_cast_bf16.py` dequant→bf16 全程 bf16。详见 KB §13.1 + memory `fp8_is_a3_hardware_limit_not_software`。

**诚实定位**:推理侧 tilelang-port **路通**(sinkhorn 全 shape 数值已验对),差异是有限工程量(idiom 重写 + 补 dtype + 绕/修后端 segfault),**不是闭源死墙**;但比"几个 backport"重(每 kernel 重写 + 两个共性 backend/dtype 缺口要上游修)。对 **V4-Flash 运行层**这窄目标仍是 torch-fallback/CANN 更直接(同 §4.0),tilelang-port 是性能恢复 / CANN 无覆盖时的路。

**性能实测(2026-06-03,sinkhorn,真 NPU,vs torch-NPU):tilelang 当前比 pytorch 慢 —— 因为 #100 逼出的 multi-launch workaround 吃掉性能。**
| N | tilelang(multi-launch)ms | torch-NPU ms | speedup |
|---|---|---|---|
| 256 | 2.08 | 0.94 | **0.45×** |
| 1024 | 2.46 | 1.11 | **0.45×** |
| 4096 | 23.2 | 1.38 | **0.06×** |
- **根因(实测分解)**:单个 `sink_iter` launch = 0.09ms(N=256)/1.30ms(N=4096),×20 launch = 1.8ms/26ms,与实测全程一致 → **慢全在"20 次独立 launch + 每次 global 读写整张 tensor"**(#100 workaround 税),不是 kernel 计算本身慢。N=4096 每 launch 1.3ms(读写整张 4096×4×4),launch 开销主导。
- **关键含义**:**#100 不只是 compile/正确性 blocker,更是 tilelang-port 的性能 blocker。** 高性能的单 kernel 形态(迭代间数据留 on-chip、~1 次 launch)本应快过 torch,但它 segfault(#100)。→ **tilelang-port 的性能价值被 #100 锁死;修好 #100(单 kernel 可编)才谈得上比 torch/CANN 快。**
- **vs CANN-替换**:sinkhorn 无 native CANN op;torch-NPU 路径本身就走 torch_npu→CANN aclnn 派发,所以这里的 "torch-NPU" 基线即"CANN-backed 执行"。真正独立的 CANN 路 = a5_ops 生成的 AscendC kernel(workspace/hc_split_sinkhorn,perf 报 51.09× vs CPU-torch,但那是 vs CPU 不是 vs NPU-torch,口径不同,不可直接比)。**结论:当前 tilelang(multi-launch)< torch-NPU(=CANN-backed);tilelang 要赢需先解 #100。**
- 详见 `workspace/task-dag-realdelta/V4_INFERENCE_TILELANG_KERNEL_SWEEP.md` 性能段 + log `_v4_runlogs/RESULT_sinkhorn_perf_*.log`。

**gemm/attention 类(sparse_mla_fwd)性能:当前无法在 NPU-capable 容器上取数(诚实 infra blocker)。**
- 想测 `sparse_mla_fwd` 的 tilelang-vs-torch 性能(它是单 kernel、不卡 #100,能看 tilelang 不被拖累时的真实力),但撞到**容器能力分裂**:
  - **sgl_probe**(有 NPU 算力 davinci1):其 tilelang build **编不过** sparse_mla_fwd —— `codegen_npuir_dev.cc:175 VcastCodegen→getBroadcastDim` 的 broadcast-dim 不匹配 compile error(默认 args,example 原样 + runpy 都崩)。
  - **tlrescue**(§4.0 报 `All check passed!` 的容器):**无 `/dev/davinciN` 算力设备**(只有 davinci_manager)→ 不能跑/计时。tilelang 装在 `/tilelang-src`(与 sgl_probe 的 build 不同)。
  - → **两个容器的 tilelang-mlir-ascend build 不一致**:tlrescue 能编 sparse_mla_fwd 但无算力,sgl_probe 有算力但编不过。当前**没有"既能编 sparse_mla_fwd 又有 NPU"的容器** → gemm/attention 类 perf 取不到数,需先对齐容器(在 sgl_probe 修 codegen broadcast bug,或给 tlrescue 挂 NPU device)。
  - **诚实**:不为了交差编造 gemm/attention 的 perf 数;sinkhorn 的 perf 是真测的(slower,#100 税),gemm/attention 的 perf 是 infra-blocked,待容器对齐。§4.0 的 sparse_mla_fwd "All check passed!" 是 tlrescue 的**正确性**结果(非本 session 重验、非 perf)。
  - **codegen broadcast bug 已定位(sgl_probe build)**:`getBroadcastDim`(codegen_npuir_dev.cc:175)的 `CHECK(shape0[i]==shape1[i])` else 分支 —— 某个 `T.vcast`/broadcast op 的两操作数 dim 既不等也不可广播(都非 1)。从 `VcastCodegen` 调用。是 sgl_probe tilelang build 的真 codegen bug(sparse_mla_fwd 触发)。修需追是 kernel 哪个 vcast 触发 + 改 codegen broadcast 逻辑(C++,中等工作量,深入 500 行 attention kernel 的 codegen)。**这是解锁 gemm/attention perf 测量的前置;但属于"修 fork C++ codegen"级别的投入,待 owner 确认是否优先(选项①)。**
  - **深挖容器/build(2026-06-03,试 ①+②):**
    - sgl_probe 与 tlrescue 的 workspace `/home/z00637938/workspace/tilelang-mlir-ascend` 是**同一挂载卷**:`codegen_npuir_dev.cc` 与编出的 `libtilelang.so` **两容器 md5 完全相同**(`72f7815d...`)→ 同一个 build。
    - workspace 当前 commit = **`a19acd5`(就是我 PR #96 的"sparse_mla_fwd heads<block_H 写回修"那个 commit)**。`git show a19acd5` 只改了 `examples/sparse_mla_fwd.py`(9 行),**没碰 codegen_npuir_dev.cc** → **broadcast codegen bug 不是我 PR #96 引入的,是 codegen 里预先存在的**(parent commit 也有)。
    - **§4.0 的 "All check passed!" 对当前 workspace build 不成立(诚实修正)**:在当前共享 build(a19acd5)上,sparse_mla_fwd **两容器都编不过**(同 getBroadcastDim 错)。§4.0 的 pass 是更早的 build/state(或 `/tilelang-src`),**不是当前 build** → "gemm/attention 有 working 先例"对当前 build 是 stale 的,需重验才算数。
    - `/tilelang-src`(commit `b925cbe`,另一条 lineage)**装不起来**(Cython 缺),且 tlrescue 无 NPU device → ② 没有现成可用的"能编+有算力"组合。
    - **净结论**:解锁 gemm/attention perf 唯一路 = 在共享 workspace build 上修 `getBroadcastDim` codegen bug(C++)+ 重编 `libtilelang.so`(共享产物,多小时编译 + 可能影响其它容器的 in-flight 工作)。**投入大且动共享 artifact**,需 owner 拍是否值得(sinkhorn perf 已表明 tilelang 被 #100 锁死,gemm/attention perf 是否还要花这个代价取)。
    - **进一步实测(2026-06-03):当前 build 的 Cube/gemm codegen 是整体坏的,不止 sparse_mla_fwd 一个 bug。** canonical `examples/gemm/matmul.py`(标准 L1/L0C tiled gemm idiom)**也编不过**,但报的是**另一个** codegen 错:`'hivm.hir.nd2nz' op expected number of tensor results (0) to equal output tensors (1)`。→ **gemm/Cube 类在当前共享 build 上有多个 codegen bug(matmul: nd2nz;sparse_mla_fwd: broadcast)**,不是单点。我手写的 naive `alloc_shared` gemm 数值还错(maxdiff 1e+1,单 block 无 proper accum)。
    - **跨容器能力地板(当前 build)**:**vector 类(sinkhorn/elementwise/rms_norm/exp)能编能跑**;**gemm/Cube 类(matmul/sparse_mla_fwd)在当前 build 编不过**(matmul nd2nz / sparse_mla broadcast 两个不同 codegen 错)。

    - **⚠️ 关键限定(2026-06-03,诚实修正,别当成"fork gemm codegen 坏"):当前 workspace 是 DIRTY 的,gemm/Cube 失败是在"被本地大改过且没改完的 codegen"上,不是干净 fork commit。**
      - `git status`:workspace **51 个文件 dirty**,其中 **`src/target/codegen_npuir_dev.cc` 有 154 行未提交本地改动**(+ 一个 `.orig.t32_emptyop` 备份 → 像是 **之前 T32 session 在改 codegen 还没完成**),还有 `src/transform/npu_loop_vectorize.cc`(+9)、`src/op/ascend.cc`(+8)。`tilelang/jit/jit_npu.py`(+2)是我加的 DUMP_NPUIR 诊断(无害)。
      - `.so`(mtime 比 .cc 新)是从**这个 dirty/打了一半补丁的源码**编的 → matmul/sparse_mla 的 codegen 错**可能是这些 in-progress 本地改动的产物,不是 upstream fork commit a19acd5 的状态**。
      - → **不能下"fork 的 gemm/Cube codegen 整体坏"这个结论**;准确说法是"**本 workspace 当前这个被本地改过的 build 上,gemm/Cube 编不过**"。要判 upstream fork 真实状态,得在干净 checkout 上重验(但那要 rebuild,且会动这个 workspace 的 in-flight 改动)。
      - **这恰恰强证"绝不擅自重编共享 .so":workspace 里有别人(T32 session)没改完的 154 行 codegen 工作,我 rebuild 会 clobber/干扰它。** [[persistent_npu_container]] / NPU 容器神圣 在这里具体化为"别动共享 workspace 的 dirty 源码 + .so"。
    - **诚实净结论(收口)**:tilelang perf 能给的硬数据 = **vector 类(sinkhorn)= 0.06–0.45× torch,被 #100 锁死**(真测)。gemm/Cube 类 perf **取不到**:不是"fork 坏",是"当前 workspace 的 dirty build 编不过 + 有别人 in-flight codegen 改动,不该擅自 rebuild"。要 gemm/Cube perf 需 owner 决定如何处理这个 dirty workspace(干净 checkout 重验 / 等 T32 codegen 改完 / 单独 build 环境)。
    - **read-only 查清了那 154 行是什么(2026-06-03,git diff,无 rebuild):T32 session 正在修的就是我撞的这两个 gemm/Cube codegen bug,而且没修完。**
      - `CreateHIVMBinaryVectorOp` 的改动带注释:*"Leave buffer_shape EMPTY so downstream `getBroadcastDim()` short-circuits via its `if (...empty()) return dims;` early-return"* → **正是在修我 sparse_mla_fwd 撞的 `getBroadcastDim` broadcast crash**。
      - `NeedGenInsertSlice` / `AllocateNode` 里的 `tensor::EmptyOp` 改动 + 那个 `.orig.t32_emptyop` 备份名 → 对应我 matmul 撞的 `nd2nz` "tensor results(0)==output(1)" 错(empty-op 相关)。
      - → **gemm/Cube 编不过 = T32 的 codegen fix in-progress 没收尾,不是 fork inherently 坏。** 等这个 fix 完成(或 owner 指示我接手/在干净环境验),gemm/Cube 大概率能编 → 那时才测得到 perf。**再次强证:绝不擅自 rebuild —— 会 clobber 没改完的 fix。**
      - 这也回答了"whose 154 lines, changed to what":T32 codegen session(= 我自己,solo workspace),在修 getBroadcastDim(broadcast)+ EmptyOp(nd2nz)两个 gemm/Cube codegen bug。

    - **RESOLVED(2026-06-03):gemm/Cube 其实能编能跑 —— 之前"编不过"是我测错 codegen 路径(Developer mode)的假象。**
      - lower.py:170 路由:`TILELANG_ASCEND_MODE` None/expert → **API codegen(默认)**;设 "Developer" → **DEV codegen**。我之前所有 gemm/attention 测试都设了 Developer → 跑在 DEV 路径,所以撞 DEV 侧未收尾的 codegen + 我的 _api.cc 修复没生效。
      - **`TILELANG_ASCEND_MODE=expert`(API 路径)下:matmul → `All check passed!` ✅(有落盘 run-log `RESULT_matmul_perf_*`);sparse_mla_fwd → API 路径 session 内观察到 `All check passed!`,但⚠️该 PASS 未单独落盘 run-log**(独立验证 2026-06-05 只找到 sparse_mla 的 Developer-mode FAILURE 日志,API-path PASS 凭当时观察未持久化)→ **sparse_mla_fwd 的 API-path 数值 PASS 标记为「session 内观察、待 A3 空闲重跑落盘复证」**(非被推翻,只是当前无法独立复证;minirl 占 chip 8–15 期间不重跑)。
      - → **gemm/Cube tilelang kernel 在 NPU 上能编能跑数值对(API 路径),不是 fork 死限。** 修正 §3.3 早先"gemm/Cube 整体编不过"的结论 —— 那是 wrong-mode 假象(我用了 Developer mode)。
      - **★ 诚实修正(2026-06-03,build-target 发现):我之前说"补 nd2nz 修复让 matmul 过"是错的。** runtime 实际加载 **`libtilelang_module.so`**(不是我一直在重编的 `libtilelang.so`);device codegen 在前者里。我早先的 matmul/sparse_mla "All check passed" 是在 `libtilelang.so`-重编(= **stale codegen,没含我的 nd2nz 修复**)上跑的 → **说明它们本来就在 stock API codegen 上能编能跑,不需要我的 nd2nz 修复**。我的 nd2nz 修复对它们 **非 load-bearing**(此前从未真正加载)。gemm/Cube 能跑是 fork 本身就支持(API 路径),不是我修出来的。
    - **gemm perf 实测(API 路径,真 NPU)**:matmul M1024×N512×K2048 fp16 → **tilelang 0.1123ms vs torch-NPU 0.0142ms = 0.13×(慢 ~8×)**,数值对。torch.matmul 派发到 CANN 手调 Cube gemm,远快于这个 basic-tiling 的 tilelang example kernel。
    - **两类 perf 完整结论(都实测)**:vector/sinkhorn 0.06–0.45× torch(#100 multi-launch 税)· gemm/matmul 0.13× torch(basic tiling vs CANN 手调)。→ **tilelang-port 功能可行(两类都能编+数值对)但都还不如 torch/CANN 快**;要 perf 竞争力需 #100 修复(vector 单 kernel)+ gemm kernel 调优(tiling/pipeline 对齐 CANN)。
    - **vector perf 大幅可恢复(2026-06-03,batching #100 workaround)**:#100 segfault 阈值下,每 kernel 可塞 3 个 sinkhorn 迭代(纯 row+col loop ≤3 安全)。改成 1 softmax-kernel + 7×(3-iter kernel)= 8 launches(代替 20 launches)→ **比 1-iter/kernel 快 2.42×**。对 torch 的比值:**N=256 1.15×(更快)、N=1024 0.96×(基本持平)、N=4096 0.17×(仍慢)**。→ **#100 的 multi-launch 税大部分能靠 batching 收回**(小-中 N 已达到/超过 torch);残留的大-N 慢是每次 launch 整张 tensor 的 global 往返(带宽 bound),只有 #100 真修(单 kernel 数据留 on-chip)才能完全解决。logs `RESULT_sink_batched*.log`。
    - **修正"vector 被 #100 锁死"的悲观说法**:更准确 = vector perf **大部分可恢复(batching)**,小-中 batch 已能持平/超过 torch;只有大 batch 还需 #100 真修。比之前"0.06–0.45× 锁死"乐观得多。

**CANN 9.1.0-beta.1 复用基座验证(2026-06-04,owner 指示"从 8.5 ML 底座 + 装 9.1.0 + 加验证"):**
> 背景:owner 定 9.1.0-beta.1 为以后实验标准基座(memory `project_cann_910beta1_standard_base`)。本段验证"在现有 8.5 ML 容器里挂 9.1.0 CANN toolkit"这条复用路是否可行,而非从裸 9.1.0 镜像重搭整个 ML 栈。容器 `cann910_test`(davinci5,避开 sgl_probe 占的 davinci1),`set_env.sh` 切 9.1.0。
| 维度 | 结论 | 证据 |
|---|---|---|
| torch_npu ABI | ✅ **兼容**(run-event 未落盘,见注) | 8.5-built torch_npu 2.8.0.post2(py3.11)在 9.1.0-beta.1 runtime 下 import + co-resident + 干净 session 已观察;⚠️独立验证 2026-06-05:`allclose/PASS` 那行未持久化(测试脚本已删),故确切口径 = 「版本栈共存 + 干净 session 已观察,matmul PASS 行未落盘」→ 复用路对 torch_npu 侧**强 plausible 但 PASS 行待重跑落盘复证** |
| bishengir 版本 | 1.1.0(2026-05-09,AscendNPU-IR 7058cef3) | verifier allow-list 含 fp8;ship `bishengir-compile-a5`/`hivmc-a5`(A5 支持) |
| **gemm/matmul** | ⚠️ **9.1.0-beta.1 回归** | apples-to-apples(同 npuir + 全套 tilelang flag):8.5.0 bishengir 编出 `o85.mix_aic.o`,9.1.0-beta.1 **SIGSEGV**(LLVM crash)。**限定**:测的 npuir 由 8.5-built tilelang 发出,故确切结论是"9.1.0 bishengir 在 8.5-tilelang 发出的 gemm npuir 上崩"——正是复用路(8.5 tilelang + 9.1.0 CANN)的失败模式;是否纯 9.1.0-bishengir bug(独立于 dialect 版本差)待 9.1.0-built tilelang 才能判 |
| **issue #100** | ⚠️ 9.1.0 **仍 segfault** | sinkhorn iters=4 在 9.1.0 上同样 exit 139 → 证实 #100 是 tilelang **in-process** PassManager bug(`libtilelang_module.so`),与 CANN/bishengir 无关 → 升级不修它 |
| fp8 | ⚠️ 仍是 A3 硬件墙 | 见上(verifier 通了,硬件仍 reject) |
- **净结论**:9.1.0-beta.1 复用-on-8.5-base 对 torch_npu 侧可行,但 **9.1.0-beta.1 bishengir 有 gemm 回归** + 不修 #100 → **对 A3 而言 9.1.0 不是无歧义的升级**(fp8 在 A3 本就 moot,是硬件)。建议:**A3 gemm 工作暂留 8.5**,直到 9.1.0 gemm 回归查清;9.1.0 真正价值在 **A5/fp8(另一块芯片)**。详见 memory `project_cann_910beta1_standard_base`。

---

## 四、训练链路(Megatron + miles)

训练链路为真实 DeepSeek-V4 模型层(MLA + compressor + indexer + 稀疏注意力 + hash-coding MoE 路由),运行于昇腾化的 Megatron(MindSpeed `core_r0.16.0` 分支适配 Megatron-Core 0.16)。

### 4.0 tilelang-ascend / bishengir 这条线(早期路线 + 为何转向 CANN + 何时仍必需)

miles 的 6 个 V4 训练算子源码全是 `@tilelang.jit`(CUDA 目标)。项目早期(T32)的路线是**让它们经 tilelang-ascend(MLIR 后端)在 NPU 上跑**——这也是项目第一个任务("修 tilelang-ascend 的 bug")的由来。这条线做了什么、现在什么状态:

- **跑通的(tilelang-ascend MLIR 后端,v0.1.1.030,tlrescue,A3 实测)**:`examples/sparse_mla_fwd.py` ✅(`All check passed!` rtol=5e-3,npu:0)、`examples/fp8_lighting_indexer.py` ✅。**注意**:跑通的是 tilelang-ascend **自带的 example 版本**(V4 同族算子),**不是 miles 源码原样**——直接编 miles 源码失败,用对了后端 API 的 example 能跑。其余(rms_norm/gemm/exp2)当时标"待确认"。
- **上游贡献(这条线的真产出,与 V4 无关也成立)**:`tile-ai/tilelang-mlir-ascend` PR #80(CheckUBBudget 早失败诊断 pass)· `Ascend/AscendNPU-IR` issue #251(R-KA-16:bishengir ExtendedCanonicalizer 在多 iter online-softmax 丢跨迭代累加器,311-pass bisect 定位 + 3 patch 方向)· `radixark/miles` PR #1246(4 个 tilelang 算子 NPU port + R-KA-16 mitigation)。KB:`bishengir-001` / `tilelang-001/002` / `miles-001`。
- **为何运行层转向 CANN**:后续发现 CANN 有 V4 核心算子的**直接 native 实现**(`npu_nsa_select_attention` 等,§4.2),比"把 miles 源码 adapt 到 tilelang DSL + 绕 bishengir R-KA-16 bug"更直接,故**运行层用 CANN native,不走 tilelang-ascend**。
- **诚实定位**:对 **V4-Flash 运行层**这个窄目标,tilelang 路线**不在关键路径**(被 CANN 取代)。但它**不是废work**:(1) R-KA-16 是真 bishengir 编译器 bug,影响任何 tilelang flash-attention online-softmax on NPU;(2) CheckUBBudget 是通用诊断改进;(3) **tilelang-ascend 是 CANN 无 native 覆盖时的 fallback**——V4-Flash 恰好 CANN 覆盖了核心算子,但 V4-Pro / 其他模型 / 未来算子 CANN 不一定有,届时 tilelang-ascend(+ 修好的 bishengir)是退路。

### 4.1 已验证结果(均为真实配置、A3 实测)

- 单个 `DeepSeekV4Attention` 层完成前向+反向训练步;
- 真实配置减层至 1 层,完成完整训练迭代(前向+损失+反向+AdamW 优化器更新,4.42B 参数,梯度全有限);
- 真实配置减层至 2 层,完成前向+反向(8.84B 参数,梯度全有限)。

> 证据级别:单层 flash 结果有冻结的 on-disk run-log(`PROTECTED_flash_attention_npu_RESULT.md`:out=(64,1,512)、loss=0.0353、grad_norm=0.173)。1 层/2 层的具体梯度计数(526/526、1051/1051)来自 commit message 里记录的 run-stdout,产出脚本在 disk(`v4_REAL_config_{1,2}layer_training_step_npu.py`),但**未把那次 PASS 的完整 stdout 单独提交为日志文件**。复现可重跑脚本。

### 4.2 算子实现路径

正在运行的训练层由两类实现构成,均经 pytorch 调用:

> 验证级别分两档,不混为一谈:**verified-run** = 有 A3 上捕获的独立 run-log;**spec-matched / coverage-confirmed** = API 签名与 V4 spec 对得上、在运行层 forward 路径上被调用,但**没有单独捕获 run-log 做数值等价**。

| V4 训练算子 | 实现 | 验证级别 |
|---|---|---|
| 稀疏 MLA(前向/反向) | `npu_nsa_select_attention`(D_qk=192/D_v=128,select_block=64,count=16,返回 attn 及 softmax max/sum 供反向) | **verified-run**(A3 捕获:attn (128,4,128) finite, 94.9us) |
| rms_norm | `npu_rms_norm` | **verified-run**(逐位等价 0.000e+00) |
| compressor | `npu_nsa_compress_attention` | **verified-run**(2026-06-02 A3 捕获:q(128,4,192)→out(128,4,128) finite, 57.3ms) |
| C4 indexer | `npu_sparse_lightning_indexer_grad_kl_loss`(带 grad+KL,明确 Atlas A3 train) | spec-matched(签名对应,层 forward 调用;独立 run-log 待补 —— 该反向算子需上游 fwd 的 softmax_max/sum 状态做输入,standalone setup 较重) |
| MLA 预处理 | `npu_mla_prolog_v3` | coverage-confirmed(dispatch 命中;独立 run-log 待补 —— 该算子需 10+ 个权重 tensor 全套,standalone setup 较重) |
| hash-coding sinkhorn | torch 组合 `_hc_split_sinkhorn_npu`(CANN 无对应原生算子) | 已接入运行层 |
| act_quant(fp8) | torch fp8-grid 模拟 `_fp8_e4m3_round`(CANN 无对应原生算子) | 已接入运行层 |

> 现状:3 个 verified-run(`npu_nsa_select_attention` / `npu_rms_norm` / `npu_nsa_compress_attention`)+ 2 个待补独立 run-log(`npu_sparse_lightning_indexer_grad_kl_loss` 需上游 fwd 状态、`npu_mla_prolog_v3` 需全套权重,standalone 捕获较重)。这 5 个在减层层的 fwd+bwd 整体跑通里都被调用过(层级 PASS)。

说明三点,避免歧义:

1. 五个核心算子使用 CANN 原生算子(经 `torch_npu` 调用),已真实接入运行中的训练层。
2. CANN 无原生对应的两个算子(sinkhorn、act_quant)运行层中以 **torch 组合**实现 —— **但两者都有 op-gen harness 生成的 AscendC kernel 作为后备方案(backup)**:精度已对齐(见下表),性能相对 pytorch 版本(统一格式,见 §4.3 表:sinkhorn 5.34× mean(min 4.02);act_quant 3.85× mean **但 min 0.38×——大张量反而更慢**)。当前运行层用 torch 组合是因为接入受阻(act_quant 的 fp8 接入被 torch_npu 限制挡住,见第 4 点;sinkhorn 待 `torch_npu` 自定义算子注册)——**AscendC 后备已就绪、验过精度,接入即可替换 torch 版以提速**。
3. **tilelang-ascend 后端在运行层中未被使用**(详见 §4.0):早期在 tilelang-ascend 上跑通过 V4 同族算子的 example(sparse_mla_fwd / fp8_indexer),但运行层最终用 CANN native(更直接);tilelang-ascend 保留为 CANN 无覆盖时的 fallback。
4. **关于 FP8(重要):A3 无 FP8 硬件,本工作全程不跑真 FP8。** 硬件能力矩阵确认 A3/a5/a2 的 VEC 单元均无 fp8 cast(无 `vconv` to/from fp8,SDK 仅有 fp8 解码无编码)。V4 设计中 act_quant 是激活的 fp8 量化优化;在无 fp8 硬件的 A3 上,运行层用 `_fp8_e4m3_round` 做**标准 fake-quant**:`round_to_fp8_grid(x/scale)*scale`,即 quant→dequant,**真注入 fp8 量化误差**,返回带误差的 fp32/bf16 高精度张量进后续高精度运算(从不以 fp8 存储或计算)。这与 QAT fake-quant 定义一致(注入量化误差 + 高精计算)。推理侧同理走 bf16,`SGLANG_OPT_FP8_*` 全关。§4.3 那个 op-gen act_quant kernel 产出的 fp8 字节是 kernel 内**软件 RNE 编码器**(scalar pipe 逐元素)生成的,用于精度 bit-match 参考,**不使用任何 fp8 硬件单元,也未接入训练层**。结论:训练与 rollout 在 A3 上是 **bf16/fp32**,fp8 仅作为被模拟的量化网格存在。
>
> **性能说明(直白)**:`_fp8_e4m3_round` 是 fp32 下的 fake-quant(QAT 式:round 到 fp8 网格、dtype 仍 fp32),它**拿不到真 fp8 的任何收益**(无带宽减半、无更快矩阵乘),反而在全宽 fp32 张量上**多了 ~7 个 round 的 elementwise op**。所以它**比直接 bf16 还慢,是负优化,不是性能路径** —— 它的唯一作用是在无 fp8 硬件的 A3 上把 fp8 量化的**数值行为**顶上去让训练逻辑跑通。要在 A3 上把 fp8 这步做出正收益,需:真 fp8 硬件支持(平台层) / 或 §4.3 的 AscendC act_quant 替换(当前被 torch_npu fp8 消费侧缺口挡住,且大张量 min 0.38× 仍偏慢)。即 **fp8 在 A3 是一个真实的性能坑,目前无正收益路径**。

### 4.3 用自动生成的 AscendC 算子替代 pytorch 算子的可行性验证

本节的关键问题不是"kernel 能不能孤立地调一下",而是:**能否用 op-gen harness 自动生成的 AscendC 算子替换运行层里的 pytorch 实现?** 这才是有意义的目标——pytorch 版是临时的功能性实现,自动生成的 AscendC 版若可替换,就能在精度对齐的同时拿到性能(§4.3 后备表:sinkhorn 5.34×)。以 `act_quant` 为对象做可行性验证,分三步,结论为"部分可行":

1. **生成 + 构建可行**:op-gen harness 产出 `act_quant` 的 AscendC 源码,在 A3 用 bisheng 构建成 pybind 扩展 `_act_quant_ext` 成功。
2. **调用 + 精度可行**:从 pytorch 在 NPU 上调用 `run_act_quant(x, block_size)`,输出与 CPU 真值逐位一致(scale 误差 0,fp8 逐值匹配 100%,反量化误差 0)。**替换的"精度等价"前提成立。**
3. **接入运行层——当前不可行(实测阻塞)**:把它接进训练层去真正替换 torch fp8 模拟时,kernel 返回 `float8_e4m3fn`,而 NPU 上对其 `.float()` 反量化报 `Float8_e4m3fn has not been supported`——torch_npu 无 fp8 device 支持(正是 torch 模拟要规避的限制)。

**可行性结论**:`act_quant` 的 AscendC 替换 = **生成/构建/调用/精度 全部可行,但"接入运行层替换 pytorch"被 torch_npu 的 fp8 消费侧缺口挡住,当前不可行**。解阻塞的三条路:(a) torch_npu 增 fp8→fp32 支持;(b) kernel 直接返回 fp32 反量化值(改输出契约);(c) NPU 上整数位运算解码 e4m3。三者任一落地后该替换即可行。对比之下,**sinkhorn 的 AscendC 替换没有 fp8 这层阻塞**(纯 vector,输出 fp32),替换可行性更高,只待 `torch_npu` 自定义算子注册接线。

> 一句话:**自动生成的 AscendC 算子替代 pytorch 的可行性,act_quant 卡在 fp8 消费侧(不可行),sinkhorn 无此阻塞(可行,待接线)**。这是比"能否孤立调用"更关键、也更难的结论。

#### AscendC 后备方案:精度对齐 + 相对 pytorch 的性能

运行层这两个算子用 torch 实现,但 op-gen harness 各生成了一个 AscendC kernel 作为**后备方案**,精度已对齐参考;性能相对 pytorch 版本的实测如下:

统一用 `mean (median, min, max, n)` 格式(AscendC vs pytorch,SYMMETRIC same-wrapper,两侧同 sync,ratio>1 = AscendC 更快):

| 算子 | 精度对齐 | 相对 pytorch 性能(统一格式) |
|---|---|---|
| `hc_split_sinkhorn` | PASS(pass_a 28/28 + pass_b 28/28 T1_STRICT,max_abs 2.38e-7) | **5.34× mean(median 5.42,min 4.02,max 7.05,n=6)** |
| `act_quant` | PASS(pass_a 24/24 + pass_b 24/24,byte-exact fp8 + bit-exact fp32 scale) | **3.85× mean(median 2.42,min 0.38,max 8.73,n=6)** |

> **读这两行的关键区别**:sinkhorn 的分布很窄(4.02–7.05),各 shape 都正收益,mean 5.34× 有代表性。**act_quant 的分布极宽(0.38–8.73)**:小张量快(max 8.73×),**大张量反而慢(min 0.38× ≈ 慢 2.6×)**。所以 act_quant 的 **mean 3.85× 不具代表性**——它跨了正收益和负收益两个区间;真正该看的是 **min 0.38×**:V4 production 跑大张量,act_quant 的 AscendC 替换很可能落在**负收益**区。统一格式没问题,但 act_quant 必须连着 min 一起看,不能只报 mean。
>
> 测量:对称 same-wrapper(两侧同 wrapper 同 sync);sinkhorn warmup=5/active=5,act_quant warmup=5/repeats=20、6 个 shape (4,256)→(128,4096)。

### 4.4 集成层适配项

| 适配点 | 处理方式 | 状态 |
|---|---|---|
| MindSpeed 适配 Megatron-Core 0.16 | 使用 `core_r0.16.0` 分支(commit 8bf0959,含 dsa TND 支持) | **production**(官方分支) |
| 两层反向 NaN(全掩码行 softmax 不稳定) | `sparse_attn_torch` 加 `nan_to_num(scores_max,neginf=0)` + `clamp(exp 参数,max=30)`,NaN 数 282→7→0 | **production**(标准 masked-softmax 守卫,可提交 miles) |
| `npu_rms_norm` 参数签名不匹配 | shim:匹配 gamma dtype + 丢弃多余参数 | 临时方案(待 MindSpeed 适配签名) |
| TransformerLayer 返回契约 | 将 `DeepSeekV4Attention.forward` 改为返回 `(output, None)` | 临时方案(miles 侧契约对齐) |
| `all_reduce_grad_fp32` 参数偏移 | Megatron-LM-miles fork 补丁(`_CopyToModelParallelRegion` 接收 + fp32 梯度 all-reduce) | 临时方案→已可提交 |
| MoE 路由依赖 `miles.utils` | 安装完整 miles 包 | 环境依赖(非代码) |
| MoE 路由 fp8_simulate(torch_npu 缺 Float8_e4m3fn cast) | fp32 下做 fp8-grid 模拟 | 临时方案(待 torch_npu 补 cast,或接入 act_quant AscendC kernel) |

### 4.5 训练→推理参数流动验证

为验证训练得到的权重能真实驱动推理(RL 循环的核心前提),在两层减层下做共享权重实验:将 megatron 模型的注意力权重作为 SGLang 推理权重的初始值(**两侧共享同一初始权重**),megatron 以定向损失训练这些权重,再将训练后权重传入 SGLang 推理引擎,与一个等幅随机扰动对照组比较。

判别结果:

- 训练后权重使推理输出相对初始值发生改变(参数流动成立);
- 且该改变**不能被等幅随机扰动复现**——即改变是该次训练**特异性**的,而非任意扰动的结果。

> 判别器设计说明:首次以平凡损失(`output.pow(2).mean()`)训练时,得到的 delta 过小且各向同性,推理输出的改变与随机扰动不可区分(判别为否)——此为真实的不通过,如实记录。改用定向损失后,delta 具备足够幅度与特定方向,判别通过。

边界:此处训练目标为定向 MSE-to-target(对真实 RL 奖励的受控替身),层数为 2(减层)。但它是**在共享权重上经真实梯度训练得到的 delta**,且推理对该训练特异性响应。替换为真实 miles RL 奖励目标即可推进到真实 RL;参数流动机制已验证。

### 4.6 显存边界(单张 61GB 芯片,实测)

| 规模 | 结果 |
|---|---|
| 1 层(4.42B)+ AdamW | 前向+反向+优化器全通过(完整训练迭代) |
| 2 层(8.84B) | 前向+反向通过;叠加 AdamW 优化器状态后显存溢出 |
| 4 层(17.68B) | 前向通过;反向显存溢出 |

更深层数需张量/流水并行或激活重计算。256 专家 MoE 在 4096 隐藏维下每层占用巨大,这是大模型常规的显存工程,非昇腾或算子层面的问题。

**关于"显存溢出"的两点说明**:
- **共享 host courtesy(有意的限制)**:A3 是多人共享主机,本工作**主动**遵循 chip-economy 纪律——只占用验证目标所需的最小芯片数、`mem_fraction_static` 设上限,**以免影响同机其他训练任务**。所以"减层到 1–2 层"既是单卡显存的客观上限,也是不挤占同机资源的主动选择。
- **OOM 的性质(以实测为准)**:本工作捕获到的是 PyTorch caching allocator 在 `opt.step()` 主动 raise 的 `RuntimeError: NPU out of memory`(进程内分配失败),**不是被外部 watchdog 强杀**。即它是一个可捕获的 runtime 错误、不会留下半死进程。

---

## 五、验证边界与限制

为避免误读,明确以下尚**未**达成的项及原因:

1. **非生产可用**:推理链路有 8 个空实现占位在执行路径上;训练链路止于 2 层前向+反向 / 1 层完整迭代(均为减层)。
2. **未验证数值正确性**:推理演示输出为形状正确的乱码(减层 + 随机权重 + 占位实现)。
3. **非全模型**:减层(1–2 层,对照全模型 43 层)。
4. **非真实 RL 训练端到端**:4.5 的训练目标为受控替身损失,非 miles 真实 RL 奖励;参数流动机制已验证,真实奖励训练为后续工作。
5. **两个 AscendC kernel 尚未接入训练链路**:sinkhorn / act_quant 的 AscendC kernel 已产出并验证精度与可调用性,但运行层当前使用 torch 组合,正式接入待 `torch_npu` 自定义算子注册。
6. **FP8 在 A3 上不可达(硬件级)**:A3(V220)无 fp8 单元;miles DSv4 的 fp8-rollout / fp8-QAT / native-fp8 推理三条 fp8 路径在 A3 上全部 fall back 到 bf16(详见 §三.3 fp8 修正 + 第八节训练精度)。真 fp8(性能收益)需 A5/GPU。本 session 已把开源软件栈(tilelang dtype + MLIR type)这层打通并验证墙在硬件层,非软件可补。
7. **CANN 9.1.0-beta.1 对 A3 非无歧义升级**:复用基座(8.5 ML 镜像 + 挂 9.1.0 toolkit)已验证 torch_npu ABI 兼容,但 9.1.0-beta.1 bishengir 在 gemm 上回归(SIGSEGV)且不修 tilelang in-process PassManager segfault(#100)→ A3 gemm 工作暂留 8.5;9.1.0 价值在 A5/fp8(详见 §三.3 末 9.1.0 验证表)。

---

## 六、上游提交计划

### 6.1 已开 / 已沉淀的上游 PR / Issue(含链接)

| # | 目标仓 | 类型 | 状态 | 链接 |
|---|---|---|---|---|
| 1 | `tile-ai/tilelang-mlir-ascend` | PR — `CheckUBBudget` 早失败诊断 pass + UT | reviewer feedback addressed,CI 待重跑,REVIEW_REQUIRED | https://github.com/tile-ai/tilelang-mlir-ascend/pull/80 |
| 2 | `Ascend/AscendNPU-IR` | Issue — R-KA-16 罪魁定位 + 311-pass bisect 报告 + 3 patch 方向 | open;Huawei 编译器组已加 `triage-review` label | https://gitcode.com/Ascend/AscendNPU-IR/issues/251 |
| 3 | `radixark/miles` | PR — `_npu/` 子包(4 NPU 算子 + dispatcher + head-split + UB cap + R-KA-16 mitigation) | reviewer feedback addressed,REVIEW_REQUIRED | https://github.com/radixark/miles/pull/1246 |
| 4 | `Ascend/MindSpeed` | PR — `apex.transformer.functional.fused_apply_rotary_pos_emb_thd` shim | ready(无 human reviewer 反馈) | https://gitcode.com/Ascend/MindSpeed/merge_requests/3509 |
| 5 | `triton-lang/triton-ascend` | Issue — triton vs triton-ascend coexistence | closed not-planned + KB `triton-ascend-002` | https://github.com/triton-lang/triton-ascend/issues/306 |
| 6 | `sgl-project/sgl-kernel-npu` | PR — `fused_split_qk_norm` RMSNorm `.bias` getattr fix | OPEN, REVIEW_REQUIRED | https://github.com/sgl-project/sgl-kernel-npu/pull/531 |
| 7 | `sgl-project/sglang` | Issue — `/update_weights_from_disk` FusedMoE `_load_w13` narrow regression(#26794) | OPEN;等 maintainer 回复 | https://github.com/sgl-project/sglang/issues/26794 |
| 8 | `radixark/miles`(本次新增,训练侧 V4 ops NPU path) | 分支 `zhshgmail/miles npu-tilelang-ops` `d03db2c` | 审计通过,待 `gh pr create` | (待开 PR 后补 URL) |
| 9 | `radixark/Megatron-LM`(miles vendored) | 分支 `Megatron-LM-miles fix/te_general_gemm_npu_fallback` `6f3209b` | 冷导入验证,随 #8 miles PR 一并提交 | (随 #8) |
| 10 | `a5_ops`(工具链,非 NPU 上游) | perf-capture canonical N/A + FA-gate 修复 `eefeaeca`(task#33) | **已合入(MERGED)** | (内部仓) |
| 11 | `tile-ai/tilelang-mlir-ascend` | PR — `sparse_mla_fwd` heads<block_H 静默写回越界修复(本 session 发现+修+验) | **OPEN(新开)** | https://github.com/tile-ai/tilelang-mlir-ascend/pull/96 |
| 12 | `tile-ai/tilelang-mlir-ascend` | Issue — MLIR PassManager(`Pipeline.run`)在合法 NPU vector kernel(softmax+sinkhorn 形)上 segfault;近乎相同的 kernel 能编;含最小复现 crash/OK 配对 + workaround(2026-06-03 tilelang-port PoC) | **OPEN(新开)** | https://github.com/tile-ai/tilelang-mlir-ascend/issues/100 |

> **唯一已合入(MERGED)的是条目 10**(a5_ops 工具链内部修复)。其余上游条目状态:1/3/6 OPEN 等 human maintainer;2/4/7 OPEN;5 CLOSED(not-planned,已 reframe 成 KB);8/9 尚未开 PR。
> **注意 #251 不是"已合入"**:它是 OPEN issue,只是我把 311-pass bisect **报告作为评论 posted 上去了**(评论 landed ≠ issue resolved ≠ fix merged);真修由华为编译器组负责,尚未开始。
> 条目 1–7 的实时 reviewer 反馈细节以 `output/miles-dsv4-flash-poc/docs/REPORT.md` §上游 PR 列表为权威;8–9 为本次 V4 训练侧新增的已准备分支。

### 6.2 本报告新识别的候选(待真实端到端后批量提交)

推理侧(`sgl-project/sglang` + `sgl-kernel-npu`):四个 production 适配项可直接提交(RMSNorm 原生、`**kwargs` 兼容、swiglu 原生、gemm cast);一个汇总 issue(缺失的 `AscendAttnBackend` V4 接口,逐项附行号与证据);NPU complex64 aclnnIndex issue;FP8 打包 KV cache scatter。

训练侧(`radixark/miles` + `Ascend/MindSpeed` + `Ascend/pytorch`):sparse-softmax 稳定化补丁(miles);MindSpeed rms_norm 签名适配 + V4 层契约;torch_npu Float8_e4m3fn cast 缺失。

工具链 / 编译器侧(本 session 2026-06-04 新识别,待真实端到端后批量提交):
- **`Ascend/AscendNPU-IR` / bishengir** — CANN 9.1.0-beta.1 bishengir 在 gemm npuir 上 SIGSEGV(8.5.0 能编),apples-to-apples 已复现;filing 前需在 9.1.0-built tilelang 上重发 npuir 排除 dialect 版本差(详见 §三.3 末 9.1.0 验证表)。
- **`radixark/miles`** — fp8-QAT(`fp8_simulate`)实现把 fake-quant 耦合到真 fp8 tensor(经 tilelang `act_quant` 物化 fp8),导致 QAT-on 在 A3 无谓地撞 fp8 硬件墙;正确 fake-quant 应是纯 16-bit round-to-fp8-grid 算术(不物化 fp8 tensor)→ 修后 QAT 在 A3 也能跑(详见第八节 + memory `miles_qat_fakequant_couples_real_fp8`)。
- **`tile-ai/tilelang-mlir-ascend`** — fork TVM datatype 注册缺 FP8/FP4(本 session 已在 workspace build 本地补 fp8 映射+解析并验证,可整理成 PR);但注意 A3 硬件不支持 fp8,该 PR 的价值在 A5 + 解锁编译期 dtype 流通,不解 A3 运行。

---

## 七、关键文件索引

| 内容 | 路径 |
|---|---|
| 推理 PoC + 适配快照 | `workspace/v4_attempt_2026_06_01/README.md`、`_*_PASS.py`、`native_op_snapshots/` |
| 训练侧算子清单 | `workspace/v4_attempt_2026_06_01/V4_TRAINING_SIDE_OP_GAP.md` |
| 训练侧 NPU 集成产物 | `workspace/v4_attempt_2026_06_01/npu_native_shims/` |
| 参数流动验证 | `workspace/task-dag-realdelta/`(`shared_weights_*`、`shared_verify2_RESULT.txt`、`RESULTS.md`) |
| AscendC 算子调用验证 | `workspace/task-dag-realdelta/call_opgen_act_quant.py` |
| 真实配置 | `workspace/T32_tilelang_rescue/v4_real_truth/v4_real_config.json` |
| 相关经验沉淀 | `docs/_meta/kb/porting_lessons/`(`sglang-004/005`、`miles-002/003`、`cross-layer-012/013`) |
| tilelang-port PoC + perf log | `workspace/task-dag-realdelta/`(`TILELANG_PORT_POC_2026-06-03.md`、`V4_INFERENCE_TILELANG_KERNEL_SWEEP.md`、`TILELANG_VS_CANN_MASTER_TABLE.md`、`CODEGEN_FIX_DEBUG_2026-06-03.md`)、`_v4_runlogs/` |
| tilelang-on-Ascend KB | `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md`(§13 cold-drive 2026-06-03) |
| **fp8 补丁可复现 Dockerfile**(2026-06-04) | `workspace/_fp8_dockerfile_2026_06_05/`:`Dockerfile.fp8patch`(从 `quay.io/ascend/cann:8.5.0-a3` 起、clone fork pinned commit、`git apply` 两个 fp8 补丁、跑 fork 既有 build)+ `data_type.h.fp8.diff` + `codegen_npuir_api.cc.fp8.diff`(精确补丁,api.cc 仅 fp8 hunk)+ `verify_fp8_patch.py`(CPU-only 自检)。**补丁已对 pinned base commit(tilelang `a19acd5` / tvm `c2921fda`)`git apply --check` 验证 APPLIES CLEAN**(2026-06-04)。pinned commit + build 命令见 Dockerfile 头部注释 |

---

## 八、miles DSv4 训练精度与 fp8 范式(2026-06-04 厘清)

本节回答"miles 这个 fp8 recipe 在 A3 上到底以什么精度跑、为什么 CANN 替换就没有 fp8 问题"。结论经读源码 + A3 实测 + 文献核实(非推测),贯通整条精度链。

**一句话:训练真高精(bf16+fp32 累加),量化只在 rollout/推理侧(真低精 fp8),训练侧的 fp8 只是 env-gated QAT 假量化且默认关——所以 A3 上(无 fp8 硬件)整条链 fall back 到 bf16 仍能跑训练。**

| 维度 | 事实 | 证据 |
|---|---|---|
| DSv4 原生权重 | **FP8**(预训练即 FP8 混精,无官方 bf16 release) | DeepSeek-V3/V4 repo;提供 `inference/fp8_cast_bf16.py` 反量化 |
| bf16 权重来源 | `fp8_cast_bf16.py` dequant(128×128 FP8 block→BF16) | miles `deepseek_v4.py:142` 硬断言 `weight.dtype==bfloat16` → 加载的是 dequant 副本 |
| 训练参数更新(miles RL) | **bf16 + fp32 累加**(高精) | 所有训练 kernel(sparse_mla fwd/bwd、indexer fwd/bwd)均 bfloat16;sparse_mla_bwd 硬断言 bf16 |
| 训练侧量化 | **仅 QAT,无 PTQ**;env-gated 默认关 | `MEGATRON_USE_KV_QAT` 默认 OFF(`deepseek_v4.py:242`);唯一 ModelOpt PTQ import 在 `glm5.py` 不在 dsv4 |
| rollout/推理精度 | 设计上**真低精 fp8** = FP8-rollout-RL 范式 | FP8-RL(arxiv 2601.18150)/ Jet-RL(2601.14243):bf16-train + fp8-rollout per-step 量化进推理引擎 |
| A3(无 fp8 硬件) | 整条链只能 bf16(dequant 权重 + bf16 训练 + bf16 rollout);真 fp8 需 A5/GPU | 见 §三.3 fp8 修正 + §五.6 |

**为什么"CANN 替换就没有 fp8 问题":**
- A3 默认路径(QAT off)= 纯 bf16,CANN native 算子(`npu_nsa_select_attention` 等)本就是 bf16/fp16 实现,根本不碰 fp8 → 自然无 fp8 问题。
- fp8 问题只在两种情况浮现:(a) 显式 `MEGATRON_USE_KV_QAT=1` 开 QAT → `fp8_simulate` 经 tilelang `act_quant` 物化真 fp8 tensor → 撞 A3 硬件墙;(b) 走 tilelang-port 推理路想编 fp8 kernel(act_quant/fp8_gemm)→ 同墙。CANN-native 路两者都不触发。

**miles fp8-QAT 实现缺陷(owner 洞察,已验证,可提 upstream 改进):** `fp8_simulate` 不是纯 16-bit 假量化——它调真 tilelang `act_quant`(`T.Cast(bf16→float8_e4m3fn)`)物化真 fp8 tensor 再 dequant 回 bf16,**把假量化无谓耦合到真 fp8 硬件**。原理上 fake-quant 只需 16-bit 算术(在 bf16 里算 round-to-fp8-grid 值,永不物化 fp8 tensor)。正确修法 = 纯 16-bit round-to-fp8-grid → QAT 在 A3 也能跑(filing target `radixark/miles`,见 §六.2)。

**术语澄清(owner 多轮追问后统一):**
- **QAT** = 训练时前向注入量化*误差*(fake-quant)+ 反向 STE,目的让模型学会容忍最终低精推理;计算仍高精。miles 的 fp8-QAT 是这个(只是实现错误地物化了真 fp8)。
- **FP8-rollout-RL**(推理真低精 + 训练参数更新仍高精 + 无 STE)是一个**独立**范式,不是 QAT;rollout 侧把当前 policy 权重 per-step 量化进推理引擎(PTQ-style 权重量化),训练参数更新本身不加噪、不 STE。owner 直觉"套用 PTQ 的名字"正确——rollout 量化确是 PTQ 式(对"这一步训练后"的权重做推理前校准)。
- `fp8_gemm`/`fp4_gemm` GEMM kernel **不在 miles 训练插件里**(已 grep 确认:miles `_miles_dsv4_preserved/.../deepseek_v4/` 里无 `fp8_gemm`/`fp4_gemm` 定义,也无 `fp4` 字样)。它们定义在 **`tilelang-mlir-ascend/examples/deepseek_v4/inference/kernel.py`**(推理示例,非训练插件)。在那个**推理**示例里:`fp8_gemm` **被调用**(`inference/model.py:121`,`Dispatches to fp4_gemm / fp8_gemm / F.linear based on weight dtype`),`fp4_gemm` **被注释掉未调用**(`model.py:118`)。→ 修正早先"训练插件里 dead code"的说法:准确说法是"GEMM kernel 属推理示例(fp8_gemm 在那被调,fp4_gemm 注释掉),不属训练插件"。**训练插件真正用到的 fp8 算子是 `act_quant`**(`qat.py:6` `act_quant(x, block_size, "ue8m0")`,经 `fp8_simulate`/`fp8_simulate_qat` 在 QAT 路调用)——这条 confirmed。

详尽链路与文献见 memory:`fp8_is_a3_hardware_limit_not_software`、`fp8_rl_bf16train_fp8rollout_paradigm`、`miles_qat_fakequant_couples_real_fp8`、`dsv4_native_fp8_dequant_bf16_for_a3`。

---

## 九、基于最新版 miles 的重新基线 + CANN-native 运行层策略(2026-06-05)

> 本节回应 owner「基于最新版 miles 重新整理用例 + upstream」。M1/M2 已完成(均独立 agent 验证),M3(CANN-native dispatcher PoC,需 A3 e2e)待 A3 排期。全程证据见 `workspace/task-dag-realdelta/`(`REBASE_ON_LATEST_MILES_PLAN_2026-06-05.md`、`M1_LATEST_MILES_USECASE_UPSTREAM_MAP_2026-06-05.md`、`MILES_REBASE_ASSESSMENT_2026-06-05.md`)。

### 9.1 版本现状(M1,本地实查)

- 最新上游 = `radixark/miles` main `74198b45`(2026-06-04,活跃)。我们 fork `npu-tilelang-ops` 落后 **115 commit**。
- 我们分支的 NPU 工作在 `glm5/ops/_npu/`(sparse_mla/lighting_indexer fwd+bwd,is_npu idiom),**不含 DSv4 plugin**——DSv4 plugin(19 文件 2393 行)是 fork 点之后才进 main 的,且**纯 GPU/CUDA,零 NPU 适配**(grep `npu|ascend|is_npu` = 0)。
- 含义:**"用最新 miles" ≠ "NPU 没问题了"**。最新 miles 把 DSv4 做得更全(sparse_mqa/indexer 全 fwd+bwd + TileKernels qat + bf16-fp32 精度对齐 linear),但它本身没有 NPU 路径;NPU 适配工作仍要做,只是基线更新更大。

### 9.2 关键策略转向:DSv4 NPU 运行层走 CANN-native(非 tilelang re-port)

M1 实测(独立 agent 验证零 REFUTED):**torch_npu(2.9.0)native 覆盖最新 DSv4 全部核心 attention/indexer/norm 算子,fwd+bwd 都有**:

| 最新 DSv4 算子 | CANN-native fwd | CANN-native bwd |
|---|---|---|
| sparse MQA (`sparse_attn_tilelang`) | `npu_nsa_select_attention` | `npu_nsa_select_attention_grad` |
| compress (`DeepSeekV4Compressor`) | `npu_nsa_compress_attention` | `npu_nsa_compress_grad` |
| lightning indexer (`V4Indexer`) | `npu_lightning_indexer` | `npu_lightning_indexer_grad` / `npu_sparse_lightning_indexer_grad_kl_loss` |
| rms_norm | `npu_rms_norm` | `npu_rms_norm_backward` |
| MLA prolog | `npu_mla_prolog_v3` | functional 变体 |
| `linear_bf16_fp32` | 纯 torch(A3 直接可用) | ✅ |
| fp8 `act_quant` | ❌ 无 native(A3 fp8 硬件墙,QAT-off 不调) | — |

→ **DSv4 NPU 运行层 = miles dispatcher 把 6 个 tilelang 调用接 torch_npu native**,绕开整个 tilelang re-port + #100/codegen + 老 fork 折腾。**tilelang-port 降级为 CANN 无 native 覆盖时的退路(cookbook miles-001 已相应降级)**。诚实限定:符号存在 ≠ A3 执行正确性,后者由 M3 的 A3 e2e 坐实。

### 9.3 cookbook / 账本重整(M2,独立 agent 验证)

- `miles-002`(CANN-native-first):加 2026-06-05 re-baseline 块,确认最新 main 同映射 + 补全 bwd op 名。
- `miles-001`(tilelang-port):**降级为 FALLBACK**,修正 stale 的"唯一路径/alternatives 不覆盖"框架。
- `miles-003`(megatron layer):修正两处对最新 main 已 stale 的结构断言(`forward` 返回裸 tensor 非 `(output,None)`;`sparse_attn_torch` 已被 `sparse_attn_tilelang` 取代)——标历史,通用教训保留。
- `sglang-004/005`、`cross-layer-012/013`:独立 agent 确认对最新 miles **仍适用**。
- `UPSTREAM_FORKS.md`:新增 miles 行(115-commit gap + CANN-native 决策)。

### 9.4 M3 — CANN-native 算子 A3 真机验证(2026-06-05,reduced+watchdog,PASS)

owner 授权用共享 A3(减层 + HBM watchdog)尽快验。复用现有 `miles_v4_npu` 容器(持 davinci0、torch_npu 2.9.0,避开新建容器 UDA ns-锁)。HBM 全程 3124MB 基线、watchdog 未触发(不扰他人 job)。**三个核心 DSv4 算子 CANN-native 在 A3 真机跑通:**

| 算子 | fwd | bwd | 证据 |
|---|---|---|---|
| `npu_nsa_select_attention`(sparse-MLA) | ✅ finite(attn (64,4,128)) | ✅ finite(dq/dk/dv) | TND layout;返回 softmax max/sum 作 bwd state |
| `npu_nsa_compress_attention`(compress) | ✅ finite(4 输出) | 待(op 名 = `npu_nsa_compress_grad`,非 `_attention_grad`) | TND |
| `npu_lightning_indexer`(indexer) | ✅ 正确(indices + 未掩码分数有效;-inf 为 mode-3 causal 掩码、非 bug) | 待(`npu_lightning_indexer_grad`) | BSND |

**三个硬发现(真机实测,供 dispatcher / 后续 PR):**
1. 精确 op schema 已全部抓取(sparse_mla/compress/indexer 的 fwd+grad)——见 `RESULT_M3_nsa_select_attention_e2e_2026-06-05.md`。
2. **`attn_sink` 风险确认为真**:native `npu_nsa_select_attention` 签名**无 attn_sink 入参**(只有 atten_mask),而最新 main `sparse_mqa_fwd_interface` 要传 `attn_sink[H]`。→ M3 真正工程点 = attn_sink 适配层(把 sink 并进返回的 softmax max/sum 重算分母)。
3. 两个坑:grad arg-order(错序报 561103 "Cannot find bin" 假象);indexer mode-3 causal `-inf` 是正确掩码(`isfinite().all()` 是错判据)。

**M3 状态**:**核心可行性已真机坐实**(DSv4 NPU 走 CANN-native 可行,sparse-MLA 训练路 fwd+bwd 全通)。**未达 PR-bar**:compress/indexer bwd、attn_sink 适配、vs-参考数值对、dispatcher 接线 + 高覆盖 UT + e2e 报告——这些是产品化(多小时),待 owner 定节奏 / 内网 agent 接(handoff doc `M3_HANDOFF_FOR_INTRANET_AGENT.md`)。

### 9.5 待办(M5)

- **M5(需 owner 二次确认)**:upstream 贡献整理——DSv4 NPU dispatcher 路是该提给 `radixark/miles` 的(upstream DSv4 = GPU-only,缺 NPU 路)。按 `feedback_pr_quality_bar` 充分验证后再提,不自动 PR。

---

## 附录 A — 范围说明与方法学备注

供需要追溯口径的读者参考;主体结论不依赖本附录。

### A.1 V4-Flash vs V4-Pro

本报告所有结果基于 **V4-Flash**。V4-Pro 是同架构(`DeepseekV4ForCausalLM`)的更大独立模型,尚未做对应移植。两者真实 config 对照:

| 维度 | V4-Flash(本报告) | V4-Pro |
|---|---|---|
| hidden_size | 4096 | 7168 |
| num_hidden_layers | 43 | 61 |
| num_attention_heads | 64 | 128 |
| q_lora_rank | 1024 | 1536 |
| n_routed_experts | 256 | 384 |
| moe_intermediate_size | 2048 | 3072 |
| index_topk | 512 | 1024 |
| o_groups | 8 | 16 |
| compress_ratios 长度 | 44 | 62 |

同架构 → 本报告的减层移植流程对 Pro 结构上适用;但 Pro 每层 footprint ≈ Flash 2–3×,单卡显存约束更紧,需更省显存的配置或并行。Pro config 已取得(`workspace/v4_pro_attempt/v4pro_real_config.json`),减层移植待排期。

### A.2 性能测量方法学

- 全部相对性能为**对称 same-wrapper**测量:参考侧与候选侧走同一计时 wrapper、同 warmup/repeats、同 `torch.npu.synchronize`;ratio = pytorch / AscendC。
- sinkhorn 5.34× mean 为此对称口径(早期一次非对称测量得到的偏大数值不作数,已以对称值为准)。
- act_quant 为 shape 相关分布(min 0.38× / max 8.73×),读 min 不读 mean(详见 §4.3)。

### A.3 tilelang-ascend vs CANN vs pytorch 对比(sparse-MLA,2026-06-02,部分结果)

回答"修复后的 tilelang-ascend / AscendNPU-IR 上,tilelang kernel 精度能否对齐、性能如何"。**当前为部分结果**:

| 实现 / config | 精度(vs pytorch 参考) | 性能 | 备注 |
|---|---|---|---|
| tilelang-ascend `sparse_mla_fwd`,**heads=32**(默认) | **PASS**(rtol=5e-3 atol=1e-2) | 297.9 us/call | B=1/seq=128/dim=512+tail64/top_k=64;**此 config 干净** |
| tilelang-ascend `sparse_mla_fwd`,**heads=4** | **FAIL**(44.3% 元素不对齐) | 285 us/call(仍能跑) | **head-count 相关精度失配**;隔离确认是 heads 触发(seq_kv=1024+heads=32 仍 PASS) |
| CANN `npu_nsa_select_attention`(契约正确 harness) | finite ✓(数值未与 tilelang 同输入对齐——两者 sparsity 粒度不同,见下) | **92.0 us/call** | Tkv=1024(≥sbs×sbc)、合法 block-topk、heads=4/Dqk=192;之前 NaN 是 harness 输入欠约束,非 CANN bug |
| tilelang-ascend `fp8_lighting_indexer`,**h=4/8/16/32** | **PASS(全部)** | ~9400 us/call @h32 | 见下"indexer 更正":之前报的"h<32 FAIL"是**我的测试错误**(example `__main__` 硬编码 H=32 忽略 `--h`),harness 修正后各 h 全 PASS,**无 indexer kernel bug** |

**结论**:
- tilelang-ascend sparse_mla 在**默认 heads=32 干净**(精度 PASS、297.9us)。
- **查实一个真 bug(已定性,不是设计边界)**:tilelang `sparse_mla_fwd` 在 **heads<16 静默输出错误**——heads=4 错 44.3%、heads=8 错 35.9%、heads=16/32 PASS(干净阈值)。根因:`padded_H=max(next_power_of_2(head_kv),16)` 在 head_kv<16 时 padding 到 16,代码走 `padded_H!=head_kv` 分支、assert 过、注释称"H-padding 自动处理",但**padded head 的 Q-copy/output-copy mask 实际算错,不报错只给错结果**。silent wrong output 比 assert 失败更危险。已起草 issue(`UPSTREAM_ISSUE_tilelang_sparse_mla_heads_lt16.md`,待提 tile-ai/tilelang-mlir-ascend)+ KB `tilelang-003`。**对 V4 相关**:V4-Flash 64 头(安全),小 head / 减层 / kv_group-split 配置会中招。
- **tilelang-vs-CANN 性能(同 heads=32,两边都合法)**:tilelang sparse_mla **297.9us** vs CANN nsa_select **148.7us** → **CANN 约快 2×**。**仍有 caveat**:两者 sparsity 粒度不同(tilelang token-level top_k=64 over seq=128;CANN block-level sbs=64×sbc=16 over Tkv=1024),KV 长度也不同,所以是"同 head 数、两边各自合法配置下"的对比,**不是 bit-identical workload**。但在各自正确的配置下,CANN 这一路更快。这给"运行层选 CANN"补上了**性能依据**(此前是"更直接"的定性):CANN 既有 native 覆盖、又(在此对比下)更快。
- **indexer 更正(撤回之前的"bug")**:之前报 `fp8_lighting_indexer` "h<32 FAIL" 是**我的测试错误,不是 tilelang bug**。根因:example `__main__` **硬编码 `H=32` 忽略 `--h`** —— `--h 4` 编译了 h=4 的 kernel 但测试数据/参考仍按 H=32 构造,比较的是 heads 不匹配的两个东西(24.7%/99% shape-dependent 就是 tell)。把 harness 改成 honor `--h` 后,indexer **h=4/8/16/32 全 PASS(exit 0)**,**无 indexer kernel bug**。我没在 sweep 前核对 example 是否把 `--h` 同时传给 kernel 和数据。issue 草稿已撤回。
- **回答"要不要修 bug"(更正后)**:**只有 1 条真 kernel bug,已修**——sparse_mla heads<block_H 静默写回越界(fix `a19acd5` 验过 4/8/16/32 PASS,无回归)。indexer 那条是我测错、已撤回。**对 V4**:V4-Flash 64 头本就安全;sparse_mla 修好后小 head 也安全;indexer 本就没问题。
- **方法学教训**:sweep 一个 `--param` 验 kernel 前,先确认 example 的 `__main__` 把该 param **同时**传给 kernel 和测试数据/参考。sparse_mla_fwd 是(其 fix 真)、fp8_lighting_indexer 否(导致假 bug)。

### A.4 sparse_mla_fwd 健壮性 sweep(dtype/tile/shape 组合,2026-06-02)

在修好 heads<16 bug 后,继续 sweep 其他参数组合(sparse_mla_fwd 的 `__main__` 数据全部由 args 构造,sweep 有效)。结果:

| 组合 | 结果 |
|---|---|
| `--top_k 128` | ✅ PASS |
| `--block_i 64 --block_k 64`(tile 加倍) | ✅ PASS |
| `--seq_len_kv 256 --top_k 128` | ✅ PASS |
| `--dim 256 --tail_dim 64` | ✅ PASS |
| `--kv_group 2 --heads 32` | ⚠️ **FAIL**(assert_close 不通过) |

- **tile/shape 健壮**:top_k / block_i / block_k / seq_len_kv / dim 的多种组合都 PASS,修过的 heads 维也 PASS。
- **kv_group>1 待定**:`kv_group=2` 精度 FAIL。**但不轻易定性为 bug** —— 该 example 的注释(line 77)表明 H-padding 的自动处理**只为 kv_group==1 设计**("when kv_group == 1, use ... would be handled automatically"),所以 kv_group>1 很可能是**未验证/未支持的配置**而非 kernel bug。需进一步分析(或上游确认 kv_group 支持范围)才能定性。记为**已知限制:kv_group>1 当前输出错误且不报错**,消费侧 kv_group>1 别用此 example,或加 assert 守卫。
- **dtype**:该 example dtype 硬编码 `float16`(非 CLI 可调),bf16 路径**未测**(需改源码)。如需 bf16 robustness,另起。

**系统性 sweep(2026-06-02,case-gen 增强)**:用 `tilelang_sweep_harness.py` 跑了 10 个组合(完整矩阵见 `workspace/task-dag-realdelta/sparse_mla_sweep_results.md`)。结果:**9/10 PASS,含 V4-Flash 真实 64 头**;唯一 ERR(`--seq_len 256`)是**无效配置**(seq_len 256 > seq_len_kv 128 默认,因果注意力下无意义)——`--seq_len 256 --seq_len_kv 256` 匹配后 PASS,确认非 bug。所以 sparse_mla_fwd 在**有效参数网格上 robust**(heads/top_k/block/seq/dim/batch/num_kernels);整个 tilelang 调查里唯一真 kernel bug 是 heads<16(已修 a19acd5)。
