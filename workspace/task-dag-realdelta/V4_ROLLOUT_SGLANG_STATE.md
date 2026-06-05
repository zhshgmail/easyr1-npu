# V4 rollout 侧(sglang-on-NPU)bring-up 状态

**日期**: 2026-06-02。完整 RL loop 的 rollout 侧(`_e2e_rl_5step_dsv4_sglang.py` 的 sglang HTTP /generate)。容器 `sgl_probe`(sglang 0.5.12 + torch 2.8 + 4 NPU 设备;`DeepseekV4ForCausalLM` 已注册进 ModelRegistry)。

## 进展(连过 3 关,第 4 关是硬边界)

1. ✅ **server 起来 + 健康**:`/host-models/dsv4_REAL_1layer_fab`,ascend attention backend,bf16,load weight 2.51GB OK,`/get_model_info` 返回正常。launch 脚本 `start_sglang_v4.sh`(server_args 镜像 2026-06-01 工作过的那次)。
2. ✅ **FP8 quant 关**:`AssertionError: FP8 quant_config must create weight_scale_inv`(`deepseek_v4.py:368` MQALayer)。根因 env `SGLANG_OPT_FP8_WO_A_GEMM` 默认 True 但 fab 模型无 FP8 scales。修:`export SGLANG_OPT_FP8_WO_A_GEMM=0`。
3. ✅ **SWA-budget admission 关**(KB `sglang-005`):`/generate` 卡住,日志 `AddReqResult.NO_TOKEN` 无限循环(`swa_size=256` 太小 admit 不进)。修:`max_total_tokens=65536 + swa_full_tokens_ratio=0.5 + mem_fraction_static=0.30` → pool `swa_size=32768`(128×),admission 不再 starve。**KB 自救了自己。**
4. 🔴 **tilelang↔torch_npu ACL precision-mode 冲突(硬边界)**:V4 forward 运行时要 import tilelang(sparse-MLA kernel)。但:
   - **tilelang 不在 path** → load + serve OK,forward 崩 `ModuleNotFoundError: tilelang`
   - **tilelang 在 path** → **weight-load 阶段**就崩 `RuntimeError: SetPrecisionMode ... AclSetCompileopt(ACL_PRECISION_MODE) error 500001`(`torch_npu LazyInitAclops.cpp:175`,在 `process_weights_after_loading → npu_format_cast → torch.ops.npu.npu_format_cast` lazy-init aclops 时)
   - 即使**不 eager-import**(只放 sys.path),sglang import 模型模块时仍会 transitively import tilelang,其 import side-effect 扰动 ACL precision-mode 状态 → npu_format_cast 的 ACL lazy-init 被拒。

## 诊断(discriminator)

catch-22 / 真实跨栈 ACL 冲突:
- tilelang 的 import 初始化它自己的 ACL/compile 状态(precision-mode / compile-opt),与 sglang 的 torch_npu ACL context 冲突。
- 这不是 config 能 paper over 的;需要要么 (a) 让 tilelang import 不动 ACL precision-mode,要么 (b) 把 tilelang 的 ACL init 推迟/隔离到 sglang 的 torch_npu ACL 已建立之后。研究级集成问题。

## 状态总览(整个 V4-on-NPU RL loop)

- ✅ **训练侧(actor)**:megatron-layer reduced-2层 fwd+bwd 训练步,真 spec builder + 真 miles 包,0 NaN,grad_norm=4.29e-2。**完整验证**(见 `V4_MEGATRON_LAYER_NPU_RUNS.md`)。
- 🟡 **rollout 侧(sglang)**:server 起来 + admit 请求(过了 FP8 + SWA 两关),但 V4 forward 卡在 tilelang↔torch_npu ACL precision-mode 冲突。**未通**。
- ⏭ 接起来需要:解决 ACL 冲突 → sglang generate 出 token → wire `_e2e_rl_5step_dsv4_sglang.py` HTTP rollout + GRPO + actor train + weight-sync。

## 下一步(ACL 冲突的可能解)—— 3 个 lead 都试过,精确定位到 torch_npu LazyInitAclops

- **lead#1(flag-gate tilelang import)— 不行**:`SGLANG_OPT_USE_TILELANG_MHC_PRE/POST` 只 gate runtime 用法;但 deepseek_v4.py 顶层 `from sglang.jit_kernel.dsv4 import (...)` + `compressor`/`indexer`(L25/45/46)在模型模块加载时就 transitively import tilelang,flag 管不到。且 `mhc.py:5` / `dsa/tilelang_kernel.py:5` / `dsv4/tilelang_kernel.py:4` 都是顶层 `import tilelang`。
- **lead#3(父进程预 init torch_npu ACL)— 不行**:在 launch 父进程 `torch.npu.set_device(0)` 预热(marker `[prelaunch] torch_npu ACL context initialized` 确实打了),但 sglang scheduler 是 **multiprocessing spawn 子进程**,子进程 fresh re-import,父进程 ACL 状态不继承 → 子进程里仍 500001。
- **精确机制**:`SetPrecisionMode` 来自 torch_npu `LazyInitAclops.cpp:175` 的 `AclSetCompileopt(ACL_PRECISION_MODE)`,在首次用 custom op(`npu_format_cast`)时 lazy-init aclops。torch_npu `npu/npu_config.py:22` `_option_map["ACL_PRECISION_MODE"]=["allow_fp32_to_fp16","must_keep_origin_dtype"]`。500001 = tilelang 的 bishengir runtime(tilelang import 时 ctypes 加载)已经 init 了 ACL 并设了不兼容的 compile-opt,torch_npu 的 lazy set 被拒。

**真正的解(需 dedicated effort,非 config)**:
1. 让 tilelang import 不在 import-time init ACL/bishengir runtime(lazy 化它的 ctypes ACL init 到首次 kernel compile)—— 改 tilelang。
2. 或 patch torch_npu `LazyInitAclops` 容忍已被设过的 ACL compile-opt(不硬 set / 忽略 500001)。
3. 或 product 路径:确认 V4 rollout 必须用 tilelang 吗?sglang 有 `aiter.ops.mhc`(L941/1005)分支 —— 若 aiter 路径不依赖 tilelang,走它避开整个冲突(待查 aiter 在 sgl_probe 是否可用 + 是否真不 import tilelang)。

**lead#2(torch attention fallback)结论**:V4 模型模块顶层就 import tilelang,单纯 runtime-flag 避不开;要么走 aiter 分支(lead 上面 #3),要么 lazy 化 tilelang import。

## ⚠ import-order 假设被证伪(discriminator,2026-06-02 后续)

试了 sitecustomize 强制子进程 torch_npu 先 init —— 但 sitecustomize 在这个 python(3.11.14 自定义 build)**根本没自动触发**(PYTHONPATH 设了、文件合法,但 `sitecustomize` 不在 sys.modules)。

更重要:**两个隔离测试都把 import-order 假设证伪了**:
- 隔离1:`torch_npu set_device → npu_format_cast → import tilelang → npu_format_cast` → **4 步全过,无 500001**
- 隔离2(失败序模拟):`import tilelang FIRST → torch_npu set_device → npu_format_cast` → **3 步全过,无 500001**
→ **不管 tilelang 先 import 还是后 import,单进程隔离里都不冲突。** 所以"import 顺序"不是根因。

**真正的差异(待深挖)**:server 崩在 **scheduler spawn 子进程**的 weight-load `npu_format_cast`(真实模型权重),隔离里是单进程 + trivial `npu_format_cast(zeros(4,4))`。区别在 sglang 子进程特有的 ACL/torch_npu 状态(可能 sglang 自己 set 了某个 compile-mode / precision option,或 spawn 子进程的 ACL context 初始化方式不同)。**需要 instrument sglang scheduler 子进程本身**(不是隔离),看它在 `npu_format_cast` 前对 ACL 做了什么 set。这是 dedicated effort。

**修正记录**:我差点顺着错误的 import-order 假设做 sitecustomize 方案 —— discriminator(两个隔离都不复现)及时拦住。根因在 sglang 子进程 ACL setup,不在 import 顺序。

## 隔离3 也不复现 → 确认是 spawn-子进程特有(2026-06-02 续)

找到 sglang 子进程特有的一个 ACL set:`hardware_backend/npu/utils.py:109` `torch_npu.npu.set_compile_mode(jit_compile=False)`。隔离3 精确复现这个序:`set_device → set_compile_mode(jit_compile=False) → import tilelang → npu_format_cast` → **4 步全过,仍无 500001**。

→ 已排除的 3 个假设(import-order×2 + set_compile_mode):**单进程隔离一律不复现**。剩下唯一变量 = **sglang multiprocessing spawn 子进程的 ACL context 初始化方式本身**(spawn 子进程怎么 init ACL,和单进程 fork/直接 init 不同)。单进程隔离根本无法复现。

**结论**:这不是能靠 isolation 试出来的,必须 **instrument sglang scheduler 子进程本身** —— 在 `hardware_backend/npu/utils.py:179` `npu_format_cast` 前加 print dump ACL 状态 + try/except 抓 500001 的完整 context。是个 dedicated 的 in-subprocess 调试 effort(要改 sglang 装好的源码,invasive)。

**product 路径备选(避免死磕子进程 ACL)**:5-step RL PoC 的 rollout 不一定非要 sglang HTTP server。可用更简单的 rollout(megatron 模型自己 generate,或 stub rollout)证明 train+weight-update 循环闭合,把 sglang-NPU-decode 当作单独跟踪的 inference/perf 问题。**但要诚实标注**(不能假装 sglang rollout 通了)—— 避免之前 owner 抓过的 synth/fake 陷阱。这条要 owner 定夺(是否接受非-sglang rollout 的 PoC 闭环)。

## ✅ ACL 死结解了 + forward 真跑起来(2026-06-02 续 2)—— fallback 方案

**关键突破**:不去死磕"为什么 spawn 子进程 ACL 冲突"(单进程无法复现),而是直接 **instrument `hardware_backend/npu/utils.py:179` `npu_format_cast`,try/except 捕获 500001 时跳过 format-cast**(它是 layout 优化、非正确性必需,跳过 = 用默认 ND layout)。fallback 打了 2 次(4096×4096 / 2048×4096 bf16 权重)→ **weight load 过、server READY、`[TRACE] V4 forward layer=0` 真正执行**。patch 备份 `utils.py.bak_fmtcast`。

**然后 forward 里连拆的 sglang-V4-vs-部署栈 version gap(都是真 fix):**
1. ✅ `PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL` + `TL_DISABLE_FAST_MATH` 缺失(mhc.py:122)→ 加进 tilelang `pass_config.py` enum(CUDA-centric,no-op)。
2. ✅ `tilelang.JITKernel` 未顶层导出(mhc.py 用作 return 注解)→ tilelang `__init__.py` 加 `from tilelang.jit.kernel import JITKernel`。
3. ✅ `deep_gemm` 没装(mhc_pre:693,`tf32_hc_prenorm_gemm`)→ env `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` 切到 tilelang splitk 路径。
4. 🔴 **当前**:tilelang `mhc_pre_gemm_sqrsum_splitk_kernel` 编译报 `TVMError: script.ir_builder.tir.Buffer ... Expected Array[PrimExpr], but got Array[index 0: Array]` —— sglang V4 的 mhc tilelang kernel DSL 写法(nested Array 传给 Buffer)和这个 TVM 版本不兼容。

**当前边界 = tilelang kernel ↔ TVM 版本不兼容**(更深的 fix:改 kernel Buffer 构造或对齐 TVM)。

**flag 不能 bypass(已验证)**:试了 `SGLANG_OPT_USE_TILELANG_MHC_PRE=0 + MHC_POST=0`(走 `hc_pre_torch_impl` 纯 torch fallback,避开 mhc splitk kernel)—— **仍报同一个 TVM Buffer error**。说明失败的 tilelang kernel 不在 mhc_pre 路径(被我 bypass 了),而在关键路径上始终会编译的某个 kernel(`hc_split_sinkhorn` / indexer / compressor / attention 之一)。→ **这个 tilelang-kernel↔TVM 不兼容不是 flag 能绕的,是根本性版本错配**:sglang V4 的 tilelang kernels 对着另一版 tilelang/TVM 写的,和部署的 tilelang-ascend TVM(`Expected Array[PrimExpr] got Array[Array]`,Buffer shape 构造 API 变了)不兼容。

**真正的解(dedicated multi-session effort)**:① 对齐 tilelang-ascend 的 TVM 版本到 sglang V4 kernels 期望的;② 或逐个 patch 失败 kernel 的 Buffer/shape 构造。两者都不小。rollout 侧 = 真实的多层版本错配 bring-up,ACL 那关(format-cast fallback)已解,剩 tilelang-kernel↔TVM 这条根本性的。

## 版本错配根因 + 2 个 proxy backport 已修(2026-06-02 续 3,owner 指令"对齐版本")

**版本调查(子 agent + 本地镜像扒)**:
- **miles 用 mainline tilelang `0.1.8`**(GitHub tile-ai/tilelang,py3.12 dist-packages)。
- **部署的是 tilelang-ascend `0.1.1.030`**(commit a19acd5,T32 fork),base 冻结在 pre-2025-09 mainline。
- **没有任何 tilelang-ascend release 对齐 0.1.8**;最新 tilelang-mlir-ascend v0.1.2 仍带同样的 proxy bug。tuple-shape fix 在 mainline PR #783(`3cfefc8e`,v0.1.7+)。
- **mainline tilelang 0.1.8 自己有 Ascend backend**(2025-09-29 起,`ascendc_pto`/`npuir` 分支)。

**精确失败点 `mhc.py:38` + 已修 2 个**(`tilelang/language/proxy.py`,backup `.bak_getitem`):
1. ✅ `__getitem__` tuple-shape:backport mainline `if all(not tuple/list/str): keys=(keys,); return self(*keys)` → `T.Tensor[(3,),dtype]` 不再 nested-Array。
2. ✅ callable-dtype 归一化:`__call__` 里若 dtype 是 func_gen 函数对象(`T.float32`,closure name 'Float32')→ 取 closure name 转 lowercase 字符串。修 `expected DLDataType got FunctionHandle`。
   隔离验证:`@T.prim_func def k(x: T.Tensor[(3,),T.float32])` **BUILT OK**。

**下一个 kernel-body 级边界**:sinkhorn kernel body 用 `T.reduce_max` → tilelang-ascend `customize_npuir.py:844 npuir_reduce` 拒绝(reduce_max 签名/语义 vs mainline 分歧)。→ 这是 **kernel-body API 分歧**(不只 proxy),长尾(sinkhorn→indexer→compressor→sparse-attn 每个 body 可能各有)。

**战略岔路 → 已查证收敛到 (a)**:
- 纠正:mainline `tile-ai/tilelang`(miles 的 0.1.8 wheel 来源)**27 分支无任何 ascend/npu 命名**,wheel 里 target.py 零 ascend 提及 → **mainline 0.1.8 = 纯 GPU,无 NPU backend**(我之前"mainline 有 Ascend backend"是 over-claim,已纠正)。
- NPU backend 只在 `tile-ai/tilelang-ascend`(`npuir`/`npuir-dev`/`npuir-mix`/`ascendc_pto`)+ 我们的 `tilelang-mlir-ascend` fork。
- (b) 查证 = **死路**:`tile-ai/tilelang-ascend` `npuir` 分支的 proxy `__getitem__` **和我们 fork 一模一样的旧 buggy 版**(无 tuple-shape fix)。没有任何 ascend-系 tilelang 带 0.1.8 DSL。
- **根本错配**:没有任何单一 tilelang 同时具备 {miles 0.1.8 DSL} + {NPU backend}。我们用 fork(唯一有 NPU 后端)是对的;gap = fork DSL 比 0.1.8 旧。
- **→ 收敛到 (a)**:把 0.1.8 DSL delta backport 到 fork 是唯一保留 NPU 后端的路。已修 2 个(proxy subscript + dtype,verified);剩 reduce_max/npuir_reduce 等 kernel-body delta,有限长尾。2 个 fix 是 `tile-ai/tilelang-mlir-ascend` PR 候选。
- git fetch:tilelang-mlir-ascend origin/main 2b8001c→e76987c + tag v0.1.2;tilelang-ascend npuir-mix HEAD 23a9804(2026-04-17)。

## 下一个 backport delta(reduce 语义)— 已精确定位,actionable

sinkhorn kernel `mhc.py:68` 调 `T.reduce_max(comb_frag, row_max, dim=1)`:`comb_frag` 是 `(hc,hc)` 2D,`row_max` 是 `hc` 1D。
我们 `customize_npuir.py` 的 `npuir_reduce`(被 reduce_max/reduce_sum 调)断言 `len(src_extent) == len(dst_extent)`(**要求 dst 保留 reduced dim,同 rank**,如 dst `(hc,1)`)。
→ **API delta**:mainline tilelang reduce 接受 **rank-reduced 输出**(dst `(hc,)`),我们要求 same-rank。mhc.py 全程用 rank-reduced 形式(reduce_max dim=1、reduce_sum dim=1/dim=0 都是)。
**fix 方向**:让 `npuir_reduce` 接受 rank-reduced dst —— 当 `len(dst)==len(src)-1` 时,内部按 reduced-dim=1 补齐 region/extent(squeeze 语义),而不是 assert 失败。比 proxy fix 复杂(动 reduce 的 region/extent 逻辑),要小心 + 验证。
**这是 kernel-body API backport 长尾的第 1 个**(后面 indexer/compressor/sparse-attn kernel 各自的 body delta 还会有)。每个要单独 fix + 验证。属于选项 (a) 的执行。

### ⚠ reduce 语义 backport 跨进 C++ codegen(不是纯 Python)— 关键成本发现

试了 Python 层在 `npuir_reduce` 里合成 rank-matched extent(reduced dim 补 1)接受 rank-reduced dst:
- Python assert 过了,但 C++ codegen 报 `RegionOp ... load->indices.size()=0 vs ndim=2`(`tvm::tl::NpuirReduce` / `RegionOp` 在 src/op/op.cc + src/target/codegen_npuir_api.cc)。
- 即 region 的 **indices 也得 rank-match**,光改 extent 不够;dst buffer 是 rank-1、access 没有 2D indexing,C++ RegionOp 拒绝。
- **结论**:rank-reduced reduce 语义要改 **C++ codegen**(NpuirReduce/RegionOp)+ 重编 `libtilelang*.so`(~10min,同之前 bf16 fix 的重编)。**已 revert Python patch(它产生更糟的 C++ 错)**。

**修正后的 option (a) 真成本**:kernel-body delta 不全是便宜的 Python proxy backport —— reduce 语义(可能还有别的)要 **C++ codegen 改 + .so 重编**,每个 delta 一轮。从"Python backport 长尾"升级为"**C++ codegen 改 + 重编 长尾**"。属于真正的 NPU 集成工程(net-new,没人干过),非琐碎。
- 已 verified 的便宜 Python fix:proxy subscript + dtype(2 个,PR 候选,留着)。
- 需 C++ 的:reduce rank-reduce(已定位 NpuirReduce/RegionOp);后续 indexer/compressor/sparse-attn 各 body delta 待逐个分类(Python-able vs C++-needed)。
- 旁注:fork 里已有 `src/transform/legalize_npuir_bf16.cc`(bf16 legalize pass,和之前 bf16 工作相关)。

### 修正:reduce rank-reduce 其实是 Python-fixable(我上轮"需 C++"是 over-escalation)

继续深挖,纠正上轮结论 —— **不需要改 C++**。HIVM `vreduce` verifier(`HIVMVectorOps.cpp:285 verifyVReduceDims`)的真实契约:**要求 same-rank dst,reduced dim = size-1**(`idx < dst.rank` && `dst.dim[idx]==1` && `src.rank==dst.rank`)。mainline tilelang kernel 传 **rank-reduced dst**(reduced dim 直接 drop,如 `reduce_max((hc,hc),(hc,),dim=1)`)。所以 bridge 方法 = 在 `npuir_reduce` Python 层把 rank-reduced dst **view 成 same-rank size-1 dst**(同 data,无 copy)。

试了 `tir.decl_buffer(new_shape, data=dst.data, ...)` 做 same-rank view:
- ✅ 清掉了 vreduce-dim 错(`invalid index 1`)和 RegionOp 错 —— IR 里 vreduce 合法了。
- 🔴 但 TVM def-use checker 报 `variable row_sum has been used before definition`:`decl_buffer` 新建 buffer-over-同-data 没被 TVM 识别为 alias → def-use 顺序错。
- **剩的精确 hurdle**:要用 TVM 认可的 buffer-aliasing 原语(不是裸 decl_buffer)把 rank-1 dst 表示成 rank-2-size-1 view。solvable 但要对的 primitive(match_buffer / buffer view / region-with-reshape)。
- **关键修正**:reduce delta 是 **Python-fixable**(reshape view + 对的 alias 原语),不是上轮说的"必须改 C++ codegen + 重编"。discriminator 又救一次 —— 读了 HIVM verifier 源码才知道契约是 same-rank,Python reshape 就够。
- 已 revert 到干净 baseline(decl_buffer 方案 trip def-use);proxy.py 2 个 fix 在另一文件,未动。

**fix 文件(sgl_probe,可提 PR / 复用)**:`hardware_backend/npu/utils.py`(format_cast fallback)、tilelang `pass_config.py`(2 keys)、tilelang `__init__.py`(JITKernel export)、launch env(`SGLANG_OPT_FP8_WO_A_GEMM=0` + `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`)。
