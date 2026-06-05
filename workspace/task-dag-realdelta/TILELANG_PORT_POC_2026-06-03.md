# tilelang-port PoC：编译 V4 推理 tilelang kernel（mhc sinkhorn）在 fork 上（2026-06-03）

> 目的(owner 要求):实测"tilelang-port 差异到底多大" —— 拿 V4 推理实际用的 tilelang kernel 在
> fork(带我 2 个 proxy backport:subscript tuple-shape + callable-dtype,均仍在位)上真编,看撞哪类墙。
> 全程 log:`_v4_runlogs/tilelang_poc_sinkhorn.log`。run-cmd:`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 python3 test_sinkhorn.py`(编 `hc_split_sinkhorn_kernel(8,5,1e-6)`)。

## PoC 结果(原始,有 log）
- 2 个 proxy backport(subscript/dtype)**生效**:kernel 签名 `T.Tensor[(n,mix_hc),FP32]` / `T.Tensor[(3,),T.float32]` 不再报 nested-Array / DLDataType 错(之前已验)。
- 当前卡点:`customize_npuir.py:844 npuir_reduce` 的 **`assert len(src_extent) == len(dst_extent)`** —— sinkhorn 用 `T.reduce_max(comb_frag(hc,hc), row_max(hc,), dim=1)`,rank-reduced dst。

## 分类（关键判定）：这是 Python-DSL-层,不是闭源 hivmc 墙
- 失败在 **Python `npuir_reduce` 的 assert**(rank-reduce 语义),不是 bishengir/hivmc 二进制。
- 之前(同 session)我对这条做过 reshape-to-same-rank-view 的 Python fix:**清掉了 vreduce-dim 错 + RegionOp 错,IR 里 vreduce 已合法生成**,只剩一个 TVM buffer-alias 的 def-use 细节(`row_sum used before definition`)—— **仍是 Python/TVM-script 层,要用对的 alias 原语(match_buffer/buffer view),不是闭源墙。**
- 对比:之前撞的**闭源 hivmc 墙是 bf16 vmul**(flash_attn bf16),那是另一条 kernel、另一类问题。**sinkhorn 这条推理 kernel 不撞那堵墙。**

## 对"差异多大"的实测结论（修正,偏乐观方向也要诚实）
- **mhc/sinkhorn 这条推理 kernel 的 tilelang-port 缺口 = Python-DSL backport**(reduce rank-reduce 的 reshape-view + 对的 alias 原语),**和我已成功做的 2 个 proxy backport 同类、同层**。→ **对这条 kernel,tilelang-port 差异不大,Python 层可解。**
- **但还没把它编 PASS**:reduce reshape-view 还差 def-use 的正确 alias 原语(上次试 decl_buffer 触发 def-use,需换 match_buffer/region 写法)。**未跑出 BUILT OK,不能声称"sinkhorn 已可编"。**
- **未验的**:indexer/attention 等其它推理 tilelang kernel 是否也都 Python-层可解,还是有的撞闭源墙 —— 要逐 kernel PoC 才知道。本 PoC 只证了 sinkhorn 这一条是 Python 层。

## 净判定（更新两条路对比）
- tilelang-port 对推理 kernel(至少 sinkhorn)**更可能是 Python-DSL backport 长尾,不是闭源死墙** —— 比我最初"撞闭源墙天花板硬"的说法乐观。**owner 的"差异没那么大"对这条成立。**
- 真正的剩余风险仍是:① 把每个 reduce-类 fix 的 def-use/alias 收干净(Python,可做)② 是否有推理 kernel 用到 bf16-vmul 那种闭源 hivmc 不支持的 op(未逐一验)。
- 仍待做出"至少一个 V4 推理 tilelang kernel 在 fork 上 BUILT OK + 数值对"的硬证据,才算 PoC 成功。当前:**差一个 alias-primitive 修复就能让 sinkhorn 编过(高可能),但还没 BUILT OK。**

## PoC 决定性结果（编 fork 自己的 V4 推理 kernel）
- 直接编 **fork 自带的** `examples/deepseek_v4/inference/kernel.py` 的 `hc_split_sinkhorn_kernel`(不是 sglang 的,是 fork 自己的 V4 推理 kernel 源码)。log `_v4_runlogs/poc_fork_sinkhorn.log`。
- **结果:fork 自己的这个 kernel 也编不过**,同一个 `npuir_reduce` `assert len(src)==len(dst)` 错(rank-reduce)。
- **结论(实测,钉死):**
  1. fork 虽然 ship 了 `deepseek_v4/inference/kernel.py` 源码,但**它当前编不过 fork 自己的 `npuir_reduce`** → 是**未完成/未测的示例源码**,不是 working kernel。(纠正任何"fork 已有 V4 推理 kernel 能用"的暗示。)
  2. rank-reduce 缺口**真实存在、且挡住 fork 自己的 V4 推理 kernel**,层级 = **Python `customize_npuir.py`**,**不是闭源 hivmc 墙**。
  3. → **tilelang-port 对 V4 推理(sinkhorn 类)的缺口 = 补 fork 自己没写完的 `npuir_reduce` rank-reduce 支持(Python)。** owner 的"差异没那么大"对这条成立;但同时暴露 **fork 的 V4 推理 kernel 从没真编过**。
- **PoC 尚未 BUILT OK**:差 `npuir_reduce` rank-reduce 的完整 Python 修(reshape-to-same-rank-view + 对的 TVM alias 原语,def-use 干净)。这是明确的、Python 层的、可做的下一步 —— 不是闭源死路。

## PoC 净答案（给 owner 的"能否做 PoC / 差异多大"）
- **能做,且已实测到关键判定**:V4 推理 tilelang kernel 在 fork 上的缺口是 **Python-DSL 层(reduce rank-reduce + 之前 2 个 proxy),不是闭源墙** → tilelang-port 对推理 kernel **比最初评估的乐观/可行**。
- **代价 = 把 fork 没写完的 npuir_reduce(+ 可能其它 DSL 算子)补全**(Python),逐 kernel 验。
- **风险残留**:① 个别 kernel 可能用到 bf16-vmul 那种闭源 hivmc 不支持的 op(本 PoC 的 sinkhorn 没用到,未逐一排查)② reduce 修复的 def-use alias 原语要找对。
- **离 BUILT-OK 硬证据:1 个 Python 修复之遥**(npuir_reduce rank-reduce)。

## PoC 进展 2（same-rank dst kernel-side 修，2026-06-03）
- 改 fork sinkhorn kernel:`row_max/row_sum` alloc `(hc,1)`、`col_sum` `(1,hc)`,索引 `[j,0]`/`[0,k]`(满足 HIVM vreduce 的 same-rank dst 契约,kernel-source 改,不动 npuir_reduce)。log `_v4_runlogs/poc_samerank_sinkhorn.log`。
- **reduce rank-reduce 这一关过了**:不再有 `npuir_reduce` assert / vreduce-dim / RegionOp / def-use 错。kernel 编译穿过 TVM-script + HIVM reduce,到达 **bishengir 后端**。
- **新边界**:bishengir 报 `'hivm.hir.copy' op Unsupported copy from cbuf to gm!`(在末尾 `T.copy(comb_frag, comb[i,:,:])`)。comb_frag 在 cbuf(fragment scope)直接 copy 到 global memory 不支持 —— 是**内存-scope/copy 路由问题**(大概率加个 ub staging buffer 或换 scope 可解),**不是 bf16-verifier 那种 op-missing 死墙**。
- **判定**:PoC 在逐层推进,每关都是 real-but-addressable 的 porting 修(proxy ✓ → reduce same-rank ✓ → 现在 copy-scope)。**到目前为止仍支持"tilelang-port 是有限 porting 工作量(Python/kernel-side),不是闭源死墙"。**
- **下一步**:加 ub staging(comb_frag → ub → comb)或调 comb_frag scope,过 copy-scope 关。
- **诚实**:仍未 BUILT OK;但已从"Python assert"推进到"bishengir copy-scope",方向是 addressable 的修,不是死墙。

## PoC 进展 3（dump npuir 定位 cbuf→gm 真根因，2026-06-03）
- 加 `comb_frag → comb_shared(alloc_shared) → comb` staging(对齐本文件里 working kernel `sparse_attn`/`fp8_gemm` 的 `frag→shared→gm` 标准写法,488-490 行)。**仍报同一 `cbuf→gm` 错,line 55 不变。**
- patch `jit_npu.py` 加 `TILELANG_DUMP_NPUIR` 把 npuir 落盘(`/tmp/sinkhorn.npuir`),实读 line 55 ↔ 源码对应。**关键发现(钉死,有 dump 为证):**
  - npuir 里 `address_space` 分布:**gm 126 / zero 76 / cbuf 12** —— 绝大多数计算落在 **gm**(global),不是 ub/cbuf 计算 scope。
  - vmul/vadd/vexp/vdiv/vsub 全部直接在 `memref<1x8xf32, gm>` 上跑(npuir 60-90 行),即 **scalar 元素循环被 lower 成 per-row gm 上的 vector 小 op**,没有 tile 进 ub。
  - 3 处 `cbuf→gm` copy(line 55/79/155):**55/79 是 `mixes_shared`(alloc_shared→被放进 cbuf)喂给 scalar 元素循环 `comb_frag[j,k]=mixes_shared[...]*..+hc_base[..]` 时,cbuf 要 copy 到 gm 才能喂 gm 上的 scalar 循环** → 不支持。155 是我加的 staging(8x8 cbuf→gm,结构正确的那个)。
- **真根因(比"加 staging"深一层,诚实修正进展 2 的乐观判断):**
  - **不是末尾那个 `T.copy(comb_frag, comb)` 的简单 staging 问题**(line 55 是开头的 `mixes_shared` load,不是末尾)。
  - sinkhorn kernel 用 **scalar Python 元素循环**(`for j: for k: comb_frag[j,k] = 标量表达式`)写。NPU MLIR 后端**不把这种 scalar-index 赋值 fuse 成 tiled-vector**,而是散落到 gm + 制造 cbuf↔gm 往返 → 撞 `cbuf→gm` 不支持。
  - = **kernel-authoring-model 不匹配**:要么把 kernel 重写成**向量化 tile 原语**(整块 `T.copy`/fragment 上的 elementwise,不用 scalar `[j,k]` 下标赋值),要么后端补 scalar-loop→ub 的 lowering(没有)。
- **对"差异多大"的诚实修正**:进展 2 我说"加个 ub staging 大概率可解"**偏乐观**。实读 npuir 后:**这条 sinkhorn 的缺口是 kernel 要按 NPU tile-model 重写(去 scalar 下标循环,改向量化原语),不是一行 staging**。仍是 **open-layer / kernel-source 层(可做)**,不是闭源 hivmc 死墙(vmul/vexp 这里都正常生成了,没撞 bf16 那堵墙)—— 但工作量比"加 staging"大,是**重写 kernel 数据流**。
- **下一步**:把 sinkhorn 改成向量化写法(`comb_frag` 整块算:`T.copy`+fragment-level `T.exp`/除法,消除 `for j,k` scalar 赋值),让计算留在 ub-tile、避免 cbuf→gm 往返,再编。
- **净判定不变**:tilelang-port 对推理 kernel **是有限的、open-layer 的 porting 工作量(现在精确到:kernel 要按 NPU tile-model 向量化重写)**,不是闭源死墙;但比我上一版乐观估计**更费工**(重写 kernel,非一行修)。

## PoC 进展 4（向量化重写到底，定位真·墙：`zero`-scope 不能 copy 到 cbuf/gm，2026-06-03）
做了完整向量化重写,逐层推过:
1. **comb 强 scalar-gather → 2D 块加载**:host wrapper 把 `mixes[:, hc*2:]` 和 `hc_base[hc*2:]` 切片+reshape 成 `(n,hc,hc)`/`(hc,hc)` 当新入参,kernel 里 `T.copy(mixes_comb_in[i,:,:], frag)` 整块加载,消掉 `comb_frag[j,k]=mixes_shared[j*hc+k+hc*2]*..` 那种 strided scalar gather。→ **comb-init 的 cbuf→gm 错消失。**
2. **pre/post scalar 写 → ub fragment + shared staging**:`pre_frag[j]=sigmoid(..)` 算进 fragment,再 `T.copy(pre_frag, pre_shared); T.copy(pre_shared, pre[i,:])`。→ **pre/post 的 `zero→gm` 错消失。**
3. **撞到真·墙(line 96)**:`error: 'hivm.hir.copy' op Unsupported copy from zero to cbuf!`

**实读 npuir 定位真根因(钉死,有 dump `/tmp/sinkhorn_v6.npuir` 为证):**
- 所有 elementwise vector 计算(`hivm.hir.vexp/vadd/vdiv/vmul` —— sigmoid=1/(1+exp(-x)) 也是这些)**全部落在 `#hivm.address_space<zero>`**(通用 local scope),**不是 `ub`**。
- 然后 staging copy 要把结果从 `zero` 搬到 `cbuf`(line 96/101 pre/post、142 comb),bishengir 报 **`zero→cbuf` 不支持**(同族还有之前的 `zero→gm`/`cbuf→gm`)。
- 对比:working kernel(sparse_attn/fp8_gemm)的 fragment 计算结果落在 `ub`/`cbuf`,直接 `→shared→gm`,**从不经过 `zero`**。

**真·墙的分类(诚实,这层比"重写 kernel"更深):**
- 这是 **fork 侧 lowering/codegen pass 的缺口**,不是 kernel 写法能绕开的:`T.Parallel` + 逐元素 vector 计算的结果**被 lower 进 `zero` scope,而该 pass 没有把 `zero` 提升(promote)到 `ub`**;同时 bishengir 不支持 `zero→{cbuf,gm}` copy。→ **无论 kernel 怎么向量化重写,只要 elementwise 计算结果进了 `zero`,就过不了 copy 关。**
- **不是闭源 hivmc 的 op-missing 墙**(vexp/vmul 都正常生成了,跟 bf16-vmul 那堵闭源墙是两回事)。这是 **fork 的开源 codegen pass(`src/target/codegen_npuir*.cc` 或 scope-promotion transform)没把 `zero`→`ub` 提升做全** —— **理论上 open-layer 可修,但要改 fork C++ codegen / MLIR pass,不是 Python DSL backport,也不是 kernel 源码改。**

## PoC 最终净答案（给 owner 的"tilelang-port 差异到底多大",基于本次逐层实测，钉死）
- **能编过的层**:proxy(subscript/dtype)✓ → reduce same-rank ✓ → scalar-gather 向量化 ✓ → pre/post staging ✓ —— 这些都是 **Python-DSL backport + kernel-source 改**,已逐一做到、推过。
- **真·墙**:`zero`-scope 的 elementwise 计算结果无法 copy 到 cbuf/gm —— 是 **fork 开源 codegen/MLIR-pass 的 scope-promotion 缺口**(`zero`→`ub` 没做全)。
  - **层级**:fork C++/MLIR pass(open-source,可改但要动 codegen,不是 DSL/kernel 层)。
  - **不是闭源墙**:hivmc 的 vexp/vmul 都正常,不是 bf16 那种 op-missing 死墙。
- **"差异多大"实测结论**:
  - 比"闭源死墙"乐观 —— 真·墙在 **fork 开源 codegen**,有源码(`Ascend/AscendNPU-IR` 的 bishengir 也开源,fork 的 codegen_npuir 更是),**理论上可修**。
  - 比"几个 Python backport"悲观 —— 不是改 DSL/kernel 能绕过的,要动 **fork 的 scope-promotion pass(C++/MLIR)**,让 `T.Parallel` elementwise 结果落 `ub` 而非 `zero`。
  - **量级**:1 个 fork codegen-pass 级别的修(中等,要懂 HIVM scope model),不是 1 行,也不是闭源绕不过。
- **下一步选项**(留给 owner 定方向):
  - (A) 钻 fork 的 `zero`→`ub` scope-promotion(改 `src/target/codegen_npuir*.cc` 或对应 transform),让 elementwise 落 ub —— 通用修,修好后所有这类 kernel 都受益。
  - (B) 找 working kernel 到底用什么 DSL 写法让计算落 `ub` 而非 `zero`(可能有 `T.alloc_fragment` 的 scope 标注 / 特定 compute 原语我没用对)—— 先排除"是我 DSL 写法没踩对",再下"必须改 codegen"的结论。
  - **诚实**:目前还没 BUILT OK;但已把墙精确定位到 `zero`-scope copy,并分类为 **fork 开源 codegen-pass 缺口(非闭源、非 kernel 写法)**。选 (B) 先验证是不是我 DSL 没写对,更稳。

## PoC 进展 5（做了选项 B —— 真相揭晓：不是 codegen 墙，是 GPU-idiom vs NPU-idiom 写法错配，2026-06-03）
**先撤回进展 4 末尾"fork codegen-pass 缺口"的判断 —— 那是过早结论。做了选项 (B) 后发现是我 DSL 没写对。**

实证链(钉死,有 log):
1. **fork 自带的 `examples/deepseek_v4/inference/kernel.py`(我 PoC 的源)用了 fork 不支持的 API** —— `GemmWarpPolicy`/`T.gemm`/`float8_e4m3`/`T.use_swizzle`(实测 `AttributeError: module 'tilelang.language' has no attribute 'GemmWarpPolicy'`、`unknown type float8_e4m3`)。→ **这个 example 是按更新版 tilelang 写的,fork(0.1.1.030)根本编不了它自己 ship 的这个 example**(sparse_attn/act_quant/fp8_gemm 都撞)。但 sinkhorn 本身没用这些 API,所以 sinkhorn 的 `zero` 问题是真的,要继续查。
2. **找到 fork-native 的对照 example `examples/vectorization_in_parallel.py`** —— 这是明确按 fork(NPU)写的 elementwise 示例。它的写法和 sinkhorn 根本不同:
   - 用 **`T.alloc_ub((block_N), dtype)`** 显式分配 Unified Buffer(**不是** `T.alloc_fragment`/`T.alloc_shared`)。
   - 用 **`with T.Kernel(n_num, is_npu=True) as (cid, _)`**(`is_npu=True` + 2-tuple,**不是** `threads=64`)。
   - 数据流:`T.copy(global, ub)` → `for i in T.Parallel: ub[i]=T.exp(...)` → `T.copy(ub, global)`。**全程 ub,不碰 shared/cbuf/fragment。**
3. **实测编 `binary_compound_elementwise`(含 `T.exp`,跟 sinkhorn 同类计算)→ `BUILT OK`**(log `_v4_runlogs/poc_vecparallel.log`)。**fork 的 vector/elementwise + exp 路径完全 work** —— 只要用对 NPU 原语。

**真相(诚实,撤回 codegen-墙误判):**
- **不是 fork codegen 缺口,不是闭源墙。是 kernel 写法的 idiom 错配:** sinkhorn(及 fork 那个编不了的 example)用 **GPU/CUDA idiom**(`alloc_fragment`/`alloc_shared`/`threads=`),fork 把它 lower 进 `zero`/`cbuf` 路由不了;**NPU-correct idiom 是 `alloc_ub` + `is_npu=True`**,fork 完全支持(实测 BUILT OK)。
- **= kernel-source 层的 porting**(把 GPU 原语换成 NPU `alloc_ub` 原语),**Python/kernel 层可做,fork 支持**。这是**最乐观但真实**的结论。

## PoC 最终净答案 v2（覆盖 v1 —— "tilelang-port 差异到底多大",钉死，最优但真实）
- **fork 的 NPU vector/elementwise 路径是 work 的**(实测 `alloc_ub`+`T.exp` BUILT OK)。
- **缺口 = 把 V4 推理 kernel 从 GPU-idiom 重写成 NPU-idiom**(`alloc_fragment`/`alloc_shared`/`threads=` → `alloc_ub`/`is_npu=True`/ub-only 数据流)。**纯 kernel-source 层,无闭源墙,fork 支持。**
- **量级**:逐 kernel 按 NPU idiom 重写(sinkhorn 这种 elementwise+reduce 已确认 fork 能编同类),不是改 fork codegen,更不是闭源绕不过。比 v1 的"codegen-pass 修"还乐观一层。
- **仍待**:把 sinkhorn 用 `alloc_ub` idiom 重写 → BUILT OK(下一步,fork 已证同类可编)+ reduce(`reduce_max`/`reduce_sum`)在 ub idiom 下是否也 OK(elementwise 已证,reduce 待验)。
- **两次自我修正记录(诚实)**:进展 4 我判"fork codegen-pass 缺口" = 过早(没先做选项 B);进展 5 做了 B,发现是 idiom 错配,fork 其实支持。**下结论前必须先排除"自己写法没踩对",这正是 owner 一直强调的。**

## PoC 进展 6（NPU-idiom 重写 sinkhorn —— 得到 BUILT OK 基线 + 精确定位剩余 backend bug，2026-06-03）
按 NPU idiom(`alloc_shared` + `is_npu=True` + block 向量 intrinsic `T.vsub/vexp/vdiv/vadd` + `T.reduce_max/sum`)重写 sinkhorn,逐块 bisect 编译(全部有 log `_v4_runlogs/poc_*.log`):

**逐块实测(每条都 BUILT OK / FAIL 钉死):**
- `T.copy` 2D in/out ✓
- `reduce_max` 单独 ✓
- `reduce_max + T.vsub(c,row_max,c)(广播)+ T.vexp` ✓  ← softmax 减最大值
- `T.vadd(c, eps标量, c)` 标量加 ✓
- `reduce_sum dim=1 + T.vdiv(c,row_sum,c)(行广播)` ✓  ← 行归一化
- `reduce_sum dim=0 + T.vdiv(c,col_sum,c)(列广播)` ✓  ← 列归一化
- **完整 sinkhorn 第一遍(softmax + 行归一 + 列归一,无循环)→ BUILT OK** ✓
- **`for _it in T.serial(N)` 循环 → FAIL**(`'ForFrame' object is not iterable`,fork 这个版本 `T.serial`/`T.unroll` 在我的 harness 里挂)→ **改用 Python `range()` 编译期展开,BUILT OK** ✓
- **完整 sinkhorn `iters=2`(range 展开)→ BUILT OK** ✓
- **`iters≥4` → Segmentation fault**(bishengir 后端崩,exit 139,无 Python traceback)
- **定位崩因**:`reduce_sum dim=1`(行)×8 单独 → BUILT OK;但**反复混用 `reduce_sum dim=0`(列/跨行 reduce)≥4 次 → segfault**。→ **崩在重复的 `reduce_sum dim=0`(NPU 上跨分块维 reduce)** —— 是 **bishengir 后端对"重复列 reduce"的具体 bug**,不是 kernel 逻辑错、不是 op-missing。

**进展 6 净结论(钉死,最关键):**
1. **sinkhorn 在 fork NPU idiom 下能编(BUILT OK 基线已拿到:iters=2 完整版 + row-reduce×8)** —— **GPU→NPU idiom 的 kernel-port 路线被证明可行**(撤回所有"撞墙编不了"的说法,这条推理 kernel 的核心算子全部 BUILT OK)。
2. **剩余唯一 blocker = bishengir 后端对反复 `reduce_sum dim=0` 的 segfault** —— 是 **fork/bishengir 的后端 bug,可提 issue / 可绕**(绕法:① 把列 reduce 改成 transpose+行 reduce ② 减少列-reduce 次数 ③ 升级 CANN/bishengir 看是否已修)。**不是闭源 op-missing 死墙(reduce/vexp/vdiv 都正常),是后端的 crash bug。**
3. **对"tilelang-port 差异多大"的最终实证答案(钉死,覆盖前面所有版本):**
   - **kernel-authoring port(GPU idiom → NPU `alloc_shared`+intrinsic idiom)= 真实但有限的工作量,已证可行(sinkhorn 核心全 BUILT OK)。**
   - **唯一硬 blocker = bishengir 对重复列-reduce 的 segfault** —— backend bug,可绕/可提 issue,**不是闭源死墙**。
   - 量级:逐 kernel 改 idiom(中等)+ 绕一个后端 reduce-bug(transpose 绕,小)。**比"闭源死墙"乐观得多;是工程量,不是不可能。**

**下一步选项(留 owner 定):**
- (A) 绕 segfault:列归一化用 `transpose(c) → reduce dim=1 → transpose back`,避免重复 `reduce_sum dim=0`,拿完整 20-iter BUILT OK。
- (B) 提 fork/AscendNPU-IR issue:重复 `reduce_sum dim=0` segfault 的最小复现(我已有:iters=4 崩、row-only×8 不崩)。
- (C) 收口本 PoC:核心结论已拿到(NPU-idiom port 可行,唯一 blocker 是可绕的 backend reduce-bug),够回答 owner"差异多大"。
- **诚实**:完整 20-iter sinkhorn 还没 BUILT OK(卡 segfault);但已拿到 iters=2 完整版 BUILT OK + 把 blocker 精确到"重复列-reduce 的后端 crash",并有最小复现。**这是 real-but-addressable backend bug,不是死墙。**

## PoC 进展 7（精确钉死 segfault 阈值 + 最小复现，2026-06-03）
继续 bisect,拿到精确阈值(flat `@tl.jit`,`poc_sink_final.py`,有 log `_v4_runlogs/poc_sink_final_*.log`):
- **完整 sinkhorn(softmax + 行/列归一循环)iters=4 → BUILT OK**(3/3 确定性,`poc_softmaxmix.py`)
- **iters=5 → Segmentation fault**(exit 139),iters=6/7/8/20 同样崩。**阈值钉死:iters≤4 编过,iters≥5 崩。**
- **对照实验定位真·trigger(钉死):**
  - `reduce_sum dim=1`(行 reduce)×20 单独 → **BUILT OK**(60 ops,op-count 不是因)。
  - `npuir_transpose ×6` 单独 → BUILT OK。
  - 行+列 reduce 各 ×3(无 softmax 前导)→ BUILT OK。
  - **完整 kernel 的列-reduce(`reduce_sum dim=0`)总数:iters=4 → 4 个(1 softmax 列归一 + 3 循环列归一);iters=5 → 5 个。** → **trigger = 一个 kernel 里 `reduce_sum dim=0`(跨分块维/列 reduce)数量 > 4 → bishengir segfault。**
  - transpose 绕法(把列 reduce 换成 transpose+行 reduce)**也在 iters=4 崩**(ops 更多)→ 绕法无效,坐实是后端对这类结构的 crash,不是 dim=0 op 本身。

**进展 7 最终净结论(钉死,这是 PoC 的硬答案):**
1. **NPU-idiom sinkhorn 能编(iters≤4 完整 BUILT OK)** —— GPU→NPU idiom 的 kernel-port **确定可行**。
2. **唯一硬 blocker:bishengir 后端在一个 kernel 内 `reduce_sum dim=0` 超过 ~4 个时 segfault。** V4 默认 `sinkhorn_iters=20` 远超此限 → 完整版编不过。
3. **最小复现(fileable)**:NPU kernel,`alloc_shared (8,8)` fp32,循环里放 5× `reduce_sum(c, col_sum, dim=0)` → bishengir-compile segfault(CANN 8.5.0)。≤4 次正常。
4. **分类**:**fork/bishengir 后端 crash bug(开源侧 bishengir-compile / HIVM pass),不是闭源 op-missing 死墙,也不是 kernel 写法错。** 可提 issue(Ascend/AscendNPU-IR 或 tilelang-mlir-ascend);可能的 workaround:① 把 sinkhorn 拆成多 kernel launch(每个 ≤4 列 reduce,host 侧循环)② 减 sinkhorn 迭代数(数值上 20→收敛后截断)③ 升级 CANN/bishengir 看是否已修。

## PoC 进展 8（诚实修正阈值 —— 确定性复测，2026-06-03）
**修正进展 7 的"iters≤4 BUILT OK":那是不同脚本结构的结果,不稳。** 用统一的 flat-jit 脚本 `poc_sink_final.py` 确定性复测(log `_v4_runlogs/RESULT_sinkhorn_npuidiom_*.log` + 背景任务):
- **iters=2 → 4/4 BUILT OK(稳定)**
- **iters=4 → 8/8 Segmentation fault(稳定崩)**
- 进展 7 里 iters=4 编过的是 `poc_softmaxmix.py`(`build()` 包装、prim_func 闭包结构不同)—— 两个"iters=4" kernel 结构有微差(闭包/参数绑定),不能混为一谈。
- **诚实钉死的最小稳定结论**:同一 kernel 结构下,**`reduce_sum dim=0`(列 reduce)≤2 个稳定编过,4 个稳定 segfault。** 阈值在 3-4 之间(未细分,3 个可能边界)。row-reduce(dim=1)单轴 ×20 始终稳定 → **确实是列-reduce(跨分块维)在 bishengir 后端的具体 crash。**
- **撤回任何"iters=4 能编"的说法**:统一脚本下 iters=4 稳定崩。**当前能稳定 BUILT OK 的是 iters=2 的完整 sinkhorn(softmax + 1 轮行/列归一)。**

## 给 owner 的 PoC 一句话总账（"tilelang-port 差异到底多大",全程实测，钉死）
**可行,差异是有限工程量,不是闭源死墙:**
- **kernel-idiom port(GPU `fragment/shared/threads` → NPU `alloc_shared`+`is_npu=True`+块向量 intrinsic):确定可做,sinkhorn 核心算子(reduce/vexp/vsub/vdiv/vadd/transpose)全部 BUILT OK。**
- **唯一拦路:bishengir 后端对"一个 kernel 内 >4 个列-reduce"segfault** —— 开源后端 crash bug,有最小复现,可提 issue / 可绕(多 kernel launch / 减迭代 / 升级 CANN)。
- **没有撞到闭源 hivmc 的 op-missing 墙**(那是 bf16-vmul 那条 attention kernel 的问题,sinkhorn 这条不撞)。
- **诚实边界**:① 完整 20-iter sinkhorn 因后端 bug 单 kernel 还没 BUILT OK(iters=2 单 kernel 稳定可)② 数值正确性(跑 NPU 对 torch ref)还没验,只验了"能编"③ 其它 V4 推理 kernel(mhc-post/indexer/compressor)未逐一 PoC,可能各有自己的 idiom/后端坑。**但"差异多大"的核心问题已答:有限工程量 + 一个可绕的后端 bug,路通。**

## PoC 进展 9（workaround 验证成功 —— 完整 20-iter sinkhorn 经 multi-launch 全部 BUILT OK，2026-06-03）
绕过 bishengir 列-reduce segfault 的 workaround **实测成功**(log `_v4_runlogs/RESULT_sinkhorn_multilaunch.log`,3/3 稳定):
- **拆成两个 kernel,每个列-reduce ≤1 个(远低于 crash 阈值):**
  - `sink_softmax`(前导:reduce_max + vsub + vexp + 行归一 + 列归一,1 个列-reduce)→ **BUILT OK**
  - `sink_iter`(单轮 sinkhorn:1 行归一 + 1 列归一,1 个列-reduce)→ **BUILT OK**
- **host 侧循环 `sink_iter` 19 次**(softmax 后)= 数值上等价 `sinkhorn_iters=20`。每个 launch 都在 crash 阈值下 → **完整 20-iter sinkhorn 算法可在 NPU 跑通(全 kernel BUILT OK)。**
- **3/3 确定性 BUILT OK** —— 不是侥幸。
- **代价**:multi-launch 比单 kernel 多 launch 开销 + 反复 global 读写(每 iter 从 gm 读、写回);性能不是最优,但**功能路径通了,且 BUILT OK 稳定**。性能可后续优化(如每 kernel 塞 2-3 iter 仍在阈值下,减 launch 次数)。

## PoC 终账 v2（覆盖前面，含 workaround，钉死）
**tilelang-port 对 V4 推理 sinkhorn:路通,有限工程量,无闭源死墙。已实测到 BUILT OK(workaround 后完整 20-iter 等价)。**
1. **kernel-idiom port(GPU→NPU `alloc_shared`+`is_npu=True`+块向量 intrinsic):确定可做,sinkhorn 核心算子全 BUILT OK。**
2. **bishengir 列-reduce >~3 segfault bug:已有最小复现 + 已验证 workaround(multi-launch,3/3 BUILT OK)。** 可提 issue 让上游修;在修之前 multi-launch 即可让完整 sinkhorn 跑通。
3. **没撞闭源 hivmc op-missing 墙**(bf16-vmul 那条是 attention kernel 的独立问题)。
4. **诚实剩余**:① 数值正确性未在真 NPU run 验(只验编译)② 其它 V4 推理 kernel 未逐一 PoC。**但 owner 的核心问题"tilelang-port 差异到底多大 / 能否做 PoC"已实测回答:能,差异 = 有限的 idiom-port 工程量 + 一个已找到 workaround 的后端 bug,不是闭源死墙。**

## PoC 进展 10（数值正确性在真 NPU 上验过 —— NUMERIC PASS，2026-06-03）
补上"只验编译没验数值"这个诚实缺口。把 multi-launch sinkhorn **在真 NPU(sgl_probe /dev/davinci1,先 precheck 空闲 + torch smoke OK)上跑,对 torch 参考实现比数值**(driver `/tmp/verify_sinkhorn_numeric.py`,log `_v4_runlogs/RESULT_sinkhorn_numeric_20260603T074741Z.log`):
- host 驱动:`sink_softmax` launch 1 次 + `sink_iter` launch 19 次(buffer ping-pong)= 完整 `sinkhorn_iters=20`。
- torch ref:`softmax(-1)+eps → 列归一 → (行归一→列归一)×19`,同一算法 fp32。
- **结果:`max_abs_diff = 8.941e-08`,`mean_abs_diff = 1.412e-08`**;`out[0,0,:4] = [0.10397, 0.07849, 0.13791, 0.06401]` 与 `ref` 5 位小数全同。
- **NUMERIC PASS**(阈值 max_abs_diff < 1e-3,实际 ~9e-8 = fp32 ULP 级)。**判别器:随机 look-alike 不可能对到 8 位小数 —— 是语义真对,不是机械跑绿。**

**→ tilelang-port PoC 端到端闭环(全有 log,this-session 实跑):编译 BUILT OK ✅ + 真 NPU run ✅ + 数值对 torch ref ✅(完整 20-iter,multi-launch)。** owner 的"能否做 PoC / 差异多大"问题:**能,已做出端到端 PoC;差异 = 有限 idiom-port 工程 + 一个可绕的后端 bug,路通且数值正确。**

## PoC 进展 11（诚实修正 segfault 归因 —— "列-reduce 计数"是错的，2026-06-03，提 issue 前 bisect）
**撤回进展 7/8/9 里"一个 kernel 内列-reduce(dim=0)> ~3 → segfault"的归因 —— 实测证伪,是错的。** 为提 issue 收紧最小复现时发现:
- **纯 `reduce_sum dim=0` × 5(循环里)→ BUILT OK**(`minrepro_colreduce.py`,1..5 全过)。
- **mixed 行+列 reduce × 5(含 softmax)→ BUILT OK**(`minrepro_mixed.py`)。
- → **不是列-reduce 计数、不是 op 计数、不是单个 op。** 进展 7/8 的"阈值在 3-4 之间"是把不同脚本结构的结果当成了 op 计数规律,错。
- **真·性质(memory-corruption / codegen-不稳定 签名)**:size/结构敏感,**两个近乎相同的 9-reduce kernel 行为相反**,各自脚本内确定性:
  - `minrepro_mixed.py 4 1`(softmax + `for range(4): 行;列`)→ 3/3 BUILT OK
  - `pin_trigger.py without`(softmax + 1行 + 1列 + `for range(3): 行;列`)→ 2/2 SEGFAULT
  - 同 9 个 reduce、同 buffer,只是展开-循环边界位置不同 → 一个崩。这是后端 codegen bug,不是语义 op 限制。
  - 完整 sinkhorn iters=2 稳过 / iters≥4 稳崩仍成立(`poc_sink_final.py`),但**原因是结构敏感的后端 crash,不是"列-reduce 太多"**。
- **bishengir-compile 版本(提 issue 用)**:`0.1.0 (e4e2ba9841d1, 2026-01-16)`,CANN 8.5.0。
- **可靠的 crash/no-crash 配对(提 issue 的最小复现)**:`pin_trigger.py without`(崩) vs `minrepro_mixed.py 4 1`(过),都在 sgl_probe `/tmp`。
- **workaround 仍有效**(multi-launch 拆小 kernel,已 NUMERIC PASS)—— 因为拆小后单 kernel 没到那个结构-阈值。
- **教训(记 memory)**:提 issue 前必须先验最小复现钉死真触发,不能把"看起来像规律"的中间观察当成结论 —— 差点提了个错的"列-reduce 计数" issue。([[feedback_check_responsibility_layer_before_filing]] / [[verify_meaning_not_just_mechanics]])

**对 owner 核心结论无影响**:tilelang-port 仍是"有限 idiom-port 工程 + 一个可绕的后端 bug,数值已验对";只是 bug 的精确描述从"列-reduce 计数"修正为"结构敏感的后端 segfault"。multi-launch workaround + NUMERIC PASS 不受影响。
