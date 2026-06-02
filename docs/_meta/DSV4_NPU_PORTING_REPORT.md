# DeepSeek-V4-Flash 昇腾 A3 NPU 移植报告

**版本**:2026-06-02 ·  **目标平台**:Ascend 910C(A3,SOC `Ascend910_9382`,CANN 8.5.2) ·  **基线**:减层(1–2 层)

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

> **唯一已合入(MERGED)的是条目 10**(a5_ops 工具链内部修复)。其余上游条目状态:1/3/6 OPEN 等 human maintainer;2/4/7 OPEN;5 CLOSED(not-planned,已 reframe 成 KB);8/9 尚未开 PR。
> **注意 #251 不是"已合入"**:它是 OPEN issue,只是我把 311-pass bisect **报告作为评论 posted 上去了**(评论 landed ≠ issue resolved ≠ fix merged);真修由华为编译器组负责,尚未开始。
> 条目 1–7 的实时 reviewer 反馈细节以 `output/miles-dsv4-flash-poc/docs/REPORT.md` §上游 PR 列表为权威;8–9 为本次 V4 训练侧新增的已准备分支。

### 6.2 本报告新识别的候选(待真实端到端后批量提交)

推理侧(`sgl-project/sglang` + `sgl-kernel-npu`):四个 production 适配项可直接提交(RMSNorm 原生、`**kwargs` 兼容、swiglu 原生、gemm cast);一个汇总 issue(缺失的 `AscendAttnBackend` V4 接口,逐项附行号与证据);NPU complex64 aclnnIndex issue;FP8 打包 KV cache scatter。

训练侧(`radixark/miles` + `Ascend/MindSpeed` + `Ascend/pytorch`):sparse-softmax 稳定化补丁(miles);MindSpeed rms_norm 签名适配 + V4 层契约;torch_npu Float8_e4m3fn cast 缺失。

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
