# DeepSeek-V4-Flash 昇腾 A3 NPU 移植报告

**版本**:2026-06-02 ·  **目标平台**:Ascend 910C(A3,SOC `Ascend910_9382`,CANN 8.5.2) ·  **基线**:减层(1–2 层)

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

### 3.2 适配项分类

下表的 14 个适配点中,标注 **production** 的为经数值实测等价、可直接向上游提交的实现;标注 **临时方案** 的仅满足短序列演示,生产环境须由上游 sgl-kernel-npu 提供真实实现。

| 适配点 | 处理方式 | 状态 |
|---|---|---|
| `fused_q_norm_rope` 的 RMSNorm 部分 → `npu_rms_norm` | 原生算子替换,逐位等价(误差 0.000e+00) | **production** |
| `forward_extend/decode` 不接收 `compress_ratio` 参数 → `**kwargs` 吸收 | 向后兼容 | **production** |
| `silu_and_mul_clamp` → `npu_clipped_swiglu` | 原生算子替换,相对误差 ≤0.77%,clamp 内逐位等价,经端到端验证 | **production** |
| `linear_bf16_fp32` 的 `torch.mm(out_dtype=fp32)` 昇腾不支持 → 去 kwarg + `.float()` | 等价 | **production** |
| RoPE 复数乘部分保留 fp32 torch 实现 | `npu_apply_rotary_pos_emb` 为 rotate-half 约定,与 V4 interleaved 复数乘不符(误差 4.22),fp32 实现精度更优 | **设计选择(非妥协)** |
| `_maybe_upgrade_forward_metadata` 缺失 | 空实现占位 | 临时方案 |
| `forward_c4_indexer` 缺失 | 空实现占位 | 临时方案 |
| `forward_core_compressor` 缺失 | 空实现占位 | 临时方案 |
| `store_cache` / `forward_compress` / `init_forward_metadata_indexer` 缺失 | 空实现占位 | 临时方案 |
| `DeepSeekV4TokenToKVPool.get_key_buffer` 未实现 | V4 dense 短路 | 临时方案 |
| `forward_decode` cache 读取不匹配 | 同上短路 | 临时方案 |
| `fused_k_norm_rope_flashmla` 写 FP8 打包字节进 paged KV cache | 演示跳过 scatter(短序列足够) | 临时方案 |
| NPU `aclnnIndex` 不支持 complex64 | 取实部后再 view-as-complex | 临时方案 |
| `mhc.hc_split_sinkhorn` 依赖 tilelang/CUDA | torch 组合实现 | 临时方案 |

**空实现占位为何能在演示中工作、代价是什么**:上述 `forward_c4_indexer` / `forward_core_compressor` / `store_cache` 等接口负责 V4 的多级 KV cache 压缩与稀疏 indexer 选择。在 1 层、序列极短(2 token)的演示中,cache 尚未被填满,跳过这些接口不影响前向能否走通。代价是**功能性而非性能**:跳过后该层注意力退化为不压缩、不稀疏选择,长序列会显存溢出或数值错误,且输出在数值上并非真实 V4(仅形状正确、能跑完)。将单个空实现升级为 torch 组合的 PoC 级实现约需半天到一天;涉及 KV cache 写入路径(FP8 打包 scatter)的约需 2–3 天;生产级(sgl-kernel-npu 真实 AscendC 实现)为上游工作,单项周级。

---

## 四、训练链路(Megatron + miles)

训练链路为真实 DeepSeek-V4 模型层(MLA + compressor + indexer + 稀疏注意力 + hash-coding MoE 路由),运行于昇腾化的 Megatron(MindSpeed `core_r0.16.0` 分支适配 Megatron-Core 0.16)。

### 4.1 已验证结果(均为真实配置、A3 实测)

- 单个 `DeepSeekV4Attention` 层完成前向+反向训练步;
- 真实配置减层至 1 层,完成完整训练迭代(前向+损失+反向+AdamW 优化器更新,4.42B 参数,梯度全有限);
- 真实配置减层至 2 层,完成前向+反向(8.84B 参数,梯度全有限)。

### 4.2 算子实现路径

正在运行的训练层由两类实现构成,均经 pytorch 调用:

| V4 训练算子 | 实现 | 状态 |
|---|---|---|
| 稀疏 MLA(前向/反向) | `npu_nsa_select_attention`(D_qk=192/D_v=128,select_block=64,count=16,返回 attn 及 softmax max/sum 供反向) | CANN 原生,已接入,实测 |
| C4 indexer | `npu_lightning_indexer` / `npu_sparse_lightning_indexer_grad_kl_loss` | CANN 原生,已接入,实测 |
| compressor | `npu_nsa_compress_attention` | CANN 原生,已接入,实测 |
| MLA 预处理 | `npu_mla_prolog_v3` | CANN 原生,已接入,实测 |
| rms_norm | `npu_rms_norm` | CANN 原生,已接入,逐位等价 |
| hash-coding sinkhorn | torch 组合 `_hc_split_sinkhorn_npu`(CANN 无对应原生算子) | 已接入运行层 |
| act_quant(fp8) | torch fp8-grid 模拟 `_fp8_e4m3_round`(CANN 无对应原生算子) | 已接入运行层 |

说明三点,避免歧义:

1. 五个核心算子使用 CANN 原生算子(经 `torch_npu` 调用),已真实接入运行中的训练层。
2. CANN 无原生对应的两个算子(sinkhorn、act_quant)运行层中以 **torch 组合**实现。这两个算子另有独立的 AscendC kernel 经算子生成流程产出并通过精度验证,但**尚未通过 `torch_npu` 自定义算子注册接入 miles 训练链路**;它们的可调用性已单独验证(见 4.3)。
3. **tilelang-ascend 后端在运行层中未被使用**。miles 的 sparse_mla 等源算子为 `@tilelang.jit`(CUDA 目标),其 tilelang API 与昇腾 tilelang 构建不兼容,因此调用被替换为上述 CANN 原生算子。

### 4.3 AscendC 算子可调用性验证(独立测试,非运行层内)

> 与 §4.2 不矛盾,明确区分两件事:**运行中的训练层用的是 CANN 原生算子 + torch 组合(§4.2),没有调用 op-gen 的 AscendC kernel**;本节是一个**独立的旁路测试**,单独验证那个 op-gen kernel 能从 pytorch 调用,不在训练层的执行路径里。

为确认算子生成流程产出的 AscendC kernel 可从 pytorch 调用,在 A3 上以 bisheng 构建 `act_quant` 算子为 pybind 扩展(`_act_quant_ext`),用一段**独立脚本**(非训练层)从 pytorch 在 NPU 上调用 `run_act_quant(x, block_size)`,输出与 CPU 真值参考逐位一致(scale 误差 0,fp8 逐值匹配 100%,反量化误差 0)。这验证了"pytorch 调用 AscendC 算子"的机制成立。

**接入训练链路的具体阻塞(实测)**:尝试以该 kernel 替换训练层中的 torch fp8 模拟时,kernel 返回 `float8_e4m3fn`(int8 字节的 view),而在 NPU 上对其做 `.float()` 反量化报 `Float8_e4m3fn has not been supported`——这正是当初写 torch 模拟要规避的限制。可行的旁路有三:(a) torch_npu 增加 fp8→fp32 支持;(b) kernel 直接返回反量化后的 fp32 值而非 fp8 字节;(c) 在 NPU 上以整数位运算解码 e4m3 字节。当前训练层因此仍用 torch 组合;接入 kernel 待上述之一落地。结论:kernel **可调用、精度已验**,但**接入 NPU 训练数值通路被 torch_npu 的 fp8 支持缺失阻塞**。

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

> 条目 1–7 沿用自 miles-dsv4-flash PoC(`output/miles-dsv4-flash-poc/docs/REPORT.md` §上游 PR 列表,以该处为权威实时状态)。8–9 为本次 V4 训练侧工作新增、尚未开 PR 的已准备分支。

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
