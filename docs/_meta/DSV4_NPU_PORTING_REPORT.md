# DeepSeek-V4-Flash 在 Ascend A3 NPU 上的移植报告

> 状态汇总 2026-06-01。本报告以 **DSv4-Flash** 为重点,记录推理(SGLang)与训练(Megatron/miles)
> 两侧在 NPU 上遇到的坑、如何解决、以及**每个修复是 walkaround 还是 production-ready**。
>
> 诚实纪律:本报告区分"已实测跑通"与"已声明",凡 walkaround 一律标 ⚠️ 并写明上游真修在哪。
> 不把 PoC 减层 / no-op stub / synth-delta 说成 production。

---

## 0. TL;DR(给赶时间的人)

| 维度 | 推理侧 (SGLang) | 训练侧 (Megatron + miles) |
|---|---|---|
| **最高已验证里程碑** | 真 V4 model class `generate()` + RL loop 闭环(rollout→weight-sync→re-rollout)在 A3 NPU bf16 跑通 | 真 DSV4 config 减层(256-expert MoE)1 层**完整训练迭代**(fwd+bwd+AdamW.step)+ 2 层 fwd+bwd 跑通 |
| **减层?** | 是(1 层 fab,gibberish 输出,形状正确非数值正确) | 是(真 config / 真 dims,层数减到 1–2,显存所限) |
| **walkaround 数** | 14 个 PoC stub/fallback(已部分换成 native op) | 4 类(rms_norm shim / sparse-softmax 稳定化 / megatron fp32-grad patch / optimizer 显存) |
| **production-ready 部分** | 2 处 native-op 替换(已实测等价,PR 形态)+ RL weight-sync plumbing | CANN-native 算子映射(NSA/indexer/compressor/MLA-prolog,全实测)+ sparse-softmax 稳定化补丁 |
| **真 e2e 缺口** | 14 stub 中约 8 个要上游 sgl-kernel-npu 补 V4 hook | 真训练 delta 替换 synth-delta(已无算子缺口——CANN-native 全覆盖,只剩多芯片显存工程) |

**一句话**:V4 在 NPU 上的**机制**(model class / KV pool / RL 闭环 / 单层训练迭代 / CANN-native 算子)已逐项实测跑通;
剩下的是**两类工程**——推理侧的上游 sgl-kernel-npu V4 hook 补全、训练侧的多芯片显存(并行/重计算)。不是算法或算子盲区。

---

## 1. DSv4-Flash 真 config(移植对象)

`deepseek-ai/DeepSeek-V4-Flash` HF config(`workspace/T32_tilelang_rescue/v4_real_truth/v4_real_config.json`):

- `hidden_size=4096`, `num_hidden_layers=43`, `num_attention_heads=64`
- MLA: `q_lora_rank=1024`, `kv_lora_rank=512`, `qk_rope_head_dim=64`, `v_head_dim=128`
- o_lora: `o_lora_rank=1024`, `o_groups=8`
- hash-coding: `hc_mult=4`, `hc_sinkhorn_iters=20`
- C4 indexer: `index_n_heads=64`, `index_topk=512`
- MoE: 256 experts, `moe_intermediate_size=2048`, `num_experts_per_tok=6`, `n_shared_experts=1`
- RoPE yarn: `factor=16`, `original_max_position_embeddings=65536`, `beta_fast=32`, `beta_slow=1`
- 每层 `compress_ratios`:44 元素 per-layer list `[0,0,4,128,...]`

V4 相对 V3/V2 的**新增结构**(决定了 NPU 适配的新坑):**hash-coding sinkhorn 路由**、**C4 多级压缩 KV(c4/c128)**、
**lightning indexer(稀疏 topk)**、**o_lora 分组输出投影**、**compressor(NSA 风格压缩注意力)**。

---

## 2. 推理侧 (SGLang) — 坑 + 解法 + walkaround/production 分类

SGLang trunk 已有 `deepseek_v4.py`(2259 行)+ `EntryClass=[DeepseekV4ForCausalLM]`,**模型类本身存在**;
坑全在 **NPU 后端(sgl-kernel-npu / AscendAttnBackend)缺 V4-specific 的 hook 和 op**。

### 2.1 已验证里程碑
- **generate() PASS**:真 `DeepseekV4ForCausalLM` + 真 V4 减层 fab,bf16,A3 NPU,0.9s 返回非空。
- **RL loop PASS**:rollout→`update_weights_from_tensor`(attn-only)→re-rollout,5/5 步 weight-sync 都改变了输出。

### 2.2 坑分类表(14 gap)

| # | 坑 | 当前解法 | 分类 | 上游真修 |
|---|---|---|---|---|
| 1 | `AscendAttnBackend._maybe_upgrade_forward_metadata` 缺失 | no-op stub | ⚠️ **walkaround** | sgl-kernel-npu 补 V4 metadata 升级 |
| 2 | `mhc.hc_split_sinkhorn` 要 tilelang/CUDA | torch fallback | ⚠️ walkaround(推理侧)/ ✅ 训练侧已有 AscendC kernel | 见 §3 sinkhorn |
| 3 | `jit_kernel/dsv4/elementwise.fused_q_norm_rope` JIT CUDA | RMSNorm 部分→`npu_rms_norm`(**bit-exact**),RoPE 部分→fp32 torch 复数乘 | ✅ **production**(RMSNorm native 替换,已实测 0.000e+00) + ⚠️ RoPE 保留 fp32 torch(见说明) | RoPE 不换是**正确选择**:`npu_apply_rotary_pos_emb` 是 rotate-half 约定,≠ V4 interleaved 复数乘,差 4.22 |
| 4 | `fused_rope_inplace` / `fused_norm_rope_inplace` 同上 | 同 #3 | 同 #3 | 同 #3 |
| 5 | NPU `aclnnIndex` 不支持 complex64 | index 实部,view-as-complex 之后 | ⚠️ walkaround | 提 NPU complex64 aclnnIndex issue |
| 6 | `fused_k_norm_rope_flashmla` 写 FP8-packed 字节进 paged kvcache | PoC 跳过 scatter(1 层 max_tokens=2 够短) | ⚠️ **walkaround(PoC-only)** | sgl-kernel-npu 补 FP8 packed kvcache scatter |
| 7 | `forward_c4_indexer` 缺失 | no-op stub | ⚠️ walkaround | sgl-kernel-npu 补 c4 indexer |
| 8 | `forward_core_compressor` 缺失 | no-op stub | ⚠️ walkaround | sgl-kernel-npu 补 compressor |
| 9 | `store_cache`/`forward_compress`/`init_forward_metadata_indexer` 缺失 | no-op stub | ⚠️ walkaround | sgl-kernel-npu 补 |
| 10 | V4 传 `compress_ratio` kwarg,NPU `forward_extend` 不收 | `**kwargs` 吸收 | ✅ **production**(back-compat,安全) | PR `**kwargs` 到 forward_extend/decode |
| 11 | `DeepSeekV4TokenToKVPool.get_key_buffer` raise NotImplementedError | V4 dense short-circuit | ⚠️ walkaround | sgl-kernel-npu 实现 V4 KV pool get_key_buffer |
| 12 | `forward_decode` 同 compress_ratio + cache-read mismatch | 同 V4 dense short-circuit | ⚠️ walkaround | 同 #11 |
| 13 | `jit_kernel/dsv4/moe.silu_and_mul_clamp` JIT CUDA | `npu_clipped_swiglu`(实测 ≤0.77% rel,clamp 内 bit-exact) | ✅ **production**(native op 已替换 + e2e gate 验过) | PR native-first + torch-fallback 守卫 |
| 14 | `jit_kernel/dsv4/gemm.linear_bf16_fp32` 用 `torch.mm(out_dtype=fp32)` NPU 不支持 | drop kwarg + `.float()` cast | ✅ **production**(等价) | PR NPU-path `.float()` |

### 2.3 production vs walkaround 小结(推理侧)
- ✅ **production-ready**(已实测等价 + e2e gate):#3 RMSNorm(bit-exact)、#10 kwargs back-compat、#13 swiglu native、#14 gemm cast。这 4 个适合直接 PR。
- ✅ **设计正确的保留**:#3/#4 RoPE 保留 fp32 torch 复数乘——native kernel 数值错(约定不符),fp32 反而更准,不是妥协。
- ⚠️ **walkaround / PoC-only**(必须上游补,不能 ship):#1/#5/#6/#7/#8/#9/#11/#12——这 8 个是 `AscendAttnBackend` 缺的 V4 hook + V4 KV pool,no-op stub 只够让 1 层短序列 PoC 跑形状,**生产必须由 sgl-kernel-npu 真实现**。
- RL loop 的 `update_weights_from_tensor`(attn-only)绕开 #26794 MoE reload bug——✅ plumbing production-ready,但 ⚠️ weight delta 当前是 synth(占位),真 e2e 要换 miles 训练梯度。

---

## 3. 训练侧 (Megatron + miles) — 坑 + 解法 + walkaround/production 分类

训练侧 = 真 DSV4 model layer(MLA + compressor + indexer + sparse attention + hash-coding MoE 路由)
在 **Megatron-on-NPU**(MindSpeed `core_r0.16.0` 适配)上做 fwd+bwd+optimizer。

### 3.1 已验证里程碑(全部真 config,实测 NPU)
- ✅ 单 `DeepSeekV4Attention` megatron layer **fwd+bwd 训练步**(MLA+稀疏注意力+indexer+compressor)— **protected/tagged**
- ✅ 真 DSV4 config 减层 **1 层完整训练迭代**(forward+loss+backward+**AdamW.step**,4.42B,grad finite 526/526)
- ✅ 真 DSV4 config **2 层 fwd+bwd**(8.84B,grad_norm=0.043,1051/1051 finite)

### 3.2 算子映射 —— 关键纠正:CANN 已覆盖,**不需要手写 op-gen**

> 本 session 的核心纠正(owner 指出):V4 训练侧这些是**基本算子,CANN 已提供**。
> 先查 CANN-native / tilelang-ascend 覆盖,查不到再 op-gen。实测结果:**核心算子 CANN 全有**。

| V4 训练算子 | CANN-native 对应 | 状态 | 分类 |
|---|---|---|---|
| sparse-MLA fwd/bwd | `npu_nsa_select_attention`(D_qk=192/D_v=128,select_block=64,count=16,返回 attn+softmax max/sum 供 bwd) | ✅ 实测 | ✅ **production**(CANN-native) |
| C4 indexer | `npu_lightning_indexer` / `npu_sparse_lightning_indexer_grad_kl_loss` | ✅ 实测 | ✅ production |
| compressor | `npu_nsa_compress_attention` | ✅ 实测 | ✅ production |
| MLA prep | `npu_mla_prolog_v3` | ✅ 实测 | ✅ production |
| rms_norm | `npu_rms_norm` | ✅ 实测(bit-exact) | ✅ production |
| **hash-coding sinkhorn** | 无 native 对应 | ✅ AscendC op-gen 完成(28/28+28/28,perf 5.34× symmetric,honesty-gate 纠正后) | ✅ production(精度验过) |
| **act_quant (fp8)** | 无 native 对应 | ✅ AscendC op-gen 完成(24/24+24/24 byte-exact fp8 + bit-exact fp32) | ✅ production |

**结论**:训练侧算子**已无盲区** —— 核心 5 个走 CANN-native(全实测),CANN 真没有的 2 个(sinkhorn/act_quant)
已用 op-gen 生成且精度验过。这推翻了"V4 ops 都要手写 op-gen kernel"的早期误判。

### 3.3 集成层的坑(让 megatron layer 在 NPU 上 fwd+bwd 跑起来)

| 坑 | 解法 | 分类 | 上游真修 |
|---|---|---|---|
| `import mindspeed.megatron_adaptor` 适配 Mcore 0.16 | 用 `core_r0.16.0` 分支(commit 8bf0959,dsa TND support) | ✅ **production**(官方分支已支持) | 已是 MindSpeed 官方路径 |
| `npu_rms_norm` "too many args" + dtype mismatch | rms shim:match gamma dtype + drop 多余 args | ⚠️ **walkaround**(driver shim) | MindSpeed 补 rms_norm 签名适配 |
| TransformerLayer 契约 `return output` vs `(output, None)` | patch `DeepSeekV4Attention.forward` 返回 `(output, None)` | ⚠️ walkaround(miles 侧契约对齐) | miles PR |
| `all_reduce_grad_fp32` kwarg skew(megatron-fork) | Megatron-LM-miles fork patch(`_CopyToModelParallelRegion` 收 + fp32 grad all-reduce) | ⚠️ walkaround → ✅ PR-ready | Megatron-LM-miles PR(已 cold-import 验证) |
| **2 层 backward nan** | `sparse_attn_torch` all-masked-row softmax 稳定化:`nan_to_num(scores_max,neginf=0)` + `clamp(exp args,max=30)`(nan 282→7→0) | ✅ **production**(标准 masked-softmax guard,PR-worthy) | miles PR |
| `miles.utils` ModuleNotFoundError(MoE router) | 装全量 miles pkg `/opt/miles_full` | ⚠️ env(非代码) | 文档化依赖 |
| MoE 路由用 fp8_simulate(torch_npu 缺 Float8_e4m3fn cast) | fp8-grid sim in fp32(`_fp8_e4m3_round`) | ⚠️ walkaround | torch_npu 补 Float8_e4m3fn cast(或 act_quant AscendC kernel 替) |

### 3.4 显存边界(单张 61GB 芯片,实测)

| 规模 | 结果 |
|---|---|
| 1 层(4.42B)+ AdamW | ✅ fwd+bwd+optim **全跑通**(完整训练迭代) |
| 2 层(8.84B) | ✅ fwd+bwd 跑通;⚠️ +AdamW states 就 **OOM**(8.84B + Adam m/v ~70GB) |
| 4 层(17.68B) | forward OK;⚠️ backward **OOM** |

更深 = **张量/流水并行 或 activation checkpointing**(256-expert MoE 在 4096 hidden 每层巨大,标准大模型显存现实)。
这是**分布式工程,不是 NPU/算子问题**。

### 3.5 production vs walkaround 小结(训练侧)
- ✅ **production-ready**:CANN-native 5 算子映射(全实测)、sinkhorn/act_quant AscendC kernel(精度验过)、sparse-softmax 稳定化补丁、MindSpeed `core_r0.16.0`(官方分支)。
- ⚠️ **walkaround / 待上游**:rms_norm shim(MindSpeed 签名)、`(output,None)` 契约、megatron fp32-grad patch(PR-ready)、fp8_simulate(torch_npu 缺 cast)。
- 显存 OOM 不是 walkaround,是**还没做的分布式工程**。

---

## 4. PR 列表(汇总)

### 4.1 已 prepared / 已 landed

| Target | Branch/Ref | Commit/ID | 状态 | 类型 |
|---|---|---|---|---|
| `radixark/miles` | `zhshgmail/miles npu-tilelang-ops` | `d03db2c` | audit-clean,待 user `gh pr create` | 训练侧 V4 ops NPU path |
| `radixark/Megatron-LM`(via miles vendored) | `Megatron-LM-miles fix/te_general_gemm_npu_fallback` | `6f3209b` | cold-import 验证,bundle 进 miles PR set | megatron NPU fallback |
| `Ascend/AscendNPU-IR` issue #251 | issue comment | comment `1.73358592e+08` | **landed** | bishengir HIVM pass softmax 累加器 bug(Huawei 补 C++) |
| `tile-ai/tilelang-mlir-ascend` PR #80 | `zhshgmail/... npuir-check-ub-budget` | `df7431e` | CI 全绿,MERGEABLE,待 maintainer | tilelang-mlir UB budget check |
| `a5_ops`(harness) | task#33 | `eefeaeca` | **merged** | perf-capture canonical N/A + FA-gate |

### 4.2 本报告新识别的 PR 候选(批量 e2e 后开,per held-PR 纪律)

**推理侧 → `sgl-project/sglang` + `sgl-project/sgl-kernel-npu`**:
1. ✅ production 4 个直接可 PR:RMSNorm native(#3)、`**kwargs` back-compat(#10)、swiglu native(#13)、gemm `.float()`(#14)
2. ⚠️ 一个 clean issue:"V4 (`DeepseekV4ForCausalLM`) on `device=npu` 缺 N 个 AscendAttnBackend V4 hook"(#1/#7/#8/#9/#11/#12,每个带 line-number + PoC-workaround 证据)
3. NPU complex64 aclnnIndex issue(#5)
4. FP8 packed kvcache scatter(#6)

**训练侧 → `radixark/miles` + `Ascend/MindSpeed` + `Ascend/pytorch(torch_npu)`**:
5. ✅ sparse-softmax 稳定化补丁(§3.3 nan fix)→ miles
6. ⚠️ MindSpeed rms_norm 签名适配 + `core_r0.16.0` V4 layer 契约
7. ⚠️ torch_npu Float8_e4m3fn cast 缺失(fp8_simulate 当前绕)

> **纪律**:训练侧 PR 批量在**真 e2e**(真 miles 训练 delta,非 synth)之后一次开,不 piecemeal。
> issue body 草稿:`workspace/v4_attempt_2026_06_01/UPSTREAM_ISSUE_sglang_v4_npu.md`。

---

## 5. 诚实边界(不能对外说的话)

- ❌ "DSv4-Flash 在 NPU 上 production-ready" —— 推理侧 8 个 no-op stub 在路径上;训练侧只到 2 层 fwd+bwd / 1 层完整迭代(减层)。
- ❌ "数值正确性已验证" —— 推理 PoC 输出是形状正确的 gibberish(减层 + 随机权重 + stub)。
- ❌ "全模型 PASS" —— 减层(1–2 层 vs 43 层)。
- ❌ "真 RL 训练 e2e 跑通" —— RL loop 的 weight delta 当前是 synth 占位,不是 miles 真训练梯度。

## 6. 可以诚实说的话

- ✅ "真 V4 model class(SGLang `DeepseekV4ForCausalLM`)在 A3 NPU bf16 端到端执行,1 层减层 fab + 14 个文档化 PoC workaround"。
- ✅ "RL 循环 plumbing(rollout→weight-sync→re-rollout)闭环验证,weight-sync 真改变 inference"。
- ✅ "真 DSV4 config 减层 1 层完整训练迭代(fwd+bwd+optimizer)+ 2 层 fwd+bwd 在 NPU 跑通,所有梯度 finite"。
- ✅ "V4 训练侧算子在 NPU 上无盲区:核心 5 个 CANN-native(实测),CANN 缺的 2 个 AscendC kernel 精度验过"。
- ✅ "每个 workaround 对应一个具体的 sgl-kernel-npu / MindSpeed / torch_npu 上游 gap;PoC 是上游 gap inventory 的 forcing function,不是交付物"。

---

## 7. 关键文件索引(disk ground truth)

- 推理 PoC + 14 workaround:`workspace/v4_attempt_2026_06_01/README.md` + `_*_PASS.py`/`native_op_snapshots/`
- 训练侧算子 gap:`workspace/v4_attempt_2026_06_01/V4_TRAINING_SIDE_OP_GAP.md`
- 训练侧 NPU 集成 artifact:`workspace/v4_attempt_2026_06_01/npu_native_shims/`(含 protected flash result)
- 真 config:`workspace/T32_tilelang_rescue/v4_real_truth/v4_real_config.json`
- SGLang 上游 issue 草稿:`workspace/v4_attempt_2026_06_01/UPSTREAM_ISSUE_sglang_v4_npu.md`
- git tags:`v4-flash-attention-npu-working`、`v4-real-config-1layer-training-step-npu`
