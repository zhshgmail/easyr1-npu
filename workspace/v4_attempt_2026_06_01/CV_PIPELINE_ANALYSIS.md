# V4 14-gap kernels — Cube/Vector pipeline classification

A3 (Ascend 910C) 架构分两路:
- **Cube unit** — 矩阵 / GEMM 类(`Mmad`, `matmul`)
- **Vector unit** — 逐元素 + reduce + 激活(`Add`, `Mul`, `Exp`, `Sigmoid`, `Sum`, `Max`, `Softmax`, `RmsNorm`, `Cast`)
- **Cube+Vector 融合(CV pipeline)** — gemm 后跟 epilogue(scale / bias / activation / quant);需要 cube 算完 → vector 并行流水;A5 ops 生成能力较弱
- **Scalar / control** — 索引、mask、loop control

## 14-gap 分类

| # | Gap kernel | 分类 | A5 ops 可生成? | Native NPU 替代 | torch fallback 是否复杂(可能引发 CPU drop) |
|---|---|---|---|---|---|
| 1 | `_maybe_upgrade_forward_metadata` | **元数据 / Python 控制流**(非算子)| N/A | 永久 no-op stub | N/A |
| 2 | `hc_split_sinkhorn` | **Vector-only**(sigmoid + reduce_sum + reduce_max + exp + div + 多次 sinkhorn 迭代);没有 matmul | ✅ Vector 算子,A5 ops 可生成 | 无 native 单算子;`torch_npu.npu_clipped_swiglu` 不是这个语义 | torch fallback 全在 NPU 上(reduce/exp/div NPU 都支持),但**性能差** + 数值精度低 |
| 3 | `fused_q_norm_rope` | **CV 融合**: rmsnorm(vector reduce_sum+rsqrt+mul) + rope(vector mul+swap) 写入 `q_output`;有 epilogue 但无 cube | ✅ **纯 Vector**,不是 CV 融合;A5 ops 可处理 | `torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin, layout)` 涵盖 RoPE 部分;rmsnorm 用 `torch_npu.npu_rms_norm`;**两者级联即可 native NPU** | 现 torch fallback 用 `view_as_complex` + 复数乘 — 已验证 NPU 上跑(刚 prove `xc * xc on npu:0`),但**多个 op 跨 stream 可能引入额外 d2d 同步** |
| 4 | `fused_rope_inplace` | **Vector-only**: 纯 rope; 无 cube,无 rmsnorm | ✅ Vector,A5 ops 可生成 | `torch_npu.npu_apply_rotary_pos_emb` 直接对应 | 同上 — 现 fallback NPU 跑通 |
| 5 | `fused_norm_rope_inplace` | **Vector-only**: rmsnorm + rope inplace | ✅ Vector | `torch_npu.npu_rms_norm` + `torch_npu.npu_apply_rotary_pos_emb`,或 `torch_npu.npu_kv_rmsnorm_rope_cache_v2`(已经 fused!)| 同 #3 |
| 6 | `fused_k_norm_rope_flashmla` | **Vector + Scatter / addr 计算**: rmsnorm + rope + 算 page 地址 + scatter 到 paged KV cache | 部分 Vector,部分 scalar 地址;CV 没有 cube | ⚠ 部分:**`torch_npu.npu_kv_rmsnorm_rope_cache_v2` 直接是这个 op**(fused rmsnorm + rope + 写 KV cache,目标格式可配置)— 但只支持 sglang 的 packed FP8/BF16 layout 时需 layout adapter | 当前 PoC 跳过 scatter,torch fallback 的 rmsnorm + rope 是 NPU 跑 |
| 7 | NPU `aclnnIndex` complex64 不支持 | **torch_npu 算子 gap**(aclnn 算子库覆盖问题)| N/A | workaround: index real domain after | NPU 上跑 |
| 8 | `forward_c4_indexer` | **Python 控制流 + 多个 sub-kernel**:fp8 Q 量化、Hadamard、RoPE、paged write | **不是单一算子**,是 dispatch hook | 永久 no-op stub(PoC) / 完整 production 实现需多个 native NPU op:`torch_npu.npu_dequant_rope_quant_kvcache` + Hadamard + scatter | N/A,本 PoC 直接 no-op |
| 9 | `forward_core_compressor` | 同上 | dispatch hook,不是单一算子 | no-op | N/A |
| 10 | `store_cache` / `forward_compress` / `init_forward_metadata_indexer` / `forward_compressor` | 同上 | dispatch hook | no-op | N/A |
| 11 | `forward_extend` kwarg ABI 不匹配 | **Python ABI 问题** | N/A | `**kwargs` 吸收 | N/A |
| 12 | V4 KV pool API NotImplementedError | **存储 API 不兼容**,不是算子 | N/A | V4 dense short-circuit | N/A |
| 13 | `forward_decode` 同 #11 #12 | 同上 | N/A | `**kwargs` + dense short-circuit | N/A |
| 14 | `silu_and_mul_clamp` | **Vector-only**: clamp + sigmoid + mul + clamp | ✅ Vector | **`torch_npu.npu_clipped_swiglu`** 直接覆盖语义(`alpha=1.0`/`limit=swiglu_limit`/`bias=0`) | 现 fallback 全 NPU 跑 |
| 14+ | `linear_bf16_fp32` `out_dtype=` 不支持 | **GEMM + cast 融合 = CV 融合**| ⚠ A5 ops 较弱(GEMM+cast 流水线) | `torch_npu.npu_matmul(a, b, out_dtype=fp32)` 可能存在;或两 op 级联 `torch.mm(a, b.t()).float()` | 现 fallback `.float()` cast 显式,NPU 跑 |

## 结论 — A5 ops 适配优先级

**A5 ops 可生成、且没有现成 native NPU op 的(优先级最高,真有价值)**:
- **`hc_split_sinkhorn`** — 纯 vector(sigmoid + reduce_max + exp + reduce_sum + div + sinkhorn 迭代),A5 ops vector 路径可以做;native NPU 库里**没有这个组合算子**;是 V4 hash-coding 的核心,torch fallback 会引入精度损失 + 性能差

**Native NPU 库已经有的(用 native 即可,不用 A5 ops 生成)**:
- `fused_q_norm_rope` → `npu_rms_norm` + `npu_apply_rotary_pos_emb` (或 `npu_kv_rmsnorm_rope_cache_v2` 一步)
- `fused_rope_inplace` → `npu_apply_rotary_pos_emb`
- `fused_norm_rope_inplace` → `npu_kv_rmsnorm_rope_cache_v2` 或 `npu_rms_norm + npu_apply_rotary_pos_emb`
- `fused_k_norm_rope_flashmla`(rmsnorm + rope 部分) → `npu_kv_rmsnorm_rope_cache_v2`(直接对应,packed kv cache write 也含)
- `silu_and_mul_clamp` → `npu_clipped_swiglu`
- `linear_bf16_fp32` → `torch.mm(...).float()` 已经 NPU 上,不需要 A5 ops

**纯 Python / dispatch / ABI 类(不是算子,不需要 A5 ops)**:
- `_maybe_upgrade_forward_metadata`, `forward_c4_indexer`, `forward_core_compressor`, `store_cache`, `forward_compress`, `init_forward_metadata_indexer`, `forward_compressor` — 全部是 V4 attention dispatch hook,no-op stub 是 short-term 正确答案,production 时需要 sglang NPU adapter 团队写完整实现(可以用 native NPU ops 拼装)
- `forward_extend` / `forward_decode` 的 `**kwargs` 吸收 — ABI 问题
- V4 KV pool API — 存储格式问题,不是算子
- aclnnIndex complex64 — torch_npu 算子库覆盖问题(可以 issue 到 torch_npu)

## CV 融合(Cube + Vector pipeline)算子盘点

V4 真正的 **CV 融合** 算子只在 fp8 路径上:
- `tf32_hc_prenorm_gemm` — fp8 GEMM + prenorm
- `fp8_einsum` — fp8 GEMM (Wo)
- `deep_gemm.fp8_einsum` 等

**本 PoC 的 bf16 path 上没有 CV 融合算子**。所有 vector/scalar 类的算子 native NPU 都已经有,或 A5 ops 可以生成。

A5 ops 应该集中在:
1. `hc_split_sinkhorn` 一个 vector 算子(value clear,native NPU 没有)
2. 后续 fp8 path 的 CV 融合算子(当前不在 critical path)
