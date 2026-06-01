# V4 训练侧算子 — tilelang-ascend pivot(2026-06-01 owner-corrected strategy)

## 核心纠正(owner 2026-06-01 ~16:14-16:29Z)
之前的 op-gen 手写 AscendC 路线**是错的第一步**。owner 连续纠正:
1. "这都是基本算子,cann 应该都包括啊" — 别盲目手写。
2. "我们第一个任务不就是修 tilelang ascend 的 bug 么?tilelang 的能都在 tilelang-ascend 上跑?" — TileLang 算子的正路是 **tilelang-ascend 后端编译**,不是 op-gen。
3. "tilelang 编译器只有你最清楚" — 这块我(blue)是 expert,自己解,别甩 team。
4. "看看 tilelang op 和 cann 的 op 能不能完全匹配上。看看性能差异(找能跑的)" — 当前 directive。

## 事实(verified)
- **6 个 V4 训练 kernel 全是 `@tilelang.jit` TileLang,0 个 raw CUDA**(grep 确认):
  sinkhorn / act_quant / indexer_fwd / indexer_bwd / sparse_mla_fwd / sparse_mla_bwd
  (都在 `miles_plugins/models/deepseek_v4/ops/kernel/`)
- TileLang 是 backend-portable DSL(默认 CUDA 后端,可换 Ascend MLIR 后端)。
- **tilelang-mlir-ascend(T32 建,tlrescue 容器 `/home/z00637938/workspace/tilelang-mlir-ascend/`,v0.1.1.030,MLIR 后端)已经有 V4 类算子的 example,且实测在 NPU 上跑通**:

| tilelang-ascend example | = 我的 V4 op | NPU 实测 |
|---|---|---|
| `examples/sparse_mla_fwd.py` (+_dynamic_shape) | sparse_mla_fwd(最难) | ✅ **PASS**(`All check passed!` rtol=5e-3 atol=1e-2, device=npu:0, EXIT=0) |
| `examples/fp8_lighting_indexer.py` | indexer_fwd | ✅ **PASS**(causal -inf mask 输出, assert_close rtol=3e-2, EXIT=0) |
| `examples/flash_attn_npuir.py` | attention 基座 | 待确认 |
| `examples/norm/example_rms_norm.py` | sinkhorn/compressor 的 norm | 待确认 |
| `examples/gemm/*`, `exp2.py`, `log2.py`, `elementwise/*` | sinkhorn/act_quant primitive | 待确认 |

## API gap(miles 源码 → 我们的 Ascend 后端)
- miles 写法:`T.Tensor[(n, mix_hc), FP32]`(**方括号下标**)+ `T.symbolic("n")`
- 我们 v0.1.1.030 后端:`T.Tensor((shape), dtype)`(**圆括号调用**);`T.symbolic` **支持**(example 在用)
- 报错:`tir.Buffer ctor: Expected Array[PrimExpr], got Array[index 0: Array]` = 方括号 vs 圆括号的 shape 传法差异
- → 直接编 miles 源码失败,但 tilelang-ascend **自带的 example 版本能跑**(用对了 API)。所以**用 example 版,不用 adapt miles 源码**(除非要 miles 的精确 spec/shape)。

## op-gen 手写的产出(现在理解为 redundant-but-not-wasted)
- sinkhorn / act_quant / indexer_fwd 三个我 op-gen 手写了 AscendC(都 PASS 精度)。**对"V4 在 NPU 跑通"是绕路**(tilelang-ascend example 现成能跑)。
- **但不是白干**:① 真 kernel + 精度验证 ② 趟出并 fix 了 3 个 a5_ops harness bug(kw_brief FA-gate / perf-capture canonical-N/A / O5-sync-timeout env)+ 4 个 backlog repro ③ probe 钉死的 Nd2Nz srcDValue uint16 overflow 是给 triton FA cube 的高价值 KB pattern。

## 正路(当前)— owner directive: tilelang-ascend vs CANN match + perf
1. 逐个确认 tilelang-ascend example 在 NPU 跑通(sparse_mla ✅ / indexer ✅ / 其余待跑)
2. 找每个的 **CANN 等价**:sparse_mla → `FlashAttentionScore`/FA 族;indexer → aclnn matmul 组合;rms_norm → `aclnnRmsNorm`;act_quant → `aclnnDynamicQuant`
3. **对比**:算的是不是一回事(match)+ 性能差(tilelang-ascend vs CANN)→ 选快的/能跑的接进 miles V4 训练
4. miles 训练侧把这些 op 的实现指向 tilelang-ascend(换后端)或 CANN — 让 V4 训练在 NPU e2e 跑起来
