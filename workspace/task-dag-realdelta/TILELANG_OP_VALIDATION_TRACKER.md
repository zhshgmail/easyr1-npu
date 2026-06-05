# tilelang-ascend 算子有效性验证 —— 状态追踪

Standing mandate: 系统性验证每个 example kernel 的正确性(相对 pytorch/CANN 参考),修真 bug,开 PR。
诚实分类:real bug / invalid-config / harness-error / fp16-noise。环境:tlrescue 容器,A3 NPU(1 chip)。

验证方法:跑 example 的 `__main__`(确认它把 swept param 传给 kernel + data/reference 两边,
见 [[verify_harness_propagates_swept_param]]),比对其内置参考;dtype/shape 扫描;失败先过 discriminator。

## 状态表

| Example | 路径 | 状态 | 备注 |
|---|---|---|---|
| exp2 | examples/exp2.py | ⚠ bf16-blocked | fp16 PASS;bf16 撞 HIVM vmul verifier(→ PR #1199)。fp16 基线对 |
| log2 | examples/log2.py | ✅ PASS | accuracy check passed (2026-06-02) |
| flash_attn_npuir | examples/flash_attn_npuir.py | ⚠ bf16-wrong | fp16 PASS;bf16 输出错(根因 = vmul/vexp bf16 verifier, #253/#1199) |
| flash_attn_npuir_dev | examples/flash_attn_npuir_dev.py | ✅ PASS | All check passed (fp16) (2026-06-02) |
| fp8_lighting_indexer | examples/fp8_lighting_indexer.py | ✅ PASS(h=4/8/16/32) | 早先"h<32 bug"是 harness 硬编码 H=32 假象,已纠正 |
| mixcv_mixkernel | examples/mixcv_mixkernel.py | ✅ PASS | all checks passed (2026-06-02) |
| sparse_mla_fwd | examples/sparse_mla_fwd.py | ✅ FIXED | heads<block_H 写越界,PR #96(verified heads=4/8/16/32) |
| sparse_mla_fwd_dynamic_shape | examples/sparse_mla_fwd_dynamic_shape.py | ✅ PASS | case_0/1/2 all passed (2026-06-02) |
| vectorization_in_parallel | examples/vectorization_in_parallel.py | ✅ PASS | all 5 checks passed (2026-06-02) |
| dsv4/act_quant | examples/deepseek_v4/example_act_quant_kernel.py | ✅ PASS | Comparison passed (2026-06-02) |
| dsv4/fp8_gemm | examples/deepseek_v4/example_fp8_gemm_kernel.py | ✅ PASS | FP8 GEMM NPU test passed (2026-06-02) |
| dsv4/hc_split_sinkhorn | examples/deepseek_v4/example_hc_split_sinkhorn_kernel.py | ✅ PASS | pre/post/comb checks passed (2026-06-02) |
| dsv4/lighting_indexer_fwd | examples/deepseek_v4/example_lighting_indexer_fwd_kernel.py | ✅ PASS | max abs err 0.000000, PASS (2026-06-02) |
| dsv4/lighting_indexer_bwd | examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py | ✅ PASS | dQ=0.00013 dKV=0.00004 dW=0; nan rows=[] (clean). NOW COMPILES (T33 P1.6 bishengir-compile failure resolved in current toolchain) (2026-06-02) |
| dsv4/mhc_post | examples/deepseek_v4/example_mhc_post.py | ✅ PASS | out check passed (2026-06-02) |
| dsv4/miles_indexer_integration | examples/deepseek_v4/example_miles_indexer_integration.py | 🔴 REAL BUG (dk) | dq/dw correct (0==0) but **dk gradient garbage: 8.23e+33 vs ref 0.285**, max_abs_err dk=8.23e+33. Per-seq-position kernel_s1 call (S=1 slice) dk path via atomic_addx4 writes garbage. Standalone bwd (multi-seq, one call) is clean → bug is in per-position S=1 dk path. Related to documented R-KA-15 atomic_addx4 0-grad garbage (workaround present but insufficient). NEEDS isolation: kernel_s1 alone vs accumulation. (2026-06-02) |
| dsv4/sparse_attn | examples/deepseek_v4/example_sparse_attn_kernel.py | ✅ PASS | All check passed (2026-06-02) |
| dsv4/sparse_attn_highperf | examples/deepseek_v4/example_sparse_attn_kernel_highperf.py | ✅ PASS | case_0/1 + all checks passed (2026-06-02) |
| dsv4/sparse_mla_bwd | examples/deepseek_v4/example_sparse_mla_bwd_kernel.py | ⚠ partial | T33 P1.4 topk=16 撞闭源 bisheng clang(exit 70);topk=8 OK |
| dsv4/sparse_mla_fwd | examples/deepseek_v4/example_sparse_mla_fwd_kernel.py | ✅ (= PR #96 同源) | 与顶层 sparse_mla_fwd 同算法 |
| dsv4/_r_ka_13_repro | examples/deepseek_v4/_r_ka_13_repro.py | (repro) | R-KA-13 复现脚本,非待验算子 |

## 已开 PR / issue
- `tile-ai/tilelang-mlir-ascend` PR #96 — sparse_mla heads<block_H 修复(verified)
- `tile-ai/tilelang-mlir-ascend` issue #97 — flash_attn bf16 wrong output(根因转 #253）
- `Ascend/AscendNPU-IR` issue #253 — HIVM vmul/vexp 漏 BF16(根因）
- `Ascend/AscendNPU-IR` PR #1199 — 补 BF16 修复(verifier 放行已验，数值待维护方验)
- `tile-ai/tilelang-mlir-ascend` issue #99 — indexer-bwd dk 垃圾值(8e33),完整集成才复现,8 假设排除链;非 V4 blocker(V4 走 CANN-native indexer)

## 下一步
按表跑 ❓ 未验的 fp16/fp32 基线(bf16 已知 blocked,不重复)。优先简单的 log2 / vectorization /
mixcv 先把"健康面"做实,再逐个 dsv4 算子。
