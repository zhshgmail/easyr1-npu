# 评估：tilelang-0.1.8-移植 vs CANN-替换 —— 哪条走得通（2026-06-03，基于本 session 实测）

## 两条路逐项对比（推理侧,性能恢复目标）

| 维度 | CANN-替换 | tilelang-0.1.8-port |
|---|---|---|
| 最重 kernel(attention) | ✅ CANN `npu_sparse_flash_attention` 现成存在(AscendAttnBackend)、接口兼容;需接 indexer→topk dataflow + V4 KV-pool(upstream-sglang 改,**纯 open-layer Python**) | tilelang attention kernel 编译,需 0.1.8 DSL + 撞过的 reduce/bf16 闭源墙 |
| indexer | ✅ CANN `npu_lightning_indexer` 存在(DSA path,需接进 V4) | tilelang/triton |
| rope | ✅ CANN `npu_rotary_mul` 存在 | tilelang/triton |
| mhc | ❌ 无 CANN op,**但有纯 torch fallback**(`hc_pre_torch_impl`:900 / `hc_post_torch_impl`:1016,已在用,功能 OK 性能低) | tilelang(原生) |
| compressor | ❌ 无 CANN,NPU 上 stub/torch 兜底 | tilelang/triton |
| fp8 quant_k_cache | ❌ 无 CANN,但 **bf16 路径可跳过**(只 kv_cache_dtype=fp8 才需) | triton |
| **是否撞闭源墙** | ❌ **不撞**(全 open-layer:torch_npu CANN op + Python dataflow 接线) | ✅ **撞**(本 session 实测:reduce 语义需 C++ codegen、bf16 vmul 卡闭源 hivmc 0.1.0,**open 层修不动**) |
| 工作量级 | 上游 sglang dataflow 接线(indexer→topk→forward_sparse + V4 KV-pool),bounded,几轮 | 1212-commit TVM rebase + 适配 ~10 codegen cc + bishengir 重编 + 闭源墙绕不过 |
| 已验证的功能底座 | ✅ 推理已在 torch-fallback 下功能跑通(产 token);CANN 是在此之上替性能关键 kernel | 需先把整个 tilelang-ascend rebase 到 0.1.8 才谈得上 |

## 结论：CANN-替换更走得通

**理由(都来自本 session 实测,非推断):**
1. **CANN 路全在 open layer**:torch_npu 的 CANN 算子(npu_sparse_flash_attention / npu_lightning_indexer / npu_rotary_mul)+ Python dataflow 接线,**不碰闭源 bishengir/hivmc**。而 tilelang-port 这 session 实测撞了两堵闭源墙(reduce 要 C++ codegen、bf16 vmul 卡 hivmc 0.1.0 verifier,open 层修不动)—— **tilelang-port 有我已确认的硬天花板。**
2. **最重的 attention 有 CANN 路**(性能价值最大的那个),且接口兼容。
3. **没 CANN 的那几个(mhc/compressor/fp8)都有退路**:mhc 有 torch fallback(已在用)、compressor NPU 上本就 stub/torch、fp8 在 bf16 路径可跳。即"CANN 覆盖不全"不阻塞 —— 关键 kernel 用 CANN,边角用 torch。
4. **功能底座已在 CANN/torch 侧**(推理 torch-fallback 已产 token),CANN-替换是增量替性能 kernel;tilelang-port 要先做完整 rebase 才有任何产出。

**但诚实标注必要性边界:**
- CANN-替换"走得通"指**路径无闭源墙、增量可做**;它仍需实做 indexer→topk dataflow + V4 KV-pool 接线(几轮 upstream-sglang 改 + 实测),**尚未跑出"走到 npu_sparse_flash_attention"的证据**(尝试 1 的 flag-only 已证伪)。
- 它**不能覆盖 mhc/compressor 的高性能**(那俩无 CANN,只能 torch 兜底或 triton)—— 若这俩是性能瓶颈,CANN 路也留缺口;但 attention 通常是大头。
- tilelang-port 的唯一独有价值 = 给 mhc/compressor 也上高性能 kernel + 跟 miles 完全对齐;代价是 rebase + 闭源墙。

## 一句话
**CANN-替换走得通(open-layer、无闭源墙、关键 kernel 有 CANN、边角有 torch 退路),是更稳的路;tilelang-port 撞已确认的闭源墙(reduce/bf16),天花板硬。建议优先 CANN-替换接 attention,把 tilelang-port 留作 mhc/compressor 高性能或完全对齐 miles 时再评估。** —— 全部基于本 session 实测,不是拍脑袋。

## 修正(2026-06-03,owner push "差异没那么大?"——我之前往悲观偏):tilelang-port 难度被我夸大

仔细看 NPU codegen(~9660 行 C++,`src/target/codegen_npuir*.cc`)实际 TVM-API 依赖:
- 绝大部分是 **`tvm::tl::Npuir*`**(fork 自有命名空间算子,跟 codegen 走,不受 mainline 漂移影响)。
- 少数 mainline 用 `tvm::PrimExpr` / `tvm::tir::{IntImm,Var,FloatImm}Node` / `tvm::arith::analyzer` / `tvm::runtime::registry`(极稳定核心类型,跨版本几乎不变)。
→ **真实"会 break 的接口面"远小于 1212-commit。** "1212 commit"是原始距离,不代表 NPU-codegen 相关的 API 破坏。我之前用 commit 数当难度 = 夸大。

**两条路的真实区别(修正后,分两层别混):**
1. **TVM-版本对齐**:tilelang-port 这层可能比我说的轻(fork 自有 op + 少量 Python DSL backport,我已做 2 个 proxy fix);不是全量 rebase。
2. **闭源 bishengir/hivmc 墙**:独立、真实,与 TVM 版本无关。本 session 实测撞:reduce 语义需 C++ codegen 改、bf16 vmul 卡死闭源 hivmc 0.1.0 verifier(open 层修不动)。**这是 tilelang-port 的真天花板**;升级 CANN 9.1.0 可能解可能不解(取决于新 hivmc 是否支持这些 op)。
3. CANN-替换仍不撞这堵墙(走 torch_npu 算子)。

**最公允**:两条路差距没我上一版说的那么悬殊。tilelang-port 的版本对齐可控;真正变量 = 闭源 hivmc 对 V4 推理 kernel 用到的 op(reduce/bf16/sinkhorn 等)支不支持 —— **这个没逐 op 验过(只验了 bf16/reduce 撞墙),需要 PoC 实测才知道"差异到底多大"。**

## tilelang-port PoC（owner 要求,见下方 PoC 记录文件）
PoC 定义:拿 V4 推理实际用的 tilelang kernel(mhc `hc_split_sinkhorn_kernel`)在 fork(带我 2 个 proxy backport)上真编一遍,判定:① Python DSL backport 能搞定(轻,tilelang-port 可行)② 撞闭源 hivmc(重,天花板硬)。结果记于 `TILELANG_PORT_POC_<date>.md`。
