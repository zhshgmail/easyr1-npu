# V4 推理侧 kernel ↔ 后端可用性映射（回答：推理侧 tilelang 算子在 CANN/torch_npu 上有无替代）

> 方法:read-only 读 sglang 源码(sgl_probe 容器 `/sgl-workspace/sglang/python/sglang/srt/`)。
> **这是代码静态分析,不是 runtime 验证**(标注以防当成已验证结果)。关键一条我亲自 spot-check 过(见底部)。

## 映射表（每个 V4 推理 kernel × 后端）

| 推理 kernel | tilelang | triton | torch_npu/CANN | 默认走哪条 | CANN 替代可用? |
|---|---|---|---|---|---|
| 1. sparse-MLA / 核心 attention | ✗ | ✗ | **npu_sparse_flash_attention 在 `AscendAttnBackend`(ascend_backend.py:256→988)`do_cp_balance_attn`,接口接收 q_nope/q_rope + block_table(paged KV)+ sparse_indices(topk)+ sparse_mode=3 —— 正是 V4 sparse-MLA 的输入** | flash_mla(deepseek_v4_backend.py:1040,NVIDIA) | ⚠️ **接口兼容,是 wiring 问题不是不兼容**(纠正:我之前说"与 V4 不兼容"是错的) |
| 2. compressor | ✗ | ✓(compress_old.py) | **无**(ascend_backend.py:280 V4 compressor hook = no-op stub) | triton | ❌ 无 |
| 3. indexer / C4Indexer | ✗ | ✓(dsv4/indexer.py,jit_kernel.dsv4 topk) | **npu_lightning_indexer 存在**(dsa_indexer.py:1794)**但 DSA-only** | dsv4 C4Indexer(triton,deepseek_v4.py:46) | ⚠ 存在但未对 V4 wired |
| 4. mhc_pre/post | ✓(@tilelang.jit,mhc.py) | ✗ | **无** | tilelang | ❌ 无(只有 hc_pre_torch_impl 兜底参考) |
| 5. rope / fused_q_norm_rope | ✗ | ✓(deepseek_v4_rope.py) | npu_rotary_mul 存在(dsa_indexer.py:1583/1603)**但 DSA-only** | triton | ⚠ 存在但未对 V4 wired |
| 6. quant_k_cache / fp8 | ✗ | ✓(dsv4/quant_k_cache.py) | **无** | triton | ❌ 无 |

## 底线
- **6 个推理 kernel,5 个是 tilelang/triton-locked、没有 wired-in 的 CANN 路径。**
- 存在的 CANN op(`npu_lightning_indexer`、`npu_rotary_mul`、`npu_sparse_flash_attention`)都在 **DSA / 非-DSV4** 代码路径里,**没有接进 V4 模型**。compressor / fp8 / mhc 连存在的 CANN op 都没有。
- **对比训练侧**(6 个 op,4 个有 CANN-native 替代):**推理侧的 CANN 覆盖明显更差** —— 你的判断对。

## 对"tilelang 0.1.8 移植必要性"的含义
- 推理侧要恢复性能,只有两条:① triton 路径(NPU 上 triton-ascend,perf 可变、且本身也有版本/编译坑)② tilelang 路径(需要 0.1.8 DSL 能在 tilelang-ascend 上编 = 那个 port)。
- **CANN-native 替代在推理侧基本不可用(5/6 无),所以"用 CANN 绕开 tilelang"在推理侧走不通** —— 这正是推理侧 tilelang-port 必要性比训练侧高的原因。
- 注:torch-fallback(我复现推理用的那条)能让推理**功能跑通**(已验证产 token),但它走的是 sglang 的纯 torch/triton fallback,**不是高性能路径** —— 功能 PoC ✓,性能 ✗。

## spot-check(我亲自核的关键一条,不全信 subagent)
- `dsv4/indexer.py`(V4 的 C4Indexer):只 import triton + jit_kernel.dsv4,**无 torch_npu/npu_lightning** → V4 indexer 确实 triton-locked。✅
- V4 对 `dsa.dsa_indexer` 的引用只是 `rotate_activation` 辅助函数,**不是 npu_lightning_indexer 那条 CANN 路径** → CANN indexer 确实没接进 V4。✅
- `mhc.py`:`torch_npu`/`npu_` 出现 **0 次**,多个 `@tilelang.jit`(:27/118/250/337)→ **mhc 确实 tilelang-only,无 CANN/torch_npu**。✅ 亲核
- compressor:ascend_backend 的 `forward_compressor`/`forward_core_compressor` 是**显式 no-op stub**("V4 compressor hook stub: no-op on NPU" / "for the bf16 reduced-fab PoC we route through the full attention path")→ **NPU 上无 CANN compressor,被 stub 绕过**。✅ 亲核
- 仅 attention(npu_sparse_flash_attention non-DSV4)、rope/fp8 triton 这几条来自 subagent 代码读,未逐条亲核,但不影响底线结论(关键的 indexer/mhc/compressor 三条已亲核,均无 wired CANN)。

## 结论(spot-check + attention 接口核对后,修正版)
逐 kernel CANN 可用性(亲核):
- **核心 attention(最重的 kernel):CANN 有路 ⚠️** —— `npu_sparse_flash_attention`(AscendAttnBackend)接口接收 V4 的 q_nope/q_rope + paged-KV block_table + topk sparse_indices,**接口兼容**;V4 默认走 flash_mla 没接它,但这是 wiring,不是不兼容。(纠正我之前"CANN 做不了 V4 sparse-MLA"的错误声称。)
- **indexer**:CANN `npu_lightning_indexer` 存在(dsa_indexer),V4 默认走 triton C4Indexer 没接 —— 同样可能是 wiring。
- **rope**:CANN `npu_rotary_mul` 存在(dsa_indexer),V4 默认 triton —— 同样可能 wiring。
- **mhc**:tilelang-only,无 CANN ✅亲核。
- **compressor**:NPU 上 no-op stub,无 CANN compressor ✅亲核。
- **fp8 quant_k_cache**:triton,无 CANN(未亲核,subagent)。

**修正后的真实图景**:推理侧不是"CANN 覆盖≈0"(我上一版的过度结论也错了)。更准确:**attention/indexer/rope 三个 CANN op 都存在且接口大体兼容,只是 V4 默认没 wire(默认走 flash_mla/triton);mhc + compressor + fp8 确实无 CANN。** → 推理侧"用 CANN 替代"是**接口适配/wiring 工作量**的问题(attention/indexer/rope 可试接 CANN),不是"根本没有 CANN"。**所以 0.1.8-port 的必要性比我上一版说的低** —— attention 这种最重的 kernel 可能用 AscendAttnBackend 的 CANN 路恢复性能,不一定要 tilelang。

**两次纠错记录(诚实)**:① 先说"CANN 做不了 V4 sparse-MLA"= 没核实瞎说;② 又说"推理 CANN 覆盖≈0/6"= 也过度(忽略了 attention/indexer/rope 的 CANN op 存在只是没 wire)。真相在中间:部分有 CANN-op-可试接、部分(mhc/compressor/fp8)真没有。**下结论前必须逐 kernel 核接口,不能拍脑袋两个方向都偏。**

## RESOLVED via runtime trace (2026-06-03): what V4-NPU attention actually does

Runtime evidence (instrumented MQALayer + backend, logs in `_v4_runlogs/`):
- V4-NPU inference attention runs on **`AscendAttnBackend`** (`hardware_backend/npu/attention/ascend_backend.py`) — NOT flash_mla (flash_mla not even installed), NOT DeepseekV4AttnBackend (its forward never called — instrumented, zero hits). The "dsv4 backend" registry log was misleading; NPU auto-selects AscendAttnBackend.
- `AscendAttnBackend.forward_extend` branch logic:
  - `topk_indices is not None` → `forward_sparse` → **`torch_npu.npu_sparse_flash_attention` (CANN sparse, :999)**.
  - else `"compress_ratio" in kwargs` (V4's path) → **"V4 PoC dense short-circuit"** (:1083): plain torch SDPA on fresh q/k/v, no pool reads, no sparse. Comment: *"won't be correct past the first chunked step but lets us validate end-to-end shapes."*
- **So today V4-NPU generate runs attention via a torch dense-SDPA PoC short-circuit, NOT the CANN sparse op.** It produces tokens (verified) but is the slow/approximate fallback.

## THE actual "wire CANN" task (precise, bounded, evidence-based)
The CANN sparse path (`forward_sparse`→`npu_sparse_flash_attention`) already exists in AscendAttnBackend; it's gated on `topk_indices is not None`. V4 currently doesn't reach it (falls into the dense short-circuit) → because V4's MQALayer/indexer isn't feeding `topk_indices` into `attn_backend.forward`, OR the indexer (C4Indexer, triton) output isn't wired to it on this path.
**Wiring = make V4 produce + pass topk_indices so forward_sparse (CANN npu_sparse_flash_attention) is taken instead of the dense-torch short-circuit.** That also requires the indexer to run (C4Indexer triton, or npu_lightning_indexer CANN) to produce the topk. So attention-CANN-wiring is coupled to indexer-wiring.
Status: not yet done; this is the precise, scoped task (verified by runtime trace, not assumed).

## Wiring attempt 1 (2026-06-03): enabling indexer flag alone does NOT reach CANN sparse path
- Experiment: took the working torch-fallback recipe, REMOVED `SGLANG_OPT_USE_TILELANG_INDEXER` from the disable list (so indexer enabled), kept torch topk fallback, instrumented `forward_sparse` entry. Ran, log `_v4_runlogs/v4_wire_indexer.log`.
- Result: generate still works (same gibberish tokens), but **`[FS] forward_sparse ENTERED` never printed** → CANN `npu_sparse_flash_attention` path STILL not reached; still on dense short-circuit.
- Conclusion: just enabling the indexer env flag does not thread topk_indices into `attn_backend.forward`. The gap is in sglang's V4 indexer→topk→backend DATAFLOW: MQALayer's `attn_backend.forward(...)` call (deepseek_v4.py ~721) does NOT pass `topk_indices`; the indexer writes topk into metadata/forward_batch as a side-effect, but `forward_extend` reads its `topk_indices` param (None here) → dense short-circuit. Wiring requires modifying this upstream-sglang dataflow (make the produced topk reach forward_extend), plus the CANN sparse path needs the V4 paged-KV-pool API that the PoC dense-short-circuit comment says is "unbuilt".
- **Honest scope**: wiring CANN sparse attention = real upstream-sglang V4 integration (indexer topk dataflow + V4 KV-pool for forward_sparse), NOT a flag/config change. Possible (op + path exist) but substantive. Attempt-1 (flag) verified insufficient by runtime trace.
