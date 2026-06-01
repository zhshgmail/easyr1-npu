# V4 真路径 attempt — 2026-06-01

> 这个目录是 2026-06-01 sglang V4 真路径在 NPU 上 PoC 跑通过程的 raw artifact。
>
> **两个 milestone PASS(2026-06-01)**:
> 1. **generate() PASS** — `DeepseekV4ForCausalLM` 真 V4 model class 在 Ascend A3 NPU bf16 跑通 `llm.generate()`,非空输出(shape-correct,gibberish 是减层预期)。证据:`v4_PASS_log_2026_06_01.txt` + `_*_PASS.py` patched sources。
> 2. **RL loop PASS** — rollout → weight-update → re-rollout 完整闭环,5/5 步 weight-sync 都改变了 rollout 输出。证据:`v4_RL_LOOP_PASS_log_2026_06_01.txt` + `_v4_rl_loop_tensor_PASS.py`。weight-sync 用 `update_weights_from_tensor`(attention-only)绕开 #26794 MoE reload bug。
>
> gibberish 文本是极限减层 PoC 的预期结果(user 2026-06-01 确认);本 PoC 验证的是**循环闭合 + 形状正确性**,不是数值/文本质量。

## 状态

| Step | Result |
|---|---|
| HF DSv4-Flash 真 config 拿到(`deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json`) | ✓ |
| sglang trunk 有 `deepseek_v4.py`(2259 行)+ `EntryClass = [DeepseekV4ForCausalLM]` | ✓ |
| 真 V4 减层 fab(`architectures=["DeepseekV4ForCausalLM"]`,真 V4 schema 字段全 43 条 + 减层) | ✓(`fabricate_dsv4_REAL_1layer_ckpt.py`) |
| sglang Engine 完整 init,所有 weights 加载 0 missing | ✓ |
| V4 KV pool + SWA + c4 + c128 全部分配 | ✓(`full=4096 swa=256 c4=1024 c128=32 c4_state=8 c128_state=128`) |
| Engine init OK time | ✓(22-32s) |
| `llm.generate(["Hi"], max_new_tokens=2)` | ✓ — 2026-06-01: generate returns non-empty text in 0.9s after 14 PoC workarounds (see Gap table + PASS section below) |
| 给 customer 看的 "PoC PASS" 证据 | **有 — `v4_PASS_log_2026_06_01.txt` + frozen patched-source snapshots** |

> ⚠ **下面 "已尝试的关闭路径 / generate hang / NO PASS / IPC contradiction" 等段落是 attempt 过程中的历史 narrative,记录调试时序,对上游 PR 有用。最终结论已被顶部两个 milestone PASS + 文末 "RL LOOP PASS" / "PASS" 段落取代 —— generate() 和 RL loop 都已跑通。阅读时以顶部 + 文末为准。**

## 已尝试的关闭路径(都不解决 generate hang)

- `SGLANG_OPT_FP8_WO_A_GEMM=0` — bf16 path,fp8 关闭
- `SGLANG_OPT_USE_TILELANG_MHC_PRE=0` — 不走 tilelang mhc kernel
- `SGLANG_OPT_USE_TILELANG_MHC_POST=0`
- `SGLANG_OPT_USE_FUSED_QK_NORM_ROPE=0`
- `SGLANG_OPT_FUSE_WQA_WKV=0`
- `SGLANG_OPT_USE_FUSED_COMPRESS=0`
- ... 16 个 SGLANG_OPT_* env 全关
- 同时 stub `sglang.srt.layers.mhc` 整模块 + stub `deepseek_common.amd.deepseek_v4_fused_mhc`

## 下一步候选(下次 attempt 接着做)

1. **trace V4 attn backend hang 点** — sglang `attention_backend='ascend'` 对 V4 RadixAttention 的 c4/c128 多级 KV cache write path 是否完整;`enable_dsa_prefill_context_parallel=False` 是否触发了 broken path
2. **forward path 手工 instrument** — 在 `deepseek_v4.py:DeepseekV4DecoderLayer.forward` 加 `print(flush=True)` 看走到哪一行
3. **不通过 sglang.Engine,直接 model.forward(token_ids)** — bypass scheduler IPC,如果 forward 跑动那 hang 在 sglang scheduler / detokenizer
4. **比较 vllm-ascend V4 — 它已有 native 路径** — 即使要解开 fp8 quant_method 的 validator,模型 forward 算子全是 `torch.ops._C_ascend.npu_hc_*` Huawei 写好的,可能跑动

## Artifact

| File | What |
|---|---|
| `fabricate_dsv4_REAL_1layer_ckpt.py` | 真 V4 减层 fab(MoE_ACTIVE=1, 43 weights, 1.3B params) |
| `config_actual_on_a3.json` | fab 跑完 + 4 个 post-tweak 后 A3 上 `/host-models/dsv4_REAL_1layer_fab/config.json` 实际状态 |
| `_sglang_v4_minimal.py` | 最简 sglang Engine 启动脚本,关掉所有 OPT_* env + stub mhc/amd |
| `_v4_audit.py` | V4 模块 importability audit script — sglang 上 `mhc` import FAIL no tilelang,其他 OK |
| `v4_engine_log.txt` | 最后一次跑的完整 log — Engine init OK + Tree cache initialized + Engine init OK in 22.2s + CALLING generate + 5min hang + SIGTERM |

## 不能向 customer 说的话

- "DSv4-Flash 真路径在 NPU 跑通了"
- "Engine init OK = PoC PASS"

## 可以向 customer 说的话(诚实)

- "DSv4-Flash 真 config 我们 wire 到 sglang V4 model class 的入口,真 V4 减层 fab 包含 43 个真 V4 schema 权重,sglang Engine init + V4 KV pool 全部成功,但 forward 在 NPU 上 hang,sglang main 缺一个或多个 V4 NPU-side op,需要上游 sgl-project/sgl-kernel-npu 补"
- 这是 V4 NPU 真适配工作的真起点,不是终点

## 2026-06-01 update — root cause narrowing

After instrumenting `DeepseekV4DecoderLayer.forward` with 4 `[TRACE]` prints
and re-running: **zero TRACE prints appeared in the log**. generate() was
called from main but V4 forward was never invoked. Confirmed the hang is
NOT in V4 algorithm code — it's in sglang's multi-process Engine
orchestration (scheduler / tokenizer / detokenizer IPC) for V4 on NPU.

`_sglang_v4_direct.py` confirms in isolation (no Engine multi-proc):
- sglang.srt.models.deepseek_v4 importable
- EntryClass=[DeepseekV4ForCausalLM]
- ModelConfig.from_server_args parses our fab cleanly
- arch=['DeepseekV4ForCausalLM']

So we have isolated:
- ✓ V4 model class on NPU works at import / class level
- ✓ V4 config / weight load / KV pool all OK
- ✗ sglang Engine multi-process pipeline for V4 on NPU hangs before calling forward

This is upstream issue territory: V4 on NPU via sglang.Engine needs a
sgl-project/sglang fix to the V4-specific scheduler/tokenizer wiring
when device=npu.

## 2026-06-01 update — IPC isolation

Added [SCHED] traces to `event_loop_overlap` in scheduler.py:
- `[SCHED] entered event_loop_overlap` — prints ✓ scheduler is alive
- `[SCHED] RECV N reqs` — **NEVER prints** even after generate() called

So the scheduler subprocess **never receives any request via its ZMQ socket**.
This narrows the hang to one of:
1. tokenizer_manager.generate_request main-side hangs before `_send_one_request`
   (i.e. tokenization, normalize_batch_and_arguments, _validate_and_resolve_lora,
   or model_update_lock.reader_lock all run first)
2. ZMQ pub/sub binding mismatch between tokenizer_manager and scheduler
3. asyncio event loop blocked in `auto_create_handle_loop` or `is_pause_cond.wait_for`

Tested workaround `input_ids=[[1,2,3]]` to skip tokenization → same hang. So
tokenization itself is not the cause. It's somewhere in the pre-tokenize
plumbing (asyncio setup, lock acquire, or ZMQ wiring).

## Honest state to land

V4 真路径在 NPU 的 PoC 没有跑通 generate(). Engine 起来,scheduler subprocess
活着,V4 KV pool + cache + weights 全部 ready,但 generate() 卡在 main process
的 sglang.Engine 异步 plumbing。卡点在 tokenizer_manager 或 asyncio event
loop,不是 V4 model forward 或 V4 attn backend(model forward 根本没被调用过)。

## 2026-06-01 update — narrower still

Traces injected through whole pipeline. Findings:
- `[TM] generate_request entered` ✓ — Engine.generate IS dispatching
- Goes to `_handle_batch_request` (obj.is_single=False because input is list-of-list)
- `[TM-B] after _send_batch_request` ✓ — request ZMQ-pushed from TM
- `[RR] _pull_raw_reqs got 1` ✓ — scheduler's request_receiver pulled it
- `[RR] after broadcast/mm/finalize: 1` ✓ — all stages pass
- BUT `[SCHED-LOOP] recv ret truthy=False` — main scheduler loop's
  `recv_requests()` returns empty list at all iterations including
  the timeframe `[RR]` printed.

These observations are contradictory. Possible explanations:
1. `[RR] got 1` printed from a DIFFERENT request flow (maybe an RPC
   health check or a sub-request) than the [SCHED-LOOP] loop's call
2. There are 2 SchedulerRequestReceiver-equivalent code paths active
3. The scheduler subprocess and the [RR]-printing process are different
   but the TM-sent request ends up at the RR but not the SCHED-LOOP one

Did not finish further isolation tonight (4h in). The hang is somewhere
in scheduler IPC dispatch that doesn't surface where the request goes
between `_pull_raw_reqs` and the scheduler's main loop seeing it.

This is sglang's NPU adapter or scheduler IPC bug, not V4 model bug.
V4 forward path never gets touched, so the V4 algorithm on NPU is
NOT proven broken (or fixed).

## What's a real PoC PASS look like (for whoever picks this up)

`llm.generate(["Hi"], sampling_params={"temperature":0, "max_new_tokens":2})`
returns a string. Currently returns nothing (hang).

To get there:
1. find the [RR] vs [SCHED-LOOP] contradiction
2. or: write a vllm-ascend V4 wrapper that bypasses fp8 quant validator
3. or: file sglang issue about V4 + device=npu + non-fp8 quant config

Status: NO PASS as of 2026-06-01 ~05:00 Beijing. ~4h on this, V3.2 PoC
report still has the §0 Disclosure as the truthful header.

## 2026-06-01 update — IPC unblocked, V4 forward now reachable

Root cause of [RR]/[SCHED-LOOP] mismatch:
- `PrefillAdder` admission was returning `AddReqResult.NO_TOKEN` and the
  request was being silently dropped/re-queued instead of run.
- `_swa_budget_for_req` requires `max(input_len, sliding_window) + page_size`
  but the SWA token pool only had ~256 slots.
- Bumped `max_total_tokens=65536` and added `swa_full_tokens_ratio=0.5`,
  PrefillAdder now returns CONTINUE.

After unblocking, V4 forward is invoked on NPU. Each step now uncovers a
discrete sgl-kernel-npu / sglang-NPU V4 adapter gap. Discovered so far:

| Gap | Site | PoC workaround |
|---|---|---|
| 1 | `AscendAttnBackend._maybe_upgrade_forward_metadata` missing | no-op stub |
| 2 | `sglang.srt.layers.mhc.hc_split_sinkhorn` requires tilelang/CUDA | torch fallback in driver stub |
| 3 | `sglang.jit_kernel.dsv4.elementwise.fused_q_norm_rope` JIT-compiles CUDA | torch fallback in source |
| 4 | Same for `fused_rope_inplace`, `fused_norm_rope_inplace` | torch fallback |
| 5 | NPU `aclnnIndex` doesn't support complex64 | index real, view-as-complex after |
| 6 | `fused_k_norm_rope_flashmla` writes FP8-packed bytes to paged kvcache | PoC: skip scatter (1-layer max_tokens=2 is short enough) |
| 7 | `AscendAttnBackend.forward_c4_indexer` missing | no-op stub |
| 8 | `AscendAttnBackend.forward_core_compressor` missing | no-op stub |
| 9 | `AscendAttnBackend.store_cache`, `forward_compress`, `init_forward_metadata_indexer` missing | no-op stubs |
| 10 | V4 dispatch passes `compress_ratio` kwarg, NPU `forward_extend` doesn't accept it | absorb via `**kwargs` |
| 11 | `DeepSeekV4TokenToKVPool.get_key_buffer` raises NotImplementedError but NPU `forward_extend` calls it | Bypassed via V4 dense short-circuit in `forward_extend` |
| 12 | `forward_decode` same `compress_ratio` kwarg + cache-read mismatch | Same V4 dense short-circuit in `forward_decode` |
| 13 | `jit_kernel/dsv4/moe.silu_and_mul_clamp` JIT-compiles CUDA | torch fallback (`silu(gate).clamp * up.clamp`) |
| 14 | `jit_kernel/dsv4/gemm.linear_bf16_fp32` uses `torch.mm(out_dtype=fp32)` not supported on NPU | drop kwarg, `.float()` cast |

## 2026-06-01 PASS

```
[v4-min] sgl 0.5.12.post2.dev434+gb13d3d18c
[v4-min] Engine init OK in 27.7s
[v4-min] generate done in 0.9s
[v4-min] output: [{'text': '醺报废', 'output_ids': [122081, 112435],
  'meta_info': {'finish_reason': {'type': 'length', 'length': 2},
                'prompt_tokens': 3, 'completion_tokens': 2,
                'e2e_latency': 0.87s, ...}}]
```

**真路径 V4 (`DeepseekV4ForCausalLM`) on Ascend A3 NPU, sglang Engine,
bfloat16, generate() returns non-empty string in 0.9s.**

The text `'醺报废'` is gibberish — expected since:
- 1-layer reduced fab vs. full 60+ layer model
- random initialization rather than trained weights
- 8 PoC no-op stubs in K/V cache path + MoE routing
- 2 V4 dense short-circuits replace sparse compressor/indexer

This is shape-correct end-to-end execution, not numerical correctness.
That distinction is honestly stated. A real PoC PASS for *correctness*
would require:
1. real DSv4-Flash weights (not 1-layer fab)
2. NPU-native impl of compressor, c4_indexer, hc_split_sinkhorn,
   fused_q_norm_rope/k_norm_rope_flashmla, paged FP8/BF16 packed
   kvcache scatter
3. removal of the 14 PoC workarounds documented above

## Artifact snapshot (frozen at PASS)

| File | What |
|---|---|
| `_sglang_v4_minimal_PASS.py` | The driver with mhc stub + hc_split_sinkhorn torch fallback + Engine ctor args (`max_total_tokens=65536`, `swa_full_tokens_ratio=0.5`) |
| `_elementwise_PASS.py` | Patched `sglang/jit_kernel/dsv4/elementwise.py` with torch fallbacks for `fused_q_norm_rope`, `fused_rope_inplace`, `fused_k_norm_rope_flashmla` (kvcache scatter skipped) |
| `_moe_PASS.py` | Patched `silu_and_mul_clamp` torch fallback |
| `_gemm_PASS.py` | Patched `linear_bf16_fp32` to drop `out_dtype=` keyword |
| `_ascend_backend_PASS.py` | Patched `AscendAttnBackend` with 7 V4 hook stubs + V4 dense short-circuits in `forward_extend`/`forward_decode` |
| `v4_PASS_log_2026_06_01.txt` | Full reproducible log of Engine init + generate() PASS |

## Honest customer message (post-PASS)

We can now say:
- "DeepSeek-V4-Flash 真路径 (sglang trunk `DeepseekV4ForCausalLM` model class)
  has been demonstrated to execute end-to-end on Ascend A3 NPU bf16, with
  a 1-layer reduced fabrication and 14 documented PoC workarounds."
- "Each workaround corresponds to a concrete sgl-kernel-npu / sglang V4
  NPU-adapter gap. The PoC is a forcing function for the upstream gap
  inventory, not a production deliverable."

We CANNOT say:
- "V4 on NPU is production-ready" — it isn't; 14 PoC stubs are in the path
- "Numerical correctness verified" — output is shape-correct gibberish
- "Full-model PASS" — this is 1-layer reduced fab

## Next steps (post-PASS, for upstream PR work)

1. File a clean sgl-project/sglang issue: "V4 (DeepseekV4ForCausalLM) on
   `device=npu` requires N missing AscendAttnBackend V4 hooks", with each
   of the 14 gaps as a concrete line-number citation and a PASS-of-workaround
   evidence link.
2. PR `_maybe_upgrade_forward_metadata` + `forward_c4_indexer` +
   `forward_core_compressor` + `store_cache` + `init_forward_metadata_indexer`
   + `forward_compress` as no-op stubs into `AscendAttnBackend` (default safe).
3. PR `**kwargs` absorption to `forward_extend` / `forward_decode` (back-compat).
4. PR `linear_bf16_fp32` NPU-path: replace `out_dtype=fp32` with `.float()` cast.
5. File NPU complex64 aclnnIndex issue (real-domain index workaround).

## 2026-06-01 RL LOOP PASS — full closure (rollout → weight-update → re-rollout)

第二个 milestone:在 generate() PASS 之上,闭合完整 RL 循环。

```
step0 ids=[122081,112435] / [127281,77292]
step1 update_weights_from_tensor(5 attn tensors) -> (True,'Success')
step1 ids=[108542,12794]  / [27153,21390]    DIFF
step2 ids=[109679,119564] / [72919,65113]    DIFF
step3 ids=[38768,13815]   / [120381,128901]  DIFF
step4 ids=[49227,121506]  / [53545,16520]    DIFF
step5 ids=[57028,44658]   / [106928,85967]   DIFF
distinct_vs_step0=5/5  step_to_step_changes=5/5
=== V4 RL LOOP PASS — attention weight-sync changes inference ===
```

**证明的命题**:weight update 真的把权重 sync 进了运行中的 sglang V4 engine,且改变了 inference 行为。这是 RL 循环闭合的核心证据(rollout 端用真训练后权重,而非陈旧权重)。

**绕开 #26794 的方法**:用 `Engine.update_weights_from_tensor(named_tensors)` 只推 5 个 attention 权重(`self_attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight`),**不碰 MoE experts**。这样 FusedMoE 的 `_load_w2` reload narrow 崩溃路径根本不触发。`/update_weights_from_disk`(全量 reload)会撞 #26794(`narrow length 4096 > dim 2048`),`update_weights_from_tensor`(选择性 in-memory)不会。

**dense fab 不能绕 #26794**:sglang V4 的 `DeepseekV4DecoderLayer` 在 `first_k_dense_replace=1` + 单层时仍无条件用 `DeepseekV2MoE`(deepseek_v4.py:795),所以 dense fab 反而撞 `gate_up_proj` KeyError。结论:绕 #26794 必须走 `update_weights_from_tensor` 路径,不是换 fab。

**诚实边界**:
- weight delta 是 seeded synthetic(占位),不是真 miles 训练梯度 —— 因为 miles 训练侧的 V4 算子(hash-coding sinkhorn / Compressor / C4Indexer / o_lora)还没移植。这与 T32 `sglang_2step_real_update.py` 的 V3.2 PoC 同样的 "先证 plumbing,后接真训练" 纪律。
- 真 RL loop 把 synth delta 换成 miles V4 actor 的训练输出即可,plumbing 已 prove。

**Artifact**:
- `v4_RL_LOOP_PASS_log_2026_06_01.txt` — 完整 5-step log
- `_v4_rl_loop_tensor_PASS.py` — driver(attention-only `update_weights_from_tensor`)

## hc_split_sinkhorn AscendC op-gen(/ascendc-op-gen,进行中)

唯一需要 A5 ops 生成的 vector 算子(native NPU 无对应):
- `model.py` PyTorch CPU-truth reference(从 tilelang kernel 反推,验证输出 doubly-stochastic)
- op-gen O2.5 已生成 `input_gen.py` + `edge_inputs.pt` + `edge_dataset.pt` + `op_classification.json`(tags 正确:fused/transcendental/softmax/reduction/normalization)
- **blocker**:ref_preflight 在 a5ops-a3 容器跑 Model.forward 时撞 `aclInit 107001 device id error` —— CPU-truth reference 不该碰 NPU,但 ref harness 强制 device init。需要设 `ASCEND_RT_VISIBLE_DEVICES` 或让 ref 跑 CPU-only。

## native NPU op 替换路径(verified 可行,后续质量提升)

V4 PASS 用的是 torch fallback。已 verify native NPU op 存在,可一对一替换(提升精度+性能):
- `silu_and_mul_clamp` → `torch_npu.npu_clipped_swiglu(x, dim, alpha, limit, bias, interleaved)` — verified bf16 max_diff < 1e-3 vs torch ref
- `fused_q_norm_rope` / `fused_rope_inplace` / `fused_norm_rope_inplace` → `torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin, layout="BSND")`(+ `npu_rms_norm` for the norm 部分)— verified runs on npu:0
- `fused_k_norm_rope_flashmla` → `torch_npu.npu_kv_rmsnorm_rope_cache_v2`(fused rmsnorm+rope+KV cache write,直接对应)
- `hc_split_sinkhorn` → 无 native 对应,走 a5_ops `/ascendc-op-gen` 生成 AscendC kernel(进行中)

torch ops verified 全在 NPU 上跑(`torch.cuda.is_available()=False`,`torch.npu.is_available()=True`,复数乘 `xc*xc` on npu:0),不是 CPU drop。但 native fused op 更快 + 数值更接近真模型。
