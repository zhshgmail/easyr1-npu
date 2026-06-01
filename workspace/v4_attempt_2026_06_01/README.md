# V4 真路径 attempt — 2026-06-01

> 这个目录是 2026-06-01 sglang V4 真路径在 NPU 上尝试 PoC 跑通过程的 raw artifact。
> **没有可执行 milestone — Engine 起来了,generate() 不返回**。本目录保留是为了让下一次 attempt 不从零开始,有完整证据链。

## 状态

| Step | Result |
|---|---|
| HF DSv4-Flash 真 config 拿到(`deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json`) | ✓ |
| sglang trunk 有 `deepseek_v4.py`(2259 行)+ `EntryClass = [DeepseekV4ForCausalLM]` | ✓ |
| 真 V4 减层 fab(`architectures=["DeepseekV4ForCausalLM"]`,真 V4 schema 字段全 43 条 + 减层) | ✓(`fabricate_dsv4_REAL_1layer_ckpt.py`) |
| sglang Engine 完整 init,所有 weights 加载 0 missing | ✓ |
| V4 KV pool + SWA + c4 + c128 全部分配 | ✓(`full=4096 swa=256 c4=1024 c128=32 c4_state=8 c128_state=128`) |
| Engine init OK time | ✓(22-32s) |
| `llm.generate(["Hi"], max_new_tokens=2)` | ✗ — call 进 scheduler 后 5+ min 无响应,NPU utilization 0% |
| 给 customer 看的 "PoC PASS" 证据 | **没有** |

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
| 11 | `DeepSeekV4TokenToKVPool.get_key_buffer` raises NotImplementedError but NPU `forward_extend` calls it | **CURRENT — fundamental layout mismatch, attempting bypass** |

The pattern is clear: NPU `AscendAttnBackend` was written for V3.x classic MLA;
V4 introduces a heap of new hooks (compressor, c4_indexer, hash-coding, swa
kvcache write) the NPU backend has never seen. PoC stubs unblock forward
propagation; production needs each hook implemented natively (or replaced
with `torch_npu.npu_*` calls).

This is now a real concrete upstream issue with line numbers and signatures
for sgl-project/sglang #npu V4 adapter team. Will continue iterating until
generate() returns OR hit a layout-fundamental block that needs a multi-day
backend rewrite.
