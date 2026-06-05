---
id: sglang-005
date: 2026-06-01
layer: sglang
title: V4 generate() "scheduler IPC hang" was PrefillAdder SWA-budget starvation, not a ZMQ/asyncio bug
trigger:
  - "DeepseekV4ForCausalLM generate() hangs on NPU, Engine init succeeds but no output returns"
  - "sglang V4 on device=npu: scheduler subprocess alive but request never runs"
  - "[RR] got 1 reqs but [SCHED-LOOP] recv ret truthy=False contradiction in scheduler traces"
  - "any V4 + small SWA token pool generate() that hangs and looks like ZMQ/asyncio plumbing"
symptom_in_wild:
  - "Engine init OK in ~22-32s, V4 KV pool + SWA + c4 + c128 all allocated (full=4096 swa=256 ...), then generate() hangs ~5min until SIGTERM"
  - "Injecting [TRACE] prints into DeepseekV4DecoderLayer.forward shows ZERO prints — V4 forward never reached"
  - "Scheduler traces are self-contradictory: [TM] generate_request entered ✓, [TM-B] after _send_batch_request ✓, [RR] _pull_raw_reqs got 1 ✓, [RR] after broadcast/mm/finalize: 1 ✓ — BUT [SCHED-LOOP] recv ret truthy=False at every iteration including the same timeframe"
  - "input_ids=[[1,2,3]] (skip tokenization) hangs identically → wrongly concluded the bug is in pre-tokenize asyncio/lock/ZMQ wiring"
  - "Hours spent hypothesizing 'two SchedulerRequestReceiver paths' / 'ZMQ pub-sub binding mismatch' / 'asyncio event loop blocked in auto_create_handle_loop'"
root_cause: >
  PrefillAdder admission returned AddReqResult.NO_TOKEN, so the request was
  silently re-queued instead of run. _swa_budget_for_req needs
  max(input_len, sliding_window) + page_size slots, but the V4 SWA token pool
  was allocated only ~256 slots. The [RR]/[SCHED-LOOP] "contradiction" was the
  request being pulled then bounced by admission, never entering the run batch —
  not a ZMQ or asyncio bug at all.
mistake_pattern: "treating an admission-budget starvation as a transport/IPC hang; chasing ZMQ/asyncio before instrumenting PrefillAdder admission result"
correction:
  - "When V4 generate() hangs and forward is never reached, instrument the PrefillAdder admission RESULT first — log AddReqResult (CONTINUE vs NO_TOKEN vs OTHER) before touching ZMQ pub/sub, asyncio loops, or tokenizer locks."
  - "Fix that unblocked it: Engine ctor args max_total_tokens=65536 + swa_full_tokens_ratio=0.5. PrefillAdder then returns CONTINUE and V4 forward is invoked on NPU."
  - "Classify the fix: this is a PoC config tune, not a production fix — it gives the SWA pool enough headroom to admit the short PoC request. A production deployment must size the SWA pool against real input_len / sliding_window, not paper over admission with a giant max_total_tokens."
  - "Verify the unblock by confirming a TRACE print INSIDE DeepseekV4DecoderLayer.forward fires — 'request admitted' is only proven when forward is actually entered, not when the scheduler stops logging NO_TOKEN."
  - "Discriminator vs a real IPC bug: if [RR] pulls the request but [SCHED-LOOP] never runs it, suspect admission bounce-and-requeue before suspecting transport. A genuine ZMQ failure would not show [RR] _pull_raw_reqs got N > 0."
evidence:
  - "workspace/v4_attempt_2026_06_01/README.md §'2026-06-01 update — IPC unblocked, V4 forward now reachable' (lines 152-163): the root-cause resolution"
  - "workspace/v4_attempt_2026_06_01/README.md §'2026-06-01 update — narrower still' (lines 112-137): the misleading [RR] got 1 vs [SCHED-LOOP] recv ret truthy=False contradiction, logged as symptom_in_wild before the cause was found"
  - "workspace/v4_attempt_2026_06_01/README.md SWA pool line: 'full=4096 swa=256 c4=1024 c128=32 c4_state=8 c128_state=128' — the 256-slot SWA pool that starved admission"
  - "Fix args live in workspace/v4_attempt_2026_06_01/_sglang_v4_minimal_PASS.py Engine ctor (max_total_tokens=65536, swa_full_tokens_ratio=0.5)"
  - "PASS log: workspace/v4_attempt_2026_06_01/v4_PASS_log_2026_06_01.txt (generate done in 0.9s after unblock)"
  - "docs/_meta/DSV4_NPU_PORTING_REPORT.md §2 — this hang was the gate before the 14 V4-hook gaps could even be discovered"
applies_to: ["sglang trunk DeepseekV4ForCausalLM @ 2026-06-01 (sgl 0.5.12.post2.dev434+gb13d3d18c), device=npu, AscendAttnBackend, bf16"]
verified_on: ["Ascend A3 NPU, tlrescue-class container, 1-layer reduced DSv4-Flash fab, 2026-06-01 — generate() PASS in 0.9s once PrefillAdder returned CONTINUE"]
unverified_on: ["full 43-layer DSv4-Flash (only 1-layer reduced fab validated)", "real (non-fab) weights", "production-sized SWA pool against real input_len"]
deprecated_after: ""
---

# sglang-005 — V4 "scheduler IPC hang" is PrefillAdder SWA-budget starvation

> **2026-06-05 re-baseline (M2)**: 独立 agent 对最新 miles main 校对——本条是 **sglang 推理侧**(PrefillAdder/SWA pool,sglang-internal),不依赖 miles。**对最新 miles 仍适用,无需改**。

## Why this matters

This single misdiagnosis burned ~4 hours of an attempt. The hang *looked* exactly
like a multi-process transport bug: Engine inits, the scheduler subprocess is
demonstrably alive, the request is demonstrably ZMQ-pushed and pulled — yet the
run loop never executes it and V4 `forward` is never reached. The natural (wrong)
conclusion was "sglang's NPU scheduler IPC / asyncio wiring is broken for V4."

The actual cause was mundane: admission. `PrefillAdder` was returning
`AddReqResult.NO_TOKEN` because `_swa_budget_for_req` needs
`max(input_len, sliding_window) + page_size` slots and the V4 SWA token pool had
only ~256. The request was silently re-queued every loop iteration. No transport
was broken; the request just never qualified to run.

Unblocking it (`max_total_tokens=65536` + `swa_full_tokens_ratio=0.5`) immediately
got V4 forward executing on NPU — and *that* is what then exposed the real V4
work: the 14 `AscendAttnBackend` / sgl-kernel-npu V4-hook gaps catalogued in
`sglang-004` and `docs/_meta/DSV4_NPU_PORTING_REPORT.md §2`. The admission hang was
the gate in front of all of it.

## The misleading trace contradiction (symptom_in_wild)

The forensic dead-end worth recognizing next time — full-pipeline traces that
*contradict each other*:

```
[TM] generate_request entered          ✓   Engine.generate IS dispatching
[TM-B] after _send_batch_request       ✓   request ZMQ-pushed from tokenizer_manager
[RR] _pull_raw_reqs got 1              ✓   scheduler's request_receiver pulled it
[RR] after broadcast/mm/finalize: 1    ✓   all receive stages pass
[SCHED-LOOP] recv ret truthy=False     ✗   main scheduler loop sees empty list — every iteration
```

The contradiction (`[RR] got 1` but `[SCHED-LOOP]` always empty) tempts you to
invent transport-layer theories: two receiver code paths, ZMQ pub/sub binding
mismatch, asyncio loop blocked in `auto_create_handle_loop`. Earlier notes even
record testing `input_ids=[[1,2,3]]` to "skip tokenization" — same hang — and
concluding the bug lived in pre-tokenize plumbing. All wrong. The request *was*
arriving; it was being pulled and then bounced by admission before it ever
entered the run batch, so the run loop legitimately saw nothing to run.

## How to know it's this and not a real IPC bug

1. Is V4 `forward` ever entered? Inject a print in `DeepseekV4DecoderLayer.forward`.
   Zero prints + alive scheduler + request demonstrably pulled (`[RR] got N>0`) →
   suspect admission, not transport.
2. Instrument the `PrefillAdder` admission **result** before anything else. If it
   returns `AddReqResult.NO_TOKEN`, you have budget starvation, full stop.
3. Check the SWA pool size against `max(input_len, sliding_window) + page_size`.
   The PoC log line `full=4096 swa=256 ...` is the tell — 256 SWA slots cannot
   admit a request whose `_swa_budget_for_req` exceeds it.
4. A genuine ZMQ transport failure would NOT show `[RR] _pull_raw_reqs got N` with
   `N > 0`. Receipt proves transport; non-execution after receipt points at
   admission/scheduling, not the socket.

## Honesty / scope

- **Walkaround, not production.** `max_total_tokens=65536` + `swa_full_tokens_ratio=0.5`
  is a PoC headroom tune that admits the short 1-layer PoC request. A production
  deployment must size the SWA pool to the real `input_len` / `sliding_window`
  budget; do not ship the giant `max_total_tokens` as the "fix."
- **1-layer reduced fab only.** Verified on a 1-layer reduced DSv4-Flash fab with
  random weights (gibberish output is expected — this proves loop closure / shape,
  not numerical correctness). Not validated on the full 43-layer model or real
  weights.
- This entry is about the *admission misdiagnosis*. The downstream V4 forward gaps
  it unblocked are separate (see `sglang-004` + the porting report §2 14-gap table).
