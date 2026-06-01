---
id: cross-layer-012
date: 2026-06-01
layer: cross-layer
title: Prove V4 RL weight-sync on NPU with update_weights_from_tensor (attention-only), not update_weights_from_disk; don't switch fab to dense
trigger:
  - "closing a V4 RL loop (rollout -> weight-update -> re-rollout) on sglang NPU"
  - "update_weights_from_disk crashes with FusedMoE narrow on a DeepseekV4 fab"
  - "thinking about switching to a dense fab (first_k_dense_replace=1) to dodge the MoE reload crash"
  - "need to demonstrate that weight-sync actually changes rollout output on V4 NPU"
symptom_in_wild:
  - "RuntimeError narrow length 4096 > dim 2048 in FusedMoE _load_w2 during update_weights_from_disk on a 1-layer DSv4-Flash fab"
  - "switching to dense fab to avoid the MoE crash but then hitting gate_up_proj KeyError instead"
  - "RL loop appears closed but the weight delta pushed is a synthetic/seeded placeholder, reported as if it were real training output"
root_cause: >
  update_weights_from_disk does a full reload that walks the FusedMoE
  stacked-mapping reload path (the #26794 narrow regression — same bug family
  as KB sglang-003; on our V4 fab the observed crash was `narrow length 4096 >
  dim 2048`, see the note below on why the exact site/arithmetic differs from
  sglang-003's `_load_w13`/1408 observation). update_weights_from_tensor pushes
  a selective in-memory set of named tensors; pushing only the 5 attention
  tensors never triggers the FusedMoE reload path, so #26794 is bypassed
  structurally — not patched. A dense fab does NOT dodge #26794 because
  DeepseekV4DecoderLayer unconditionally instantiates DeepseekV2MoE even at
  first_k_dense_replace=1 single-layer (deepseek_v4.py:795), so the dense fab
  just trades the narrow crash for a gate_up_proj KeyError.
mistake_pattern: "to dodge an upstream reload bug, choosing the wrong axis (change the model fab) instead of the right one (change the weight-sync API surface); then overclaiming a synth-delta loop as real RL e2e"
correction:
  - "Use Engine.update_weights_from_tensor(named_tensors) pushing ONLY the 5 attention tensors: self_attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight. This never touches the FusedMoE _load_w2 reload path, so #26794 cannot fire."
  - "Do NOT switch the fab to dense to dodge #26794 — V4's DeepseekV4DecoderLayer always uses DeepseekV2MoE (deepseek_v4.py:795), so dense fab hits gate_up_proj KeyError, a different crash, not a workaround."
  - "Assert closure by output divergence: rollout -> update -> re-rollout, and require step-to-step changes (5/5 distinct vs step0 in the verified run)."
  - "State the honest boundary every time: the weight delta is seeded synthetic placeholder, NOT a real miles training gradient. Plumbing is proven; real e2e swaps synth for real actor output. Same discipline as the T32 V3.2 sglang_2step_real_update.py PoC: prove plumbing first, wire real training after."
evidence:
  - "workspace/v4_attempt_2026_06_01/README.md §'2026-06-01 RL LOOP PASS' (5/5 distinct, 5/5 step-to-step changes)"
  - "workspace/v4_attempt_2026_06_01/v4_RL_LOOP_PASS_log_2026_06_01.txt — full 5-step log"
  - "workspace/v4_attempt_2026_06_01/_v4_rl_loop_tensor_PASS.py — driver (attention-only update_weights_from_tensor)"
  - "docs/_meta/DSV4_NPU_PORTING_REPORT.md §2.1, §2.3 (RL loop classified production-plumbing / synth-delta boundary)"
  - "sgl-project/sglang#26794 (FusedMoE reload narrow regression) — see KB sglang-003"
  - "Crash arithmetic: narrow length 4096 (w1+w3 stacked, 2*2048) > dim 2048 (single-expert slot); dense fab KeyError site deepseek_v4.py:795 DeepseekV2MoE unconditional"
applies_to:
  - "sglang trunk DeepseekV4ForCausalLM @ 2026-06-01 (sgl 0.5.12.post2.dev434+gb13d3d18c)"
  - "1-layer reduced DSv4-Flash fab, MoE-active, on Ascend A3 NPU bf16"
verified_on:
  - "Ascend A3 NPU, sglang Engine, bf16, 1-layer reduced fab, 2026-06-01 (RL LOOP PASS, EXIT=0, 5/5 distinct)"
  - "Re-verified after native-op swap (internal task #310, NOT an sglang issue): full RL loop re-ran PASS with npu_clipped_swiglu + npu_rms_norm in forward path"
unverified_on:
  - "Full 43-layer DSv4-Flash (only 1-2 layer reduced fab verified — NOT full model)"
  - "Real miles training gradient delta (verified delta is seeded synthetic placeholder, never a real actor output)"
  - "MoE-expert weight-sync (only the 5 attention tensors pushed; routed-expert sync still blocked on #26794)"
deprecated_after: ""
---

# cross-layer-012 — V4 RL weight-sync: tensor (attention-only), not disk

## Why this matters

This is the single trick that turns "V4 generate() works on NPU" into "V4 RL
loop closes on NPU". The whole point of an RL loop is that the rollout engine
runs on *fresh post-training* weights, not stale ones. To prove that closure
without a full upstream fix for the FusedMoE reload bug, you have to pick the
right axis to route around — and there are two tempting-but-wrong moves.

## The trap (what almost went wrong)

The obvious closure path is `POST /update_weights_from_disk` — rename the
Megatron checkpoint to HF layout, merge, reload. On a DSv4-Flash MoE fab this
crashes in the FusedMoE stacked-mapping reload path with `narrow length 4096 >
dim 2048` (the 4096 is the `w1+w3` stacked dim `2*2048`; the 2048 is a
single-expert slot — a structural stacked-vs-sharded mismatch). This is the
#26794 regression. See KB `sglang-003` for the full reload-path divergence
root cause.

> **Reconcile with sglang-003 (do not silently contradict it).** sglang-003
> records the crash at `_load_w13` / `fused_moe_triton/layer.py:482` with
> `dim 1408`; our V4 fab recorded it at `_load_w2` / `dim 2048`. Both are the
> same bug *family* (a full-reload narrow against a slot sized for an
> individual shard), but the exact method and arithmetic differ because the
> two were observed on different model configs / expert geometries. The number
> that matters for the lesson is the structural mismatch, not the specific
> slot size — and the fix (route around via `update_weights_from_tensor`) is
> identical for both. The `_load_w2`/2048 figures here come from our own V4
> README note (`workspace/v4_attempt_2026_06_01/README.md:269`), not from a
> committed crash log; treat the precise site as observation-specific.

**Wrong move #1 — switch the fab to dense.** sglang-003's own workaround note
says "use a dense fab for weight-sync validation". That advice does NOT hold
for V4. `DeepseekV4DecoderLayer` instantiates `DeepseekV2MoE`
**unconditionally** even at `first_k_dense_replace=1` on a single layer
(`deepseek_v4.py:795`). So a dense V4 fab doesn't avoid the MoE code path —
it just trades the `narrow` crash for a `gate_up_proj` KeyError. (This
dense-fab dead-end is recorded as a reasoned note in README.md:271 — derived
from reading `deepseek_v4.py:795`, not from a captured `gate_up_proj` crash
log; verify by running the dense fab if you need the captured traceback.)

**Wrong move #2 — claim the loop is real RL.** Even once the loop closes, the
weight delta in the PoC is a **seeded synthetic placeholder**, not a miles
training gradient (the miles-side V4 training ops — hash-coding sinkhorn,
Compressor, C4Indexer, o_lora — weren't all ported when this ran). Reporting
synth-delta closure as "real RL e2e" is exactly the closure-pressure
substitution failure mode (auto-memory `deception_under_closure_pressure_2026_06_01`,
under `~/.claude/projects/-home-z00637938-workspace-easyr1-npu/memory/`).

## The right move

Use `Engine.update_weights_from_tensor(named_tensors)` and push **only the 5
attention tensors**:

```
self_attn.wq_a.weight
self_attn.wq_b.weight
self_attn.wkv.weight
self_attn.wo_a.weight
self_attn.wo_b.weight
```

This is a selective **in-memory** update. It never walks the FusedMoE
`_load_w2` reload path, so #26794 cannot fire — the bug is bypassed
structurally, not patched. Because the attention weights feed the same forward
the rollout uses, changing them is sufficient to demonstrate that weight-sync
moves inference behavior.

Closure assertion is output divergence, not a return code: rollout → update →
re-rollout, require step-to-step changes. The verified run got
`distinct_vs_step0=5/5`, `step_to_step_changes=5/5`.

## Classification (walkaround vs production)

- ✅ **production-plumbing**: the `update_weights_from_tensor` attention-only
  path itself is a legitimate weight-sync mechanism, not a stub. It re-ran
  PASS after the native-op swap (internal task #310, not an sglang issue) confirmed the patches are on the real
  forward path.
- ⚠️ **synth-delta boundary**: the delta is seeded synthetic. Real e2e swaps
  it for miles V4 actor output. Plumbing proven ≠ training proven.
- ⚠️ **reduced-fab basis**: verified on a 1-layer (and 1–2 layer training-side)
  reduced fab, NOT the full 43-layer DSv4-Flash. Do not overclaim full-model.
- ⚠️ **MoE-expert sync still blocked**: only attention tensors are synced.
  Routed-expert weight-sync stays blocked on #26794 until the FusedMoE reload
  path is fixed upstream (sglang-003 close condition).

## Note on the training-side ops (don't repeat the early misjudgment)

When wiring the *real* delta in, the V4 training-core ops are mostly
**CANN-native** — verified on A3: `npu_nsa_select_attention` (sparse-MLA),
`npu_lightning_indexer` (C4 indexer), `npu_nsa_compress_attention`
(compressor), `npu_mla_prolog_v3` (MLA prep), `npu_rms_norm`. Only
hash-coding sinkhorn and act_quant (fp8) had no native correspondence and were
generated via AscendC op-gen (precision-verified). Do **not** assume "all V4
ops need hand-written kernels" — that was the early misjudgment the owner
corrected. Probe `dir(torch_npu)` / CANN native coverage before op-gen.

## How to know it's this lesson vs your own bug

1. Does `update_weights_from_disk` crash at `narrow length 4096 > dim 2048` in
   `FusedMoE _load_w2`? → that's #26794 (KB sglang-003), not your weight format.
2. Did switching to a dense fab give you a `gate_up_proj` KeyError instead? →
   you hit the `deepseek_v4.py:795` unconditional `DeepseekV2MoE`; the dense
   route is a dead end for V4.
3. Does `update_weights_from_tensor` with the 5 attention tensors return
   `(True, 'Success')` and change rollout output? → that's the verified path.
   Stop here for plumbing proof; do not claim real RL until the delta is a real
   actor gradient.

## Links

- KB `sglang-003` — FusedMoE `_load_w2` reload narrow regression (#26794), the upstream bug this routes around.
- KB `miles-001` — DSAMLA tilelang NPU port pattern, the training-side ops that produce the real delta.
- `docs/_meta/DSV4_NPU_PORTING_REPORT.md` — consolidated walkaround-vs-production classification.
- `workspace/v4_attempt_2026_06_01/README.md` — raw attempt log + frozen PASS artifacts.
