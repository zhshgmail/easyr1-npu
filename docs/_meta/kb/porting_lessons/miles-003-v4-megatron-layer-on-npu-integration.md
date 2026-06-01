---
id: miles-003
date: 2026-06-01
layer: miles
title: Getting a real DeepSeekV4Attention megatron layer (+ reduced 1-2 layer TransformerBlock) to fwd+bwd+optimizer.step on NPU
trigger:
  - "running a real DeepSeekV4Attention megatron layer fwd+bwd on Ascend NPU"
  - "wiring miles DeepSeek-V4 ops into a Megatron-on-NPU TransformerLayer for training"
  - "MindSpeed core_r0.16.0 dsa TND adapting Megatron-Core 0.16"
  - "npu_rms_norm too many args / gamma dtype mismatch inside a megatron layer"
  - "2-layer DSV4 backward produces NaN in sparse_attn_torch masked softmax"
  - "single 61GB NPU chip OOM when adding AdamW states to a 2-layer DSV4 reduced model"
symptom_in_wild:
  - "import order error: megatron-core 0.16 APIs missing until mindspeed.megatron_adaptor imported"
  - "npu_rms_norm raises 'too many arguments' or silently wrong because gamma dtype != x dtype"
  - "TransformerLayer plumbing crashes because DeepSeekV4Attention.forward returned a bare tensor, not (output, bias)"
  - "all_reduce_grad_fp32 TypeError unexpected kwarg in vendored Megatron-LM-miles"
  - "ModuleNotFoundError: miles.utils when MoE router path executes"
  - "loss/grad NaN appears only at layer count >= 2 (all-masked rows in sparse attention softmax)"
  - "OOM at optimizer.step on a single chip after a 2-layer fwd+bwd that itself fit"
root_cause: >
  Getting a real DSV4 layer to train on NPU is mostly an INTEGRATION problem, not an op problem.
  The five V4 core compute ops are already CANN-native and verified on A3; the work that almost
  derailed it was a chain of framework-contract / driver-signature / numerical-stability /
  single-chip-memory issues, each of which masquerades as an op or kernel gap but is not.
mistake_pattern: "treating a megatron-layer integration wall as an NPU op gap; overclaiming full-model from reduced-layer"
correction:
  - "Import mindspeed.megatron_adaptor FIRST — MindSpeed core_r0.16.0 (commit 8bf0959, dsa TND) DOES adapt Mcore 0.16; this is the official path, not a walkaround."
  - "npu_rms_norm shim: cast gamma to match the input/x dtype AND drop the extra args the V4 caller passes that the CANN signature does not accept."
  - "Honor the TransformerLayer contract: DeepSeekV4Attention.forward must return (output, None), not a bare output tensor."
  - "Apply the Megatron-LM-miles all_reduce_grad_fp32 kwarg-skew fork patch (_CopyToModelParallelRegion accepts + fp32 grad all-reduce)."
  - "Install the FULL miles package (/opt/miles_full or /opt/miles_v4) — the MoE-router path hard-depends on miles.utils; it is an env dependency, not code to stub."
  - "sparse_attn_torch all-masked-row softmax stabilization (production-worthy): nan_to_num(scores_max, neginf=0) before subtract, then clamp(exp args, max=30); this took 2-layer backward NaN from 282 -> 7 -> 0."
  - "Single 61GB chip is the memory wall: 1-layer + AdamW fits a full training iteration; 2-layer fwd+bwd fits but +optimizer OOMs; 4-layer bwd OOMs. Going deeper needs TP/PP or activation checkpointing — distributed engineering, NOT an NPU/op problem."
  - "State the verified basis honestly: reduced 1-2 layer, NOT full 43-layer. Do not claim full-model PASS."
evidence:
  - "docs/_meta/DSV4_NPU_PORTING_REPORT.md §3 (training side walkaround-vs-production table + §3.4 memory boundary table)"
  - "workspace/v4_attempt_2026_06_01/npu_native_shims/PROTECTED_flash_attention_npu_RESULT.md (single-layer fwd+bwd PASS, out=(64,1,512) finite, loss=0.0353 grad_norm=0.173)"
  - "workspace/v4_attempt_2026_06_01/V4_TRAINING_SIDE_OP_GAP.md (op inventory; CANN-native coverage correction)"
  - "git tags: v4-flash-attention-npu-working, v4-real-config-1layer-training-step-npu"
  - "commit a577cc0 (v4_e2e_megatron_layer_forward_npu.py fwd+bwd driver)"
  - "1-layer 4.42B: grad finite 526/526; 2-layer 8.84B: grad_norm=0.043, 1051/1051 finite (NOTE: these counts come from the run-stdout reported in commits 5c84d22/035661d/44bb058 + the porting report; the producing drivers v4_REAL_config_{1,2}layer_training_step_npu.py exist on disk but the captured PASS stdout was not committed as a separate log — only the single-layer flash numbers out=(64,1,512)/loss=0.0353/grad_norm=0.173 have a frozen on-disk log in PROTECTED_flash_attention_npu_RESULT.md)"
  - "memory/project_v4_megatron_layer_boundary.md, memory/project_v4_ops_cann_native_mapping.md"
applies_to: "radixark/miles DeepSeek-V4-Flash training ops on MindSpeed core_r0.16.0 (commit 8bf0959) + Megatron-Core 0.16.0rc0, real DSV4-Flash config reduced to 1-2 layers"
verified_on: "Ascend A3 NPU (single 61GB chip), tlrescue / verl-8.5.2 container, 2026-06-01"
unverified_on:
  - "full 43-layer DSV4-Flash (blocked on single-chip memory; needs TP/PP or activation checkpointing)"
  - "multi-chip distributed training (TP/PP not exercised in this session)"
  - "real miles training-delta RL e2e (RL loop weight-sync was synth-delta plumbing, not real gradients)"
deprecated_after: ""
---

# miles-003 — DSV4 megatron layer fwd+bwd+optimizer on NPU: the integration walkarounds

## Why this matters

The headline-grabbing question for DeepSeek-V4 on NPU was "do we need to hand-write a pile of
AscendC kernels for sparse-MLA / indexer / compressor?" The answer turned out to be **no** — the
five V4 core compute ops are CANN-native and verified on A3 (see miles-001 and
`project_v4_ops_cann_native_mapping`). The real work to get an actual `DeepSeekV4Attention`
megatron layer to do a full fwd+bwd+optimizer.step was a chain of **integration** issues. Each one
looks, on first contact, like an NPU op gap or a kernel bug. None of them were. Mistaking any of
them for an op problem would have sent us down a multi-day op-gen rabbit hole instead of a
one-line shim.

This lesson exists so the next person wiring a miles V4 layer into Megatron-on-NPU recognizes the
five integration walls fast, applies the known fix, and does **not** overclaim full-model from a
reduced-layer result.

## The op-layer is NOT the problem (read this first)

Verified on A3, CANN-native, no hand-written kernel needed:

| V4 op | CANN-native call | status |
|---|---|---|
| sparse-MLA fwd/bwd | `npu_nsa_select_attention` (D_qk=192, D_v=128, select_block=64, count=16) | production |
| C4 indexer | `npu_lightning_indexer` / `npu_sparse_lightning_indexer_grad_kl_loss` | production |
| compressor | `npu_nsa_compress_attention` | production |
| MLA prep | `npu_mla_prolog_v3` | production |
| rms_norm | `npu_rms_norm` | production (bit-exact) |

Only **hash-coding sinkhorn** (`#311`) and **act_quant fp8** (`#315`) had no CANN-native
equivalent and were generated via AscendC op-gen (both precision-verified). Do **not** write
"all V4 ops need hand-written kernels" — that was the early misjudgment the owner explicitly
corrected. The five above are basic ops CANN already provides; probe `dir(torch_npu)` before ever
reaching for op-gen.

## The five integration walls (with walkaround vs production classification)

### (1) MindSpeed core_r0.16.0 DOES adapt Mcore 0.16 — import the adaptor first ✅ production

Megatron-Core 0.16 APIs are not present until you do:

```python
import mindspeed.megatron_adaptor   # FIRST — before any megatron.core import that needs 0.16 surface
```

Use the **`core_r0.16.0`** MindSpeed branch (commit **8bf0959**, which carries dsa TND support).
This is the official MindSpeed path, not a hack. Note this is a moving target: an earlier memory
(`feedback_npu_megatron_via_mindspeed`) recorded MindSpeed as *not yet* supporting Mcore 0.16 —
that was true for the tilelang-ops port window; for the V4 layer work the `core_r0.16.0` branch
now adapts it. Always confirm the branch's Mcore coverage against the consumer version before
choosing a PR target.

### (2) npu_rms_norm shim: match gamma dtype + drop extra args ⚠️ walkaround (driver shim)

`npu_rms_norm` raises "too many arguments" because the V4 caller passes extra args the CANN
signature does not accept, and it silently misbehaves when `gamma` dtype differs from the input.
The shim must do both: cast gamma to the x dtype, and drop the surplus args. The proper fix is
MindSpeed adapting the `rms_norm` signature upstream — until then this is a consumer-side driver
shim. (rms_norm itself is bit-exact native; the gap is purely the call signature.)

### (3) TransformerLayer contract needs `(output, None)` return ⚠️ walkaround (contract alignment)

Megatron's `TransformerLayer` plumbing expects each attention submodule to return
`(output, bias)`. V4's `DeepSeekV4Attention.forward` returned a bare tensor, so the surrounding
layer crashed. Patch `DeepSeekV4Attention.forward` to return `(output, None)`. This is a
miles-side contract alignment (miles PR target), not an NPU issue at all.

### (4) Megatron-LM-miles all_reduce_grad_fp32 kwarg-skew fork patch ⚠️ walkaround → PR-ready

The vendored `Megatron-LM-miles` has a kwarg skew on `all_reduce_grad_fp32`
(`_CopyToModelParallelRegion` needs to accept the kwarg + do the fp32 grad all-reduce). The patch
lives **uncommitted** at `npu_native_shims/megatron_npu_patches/megatron_copy_to_tp_all_reduce_grad_fp32.patch`
(touches `megatron/core/tensor_parallel/mappings.py`), with a reference copy
`mappings_patched_reference.py` alongside. Bundle it into the miles PR set rather than a
standalone Megatron PR (per user direction).

> Do NOT confuse this with commit `6f3209b` on `Megatron-LM-miles fix/te_general_gemm_npu_fallback`
> — that is a *different* NPU patch (the `te_general_gemm` None-fallback guard in `moe_utils.py`,
> 8 lines). The fp32-grad-allreduce change is the uncommitted `.patch` above, not `6f3209b`.

### (5) miles MoE-router path hard-depends on miles.utils — install the full package ⚠️ env

The MoE router triggers `ModuleNotFoundError: miles.utils`. This is NOT a stub candidate — the
router genuinely needs the full miles package. Install it at `/opt/miles_full` (a.k.a.
`/opt/miles_v4`) and set `PYTHONPATH` accordingly. Document the dependency; do not try to fake it.
This is exactly the wall that separated the protected single-`DeepSeekV4Attention`-layer flash
result (which does NOT hit the MoE router) from the full reduced-config 1-layer model (which
does): the flash result stands independently, and the MoE/env dependency is a different
workstream, not a flash/attention defect.

## The production-worthy fix: all-masked-row softmax stability ✅ production

This one is genuinely PR-worthy, not a walkaround. At layer count >= 2 the 2-layer backward
produced NaN. Root cause: `sparse_attn_torch` masked softmax hits **all-masked rows** — every
entry in the row is masked, so `scores_max` is `-inf`, and `scores - scores_max` becomes
`-inf - (-inf) = NaN`; `exp` of large positive args also overflows.

Fix (standard masked-softmax guard, applied in `sparse_attn_torch`):

```python
scores_max = torch.nan_to_num(scores_max, neginf=0)   # -inf row-max -> 0, kills the inf-minus-inf NaN
...
attn = torch.exp(torch.clamp(scores - scores_max, max=30))  # clamp exp args to avoid overflow
```

This drove the NaN count **282 -> 7 -> 0** and unblocked the 2-layer backward. It is a
production-quality numerical stability guard that belongs upstream in miles — file it as a miles
PR.

## The single-61GB-chip memory wall (NOT an NPU/op problem)

Verified on one A3 61GB chip, real DSV4-Flash config (4096 hidden, 64 heads, 256-expert MoE),
reduced layer count:

| scale | result |
|---|---|
| 1 layer (4.42B) + AdamW | full training iteration fwd+bwd+**optimizer.step** PASS (grad finite 526/526) |
| 2 layer (8.84B) | fwd+bwd PASS (grad_norm=0.043, 1051/1051 finite); **+AdamW states OOM** (8.84B + Adam m/v ~70GB) |
| 4 layer (17.68B) | forward OK; **backward OOM** |

A 256-expert MoE at 4096 hidden is enormous per layer, so this is the ordinary large-model memory
reality. Going deeper is **distributed engineering** — tensor/pipeline parallel or activation
checkpointing — **not** an NPU or op gap. Do not log the OOM as an NPU defect.

## What is the verified basis (honesty boundary)

- ✅ A single `DeepSeekV4Attention` megatron layer (MLA + sparse/flash attention + indexer +
  compressor) does a full fwd+bwd training step on A3 NPU (protected/tagged).
- ✅ Real DSV4 config, **reduced to 1 layer**, full training iteration (fwd + loss + bwd +
  AdamW.step).
- ✅ Real DSV4 config, **reduced to 2 layers**, fwd+bwd (all grads finite).
- ❌ NOT full 43-layer. NOT multi-chip distributed. NOT real miles-gradient RL e2e (the RL loop's
  weight-sync was synth-delta plumbing).

The reduced 1-2 layer result is the verified basis. State it that way. The remaining gap to
full-model is single-chip memory (distributed engineering), and the remaining gap to real RL e2e
is swapping the synth-delta for real miles training gradients — neither is an NPU/op blocker.

## Cross-references

- `miles-001` — the tilelang DSAMLA op port (alternative to the CANN-native route used here).
- `project_v4_ops_cann_native_mapping` (memory) — the five CANN-native op mappings.
- `project_v4_megatron_layer_boundary` (memory) — the recipe + the MoE/full-package boundary.
- `docs/_meta/DSV4_NPU_PORTING_REPORT.md` §3 — the authoritative training-side walkaround table.
- `feedback_cann_has_basic_ops_dont_hand_gen` (memory) — verify CANN coverage before op-gen.
