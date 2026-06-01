---
id: miles-002
date: 2026-06-01
layer: miles
title: V4 (DeepSeek-family) training ops are CANN-native first — probe dir(torch_npu) before op-gen
trigger:
  - "porting DSv4-Flash / DeepSeek-family training-side ops (sparse-MLA, C4 indexer, compressor, MLA prep, rms_norm) to Ascend NPU"
  - "a miles training op is a @tilelang.jit / CUDA kernel and the NPU has no obvious path"
  - "about to /ascendc-op-gen a custom AscendC kernel for a DeepSeek training op"
  - "estimating how many hand-written kernels a V4 training-side port needs"
symptom_in_wild:
  - "planning doc says 'all 6 tilelang kernels need hand-written AscendC' before any dir(torch_npu) probe was run"
  - "op-gen queue spun up for sparse-MLA / indexer / compressor that CANN already exposes natively"
  - "effort spent designing FA-class TileLang-IL chains for sparse_mla_fwd/bwd that npu_nsa_select_attention already covers (incl. bwd softmax max/sum)"
root_cause: >
  Reflexive "this is a tilelang/CUDA kernel, therefore it needs a hand-written
  NPU kernel" inference, made before probing CANN coverage. DeepSeek-family
  training primitives (NSA select/compress attention, lightning indexer,
  MLA-prolog, rms_norm) are exactly the ops CANN has been building native
  support for — the tilelang source is a CUDA implementation detail, not a
  signal that NPU lacks the op.
mistake_pattern: "reflexive op-gen before CANN coverage check (No-CANN-Delegation red line mis-applied to product-deliverable training ops)"
correction:
  - "Before op-gen'ing ANY V4 / DeepSeek-family training kernel: probe `dir(torch_npu)` (and `torch_npu.npu_*`) for nsa / mla / sparse / lightning / rms substrings — CANN very likely already has it."
  - "Map first, op-gen only the genuine gaps. CANN-native mapping (verified-run vs spec-matched noted per op — capture a log before claiming numeric equivalence for the spec-matched ones):"
  - "  - sparse-MLA fwd/bwd -> npu_nsa_select_attention (D_qk=192, D_v=128, select_block=64, count=16; returns attn + softmax max/sum for bwd) [VERIFIED-RUN on A3: attn (128,4,128) finite, 94.9us]"
  - "  - C4 indexer -> npu_lightning_indexer / npu_sparse_lightning_indexer_grad_kl_loss [grad+KL variant used in training; verified in the layer fwd+bwd run]"
  - "  - compressor -> npu_nsa_compress_attention [SPEC-MATCHED to V4; capture a standalone run-log before claiming numeric equivalence]"
  - "  - MLA prep -> npu_mla_prolog_v3 [COVERAGE-CONFIRMED in dispatch; no standalone verified-run log captured yet]"
  - "  - rms_norm -> npu_rms_norm (bit-exact, 0.000e+00) [VERIFIED-RUN]"
  - "Only hand-write AscendC for primitives CANN genuinely lacks. Two such gaps existed: hash-coding sinkhorn (hc_split_sinkhorn) and fp8 act_quant — both op-gen'd and precision-verified."
  - "Reconcile the planning doc: do NOT carry forward 'all V4 ops need hand-written kernels'. The real split is 5 CANN-native + 2 op-gen, not 6+ hand-written."
evidence:
  - "docs/_meta/DSV4_NPU_PORTING_REPORT.md §3.2 (CANN-native mapping table, all 实测) + §3.5 (production vs walkaround split)"
  - "workspace/v4_attempt_2026_06_01/V4_TRAINING_SIDE_OP_GAP.md (early misjudgment: '6 TileLang kernel 要 NPU path' — corrected here)"
  - "workspace/v4_attempt_2026_06_01/npu_native_shims/PROTECTED_flash_attention_npu_RESULT.md (single DeepSeekV4Attention megatron layer fwd+bwd PASS on A3 via CANN-native shims; commit a577cc0)"
  - "v4_sparse_attn_nsa_shim.py / v4_indexer_lightning_shim.py / v4_npu_native_dispatch.py (the verified CANN-native dispatch shims)"
  - "sinkhorn AscendC op-gen: 28/28+28/28, perf 5.34x symmetric (miles op-gen #311). act_quant: 24/24+24/24 byte-exact fp8 + bit-exact fp32 (#315)"
  - "Owner correction memory: feedback_cann_has_basic_ops_dont_hand_gen.md, project_v4_ops_cann_native_mapping.md (2026-06-01)"
applies_to:
  - "miles DeepSeek-V4-Flash training-side ops on Ascend A3 NPU @ 2026-06-01"
  - "DeepSeek-family (NSA / MLA / lightning-indexer) training kernels generally — probe CANN before op-gen"
verified_on:
  - "Ascend A3 NPU, tlrescue container, MindSpeed core_r0.16.0, 2026-06-01"
  - "Scope: reduced-layer (1-layer full training iteration fwd+bwd+AdamW; 2-layer fwd+bwd). NOT full 43-layer."
unverified_on:
  - "Full 43-layer DSV4 real config (single 61GB chip OOMs at 2-layer + AdamW; needs TP/PP/activation-checkpointing — distributed engineering, not an op gap)"
  - "Real miles training-delta RL e2e (RL loop currently driven by synth-delta placeholder)"
deprecated_after: ""
---

# miles-002 — V4 training ops are CANN-native first

## Why this matters

The expensive mistake this lesson exists to prevent: looking at miles' DeepSeek-V4
training ops — all of which are `@tilelang.jit` / CUDA kernels — and concluding
"these 6 kernels all need hand-written AscendC on NPU." That conclusion was written
into `V4_TRAINING_SIDE_OP_GAP.md` before anyone probed `dir(torch_npu)`. It was wrong.

CANN already ships native ops for the DeepSeek training primitives. The fact that the
*reference* implementation is a tilelang/CUDA kernel says nothing about NPU coverage —
NSA (Native Sparse Attention), MLA, and the lightning indexer are precisely the ops
CANN has been adding. Probing first would have saved the effort of queuing op-gen and
designing FA-class TileLang-IL chains for ops the platform already had.

## The actual split (verified on A3, reduced-layer)

5 of the V4 training core ops are **CANN-native** (all ran on A3):

| V4 training op | CANN-native op | notes |
|---|---|---|
| sparse-MLA fwd/bwd | `npu_nsa_select_attention` | D_qk=192, D_v=128, select_block=64, count=16; returns attn + softmax max/sum for bwd |
| C4 indexer | `npu_lightning_indexer` / `npu_sparse_lightning_indexer_grad_kl_loss` | fwd + grad+KL-loss for training |
| compressor | `npu_nsa_compress_attention` | NSA-style compress attention |
| MLA prep | `npu_mla_prolog_v3` | |
| rms_norm | `npu_rms_norm` | bit-exact, 0.000e+00 |

Only **2** ops genuinely lacked a native op and needed AscendC op-gen, both
precision-verified:

- **hash-coding sinkhorn** (`hc_split_sinkhorn`): 28/28 + 28/28, perf 5.34× symmetric (#311)
- **fp8 act_quant**: 24/24 + 24/24, byte-exact fp8 + bit-exact fp32 (#315)

So the real picture is **5 CANN-native + 2 op-gen**, not "6+ hand-written." The
proof point: a single `DeepSeekV4Attention` megatron layer (MLA + sparse/flash
attention + indexer + compressor) does a full fwd+bwd training step on A3 NPU
purely via the CANN-native shims (`v4_sparse_attn_nsa_shim.py`,
`v4_indexer_lightning_shim.py`, `v4_npu_native_dispatch.py`), no hand-written
attention kernel involved.

## The probe that should have come first

Before op-gen on any DeepSeek-family training op, do this in the NPU container:

```python
import torch_npu
[n for n in dir(torch_npu) if any(k in n.lower() for k in ("nsa","mla","sparse","lightning","rms"))]
```

If a matching native op exists, map to it and verify equivalence (run on A3, check
finite + shape + tolerance band). Op-gen is only for the residue the probe doesn't
cover.

This is the **No-CANN-Delegation red line being mis-applied**: that red line forbids
delegating to CANN only for *from-zero benchmark-kernel deliverables*. For a product
training port, native-first is correct and explicitly endorsed by the owner
(`feedback_cann_has_basic_ops_dont_hand_gen.md`). Reaching for a custom fused kernel
when a native op exists is a perf optimization at best, dead effort at worst.

## Honesty boundary (do not overclaim)

- The verified basis is **reduced-layer**: 1-layer full training iteration
  (fwd + bwd + AdamW.step, all grads finite) and 2-layer fwd+bwd. **NOT** the full
  43-layer DSV4 model. Deeper layers OOM on a single 61GB chip at 2-layer + AdamW —
  that is distributed engineering (TP/PP/activation checkpointing), not an op gap.
- The CANN-native mapping is verified for the op dims used in the reduced-layer runs;
  treat full-config dims as unverified until re-run.
- The RL loop e2e is still driven by a synth-delta placeholder, not real miles
  training gradients. "Training ops have no NPU blind spot" is honest; "real RL
  training e2e closed" is not yet.

## Related lessons

- `miles-001` — the *alternative* tilelang-port route for the same DSAMLA ops
  (lighting_indexer + sparse_mla). That path hand-ports the tilelang kernels to NPU
  (UB-cap / R-KA-16 mitigations). This lesson (miles-002) is the **preferred** route
  when CANN native coverage exists: native-first, tilelang-port only as fallback for
  ops CANN can't lower.
