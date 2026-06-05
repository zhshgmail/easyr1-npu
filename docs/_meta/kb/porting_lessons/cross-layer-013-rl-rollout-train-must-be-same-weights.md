---
id: cross-layer-013
date: 2026-06-02
layer: cross-layer
title: A "real trained delta" pushed from the training model into an INDEPENDENT rollout model is not a meaningful transfer — RL rollout and train must be the same weights
trigger:
  - "bridging a training-side weight delta into a separate inference/rollout engine to 'close the RL loop'"
  - "exporting a delta from a megatron/mindspeed model and update_weights_from_tensor into an sglang Engine"
  - "claiming 'real training drives the rollout' after pushing a cross-stack delta"
  - "two models share tensor shapes so a delta 'applies cleanly' — assuming that means it's meaningful"
symptom_in_wild:
  - "applied delta L2 exactly equals the exported delta L2 (transform() was identity) and you conclude the transfer worked"
  - "re-rollout output changes after the push, and you call it 'real-training weight-sync changes inference'"
  - "the training model and the rollout model were each random-initialized independently (different init, no shared ckpt)"
  - "RL loop 'closes' but rollout quality is indistinguishable from a synthetic-perturbation run"
root_cause: >
  A gradient/AdamW delta Δ is only meaningful relative to the specific weights W_train it was
  computed on. If the rollout model W_roll is an INDEPENDENT random init (merely same shape),
  then W_roll + Δ is W_roll plus an arbitrary fixed vector — mathematically no different from a
  synthetic perturbation. Same tensor shape ≠ same model. A real RL loop requires the rollout
  weights and the training weights to be the SAME parameters (shared, or synced from a common
  checkpoint), so that Δ moves W_roll along the direction training actually improved.
mistake_pattern: "treating shape-compatible cross-model delta application as a meaningful trained-weight transfer; 'plumbing works' mis-stated as 'real training drives rollout'"
correction:
  - "Before claiming an RL loop is driven by real training, verify rollout and train share weights: same ckpt initializes both, OR a verified train->rollout weight conversion maps W_train onto W_roll (not two independent random inits)."
  - "If you only have a cross-stack delta push that changes inference, state it as PLUMBING-ONLY: 'update_weights_from_tensor transports a tensor into the Engine and changes inference' — NOT 'real training drives rollout'."
  - "The honest discriminator: would adding the SAME-magnitude synthetic random delta change rollout the same way? If yes (and it does, when the base is unrelated), your 'real delta' carries no training signal into THIS model."
  - "For V4 specifically: to make megatron-trained weights drive the sglang rollout, initialize both from one real DSv4 ckpt (or convert megatron->HF/sglang layout), then the attention-only update_weights_from_tensor path (KB cross-layer-012) becomes a meaningful sync — the #26794 dodge is still valid, but only on shared weights."
  - "Same-shape is necessary but NOT sufficient. transform() shape-asserting and applied-L2-matching prove layout correctness, NOT semantic correctness."
evidence:
  - "2026-06-02 owner catch (Discord): '如果参数在两个不同层数模型间移动是怎么做到的？是不是没有移动？' — exposed that the task-dag-realdelta n3 bridge moved a megatron delta onto an unrelated sglang random-init fab."
  - "workspace/task-dag-realdelta/RESULTS.md (CORRECTION block) + n3 log: applied L2 == megatron delta L2 (identity transform) but base weights are independent random inits"
  - "n1_export_real_delta.py: megatron TransformerBlock random-init (model_parallel_cuda_manual_seed 1234); fabricate_dsv4_REAL_1layer_ckpt.py: sglang fab _randn — two independent inits, same MLA shapes"
  - "docs/_meta/DSV4_NPU_PORTING_REPORT.md §2.4 (the retraction) + §0 真e2e缺口 row (1): shared-weights requirement"
applies_to:
  - "any cross-framework RL loop where rollout (sglang/vllm) and train (megatron/miles) are separate processes"
  - "V4 DSv4-Flash RL-on-NPU specifically (megatron train + sglang rollout)"
verified_on:
  - "Reasoning verified against on-disk artifacts 2026-06-02 (independent random inits confirmed in both fab + training driver)"
unverified_on:
  - "the FIX (shared-ckpt init of both sides) — not yet built; this lesson documents the requirement, not a demonstrated shared-weight RL loop"
deprecated_after: ""
---

# cross-layer-013 — RL rollout and train must be the SAME weights, not just same-shape

> **2026-06-05 re-baseline (M2)**: 独立 agent 对最新 miles main 校对——纯机制级教训(delta 只在共享权重上有意义),无版本特定面,最新 main 不与之冲突。**仍适用,无需改**。

## Why this matters

This lesson exists because I (the agent) shipped a misleading "PASS": a cross-stack bridge that
pushed a *real trained delta* from a megatron training iteration into an sglang RL rollout loop,
and I framed it as "real-training weight-sync changes inference — RL loop closed with REAL delta."
The owner caught it in one question: if the two models have different structure, did anything
actually *move*? The answer is no — not meaningfully.

## The trap

1. You have a training model `W_train` (megatron) and a rollout model `W_roll` (sglang fab).
2. They share tensor shapes (same MLA dims: wq_a 1024×4096, etc.).
3. You compute a real AdamW delta `Δ = W_train_post − W_train_pre`.
4. You push `W_roll + Δ` via `update_weights_from_tensor`. It succeeds; applied L2 == `‖Δ‖`
   exactly (the layout `transform()` is identity); re-rollout output changes.
5. You conclude "real training drives the rollout." **Wrong.**

`W_train` and `W_roll` were **independently random-initialized**. `Δ` is the direction that
improved `W_train`'s loss; it has no relationship to `W_roll`'s loss landscape. `W_roll + Δ` is
`W_roll` plus an arbitrary fixed vector — exactly what a synthetic perturbation is. The rollout
changes for the same reason a random delta changes it, not because training taught the rollout
model anything.

## What was actually proven vs claimed

- ✅ **Proven (plumbing)**: a megatron-derived tensor can be transported across the stack
  boundary (a `.pt` on the shared mount) and applied to the sglang Engine via
  `update_weights_from_tensor`, changing inference. The layout remap (KB cross-layer-012) and
  shape-assert are correct.
- ❌ **Not proven (and retracted)**: that real training drives the RL rollout. It does not —
  the delta lands on an unrelated model.

## The requirement for a real RL loop

Rollout weights and training weights must be the **same parameters**:
- initialize both sides from **one real DSv4 ckpt** (so `W_roll` and `W_train` start identical and
  stay related), OR
- implement a verified **megatron → sglang/HF weight conversion** so each rollout weight is the
  transformed image of the corresponding training weight.

Only then does `W_roll + remap(Δ)` move the rollout model along the direction training improved.
The attention-only `update_weights_from_tensor` #26794-dodge (cross-layer-012) is still the right
sync mechanism — but it's only meaningful on shared weights.

## The honest discriminator (use this before claiming RL closure)

> Would adding a same-magnitude *synthetic random* delta change the rollout the same way?

If yes, your "real delta" carries no training signal into this model — you have plumbing, not
training. Same-shape + applied-L2-match prove **layout** correctness, never **semantic**
correctness.

## Cross-references

- `cross-layer-012` — the attention-only update_weights_from_tensor mechanism (correct sync path; this lesson adds the shared-weights precondition).
- `docs/_meta/DSV4_NPU_PORTING_REPORT.md` §2.4 — the retraction in the project report.
- `workspace/task-dag-realdelta/RESULTS.md` — the corrected DAG result.
