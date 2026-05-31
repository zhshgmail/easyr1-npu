---
id: sglang-003
date: 2026-05-30
layer: sglang
title: /update_weights_from_disk reload path skips stacked_params_mapping; FusedMoE _load_w13 narrow regression on MoE models
trigger:
  - "POST /update_weights_from_disk after initial Engine load on MoE model"
  - "RuntimeError: start (0) + length (4096) exceeds dimension size (1408) in fused_moe_triton/layer.py:482 _load_w13"
  - "RL weight sync from Megatron actor back into sglang rollout engine for any MoE model"
  - "Initial Engine load succeeds; only the reload fails"
symptom_in_wild:
  - "RuntimeError: start (0) + length (4096) exceeds dimension size (1408) at sglang fused_moe_triton/layer.py:482 _load_w13 during update_weights_from_disk"
  - "Initial Engine init+load passes; reload crashes"
  - "Dense fab ckpt (first_k_dense_replace=1) reloads fine; same fab with MoE-active (first_k_dense_replace=0) crashes"
  - "Verified on 1-layer DSv4-Flash fab ckpt with 4 routed + 1 shared experts, top-k=2"
root_cause: >
  Initial Engine load walks `stacked_params_mapping` which is a list of
  (shard_name, weight_name, shard_id) triples telling FusedMoE _load_w13 how
  to narrow the w1/w3 tensors into separate slots. Reload via
  `/update_weights_from_disk` does NOT consult the same mapping. The weight
  tensor arrives at _load_w13 already-stacked but the slot is sized for an
  individual shard. narrow(0, 4096) blows up against dim 1408.

  This is a regression — at some earlier sglang version the reload path
  worked. Maintainer should decide whether reload path should honor
  stacked_params_mapping or whether _update_weights_from_disk should
  pre-split.
mistake_pattern: "code path divergence between initial-load and reload paths in a state-mutating API"
correction:
  - "Workaround on consumer side: use dense-only fab ckpt for weight-sync validation. miles MoE-active path stays blocked until upstream fixes."
  - "Filed Issue sgl-project/sglang#26794 with full repro, dense PASS evidence, and the dim arithmetic."
  - "Reproducer: workspace/T32_tilelang_rescue/test_5step_weight_sync.py (5-step driver synthesizes seeded delta, renames Megatron->HF, merges, POSTs /update_weights_from_disk, byte-checks)"
  - "When the upstream fix lands: re-run on MoE-active fab to validate full path; close this entry."
evidence:
  - "Issue: https://github.com/sgl-project/sglang/issues/26794 (OPEN, awaiting maintainer)"
  - "Dense path PASS evidence: 5/5 step rollout outputs distinct (6/6 distinct rollout sequences)"
  - "MoE-active path crash: in narrow(0, 4096), the 4096 is w1+w3 stacked dim (2*2048 for 2-expert top-k stack); slot 1408 is single-expert slot — the mismatch is structural"
  - "Two-layer R3 flag system (enable_return_routed_experts engine ctor + return_routed_experts per-request) confirmed working on same MoE fab — R3 plumbing PASS"
---

# sglang-003 — FusedMoE _load_w13 reload narrow regression

## Why this matters

This is the only thing blocking MoE-active full-path weight-sync validation. Dense path is fully validated (5-step weight sync PASS). The MoE path's load+rollout-with-R3 also works — only the reload step crashes. The structural fix has to be on sglang side.

## How to know it's this bug vs your own bug

Quick discriminators:
1. Does initial Engine load succeed? If yes, it's not your weight format
2. Does dense version of the same model reload fine? If yes, MoE-specific
3. Is the crash at `narrow(0, X) > dim Y` with X being a power-of-2 multiple of Y? If yes, stacked vs sharded mismatch (this bug)
4. Did you do the rename Megatron-to-HF correctly? Compare with `workspace/T32_tilelang_rescue/megatron_to_hf_rename.py` byte-clean closure

If 1+2+3 all match, you're looking at #26794.

## R3 plumbing is independent

Note: the R3 (return_routed_experts) flag system works independently of this bug. Engine init with `enable_return_routed_experts=True` + per-request `return_routed_experts=True` returns base64-encoded per-token expert IDs in meta_info correctly. Only the reload path is broken.
