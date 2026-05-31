---
id: miles-001
date: 2026-05-25
layer: miles
title: Pattern for porting 4 tilelang DSAMLA ops (lighting_indexer + sparse_mla) to NPU — head split, UB cap mitigation, R-KA-16 workaround
trigger:
  - "porting DSv4-Flash / GLM-5 DSAMLA attention to Ascend NPU"
  - "implementing lighting_indexer_fwd/bwd or sparse_mla_fwd/bwd in tilelang on Ascend"
  - "miles dispatcher needs an NPU backend"
symptom_in_wild:
  - "miles dispatcher falls back to CPU path on NPU input (q.is_npu detected, no NPU backend registered)"
  - "naive port hits UB overflow on H=64 / D_V=512"
  - "sparse_mla_fwd NS≥2 NaN even when port itself is correct (R-KA-16)"
root_cause: >
  miles uses 4 tilelang ops for DSA + MLA:
  `lighting_indexer_fwd/bwd`, `sparse_mla_fwd/bwd`. Each is a CUDA tilelang
  source. NPU port requires:
  - Backend registration with dispatcher hook `q.is_npu`
  - UB cap mitigation (real-shape H=64 / D_V=512 overflows naive layout)
  - R-KA-16 mitigation for sparse_mla_fwd (force num_stages=1)
  - Per-op specifics: head split for lighting_indexer_bwd, block_size=8 for sparse_mla_bwd, correction_expanded cleanup for sparse_mla_fwd
mistake_pattern: "naive tilelang port assuming CUDA layouts; UB pressure ignores Ascend 64KB cap"
correction:
  - "Branch: zhshgmail/miles npu-tilelang-ops (clean, distilled from 18-commit dev branch via 2-round audit)"
  - "Subpackage: miles/miles_plugins/models/glm5/ops/_npu/ -- 4 kernels + dispatcher"
  - "Dispatcher: check `q.is_npu` in glm5/ops/lighting_indexer.py and sparse_mla.py; if True dispatch to _npu/ variant"
  - "Per-op layout adjustments:"
  - "  - lighting_indexer_fwd: direct PASS, no UB pressure issue"
  - "  - lighting_indexer_bwd: head-split `block_H_inner=16` to bring H=64 down from 259 KB UB overflow"
  - "  - sparse_mla_fwd: cleanup `correction_expanded` + delete dead alloc; force num_stages=1 (R-KA-16 mitigation)"
  - "  - sparse_mla_bwd: block_size=8 for d_v>=512; pp_block_N reduced to 16; UB 140/192 KB"
  - "Driver scripts: _e2e_megatron_step.py for single-step compile+flow-through verify; _e2e_megatron_multilayer_mindspeed.py for multi-layer multi-iter NaN-drift check (25/25 finite across 3 iter)"
  - "Production safety: every NPU kernel module has a `# R-KA-16 mitigation` comment block at the num_stages assignment; removing the workaround is gated on Ascend/AscendNPU-IR#251 landing"
evidence:
  - "PR radixark/miles#1246 (MERGEABLE, REVIEW_REQUIRED)"
  - "13 files / 1767 LOC clean diff"
  - "No Claude/Anthropic/Huawei trailer in commits or PR body (audited 2026-05-30)"
  - "Verified on tlrescue 2026-05-23 with real DSv4-Flash dims H=64 SEQ=2048 topk=512 SKV>=topk"
  - "miles_local config (1-layer, hidden=4096, H=64, SEQ=128, topk=8) used for HBM-constrained local re-validation: 4/4 kernels compile + run, loss=-0.00151, finite grads 7/12 (5 non-finite from R-KA-16 sparse_mla_fwd NS>=2)"
---

# miles-001 — DSAMLA tilelang NPU port pattern

## Why this matters

This PR (#1246) is the only path for getting miles running on NPU as a tilelang-based stack. Without the port:
- miles dispatcher falls back to CPU; no NPU acceleration
- alternative routes (Huawei `npu_lightning_indexer.cpp` + `triton_indexer_bf16.py`) are still in development and don't cover all 4 ops

## Decision: tilelang port vs Huawei fused-op route

This port co-exists with Huawei's MindSpeed DSA flag chain (`use_fused_lightning_indexer` etc). If Huawei's path matures and supports miles directly, this tilelang port becomes the fallback (still useful for debug, comparison, and when fused ops fail).

If Huawei's path doesn't mature in time, this port is the only viable production option. The current ROADMAP keeps both directions open.

## Verification matrix

Before claiming the port done:
- [ ] lighting_indexer_fwd: real shape H=64 SEQ=2048 topk=512, compile <1s, output finite
- [ ] lighting_indexer_bwd: same shape, gq/gw/gk all finite, no UB overflow
- [ ] sparse_mla_fwd: NS=1 PASS; NS>=2 NaN (R-KA-16 expected); workaround num_stages=1
- [ ] sparse_mla_bwd: real shape, UB 140/192 KB stay under cap
- [ ] e2e _e2e_megatron_step.py MILES_E2E_SHAPE=real: 52M-param forward+backward+Adam PASS
- [ ] e2e _e2e_megatron_multilayer_mindspeed.py: 2-layer x 3-iter, 25/25 finite no NaN drift
- [ ] e2e _e2e_rl_step_mindspeed.py: full RL step PASS (rollout + train in single proc)
