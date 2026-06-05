---
id: tilelang-003
date: 2026-06-02
layer: tilelang
title: tilelang-mlir-ascend sparse_mla_fwd silently produces wrong output at heads < 16 (H-padding auto-handle is incorrect)
trigger:
  - "running tilelang-mlir-ascend examples/sparse_mla_fwd.py with heads < 16"
  - "sparse-MLA / NSA-style attention on tilelang-ascend with a small attention-head count"
  - "tilelang sparse_mla output ~40% mismatch vs pytorch reference but no error raised"
symptom_in_wild:
  - "examples/sparse_mla_fwd.py --heads 4 -> assert_close FAIL, 44.3% elements mismatch (vs its own pytorch ref)"
  - "--heads 8 -> 35.9% mismatch; --heads 16 and --heads 32 -> All check passed!"
  - "no exception/assert fires — the kernel runs to completion and returns a wrong tensor"
root_cause: >
  For heads (head_kv when kv_group==1) below 16, padded_H = max(next_power_of_2(head_kv), 16)
  pads the head dim up to 16. The example takes the `padded_H != head_kv` branch whose assert
  (kv_group==1) PASSES and claims H-padding is "handled automatically" — but the Q-copy /
  output-copy masking for the padded heads is in fact wrong, so the kernel computes over the
  padding lanes incorrectly and returns a silently-wrong result. Threshold is clean: wrong at
  heads ∈ {4,8}, correct at heads ∈ {16,32}.
mistake_pattern: "silent wrong output below a tiling threshold — the code asserts the config is auto-handled but the padding mask is incorrect; no error, just wrong numbers"
correction:
  - "Treat heads<16 sparse_mla_fwd on tilelang-ascend as UNSAFE until fixed: either it should assert-out (refuse) or correctly mask the padded heads. Currently it does neither — it returns wrong output."
  - "For consumers (V4 etc.): if an attention config has heads<16 (e.g. reduced/kv_group splits), do NOT trust tilelang-ascend sparse_mla_fwd output; use heads>=16 or a different impl (CANN native) until the padding fix lands."
  - "Upstream: file at tile-ai/tilelang-mlir-ascend — the `padded_H != head_kv` branch's Q-copy/output-copy mask handling is incorrect for head_kv<16. Repro: examples/sparse_mla_fwd.py --heads 4 (and --heads 8). Fix should make the auto-H-padding mask correct, OR assert-refuse heads<16 instead of silently mis-computing."
  - "Diagnostic: sweep --heads {4,8,16,32}; a clean pass-threshold at 16 + the padded_H=max(...,16) logic localizes it to the sub-16 padding path."
evidence:
  - "A3 2026-06-02 (tilelang-mlir-ascend v0.1.1.030, tlrescue): --heads 4 FAIL 44.3% / --heads 8 FAIL 35.9% / --heads 16 PASS / --heads 32 PASS, all vs the example's own pytorch ref (rtol=5e-3 atol=1e-2)"
  - "code: examples/sparse_mla_fwd.py — `padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)` + `if padded_H != head_kv: assert kv_group==1, 'H padding handled automatically...'`"
  - "report §附录 A.3 (DSV4_NPU_PORTING_REPORT.md)"
applies_to:
  - "tile-ai/tilelang-mlir-ascend examples/sparse_mla_fwd.py @ v0.1.1.030 (MLIR/bishengir backend, A3)"
verified_on:
  - "Ascend A3 NPU (Ascend910_9382), tlrescue, CANN 8.5.2, 2026-06-02"
unverified_on:
  - "whether the same sub-16-head padding bug affects other tilelang-ascend attention examples (flash_attn_npuir / fp8_indexer) — only sparse_mla_fwd tested"
  - "whether a newer tilelang-mlir-ascend revision fixes it"
deprecated_after: ""  # FIXED 2026-06-02 (fork blue/fix/sparse-mla-heads-lt-block-h a19acd5): bound output store to valid head rows; heads=4/8/16/32 all PASS
---

# tilelang-003 — sparse_mla_fwd silently wrong at heads < 16

## Why this matters

Surfaced while doing the tilelang-ascend-vs-CANN comparison the owner asked for (report §A.3). The
tilelang `sparse_mla_fwd` example PASSES at the default heads=32, which is what earlier "tilelang
example runs on NPU ✓" claims were based on. But sweeping the head count revealed it **silently
returns wrong output at heads < 16** (44% mismatch at 4, 36% at 8), with no error raised. Silent
wrong output is more dangerous than a crash — a consumer gets a plausible-looking tensor that is
~40% incorrect.

## The threshold + root cause

| heads | result |
|---|---|
| 4 | FAIL — 44.3% mismatch vs pytorch ref |
| 8 | FAIL — 35.9% mismatch |
| 16 | PASS |
| 32 | PASS |

`padded_H = max(next_power_of_2(head_kv), 16)`. For head_kv < 16 the head dim is padded to 16 and the
code takes the `padded_H != head_kv` branch, whose `assert kv_group == 1` passes and a comment claims
the H padding is "handled automatically." It is not — the Q-copy / output-copy masking over the
padded lanes is wrong, so the result is computed incorrectly but returned without error.

## What to do

- **Consumer side**: do not trust tilelang-ascend `sparse_mla_fwd` for heads<16; use heads≥16 or CANN
  native until fixed. (V4-Flash is 64 heads = safe; small-head / reduced configs are not.)
- **Upstream**: file at tile-ai/tilelang-mlir-ascend; the fix should either correctly mask the padded
  heads or assert-refuse heads<16 (don't silently mis-compute). Repro: `examples/sparse_mla_fwd.py
  --heads 4`.

Related: `tilelang-001` (UB budget), `tilelang-002` (vbrc), `bishengir-001` (R-KA-16) — other
tilelang-ascend / bishengir correctness/diagnostic issues found in this project.
