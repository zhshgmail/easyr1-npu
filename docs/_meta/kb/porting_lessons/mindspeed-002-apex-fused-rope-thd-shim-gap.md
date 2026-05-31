---
id: mindspeed-002
date: 2026-05-28
layer: mindspeed
title: MindSpeed adapter for Mcore 0.16 lacks apex.transformer.functional.fused_apply_rotary_pos_emb_thd shim; miles glm5.fuse_rope can't import
trigger:
  - "miles glm5.py fuse_rope path on Ascend NPU with mindspeed installed"
  - "ModuleNotFoundError: No module named 'apex.transformer' (even after mindspeed install)"
  - "TE / Megatron downstream code that calls apex's THD-layout fused rope kernel directly"
  - "porting any GLM-5-family model with rope on NPU"
symptom_in_wild:
  - "ModuleNotFoundError: No module named 'apex.transformer' at miles glm5.py:fuse_rope"
  - "Stack trace from MindSpeed adaptor.apply() down to apex import"
  - "`pip show apex` shows no apex, `dir(mindspeed.<...>)` shows no fused_apply_rotary_pos_emb_thd register"
root_cause: >
  MindSpeed's `RequirementsBasicFeature.apex_adaptation` registers stubs for
  several apex.* submodules but does not cover
  `apex.transformer.functional.fused_apply_rotary_pos_emb_thd`. miles's
  GLM-5 `fuse_rope` imports this symbol directly (THD-layout is required for
  variable-length attention).

  Root issue is incomplete coverage of MindSpeed's apex shim set against
  Mcore 0.16's actual apex surface. Not a bug in either library; a coverage
  gap that surfaces when a downstream model uses an underused apex entry.
mistake_pattern: "shim layer not exhaustive enough to cover real downstream consumers"
correction:
  - "PR Ascend/MindSpeed#3509: 38-line self-contained pure-torch `_fused_apply_rotary_pos_emb_thd_fallback` + 1 line `pm.register_patch('apex.transformer.functional.fused_apply_rotary_pos_emb_thd', _fused_apply_rotary_pos_emb_thd_fallback)`"
  - "Self-contained = NO imports from mindspeed internals; minimal Megatron checkout (no apex, no TE) can still test it"
  - "Place: `mindspeed/features_manager/megatron_basic/requirements_basic.py`"
  - "Algorithm: ROPE in THD layout = treat each batch sequence as a separate row; per-row cos/sin applied; equivalent to grouped reshape + complex-rotation. Implementation matches Megatron's reference within fp32 round-off."
  - "Verification: miles `_e2e_megatron_step.py` MILES_E2E_SHAPE=real PASSES after patch; without patch crashes at import."
evidence:
  - "PR: https://gitcode.com/Ascend/MindSpeed/merge_requests/3509 (OPEN, REVIEW_REQUIRED)"
  - "Author commit: `9aa2f75f` (zhshgmail <zhengshencn@gmail.com>) -- note force-push amend to remove huawei email per user 2026-05-30"
  - "miles call site: `miles/.../glm5.py:fuse_rope` -- direct apex import (not patched by mindspeed)"
  - "Mcore 0.16 calls apex.* directly in rope path; pure-torch fallback is sufficient because rope is not on the hot path for short SEQ"
  - "Memory: feedback_npu_megatron_via_mindspeed.md -- MindSpeed-not-Megatron-LM-direct is the canonical NPU adapter location"
---

# mindspeed-002 — apex.transformer.functional fused-rope-thd shim gap

## Why this matters

This shim is the last layer between MindSpeed-adapted Megatron and a working GLM-5/DSv4-Flash model on NPU. Without it, the entire MindSpeed approach (replacing direct Megatron patching) is gated by a one-symbol import.

## Pattern for similar shim gaps

When you see a `ModuleNotFoundError: No module named 'apex.<X>'` on a system that has MindSpeed installed:

1. Confirm MindSpeed installed correctly: `python -c "import mindspeed; print(mindspeed.__version__)"`
2. Search MindSpeed source for which apex.* it already shims: `grep -r "register_patch.*apex" mindspeed/features_manager/`
3. If your missing symbol isn't in the list, you need a new shim — follow the pattern in this PR (self-contained, no mindspeed internals, pure-torch fallback)
4. The shim must be self-contained because tests with minimal Megatron checkouts shouldn't pull in mindspeed itself

## Future: what other apex.transformer.functional symbols might surface

Likely candidates if porting other model families:
- `fused_apply_rotary_pos_emb` (the non-THD variant) — already shimmed in MindSpeed
- `fused_softmax` — already shimmed
- new fused ops added in apex 24.x — not yet shimmed

Each newly-discovered gap follows the same recipe: 30-60 lines of pure-torch fallback + 1 line register_patch.
