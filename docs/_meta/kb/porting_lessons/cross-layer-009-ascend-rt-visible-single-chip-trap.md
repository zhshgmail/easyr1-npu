---
id: cross-layer-009
date: 2026-05-30
layer: cross-layer
title: ASCEND_RT_VISIBLE_DEVICES=1 filters away the sole mounted chip on a single-chip container
trigger:
  - "container has only one davinci device mounted (/dev/davinci0)"
  - "user sets ASCEND_RT_VISIBLE_DEVICES=1 thinking 'use 1 chip'"
  - "driver reports no NPU visible despite the chip being present"
symptom_in_wild:
  - "torch_npu.npu.is_available() returns False inside container"
  - "Engine init crashes with 'no NPU device available'"
  - "host `npu-smi info` shows the chip; container `npu-smi info` shows nothing"
  - "Confusingly, host can use the chip — only the container is blind"
root_cause: >
  `ASCEND_RT_VISIBLE_DEVICES` is a *device id whitelist*, not a count. The
  id is driver-logical: on a single-chip container that maps host davinci2
  to container's exposed device, the container sees id=0 (the only id it
  has). Setting `ASCEND_RT_VISIBLE_DEVICES=1` whitelists id=1 — which the
  container doesn't have — filtering everything away.

  Confusingly, the name suggests "1 chip should be visible". The semantics
  are CUDA_VISIBLE_DEVICES-style: the value is a comma-separated id list.
mistake_pattern: "env var name suggests count, semantics is id list; user assumes wrong category"
correction:
  - "On single-chip containers: don't set ASCEND_RT_VISIBLE_DEVICES at all. Let the driver default to id=0."
  - "On multi-chip containers when you want chip 1: `ASCEND_RT_VISIBLE_DEVICES=1` is correct iff id=1 exists"
  - "Quick check: `python -c 'import torch_npu; print(torch_npu.npu.device_count())'` inside container before setting the env"
  - "When automating from a script: query device count first, then set if and only if > 1"
  - "If container has /dev/davinci0..davinci7 all mounted (8-chip), the env behaves as expected"
evidence:
  - "Tested 2026-05-30 in `quay.io/ascend/verl:verl-sglang-8.5.0-...` sidecar container with single davinci mount: setting `ASCEND_RT_VISIBLE_DEVICES=1` -> no NPU; unsetting -> device_count() == 1, works"
  - "Same trap re-discovered in sglang_npu_smoke.py and test_load_dsv4_fab.py — both eventually solved by removing the env var"
  - "Memory: feedback_sglang_npu_smoke_recipe.md (lesson #2 of 3 must-avoid pitfalls)"
---

# cross-layer-009 — ASCEND_RT_VISIBLE_DEVICES on single-chip containers

## Why this matters

Multiple users (myself included) re-discover this every time a single-chip container is created. The error message ("no NPU device available") doesn't hint at the cause. With this lesson memorized, it's instant.

## Decision table

| Container chip count | What to set | Why |
|---|---|---|
| 1 mounted | (unset) — driver defaults to id=0 | id=1 doesn't exist |
| 2+ mounted, want all | (unset) or set all ids | leave default |
| 2+ mounted, want chip 1 specifically | `ASCEND_RT_VISIBLE_DEVICES=1` | id semantics, correct |
| 2+ mounted, want chip 0 specifically | `ASCEND_RT_VISIBLE_DEVICES=0` | id semantics, correct |
| 2+ mounted, want chip N where N>=count | will be filtered away | bug — check chip count first |

## Doc opportunity (not filed)

`Ascend/torch_npu` README could clarify the semantics (id list vs count). Worth a one-line PR if encountered again.
