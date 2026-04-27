# vllm API drift — vllm-ascend specifics

> Family taxonomy (F1–F8 + F2-path-move) is **shared across all NPU
> upstream skills**. See [`_shared/patterns/F-family-taxonomy.md`](../../../../../_shared/patterns/F-family-taxonomy.md).
>
> This file holds vllm-specific concrete examples and call-out points.

## What's in shared taxonomy

The shared file defines all 8 families (F1 / F2-rename / F2-path-move
/ F3 / F4 / F5 / F6 / F7 / F8) with abstract Detect / Fix-template
sections and the cross-family meta-rule (forward-compat try/except,
no hard version gates). Read that first.

## vllm-specific notes for each family

- **F1 — removed symbol**: vllm refactors public helper functions
  about once per minor release. Recent examples logged in
  [`KB_INDEX.md`](../../../KB_INDEX.md) §"Known vllm API removals / adds".
- **F2-path-move**: vllm-ascend has a dedicated `vllm_ascend/compat/`
  module for these. Worked examples: see
  [`KB_INDEX.md`](../../../KB_INDEX.md) case rows.
- **F5 — buffer API**: vllm 0.18 → 0.19 migrated a large set of
  `.np` / `.copy_to_gpu` paths to `CpuGpuBuffer`. This is the most
  invasive class — usually needs a sweep + batch shim.
- **F6 — kv_cache contract**: vllm 0.18 → 0.19 changed kv_cache from
  `list[Tensor]` to single concatenated `Tensor`; vllm-ascend's
  attention backend reads kv_cache directly so this hits hard.
- **F7/F8 — new required field/method on base class**: vllm regularly
  adds attributes to `Platform`, `WorkerBase`, `ModelRunner` that
  vllm-ascend's NPU subclasses must implement. Often surfaces as
  `AttributeError` at first use, not at import time.

## Where to look next

- Concrete cases (date / vllm PR / vllm-ascend file / matched family):
  [`KB_INDEX.md`](../../../KB_INDEX.md) §"Known vllm API removals / adds"
- Probe / failure-minimization workflow:
  [`vllm-ascend-probe.md`](vllm-ascend-probe.md)
- Cross-upstream taxonomy:
  [`_shared/patterns/F-family-taxonomy.md`](../../../../../_shared/patterns/F-family-taxonomy.md)
