# torch private-API drift — torch-npu specifics

> Family taxonomy (F1–F8 + F2-path-move) is **shared across all NPU
> upstream skills**. See [`_shared/patterns/F-family-taxonomy.md`](../../../../../_shared/patterns/F-family-taxonomy.md).
>
> torch-specific notes below.

## torch-specific scan focus

torch_npu mostly imports from torch's **private** modules (`torch._dynamo`,
`torch._inductor`, `torch._C`). These reorganize aggressively each
release — F2-path-move is by far the dominant family.

The torch-npu scanner (`scripts/extract_imports.py` +
`scripts/check_drift.py` + `scripts/check_sig_drift.py` +
`scripts/check_f7_f8.py`) sweeps every public symbol the torch-npu
source imports from torch, then checks for:

- **F1**: import at all? (deletion)
- **F2-path-move**: same symbol identity at a different module path?
- **F2-rename**: same module, new name?
- **F3**: signature change on imported callable
- **F7/F8**: new required attribute / method on a base class
  torch-npu subclasses

F4–F6 are rare on torch-npu's surface and not part of the routine
sweep.

## Recent cases

See [`KB_INDEX.md`](../../../KB_INDEX.md) §"Known torch private-API
moves". Most recent: torch 2.11 → 2.12-rc3 surfaced 1 F2-path-move
(`Union` removal from `torch._inductor.codecache`); fixed with
`torch_npu/compat/inductor_codecache.py` shim.

## Where to look next

- Cross-upstream taxonomy: [`_shared/patterns/F-family-taxonomy.md`](../../../../../_shared/patterns/F-family-taxonomy.md)
- Per-version cases: [`KB_INDEX.md`](../../../KB_INDEX.md)
- Probe / minimization: [`torch-overlay-probe.md`](torch-overlay-probe.md)
