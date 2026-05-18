---
name: tilelang-fix
description: Orchestrator for diagnosing and fixing tilelang-mlir-ascend runtime errors. Drives the 5-step loop (sweep → triage → patch → verify → KB feedback) end-to-end OR on a specific failing op. Mirrors a5_ops's `ascendc-op-gen` pattern: single user-facing entry + atomic `tlfix-*` step skills.
---

# /tilelang-fix — diagnose-and-repair orchestrator for tilelang-mlir-ascend

User-facing single entry. Drives the 5-step cold-loop end-to-end OR
focuses on a specific failing op. Composed of atomic step skills:

| Step | Sub-skill | Purpose |
|------|-----------|---------|
| 1 | `/tlfix-sweep` | Walk `examples/`, `testing/npuir/`, `unittest/npuir/`; classify each op into status enum per KB §11 taxonomy. Output: `results.json` |
| 2 | `/tlfix-triage` | For each FAIL, match against KB §10.x deep-dive patterns. Output: triage hypothesis + candidate patch (or "unknown" → ask user/KB update) |
| 3 | `/tlfix-patch` | Apply a patch from the recipe library (`apply_*.py`) or generate a new patch proposal. Output: modified source files |
| 4 | `/tlfix-verify` | Rebuild + regression-test (small sample N=5-10 PASSing ops); confirm the patch didn't break anything |
| 5 | `/tlfix-kb-update` | If new bug class found, write back to KB §8.1 (verification matrix), §10.x (new deep-dive), §11 (taxonomy row), §12 (preventive rule) |

## Mode selection

| User invocation | What runs |
|-----------------|-----------|
| `/tilelang-fix` (no args) | Full loop: sweep → triage all fails → fix any with known recipes → verify → KB write-back |
| `/tilelang-fix <op_path>` | Focus on one op: triage → patch → verify → optional KB update |
| `/tilelang-fix --sweep-only` | Phase 1 only (delegates to `/tlfix-sweep`) |
| `/tilelang-fix --fix <op_path>` | Skip sweep+triage; jump to apply known patch from KB recipe |

## What this is NOT

- Not a compiler from scratch — uses `bishengir-compile` + tilelang as black box
- Not auto-merge to upstream — only proposes patches; human gates push
- Not a benchmark / perf tool — see `benchmark/` for that

## Repository setup

Requires:
- `tilelang-mlir-ascend` checked out at a path provided by caller
- `bishengir-compile` available (built at `3rdparty/AscendNPU-IR/build/install/bin/`)
- A3 NPU container with privileged + driver bind set per
  `knowledge/upstream-refs.md` `tlrescue` recipe

## See also

- `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md` — KB this skill reads and writes
- `workspace/T32_tilelang_rescue/AUTOPORT_SKILL_DESIGN.md` — original 5-phase design doc
- `workspace/T32_tilelang_rescue/apply_*.py` — patch recipe library (Phase 3 input)
- a5_ops's `ascendc-op-gen` skill — pattern this mirrors
