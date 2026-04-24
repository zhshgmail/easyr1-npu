# triton-ascend port-expert KB — Search Index

## Why this KB is terse

Unlike vllm-ascend / torch_npu, triton-ascend drift is resolved by
`git rebase` + manual conflict resolution, not by symbol-drift
scanners. There's no F1-F8 taxonomy here. KB records the **conflict
surfaces** historically seen during rebase.

## Known conflict surfaces

| Path | Type of conflict | Notes |
|---|---|---|
| `python/triton/backends/ascend/` | NPU backend sibling added to `backends/`. Upstream changes the backend registration protocol → NPU backend needs to match. | Watch for `register_backend()` signature changes in `python/triton/runtime/jit.py`. |
| `python/triton/language/ascend/` (if present) | NPU-specific lang extensions. Rare conflict surface — upstream language evolves slowly. | |
| `python/triton/runtime/jit.py` | NPU JIT hook, compile cache integration. Conflicts when upstream refactors `JITFunction.__call__`. | |
| `python/triton/compiler/compiler.py` | NPU compile pipeline target. Conflicts on `CompiledKernel` layout changes. | |
| `third_party/ascend/` | Native NPU kernels + MLIR passes. Standalone directory — conflicts rare. | |

## Rebase procedure

1. `git fetch origin && git fetch upstream` (upstream = community triton)
2. Identify baseline commit: the community triton tag the current
   triton-ascend main was forked from (README.md or CMakeLists.txt version line).
3. `git checkout -b <target>_auto_porting origin/main`
4. `git rebase --onto <target-community-tag> <current-baseline-community-tag>`
5. Resolve conflicts in favor of NPU backend.
6. Rebuild wheel (see triton-ascend's own build_ascend docs).
7. Run triton-ascend tests.
8. Smoke on A3: tiny triton kernel on NPU.

## What is NOT this skill's job

- Fixing C++/MLIR compile errors in the Ascend backend (AIL team).
- CANN driver / kernel library compatibility (torch_npu territory).
- Performance regressions (separate task).

## Relationship to other port-experts

- **torch_npu** upgrade may require newer triton-ascend — run both.
- **vllm-ascend** consumes triton-ascend indirectly through torch_npu.
- **transformers** doesn't touch triton-ascend directly.
