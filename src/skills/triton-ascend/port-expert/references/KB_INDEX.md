# triton-ascend port-expert KB — Search Index

## Why this KB is terse

Unlike vllm-ascend / torch_npu, triton-ascend drift is resolved by
`git rebase` + manual conflict resolution, not by symbol-drift
scanners. There's no F1-F8 taxonomy here. KB records the **conflict
surfaces** historically seen during rebase.

## Repo layout (verified 2026-04-24 on triton-ascend 3.5.0)

- `python/triton/**` — **vendored community triton source** tracked inside
  the fork. This is the primary rebase-conflict surface.
- `third_party/ascend/backend/` — **the actual NPU backend** (NOT
  `python/triton/backends/ascend/`; that directory is generated at
  install time by `setup.py:711` from the third_party source).
- `third_party/ascend/` sibling subdirs — MLIR passes + native kernels.
- `setup.py:711` — hardcoded `["ascend","nvidia","amd"]` backend list. NPU-specific.
- `AscendNPU-IR/` — git submodule (separate bump path; bump before rebase).
- `version.txt` + `python/triton/__init__.py:__version__` — the current
  community baseline. **The triton-ascend team keeps these in sync with
  the community tag they last rebased onto.**

## Known conflict surfaces

| Path | Type of conflict | Notes |
|---|---|---|
| `python/triton/**` | Vendored community source — every upstream commit is a rebase conflict candidate. | Biggest conflict mass; take-upstream by default. |
| `python/triton/runtime/jit.py` | If NPU backend hooks into `JITFunction.__call__` / compile cache, expect collisions here when upstream refactors. | Check third_party NPU code for `import triton.runtime.jit` references. |
| `python/triton/compiler/compiler.py` | `CompiledKernel` layout, backend dispatch. | Watch for `BaseBackend` signature changes propagating here. |
| `third_party/ascend/backend/compiler.py` | NPU `BaseBackend` subclass. | Must track upstream `BaseBackend` abstract signature when upstream changes it. |
| `third_party/ascend/backend/driver.py` | NPU `DriverBase` subclass. | Same as compiler.py. |
| `third_party/ascend/` (other dirs) | Native NPU kernels + MLIR passes. | Standalone — conflicts rare; keep-ours by default. |
| `setup.py` | Build system. Hardcoded `["ascend","nvidia","amd"]` list at line 711 (verify line number each rebase). | Keep-ours on the NPU hardcode; take-upstream on build-infrastructure changes. |
| `AscendNPU-IR/` submodule | Bump via `git submodule update --remote` before rebase, or resolve as pointer-update conflict. | Separate semver from triton-ascend; check `AscendNPU-IR` compatibility with target triton's MLIR version. |

## Finding the current baseline tag

The fork's last-rebased community triton tag is the baseline for
`git rebase --onto`. Find it:

```bash
cd upstream/triton-ascend
cat version.txt                          # matches community tag
cat python/triton/__init__.py | grep __version__
grep -rn "VERSION\|version" CMakeLists.txt | head
```

If `version.txt` matches a known community triton tag (e.g. "3.5.0"),
use `v3.5.0` as the baseline.

See also: `repo/knowledge/upstream-refs.md` — pins which community tag
each Ascend-fork mirrors (project-wide convention).

## Rebase procedure

1. Add community triton remote if not present:
   ```bash
   cd upstream/triton-ascend
   git remote add community https://github.com/triton-lang/triton.git || true
   git fetch origin  &&  git fetch community
   ```
2. Identify baseline commit (see "Finding the current baseline tag"
   above). Example: baseline = `v3.5.0` (community tag), target = `v3.6.0`.
3. Bump submodule if needed: `git submodule update --init --remote AscendNPU-IR`.
4. `git checkout -b <target>_auto_porting origin/main`.
5. `git rebase --onto <target-community-tag> <baseline-community-tag>`.
6. Conflict policy (per surface):
   - `python/triton/**` pure-core: **take-upstream** (community wins).
   - `third_party/ascend/**` pure-NPU: **keep-ours**.
   - Real collision (NPU backend subclass needs new abstract method
     added upstream): **re-apply against new signature**. Treat as F8
     equivalent — implement the new method with NPU semantics.
   - C++/MLIR compile errors in Ascend backend: **escalate to AIL
     team**. Not this skill's scope.
7. Rebuild prereqs:
   - Ensure LLVM prebuild is in place (see `triton-ascend/bin/build-llvm.sh`).
   - Set `TRITON_CODEGEN_BACKENDS=ascend` env var before building.
   - `bash setup.py build` or triton-ascend's own build script.
8. Run triton-ascend tests: `python/test/unit/ascend/` if present.
9. Smoke on A3: tiny triton kernel on NPU. See
   `src/skills/torch-npu/port-expert/scripts/smoke_validate.sh` for
   a concrete harness (triton-ascend is typically validated via the
   torch_npu inductor path).

## What is NOT this skill's job

- Fixing C++/MLIR compile errors in the Ascend backend (AIL team).
- CANN driver / kernel library compatibility (torch_npu territory).
- Performance regressions (separate task).

## Relationship to other port-experts

- **torch_npu** upgrade may require newer triton-ascend — run both.
- **vllm-ascend** consumes triton-ascend indirectly through torch_npu.
- **transformers** doesn't touch triton-ascend directly.
