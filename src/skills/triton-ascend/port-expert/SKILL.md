---
name: triton-ascend-port
description: >
  triton-ascend is a FORK of community triton (not a plugin). Upgrading it
  to a newer community release is a git-merge + conflict-resolution task,
  not an F1-F8 symbol-drift scan. This skill encodes the workflow and the
  concrete conflict recipes learned from the v3.5.0 -> v3.6.0 manual port.
argument-hint: >
  target-triton-version: community triton tag to merge in (e.g. v3.6.0)
context: inline
---

# /triton-ascend-port — fork upgrade to new community triton

## TL;DR for the reader who only reads this block

1. Read `references/KB_INDEX.md` first. It has the verified repo layout,
   the list of 12 conflict files from the v3.5.0 -> v3.6.0 rebase, and
   the exact resolution recipe per file.
2. Use **`git merge <target-tag> --no-commit --no-ff`**, NOT
   `git rebase --onto`. triton-ascend's main is not a descendant of any
   community tag (vendored-fork pattern). Rebase will explode.
3. Text-level clean merge is **only half the work**. C++ build on A3 +
   NPU smoke is the other half, and exposes LLVM API drift that text
   merge can't see.

## Why this skill is different from vllm-ascend / torch_npu

| Dimension | vllm-ascend / torch_npu | triton-ascend |
|---|---|---|
| Source structure | Plugin that imports from upstream | **Fork**: vendored upstream source + NPU patches baked in |
| Drift shape | F1-F8 symbol drift | Git merge conflicts on text + hidden LLVM API drift at build time |
| Fix pattern | Compat shim under `<project>/compat/` | In-place resolve-and-commit; no separate compat layer |
| Scanner | `kb_drive_test.py`, `check_f4.py`, `check_f7_f8.py` | `git merge`, `py_compile`, full C++ build |
| Validation | `/drift-port-validate` (shim branch check) | `git commit` clean -> `py_compile` pass -> C++ build -> NPU smoke |

## Workflow

### P0 — pick target community tag and confirm it is NOT already applied

```bash
cd upstream/triton-ascend
git remote add community https://github.com/triton-lang/triton.git   # if not present
git fetch community --tags
git merge-base --is-ancestor v<target> origin/main && echo "already merged"
```

If "already merged" prints, no work. Otherwise continue.

### P1 — create fork branch, verify fork-pattern sanity check

Branch naming follows `fork_branch_naming.md` memory: use
`<target-version>_manual_porting` on the user's fork
(`personal` remote = `gitcode.com/zhengshencn_hwca/triton-ascend`).

```bash
git checkout -b v<target>_manual_porting origin/main
# sanity: confirm origin/main is NOT a community descendant
git merge-base --is-ancestor v<target-target-tag> origin/main
echo $?  # must be 1 (NOT ancestor). If 0, stop — this is not a fork upgrade.
```

### P2 — merge, do not rebase

```bash
git merge v<target> --no-commit --no-ff
```

Expect conflicts. For v3.5.0 -> v3.6.0 the count was 12. See
`references/KB_INDEX.md` § "Conflict surfaces" for the file list and
per-file recipe. Resolution classes (verified 2026-04-25):

- **Trivial** (`.gitignore`, `__init__.py`, `setup.py`, `README.md`,
  `Makefile`, `docs/*`): resolve by selection rule in the KB table.
- **Compiler-level** (`cmake/llvm-hash.txt`, `TritonAttrDefs.td`,
  `python/src/ir.cc`, `python/src/ir.h`): merge-with-judgment; each has
  a specific recipe in KB.
- **Interpreter** (`python/triton/runtime/interpreter.py`): take
  community outer structure, branch on
  `hasattr(interpreter_builder, 'execute_with_sub_vec_simulation')`
  for NPU path. Full snippet in KB.

### P3 — verify text-level merge

```bash
git commit                                                # clean
python -m py_compile $(git diff --name-only HEAD~1 HEAD | grep '\.py$')
python -m py_compile third_party/ascend/backend/*.py
```

All three must pass. This is **cheap signal, not proof** — see P4.

### P4 — C++ build on A3 (the real verification)

Inside an A3 container with CANN installed:

```bash
TRITON_CODEGEN_BACKENDS=ascend pip install -e . 2>&1 | tee build.log
```

Common follow-ups that only surface here:

- **LLVM API drift**: community's new C++ calls a MLIR API that the
  NPU-pinned LLVM base (`cmake/llvm-hash.txt`) doesn't have. Either
  port the community change down to the old API, or bump LLVM base +
  regenerate `third_party/ascend/llvm_patch/*.patch`. The latter is a
  **separate multi-week task** and is out of this skill's scope.
- **AscendNPU-IR submodule mismatch**: new triton-ascend may require a
  newer AscendNPU-IR commit. Check `.gitmodules` + release notes.
- **CANN runtime compat**: A3 image's CANN version vs. what
  triton-ascend's lib expects.

### P5 — NPU smoke

```bash
# tiny @triton.jit vector_add on NPU
python python/test/unit/ascend/<smoke>.py        # if present
# else: minimal vector_add(x, y, out) via torch_npu inductor path
```

### P6 — push and file PR

```bash
git push personal v<target>_manual_porting
gc pr create ...    # against Ascend/triton-ascend
```

## Boundary: what is NOT this skill's job

- Fixing C++/MLIR compile errors after LLVM API drift — AIL team
  territory.
- Bumping LLVM base + regenerating the patch file — separate task
  with its own review process.
- CANN driver / kernel library compatibility — overlaps torch_npu
  expert.
- Performance regressions post-upgrade — separate benchmark task.

## Interaction with other port-experts

- **torch_npu upgrade may force a triton-ascend bump**: if new torch
  bumps `triton>=3.M`, check triton-ascend's `version.txt` and fire
  this skill in parallel.
- **vllm-ascend -> triton-ascend**: no direct coupling. vllm-ascend's
  triton usage goes through torch_npu's inductor path; API drift
  surfaces as F3 in vllm-ascend's scanner if at all.
- **transformers -> triton-ascend**: no path.

## Knowledge query paths

- `references/KB_INDEX.md` — **read first**. Repo layout, conflict
  surfaces, per-file resolution recipes, case registry.
- `memory/a3_server.md` — A3 host access for P4/P5.
- `memory/fork_branch_naming.md` — branch naming.
- `memory/version_aware_reviews.md` — when reviewing NPU upstream
  changes, use the matching release branch, not master.

## What this skill is NOT

- Not an F1-F8 symbol-drift scanner target. Use git-merge mental
  model.
- Not a compat-shim target. There is no compat layer — conflicts are
  resolved in place on the vendored source.
- Not a runtime install-issue fixer. A separate memory notes a
  specific triton-ascend rc wheel install issue unrelated to version
  upgrade.

## Case registry

See `references/KB_INDEX.md` § "Case registry". As of 2026-04-25:
v3.5.0 -> v3.6.0 text-merge DONE on `v3.6.0_manual_porting` on fork;
C++ build + A3 smoke PENDING.
