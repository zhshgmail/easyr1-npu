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
4. **Recommended end-to-end baseline (production-equivalent)**: vendor
   `triton_ascend-3.2.0` wheel + matching CANN image's bundled
   `bishengir-compile`. Validated 6/6 PASS on the smoke suite at
   `repo/src/scripts/smoke_triton_vector_add.py` (vec_add fp32 / masked /
   bf16 / fp16 / reduction sum / dot 16x16x16). See § "End-to-end
   baseline reproducer" below.
5. **Source-build path (main branch with new community-triton merge)
   is currently BLOCKED end-to-end**: main's libtriton.so is built
   against LLVM 22 (`cmake/llvm-hash.txt = fad32722`) and emits new MLIR
   text syntax (`bufferization.to_tensor : memref<…> to tensor<…>`).
   The bishengir-compile shipped in CANN 8.5.x and the one buildable
   from `AscendNPU-IR` (all branches) is built against LLVM 19 and
   cannot parse that syntax — fails with `custom op 'to' is unknown`.
   Code-side fixes (text merge, drift fixes, build, import) are still
   correct and ship as fork branch + KB; **end-to-end NPU smoke must
   wait for Huawei CI to release a bishengir-compile built against
   LLVM 22**. This boundary is the skill's documented blocker.

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

Recommended: run the 6-pattern suite at
`repo/scripts/smoke_triton_vector_add.py` (relative to repo root on
the A3 host clone; the dev-side master at `repo/src/scripts/` is the
authoritative source and gets sync'd to `repo/scripts/` on the A3 host
before running). It exercises elementwise fp32/bf16/fp16, masked
load/store, reduction, and `tl.dot` 16x16x16 patterns; pass criterion
is `6/6 PASS` with per-kernel atol budgets.

```bash
python /workspace/repo/scripts/smoke_triton_vector_add.py
# Expect: 6/6 PASS, max abs err per kernel within atol
```

If only the merged `triton-ascend` source build is available (no vendor
wheel), see § "Source-build path is BLOCKED" below — smoke will fail
at `bishengir-compile` parse step with `custom op 'to' is unknown`,
and that is the documented blocker, not a regression in your merge.

## End-to-end baseline reproducer (vendor-released configuration)

This block is the full reproducer including the `docker run` invocation
on the A3 host. Adjust `--device /dev/davinciN` for whichever NPU chip
you have free (precheck via `npu-smi info -t proc-mem -i <NPU_ID>`,
abort if "Process id" lines present — see `memory/a3_chip_economy.md`).

```bash
# 1) On A3 host, ensure the vendor wheel is present (one-time, ~52 MB):
mkdir -p /data/$USER/triton-ascend-vendor
cd /data/$USER/triton-ascend-vendor
[ -f triton_ascend-3.2.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ] || \
  wget -c "https://files.pythonhosted.org/packages/7f/f3/d4e6ddbaf6f07b72ceb29f0f739c4c8fba2ff476eac07aeb7ae6f654fce0/triton_ascend-3.2.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

# 2) On A3 host, ensure the smoke script is present at the path used inside the container.
#    The dev-side master is at <repo>/src/scripts/smoke_triton_vector_add.py;
#    sync to A3-clone <repo>/scripts/smoke_triton_vector_add.py if not there.

# 3) Run the smoke (one-shot --rm container; precheck device first).
docker run --rm \
  --privileged --network=host --ipc=host --shm-size=64g \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --device /dev/davinci2 \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/ascend_driver.conf:/etc/ascend_driver.conf \
  -v /etc/ascend_filelist.info:/etc/ascend_filelist.info \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /home/$USER/workspace/easyr1-npu:/workspace \
  -v /data/$USER/triton-ascend-vendor:/wheels \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5 \
  bash -c '
    pip uninstall -y triton triton-ascend
    pip install /wheels/triton_ascend-3.2.0-*.whl
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python /workspace/repo/scripts/smoke_triton_vector_add.py
  '
# Expect: =6/6 PASS=  with all kernels max abs err 0.000e+00 vs torch reference.
```

Notes:
- The bind set above is the minimum that satisfies CANN's `dcmi_init`
  inside container (per NPU-OPS-009). Removing any bind line breaks
  NPU enumeration.
- `--device /dev/davinciN` for `N` = chip index. Default to the chip
  free per `npu-smi info`.
- The vendor wheel is hosted on PyPI but A3's egress to PyPI may be
  slow (10s of minutes for 54 MB). The wget-then-bind pattern above
  keeps the wheel cached on the host so subsequent runs are fast.

Validated 2026-04-26 on `quay.io/ascend/verl:verl-8.5.2-a3-...qwen3-5`
image: 6/6 PASS, all kernels max abs err 0.000e+00 vs torch reference.

## Source-build path is BLOCKED end-to-end (until Huawei ships LLVM-22 bishengir)

The full source-build path is:
1. `git merge <community-tag>` into fork branch.
2. Resolve conflicts per `references/KB_INDEX.md` recipes.
3. `pip install -e .` against OBS-prebuilt LLVM 22 — produces
   `python/triton/_C/libtriton.so`.
4. `import triton` succeeds; backends discoverable.
5. `@triton.jit` compile: invokes `bishengir-compile <kernel.ttadapter.mlir>`.
6. **Parse error**: `custom op 'to' is unknown (tried 'func.to' as well)`
   on `bufferization.to_tensor %x : memref<…> to tensor<…>`.

Root cause: triton-ascend's libtriton.so emits MLIR text in LLVM 22
syntax; CANN 8.5.x's bundled `bishengir-compile 0.1.0` and any
`bishengir-compile` you can build from public AscendNPU-IR
(`cd708029e0`-pinned across all its branches) are built against
LLVM 19, whose parser does not accept that syntax.

Fix is on Huawei CI side, not on the skill consumer's side. Until
that ships, `P5 — NPU smoke` for a freshly-merged source build will
be skipped with documented blocker; the skill's deliverable for
that case is "fork branch with merge + drift fixes + KB updated"
(see KB_INDEX.md case registry).

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
