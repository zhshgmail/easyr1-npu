# Day-0 deploy artifacts pattern

> Authored from two rounds of Day-0 porting (torch 2.11 → Ascend NPU
> and vllm-ascend Fix B+ on top of torch 2.11) on 2026-04-23.

## When this applies

Any Day-0 session that produces a working overlay image of a new upstream
version on NPU. Before the session closes and the skill gets codified
(Phase 3), sit at Phase 2.5 and produce this artifact bundle so the
next-layer maintainer (or a downstream RL framework user) can pick up
from a deployed state without redoing the plumbing.

## The five deliverables

Every Day-0 deploy dir must contain:

### 1. `Dockerfile.overlay-<target>`
- `ARG BASE_IMAGE` parameterized on the previous layer's image
- Pure COPY / `pip install --no-deps` / `pip install git+...`; no
  secrets, no host-specific paths
- **No `import torch` or other runtime-only imports at build time.**
  Since PyTorch 2.11's `_import_device_backends()` auto-load, even
  `import torch` needs CANN libs at runtime. Use `python3 -m py_compile`
  or `ast.parse` for build-time syntax validation instead.
- `pip install` with both aliyun mirror and pytorch index:
  `--extra-index-url https://download.pytorch.org/whl/cpu/` (pip config
  already has `https://mirrors.aliyun.com/pypi/simple/` as the primary
  index on A3)

### 2. `smoke_<target>.sh`
Runtime-only smoke battery, invoked **inside** a container with NPU
devices mounted + CANN env sourced:

- sources `/usr/local/Ascend/ascend-toolkit/set_env.sh` if not already
- 6 minimum steps:
  1. pip metadata (versions)
  2. `import torch` (proves `_import_device_backends()` path works)
  3. explicit `import torch_npu` (proves the extension loads)
  4. `torch.npu.is_available()` + `device_count()` + `get_device_name(0)`
  5. minimal tensor op on NPU (`x.npu() @ x.npu().t()`, print mean)
  6. a few API-presence checks for surface the target depends on
     (e.g. `torch.npu.Stream.native_handle`)
- Prints "ALL SMOKE STEPS PASSED" as the last line on success; exit
  non-zero on any step failure (implicit via `set -euo pipefail`)

### 3. `deploy_<target>.sh`
One-shot deployment helper that:

- Takes `--host / --port / --user` for the NPU host
- Takes `--base-image / --image-tag / --target-dir / --chip`
- Step 1: `rsync -az` the artifact dir to the host
- Step 2: `docker build --build-arg BASE_IMAGE=...`
- Step 3: runtime smoke inside a container with all 4 NPU device mounts
  (`/dev/davinci<N>`, `/dev/davinci_manager`, `/dev/devmm_svm`,
  `/dev/hisi_hdc`) + 4 lib mounts (`/usr/local/Ascend/driver`,
  `/usr/local/dcmi`, `/etc/ascend_install.info`, `/usr/local/bin/npu-smi`)
- Exit codes separate rsync/build/smoke failures (1/2/3)
- `--skip-smoke` for the rebuild-only case

### 4. `ONBOARDING.md`
User-facing (downstream upstream maintainer or framework user):

- **What this gives you** — one-line summary
- **What's in the image** — layer table (base / patched layer /
  verified status)
- **How to use** — FROM directive + run-container command
- **What's validated end-to-end** — exact smoke harness output snippet
- **What's different from the regular install** — if this deploy
  introduces a behavior change (e.g. our Fix B+ auto-enables
  VLLM_BATCH_INVARIANT), document here
- **What's known-broken** — list things you haven't tested so they
  know where to be skeptical
- **Patch summary** — if we made code changes, list commits + what they
  do
- **Day-0 KB notes** — lessons the upstream maintainer should fold into
  their own KB (prevents re-discovery)
- **Where to report bugs** — route per class of bug (our patch vs
  upstream layer vs downstream layer)
- **Validated combinations table** — `<component × version × status>`
  row; add rows as more combos get tested

### 5. `PR_MATERIAL.md` *(when C-patch)*
If this Day-0 produced a patch to a Huawei-owned upstream (vllm-ascend,
torch_npu, triton-ascend, transformers NPU integrations), write the PR
description here:

- Title (follow upstream's convention, e.g. `[BugFix]`)
- Branch name (`ascend-day0-<target>-<session-tag>`)
- Description section pattern:
  - What / why
  - Detection mechanism
  - User-facing change (warnings / behavior)
  - How tested (before → after table)
  - Follow-up work
- Reproducer command (copy-paste, not prose)
- Signed-off-by line (PR submitter fills in)

## Execution order

After Phase 2 manual port PASS:

1. Copy working `Dockerfile` + `smoke.sh` from manual workspace into
   deploy dir
2. Write `deploy.sh` that embeds the exact `docker run` command used
   in manual Phase 2 smoke
3. **Cold-drive the deploy script end-to-end** with a different
   `--image-tag` than the manual one (e.g. `<tag>-deploy-validation`).
   If PASS, delete the validation image immediately (OL-04b cleanup).
   If FAIL, go back and fix artifacts before writing onboarding.
4. Write `ONBOARDING.md` — `Validated combinations` row uses the exact
   versions from step 3's PASS
5. If C-patch: write `PR_MATERIAL.md`; branch is already on personal
   fork from Phase 5 patch work

## What NOT to put in deploy artifacts

- Session-specific paths (`/tmp/z00637938/...`) — parameterize
- Your user's models dir — parameterize via `MODELS_DIR` env var
- Host-specific chip numbers — default to chip 0, parameterize
- Session TaskId / date in file contents (the dir name already has it)
- The Phase 1 analysis or Phase 2 PROGRESS.md — those are session
  artifacts; ONBOARDING only links back to them

## Why not git-commit these to repo

By session convention, workspace artifacts live at
`workspace/<session-tag>/` and are NOT tracked in git. They are
ephemeral to the session. The durable outputs are:
- This pattern file (committed to `_shared/references/patterns/`)
- Upstream patches (committed on personal fork's `ascend-day0-*` branch)
- Skill definition (committed under `src/experts/<expert>/`)
- KB updates (committed under `src/experts/<expert>/references/`)

If a future session needs to reproduce a past deploy: the upstream
patches + the skill + the deploy template (in the skill's scripts dir)
regenerate the artifacts. Don't copy session-specific artifacts into
the skill.

## Cross-reference

Concrete instances of this pattern:
- `workspace/torch-day0-deploy-20260423-0548/` — torch 2.11 + torch_npu 2.11.0rc1
- `workspace/vllm-ascend-day0-deploy-20260423-0655/` — Fix B+ on vllm-ascend

When codifying a new Day-0 skill, fork the deploy scaffolding from
either and adapt.
