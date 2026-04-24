---
name: drift-port-validate
description: >
  Validate that a forward-compat shim (F1/F2 family port patch) actually
  takes the correct branch under both OLD-upstream and NEW-upstream
  conditions, WITHOUT requiring the NEW upstream to be installed and
  WITHOUT requiring working NPU runtime/driver. Use after a port patch is
  authored but before declaring the port done. Produces binary pass/fail
  on a small set of structural checks.
argument-hint: >
  patch-summary: what compat shim you just added (e.g. "F1 SharedFusedMoE
  compat at vllm_ascend/compat/shared_fused_moe.py; 2 callsite imports
  swapped"). The skill reads from there.
---

# `/drift-port-validate` — shim-branch behavioral check

## What this skill is for

When a port fix is of the F1 (removed symbol) or F2 (renamed symbol)
family from `src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md`,
the fix is a **forward-compat shim** at `vllm_ascend/compat/<module>.py`
that tries the upstream import first and falls back to a local
equivalent on `ImportError`.

This skill verifies the shim by **exercising both branches on A3**:
- **OLD upstream path**: stub upstream modules to contain the removed
  symbol → shim must take the `try` branch and resolve to upstream
- **NEW upstream path**: stub only the parent packages, NOT the leaf
  modules that hold the removed symbol → shim's inner import raises
  `ImportError` → shim must take the `except` branch and resolve to
  its local fallback

Both branches must be exercised. One-sided tests give false confidence.

## When to invoke

1. Immediately after committing a new `vllm_ascend/compat/<module>.py`
   shim AND swapping at least one callsite's import.
2. Before merging the port branch.
3. Before claiming "KB-driven port complete" to the user.

Do NOT invoke when:
- The fix family is F3/F4/F5+ (these are not import-level shims and
  this skill's check shape doesn't apply).
- The shim file isn't under `vllm_ascend/compat/` (non-standard layout
  means this skill's structural assumptions may not hold).

## Pre-requisites

Before running:
- A3 SSH is reachable (see `memory/a3_server.md` for host:port).
- Your port commits are pushed to your personal fork branch
  (`<target-version>_auto_porting`). The harness copies files via
  `scp` / `docker cp`, not via git pull inside the container, because
  A3 host's vllm-ascend checkout is typically not present.
- Any container image with `torch` + `torch_npu` + `vllm` + the
  pre-patch `vllm-ascend` tree installed. A Fix-C image is fine.
- At least one davinci device that is NOT claimed by someone else's
  container. If ALL davinci devices are ns-locked by other users
  (check `dmesg | grep uda_occupy_dev_by_ns` on host), you can still
  run this skill in **CPU-only mode** because the checks don't open
  an NPU device. Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` in the exec
  env — see step 3.

## Target (goal criterion — this is what "done" looks like)

A single printed line `RESULT <N>/<N>` with `<N>` equal to the number
of checks that were run, and every individual check printing `PASS`.
Any `FAIL` means the shim is broken and the port is NOT done.

Expected check set per family:
- **F1/F2 single-symbol shim** (e.g. `SharedFusedMoE`): 2 OLD + 3 NEW
  = 5 checks
- **F1/F2 two-symbol shim**: 2+3 per symbol, so 5 per symbol
- **F2 with aliased new name** (e.g. `DefaultMoERunner → MoERunner`):
  2 OLD + 2 NEW = 4 checks (the NEW side doesn't need an
  `issubclass` check because it's a direct alias)

## How to run (LLM-driven, reproducible)

An LLM following this skill should produce exactly the same outputs
and decisions as a human expert did in the reference execution on
2026-04-24 (see `reference_run_20260424.md` in this skill's
`references/` dir once codified — pending).

### Step 1 — identify what you are validating

Open the compat module you just added. Note:
1. The module file path inside `vllm-ascend/` (e.g.
   `vllm_ascend/compat/shared_fused_moe.py`).
2. The symbol(s) it exports.
3. The upstream import path(s) it tries first — these are the
   `from vllm.<path> import <Symbol>` lines inside the `try` block.
4. The local fallback (either a local `class Foo:` body, or an
   aliased `from vllm.<new.path> import NewSymbol as OldSymbol`).

If the shim exports multiple symbols, handle each independently.

### Step 2 — prepare the A3 container

Choose a base image that has the pre-patch vllm-ascend installed.
Look for the most recent `easyr1-npu-*fixc*` or similar in
`docker images` on A3. The image must have `torch` + `torch_npu` +
`vllm` (any version where the target symbol still existed). It does
NOT need to have the NEW upstream version.

Start a persistent named container for this session. Use a name that
includes the session date and the drift ID (e.g.
`drift-validate-<SHA>-<DATE>`). Do NOT use `--rm`. See
`memory/persistent_npu_container.md` for rationale.

If all davinci devices are ns-locked by other users' containers, start
the container **without any `--device`** flags; CPU-only mode is
fine for this skill because the checks don't open NPU.

### Step 3 — apply your patch inside the container

The A3 host typically does NOT have your vllm-ascend fork-branch
checkout, so you cannot bind-mount it. Instead:

1. On your local machine, identify the patched files in your fork
   branch (typically: new files under `vllm_ascend/compat/` + one
   `sed`-able import swap in each existing source file).
2. `scp` the compat files and any modified source files to
   `/tmp/<user>/drift_patches/` on A3.
3. Inside the container, copy compat files to
   `/vllm-ascend/vllm_ascend/compat/` (create the dir if needed) and
   `sed -i` each import swap.
4. Quickly verify with `grep` that the new imports look right.

### Step 4 — author two verify scripts

Per-family. For an F1 removed-symbol shim, two files:

**`verify_new.py`** (NEW-upstream path — shim's except branch):
1. Import `sys, types, importlib.util`.
2. Stub parent packages via `sys.modules[<name>] = types.ModuleType(<name>)`.
   Stub every vllm package along the path **except** the leaf module
   that holds the removed symbol. You can inspect the compat file's
   `try:` block to know which leaf to leave unstubbed.
3. For any symbol the fallback code references from a DIFFERENT
   upstream path (e.g. `FusedMoE` as base class, or the new aliased
   class like `MoERunner`), stub those under their correct path with
   minimal class bodies.
4. `importlib.util.spec_from_file_location` + `exec_module` on the
   compat file — NOT `import vllm_ascend.compat.<X>`. Bypassing
   `vllm_ascend/__init__.py` avoids the `torch_npu → CANN` dependency
   chain.
5. Check: `_UPSTREAM_HAS_<X>` attribute on the loaded module is
   `False`. (If the shim doesn't expose this, add it — it's a standard
   part of the F1 template.)
6. Check: the exported symbol's `__module__` points at your loaded
   test module name (proving it's the local class, not the upstream).
7. Check: any structural property the local fallback must preserve
   (e.g. `issubclass(local, FusedMoE)`).
8. End with `RESULT <pass>/<total>` print and `sys.exit(0 if ok else 1)`.

**`verify_old.py`** (OLD-upstream path — shim's try branch):
Same structure, but in step 2 you stub the leaf module WITH the
removed symbol inside it. The shim's try branch must succeed. Check:
- `_UPSTREAM_HAS_<X>` is `True`.
- Exported symbol **is** the upstream stub class (identity check).

### Step 5 — run in container

```
scp verify_*.py root@A3:/tmp/<user>/
docker exec -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 <container> python3 /tmp/<user>/verify_new.py
docker exec -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 <container> python3 /tmp/<user>/verify_old.py
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` is REQUIRED — without it, `import torch`
tries to load `torch_npu` which needs real CANN runtime.

### Step 6 — interpret results

- Both scripts print `RESULT N/N` with N = sum of PASS lines:
  → validation complete, proceed to commit / merge.
- Any `FAIL` line: the shim is broken. Read the FAIL line to see
  which branch/assertion failed; edit the shim; re-run Step 5.
- Exception or stack trace before any check line: harness setup
  issue, not shim issue. See **Debug** below.

### Step 7 — clean up + record

- Stop + remove the container (`docker stop && docker rm`). This
  releases the udevid namespace if any was held.
- Append the result to the skill's case registry (once this skill
  has a case registry — similar to
  `src/skills/vllm-ascend/port-expert/references/KB_INDEX.md §"Concrete case registry"`).

## Debug paths (what to do when something fails)

**Symptom**: `RuntimeError: Failed to load the backend extension: torch_npu`
when running verify scripts.
**Cause**: missing `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. Re-run with the env.

**Symptom**: `ImportError: libascend_hal.so / libhccl.so: cannot open`
during verify script import chain.
**Cause**: you are accidentally going through `import vllm_ascend` (which
triggers `import torch_npu` → CANN libs). Your verify script must use
`importlib.util.spec_from_file_location` on the compat file directly,
not `import vllm_ascend.compat.<X>`.

**Symptom**: `drvRet=87` during `python3 -c "import vllm_ascend"`.
**Cause**: NPU device namespace locked by another user's container
(per `memory/a3_uda_ns_conflict.md`). Run without `--device` flags —
this skill doesn't need a device. Or wait for other user.

**Symptom**: shim takes the `try` branch even on NEW path.
**Cause**: you forgot to NOT stub the leaf module in `verify_new.py`,
or you stubbed a too-broad parent module. Check: the exact module path
in the shim's `try: from <path> import <Symbol>` line must NOT appear
in `sys.modules` when the compat file runs.

**Symptom**: shim's `except` branch raises its own `ImportError`
while trying to load the fallback class's dependencies.
**Cause**: a class used inside the `except` branch (e.g. `FusedMoE` as
base class, `MoERunner` as alias target) isn't stubbed. Stub it with
a minimal class body.

## Knowledge query paths

When this skill runs into an unknown family or an unexpected error,
read these first:

1. `src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md`
   — describes F1/F2 shim structure. Confirm the fix being validated
   follows the template.
2. `src/skills/vllm-ascend/port-expert/references/KB_INDEX.md`
   — symptom-routing table + concrete case registry. If the drift
   being validated isn't listed, add it after this skill passes.
3. `src/skills/vllm-ascend/port-expert/scripts/kb_drive_test.py`
   — the detector that **originally surfaced** the drift. Run it
   after your fix to confirm it still reports the drift as matched
   (it should — the detector is symptom-level, not fix-level). If
   `kb_drive_test.py` no longer flags your symbol after the fix,
   the detector has a bug, not your port.
4. `memory/a3_server.md`, `memory/persistent_npu_container.md`,
   `memory/a3_uda_ns_conflict.md`, `memory/a3_chip_economy.md`
   — A3 operational quirks.

## Reference execution (2026-04-24)

First real run, performed by the author on 2026-04-24:
- Target: F1 shims for `SharedFusedMoE` (commit `5e584ce9e`) and
  `DefaultMoERunner` (commit `809d83c2d`) on personal fork branch
  `vllm-main_auto_porting`.
- Base image: `easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423`.
- Container: `f1-validate-20260424` (CPU-only — all davinci devices
  held by `roll_npu_new`, `vllm_ascendreleases013`, `lynn_verl`).
- Result:
  - `verify_old.py`: 4/4 PASS (OLD vllm, shim pass-through)
  - `verify_new.py`: 5/5 PASS (NEW vllm, shim fallback)
  - Combined: 9/9 PASS.
- The reference verify scripts as executed are preserved at
  `/tmp/z00637938/f1_verify.py` and `/tmp/z00637938/f1_verify_old.py`
  on A3, and in the skill's `references/` once copied in. An LLM
  can use them as templates by (a) substituting the symbol name,
  (b) substituting the leaf module path, (c) substituting the
  base-class or alias target in the stub block.

## What this skill is NOT

- Not a V1.3 rollout test — that requires a working NPU device and
  catches C-extension / ABI issues, not Python import-level bugs.
- Not a substitute for end-to-end inference validation before a
  customer-facing release — it's a fast gate between "patch drafted"
  and "patch tested enough to ship to V1.3".
- Not applicable to ABI drift (F9 family) — those need real NPU and
  a different validation shape (actually calling the custom op).
