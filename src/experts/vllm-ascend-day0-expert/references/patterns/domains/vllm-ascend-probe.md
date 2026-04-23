# Domain — vllm-ascend reproducer minimization + call-site location

## Phase A: reproducer minimization

Start from a failing V1.3 rollout smoke or a consumer-reported traceback.
Shrink to the smallest `python3 -c '...'` that reproduces the failure.
Use `-X faulthandler` to get Python stack for native segfaults.

**Template minimum reproducer**:

```python
import faulthandler; faulthandler.enable()
import torch, torch_npu
import vllm_ascend.vllm_ascend_C  # or whatever loads the failing path
# Setup: tensors matching the failing shape/dtype from the original traceback
x = torch.randn(<shape>, dtype=<dtype>).npu()
# The single failing operation:
result = <the offending op>
```

Save under `workspace/vllm-ascend-day0-analysis-<SESSION>/isolate_*.py`.
Iterate: strip imports one at a time until either (a) the script
succeeds (last-removed import was the trigger), or (b) you're at the
minimum failing script.

## Phase B: C++ ABI drift detection

Three-way check in a single script:

```python
import torch, torch_npu
import vllm_ascend.vllm_ascend_C as ext

# 1. Did the .so load?
print("loaded:", ext is not None)

# 2. Did TORCH_LIBRARY_IMPL register?
op_ns = torch.ops._C_ascend
print("namespace present:", hasattr(torch.ops, "_C_ascend"))
# Note: dir(op_ns) is misleading — may only show ['name'].
# Check specific ops via hasattr instead:
for op in ['npu_add_rms_norm_bias', 'bgmv_shrink']:
    print(f"  {op}:", hasattr(op_ns, op))

# 3. Does calling the op work, or SIGSEGV?
# If steps 1-2 pass but step 3 segfaults → ABI drift, not a missing op
```

If steps 1-2 pass and step 3 SIGSEGV → **C++ ABI drift** is the root
cause. The `.so` was compiled against a different torch ABI. Python
patches can't fix the ABI itself; they can only bypass the op.

If step 3 raises `AttributeError` → op never registered (e.g. conditional
`m.impl(...)` in the C++ skipped due to macro) → different class of bug.

## Phase C: call-site location

Given a failing op name, find every call site:

```bash
grep -rn "<op_name>\|torch.ops._C_ascend.<op_name>" upstream/vllm-ascend/
```

Classify call sites:
- **Guard-gated**: wrapped in `if vllm_is_batch_invariant(): else:`
  or `if enable_custom_op():` — these route away safely when the
  gate is flipped. Most of the layernorm / sampler / silu paths are
  like this.
- **Unguarded**: direct `torch.ops._C_ascend.<op>(...)` call with no
  fallback. These need per-site patches.

Count each class. If **all call sites are guard-gated**, a single
env-var flip (VLLM_BATCH_INVARIANT) bypasses everything. This is the
Fix B+ pattern — the cheapest valid fix.

If any unguarded call sites exist, they need explicit per-site patches
(version-check + fallback branch).

## Phase D: fix-level selection (5 levels as of 2026-04-23 dusk)

Walk the levels in order, pick the lowest that works. Empirically we've
found levels 1-3 often fix **inference** (V1.3 rollout) but leave
**training** (V1.4) broken; level 4 C++ rebuild is usually needed for
full training path.

1. **Env-var only**: does setting `VLLM_BATCH_INVARIANT=1` (before any
   vllm import) make the failing reproducer work? If yes, outcome **B**
   (user sets it) or **C-patch** (auto-set at plugin entry).
2. **Python patch at plugin entry**: add guard in `vllm_ascend/__init__.py`
   to auto-detect the bad condition + set the env-var. Users get fixed
   behavior without CLI intervention.
3. **Python patch at call site**: if only 1-2 sites affected, a direct
   version-check branch there is cleaner than the broad env-var
   approach. Example: `linear_batch_invariant` reshape 3D→2D (covers
   training forward but not backward).
4. **C++ rebuild** (Fix C — in scope as of 2026-04-23):
   rebuild `vllm_ascend_C` against the running torch's ABI. Usually
   the "one real fix" for training, because it eliminates the need
   for batch-invariant fallback entirely, and autograd backward goes
   through native PrivateUse1 impl (where all linear/rmsnorm/etc.
   backwards are registered).
5. **C-report**: fix belongs to a different upstream (community torch,
   CANN kernel team). Blocker report only.

Each level narrows scope — favor lower levels, but skip to level 4 if
the issue is known from past sessions to be ABI-drift training-time
(2026-04-23 torch-2.11 case: levels 1-3 bought inference but we had
to do level 4 for training).

### Level 4 rebuild recipe (from 2026-04-23 torch-2.11 session)

```bash
# Inside the Fix B+ overlay container with NPU devices mounted:
docker run --rm --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /tmp/CMakeLists-widen.txt:/vllm-ascend/CMakeLists.txt:ro \
    -v /tmp/fixc-out:/host-out \
    --device=/dev/davinci0 --device=/dev/davinci_manager \
    --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    <fix-B+-overlay> \
    bash -c '
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      export SOC_VERSION=ascend910_9391  # adjust per hardware
      rm -rf /vllm-ascend/build /vllm-ascend/.deps
      cd /vllm-ascend && python3 setup.py build_ext --inplace
      mkdir -p /host-out
      cp /vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-*.so /host-out/
      cp /vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so /host-out/
      cp -r /vllm-ascend/vllm_ascend/lib /host-out/
      cp -r /vllm-ascend/vllm_ascend/_cann_ops_custom /host-out/
    '
```

Gotcha: `CMakeLists.txt:26` hard-pins `torch == 2.9.0`. Patch this
first (accept 2.x minor range you're targeting). This is itself an
upstream patch worth committing and PR-ing.

Then build Fix C image: `FROM <fix-B+-overlay> + COPY /host-out/...`.

Verify native-op reproducer no longer SIGSEGVs (now RuntimeError at
worst, usually PASS).

Re-run V1.4 training smoke with `VLLM_BATCH_INVARIANT=0` explicitly set
(force native custom-op path, bypass Fix B+'s auto-batch-invariant
that's now redundant). Expected: PASS with entropy_loss in baseline
band.

## Phase E: patch application

```bash
cd upstream/vllm-ascend
git fetch origin --tags
git checkout -B ascend-day0-<delta>-<SESSION> <image's vllm-ascend commit>
# apply edits
git commit -m "[BugFix] ..."
git push personal ascend-day0-<delta>-<SESSION>
```

Then rebuild overlay:

```dockerfile
ARG BASE_IMAGE=<torch-day0 deployed>
FROM ${BASE_IMAGE}
COPY utils.py.patched /vllm-ascend/vllm_ascend/utils.py
COPY __init__.py.patched /vllm-ascend/vllm_ascend/__init__.py
RUN python3 -m py_compile /vllm-ascend/vllm_ascend/utils.py && \
    python3 -m py_compile /vllm-ascend/vllm_ascend/__init__.py
```

Build-time smoke stays at py_compile level (no runtime import).
Runtime smoke happens in the post-build docker run with NPU mounts.

## Phase F: smoke to PASS

Run the consumer's V1.3 harness unchanged — **do not** set
`VLLM_BATCH_INVARIANT=1` manually. Passing without the manual env var
proves the patch auto-triggers correctly.

Accept when:
- `smoke_validate.sh --rung V1.3 --image-tag <patched-overlay> ... --chips 0`
  returns 0
- log contains `V1.3 ROLLOUT SMOKE PASSED` marker

## Cross-references

- `../KB_INDEX.md` — known drift surfaces + fix patterns
- `../ALWAYS_LOADED_RULES.md` — Fix level selection order (§ OL-08)
- Concrete session: `workspace/vllm-ascend-day0-{analysis,deploy}-20260423-*/`
- Upstream patch branch: `zhshgmail/vllm-ascend/ascend-day0-torch211-20260423`
