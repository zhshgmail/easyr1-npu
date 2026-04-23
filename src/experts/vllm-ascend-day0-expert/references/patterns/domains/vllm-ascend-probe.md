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

## Phase D: fix-level selection

Walk the levels in order, pick the lowest that works:

1. **Env-var only**: does setting `VLLM_BATCH_INVARIANT=1` (before any
   vllm import) make the failing reproducer work? If yes, outcome **B**
   or **C-patch** (auto-set at plugin entry).
2. **Python patch at plugin entry**: add guard in `vllm_ascend/__init__.py`
   to auto-detect the bad condition + set the env-var. Users get fixed
   behavior without CLI intervention.
3. **Python patch at call site**: if only 1-2 sites affected, a direct
   version-check branch there is cleaner than the broad env-var
   approach.
4. **C++ rebuild**: tracked as tech debt; not this expert's scope
   unless explicitly requested.

Each level narrows scope — favor lower levels.

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
