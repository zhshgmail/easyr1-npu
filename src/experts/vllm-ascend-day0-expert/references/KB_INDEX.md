# vllm-ascend-day0-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL | Phase A |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | OL-03 + OL-08 + outcome matrix + fix-level selection | Phase A |
| [patterns/domains/vllm-ascend-probe.md](patterns/domains/vllm-ascend-probe.md) | Reproducer minimization + call-site location + C++ ABI drift detection | Phase A + B |
| [../../_shared/references/patterns/domains/day0-deploy-artifacts.md](../../_shared/references/patterns/domains/day0-deploy-artifacts.md) | 5 deploy deliverables | Phase E |

## Quick symptoms → classification

| Symptom | Likely outcome | Fix level |
|---|---|---|
| vllm-ascend main tip already has the fix (the shipped release just doesn't) | **A** | ship overlay pointing at newer vllm-ascend commit; no patch needed |
| Segfault in profile_run at `torch._ops.py:1269 __call__` → `vllm_ascend/ops/*.py forward_oot` | **C-patch** | **C++ ABI drift** on `vllm_ascend_C`; route around via python-layer guard + env var (Fix B+) |
| `ImportError: cannot import name 'X' from vllm...` where vllm dropped the symbol | **C-patch** | forward-compat helper in vllm-ascend that tries both old + new import paths |
| Segfault or error only in batch-invariant-NON-gated path | **C-patch** | need code change (env-var workaround alone won't help) |
| V1.3 FAIL but fix belongs to community vllm (e.g. their dispatcher change is the actual bug) | **C-report** | file upstream, session ends at report |
| Consumer just needs to set `VLLM_BATCH_INVARIANT=1` to work | **B** | document env-var workaround in ONBOARDING |

## Known ABI drift surfaces (2026-04-23)

- **torch 2.11 dispatcher refactor** breaks pre-2.11-built
  `vllm_ascend_C.so`. All ops in `torch.ops._C_ascend.*` SIGSEGV when
  called through `torch._ops.py:1269 __call__`. Load + register OK,
  call FAIL. Affected call sites (not exhaustive):
  - `vllm_ascend/ops/layernorm.py:73` AscendRMSNorm.forward_oot
  - `vllm_ascend/sample/sampler.py:139` _apply_top_k_top_p_ascendc
  - Any other site using `torch.ops._C_ascend.*` ops
- Mitigation landed: Fix B+ patch (2 commits on
  `ascend-day0-torch211-20260423` branch of `zhshgmail/vllm-ascend`):
  1. `vllm_ascend/utils.py` — `_torch_abi_safe_for_custom_ops()` +
     guard in `enable_custom_op()`
  2. `vllm_ascend/__init__.py` — set `VLLM_BATCH_INVARIANT=1` at
     plugin import time (before vllm caches it)

## Known vllm API removals that affect vllm-ascend (2026-04-23)

- `vllm_is_batch_invariant` **removed** from
  `vllm.model_executor.layers.batch_invariant` in vllm 0.19.1. vllm-ascend
  0.17 imports it in 5 call sites (`ascend_config.py:132`,
  `batch_invariant.py:24`, `utils.py:262`, `sample/sampler.py:2`).
  **But**: vllm-ascend main already fixed this in PR #7787
  (`811271d1`, merged 2026-04-03) — switched to `vllm.envs.VLLM_BATCH_INVARIANT`.
  So vllm 0.19.1 is **not a valid Day-0 target** (upstream already handled).
  See memory `day0_real_target.md`.

## Fix pattern — auto-enable batch-invariant at plugin entry

Invaluable pattern for any Day-0 where a vllm-ascend call site breaks
under a new upstream. Source: this expert's 2026-04-23 session.

```python
# In vllm_ascend/__init__.py (plugin entry point, imported before vllm)
def _maybe_guard_against_<condition>():
    import os
    if os.environ.get("VLLM_BATCH_INVARIANT") is not None:
        return  # respect user's explicit setting
    if <condition matches>:  # e.g. torch ABI mismatch, missing symbol, etc.
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        # emit warning to stderr

_maybe_guard_against_<condition>()

def register():  # existing
    ...
```

This leverages vllm-ascend's existing batch-invariant gates at
`sampler.py:79`, `ascend_config.py:135`, `batch_invariant.py:139`,
`utils.py:274`, `layernorm.py:72`. All of them check
`vllm_is_batch_invariant()`, so setting the env-var before vllm caches
it bypasses every affected call site in one shot.

## Concrete session artifacts (2026-04-23)

- Analysis: `workspace/vllm-ascend-day0-analysis-20260423-0636/analysis.md`
- Reproducers: `workspace/vllm-ascend-day0-analysis-20260423-0636/isolate_segfault_v*.py`
- Deploy artifacts: `workspace/vllm-ascend-day0-deploy-20260423-0655/`
  (Dockerfile overlay patch, utils.py.patched, __init__.py.patched,
  smoke, deploy, ONBOARDING, PR_MATERIAL)
- Patched branch: `zhshgmail/vllm-ascend/ascend-day0-torch211-20260423`
  (commits `7c2078e7` + `caa55fed`)
- Patched overlay image on A3: `easyr1-npu-torch211-vllmascend-fixb:ascend-day0-torch211-20260423`

## Validated smoke matrix for Fix B+ patch (2026-04-23 late dusk)

| Rung | Result | Notes |
|---|---|---|
| V1.3 (Qwen2-0.5B rollout, 1 chip, enforce_eager, no training) | **PASS** | marker `V1.3 ROLLOUT SMOKE PASSED`; 3/3 prompts produce text. See `workspace/.../deploy/smoke.log` |
| V1.4 (Qwen2-0.5B GRPO training, 2 chips, full trainer loop) | **FAIL** | `AssertionError: x must be a 2D tensor` in `vllm-ascend/ops/triton/batch_invariant/matmul.py:261 linear_persistent`. Training path passes 3D `[batch, seq, hidden]` to `F.linear`, but batch-invariant's `linear_batch_invariant` Triton kernel only supports 2D inputs. Log: `/tmp/z00637938/easyr1-logs/V1.4-easyr1-npu-torch211-vllmascend-fixb-*20260423-011007.log` |

**Limitation of Fix B+ batch-invariant escape hatch**: works for inference
paths (V1.3) where linear inputs are 2D-flattened, breaks on training
paths (V1.4) where FSDP + trainer emit 3D tensors to F.linear directly.
**Root cause**: not an ABI-drift issue anymore — the triton-ascend
batch-invariant linear kernel itself has a dim-restriction assert
(`matmul.py:261`). Presumably written with inference shapes in mind.

### Option 1 (quick): loosen the assert to handle 3D via reshape

vllm-ascend patch candidate (untested, but clean):

```python
# vllm_ascend/ops/triton/batch_invariant/matmul.py linear_persistent
def linear_persistent(x, weight, bias=None, ...):
    orig_shape = x.shape
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])  # flatten batch*seq × hidden
        output = <existing 2D kernel call>
        return output.reshape(*orig_shape[:-1], output.shape[-1])
    assert x.dim() == 2, "x must be a 2D tensor"
    ...
```

### Option 2 (simpler for user): skip batch-invariant on linear

Add per-op check in Fix B+ guard: if running torch ABI-unsafe AND
training mode (which can be detected via `torch.is_grad_enabled()`
or an env var), don't register the batch-invariant `linear` impl at
all — fall back to torch's native `F.linear` (slower but correct).

### Option 3 (tracks separately): this is actually a triton-ascend-day0
issue, not vllm-ascend — the `linear_batch_invariant` kernel lives in
`vllm_ascend/ops/triton/batch_invariant/matmul.py` but is a triton
kernel. Could be routed to a future `triton-ascend-day0-expert` when
that skill exists.

## Session status (as of dusk 2026-04-23)

- Outcome **A-with-note (V1.3 only)**: Fix B+ patch PR-material ready,
  V1.3 PASS, but V1.4 training mode surfaces a separate 3D-tensor
  issue in batch-invariant linear. Documented as known-broken in
  ONBOARDING.md + PR_MATERIAL.md should mention it as follow-up.

## Related KB (sibling experts)

- `torch-day0-expert/` — produces the base image this expert builds on
- `vllm-day0-expert/` — Day-0 for vllm itself on a stable vllm-ascend
  (usually runs after this expert's Day-0 on vllm-ascend)
- `transformers-day0-expert/` — sibling scaffold
- `../../_shared/references/patterns/domains/day0-deploy-artifacts.md`
  — Phase E deploy deliverables
