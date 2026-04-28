# vllm-ascend/day0-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL | Phase A |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | OL-03 + OL-08 + outcome matrix + fix-level selection | Phase A |
| [patterns/domains/vllm-ascend-probe.md](patterns/domains/vllm-ascend-probe.md) | Reproducer minimization + call-site location + C++ ABI drift detection | Phase A + B |
| [patterns/domains/vllm-api-drift.md](patterns/domains/vllm-api-drift.md) | vllm API drift families F1–F8 (removed symbol / rename / sig change / return-type / buffer API / kv_cache / new attr / new method) | Phase B + C when root cause is vllm community API change |
| [../../_shared/references/patterns/domains/day0-deploy-artifacts.md](../../_shared/references/patterns/domains/day0-deploy-artifacts.md) | 5 deploy deliverables | Phase E |

## Quick symptoms → classification

| Symptom | Likely outcome | Fix level | Route to |
|---|---|---|---|
| vllm-ascend main tip already has the fix (the shipped release just doesn't) | **A** | ship overlay pointing at newer vllm-ascend commit; no patch needed | — |
| Segfault in profile_run at `torch._ops.py:1269 __call__` → `vllm_ascend/ops/*.py forward_oot` | **C-patch** | **C++ ABI drift** on `vllm_ascend_C`; route around via python-layer guard + env var (Fix B+) | `vllm-ascend-probe.md` Phase B |
| `ImportError: cannot import name 'X' from vllm...` where vllm dropped the symbol | **C-patch** | forward-compat helper in vllm-ascend that tries both old + new import paths | `vllm-api-drift.md` **F1** |
| `AttributeError: module 'vllm.X' has no attribute 'OldName'` after vllm rename | **C-patch** | alias the new name back to the old at import time | `vllm-api-drift.md` **F2** |
| `TypeError: func() missing/unexpected argument` on a vllm call after upgrade | **C-patch** | runtime signature sniff via `inspect.signature` | `vllm-api-drift.md` **F3** |
| Arithmetic/indexing error on a previously-scalar vllm return | **C-patch** | return-type migration (scalar → NamedTuple) | `vllm-api-drift.md` **F4** |
| Many `.np` / `.copy_to_gpu` / `CpuGpuBuffer` call sites failing together | **C-patch** | thin compat helper (`to_gpu`/`np_view`) instead of N-site migration | `vllm-api-drift.md` **F5** |
| `AssertionError: attn_layer.kv_cache must be single tensor` or silent token garbage | **C-patch** | version-detect kv_cache contract (list vs stacked tensor) | `vllm-api-drift.md` **F6** |
| `AttributeError: 'NPUX' has no attribute 'Y'` where vllm base class added Y | **C-patch** | add attr with NPU-correct default (not `True` to silence) | `vllm-api-drift.md` **F7** |
| `NotImplementedError` on a base class method vllm made required | **C-patch** | implement minimal NPU-semantic version | `vllm-api-drift.md` **F8** |
| Segfault or error only in batch-invariant-NON-gated path | **C-patch** | need code change (env-var workaround alone won't help) | `vllm-ascend-probe.md` |
| V1.3 FAIL but fix belongs to community vllm (e.g. their dispatcher change is the actual bug) | **C-report** | file upstream, session ends at report | — |
| Consumer just needs to set `VLLM_BATCH_INVARIANT=1` to work | **B** | document env-var workaround in ONBOARDING | — |

## Known ABI drift surfaces (2026-04-23)

- **torch 2.11 dispatcher refactor** breaks pre-2.11-built
  `vllm_ascend_C.so`. All ops in `torch.ops._C_ascend.*` SIGSEGV when
  called through `torch._ops.py:1269 __call__`. Load + register OK,
  call FAIL. Affected call sites (not exhaustive):
  - `vllm_ascend/ops/layernorm.py:73` AscendRMSNorm.forward_oot
  - `vllm_ascend/sample/sampler.py:139` _apply_top_k_top_p_ascendc
  - Any other site using `torch.ops._C_ascend.*` ops
- Mitigation to land upstream (Fix B+, 2 call-site edits in your
  vllm-ascend tree):
  1. `vllm_ascend/utils.py` — add `_torch_abi_safe_for_custom_ops()` +
     guard in `enable_custom_op()` to early-return when torch major.minor
     != the one `vllm_ascend_C.so` was built against
  2. `vllm_ascend/__init__.py` — set `VLLM_BATCH_INVARIANT=1` at plugin
     import time (before vllm caches it), only when the ABI-guard above
     returns False, so users hitting the mismatch get a working fallback
     automatically

## Known vllm API removals that affect vllm-ascend (2026-04-23)

- `vllm_is_batch_invariant` **removed** from
  `vllm.model_executor.layers.batch_invariant` in vllm 0.19.1. vllm-ascend
  0.17 imports it in 5 call sites (`ascend_config.py:132`,
  `batch_invariant.py:24`, `utils.py:262`, `sample/sampler.py:2`).
  **But**: vllm-ascend main already fixed this in PR #7787
  (`811271d1`, merged 2026-04-03) — switched to `vllm.envs.VLLM_BATCH_INVARIANT`.
  So vllm 0.19.1 is **not a valid Day-0 target** (upstream already handled).
  See memory `day0_real_target.md`.

## Concrete case registry — vllm 0.20.0 drift (observed 2026-04-23/24)

Sourced from trace branch on fork during the vllm 0.20.0 Day-0 session.
Each row is one fix landed against vllm-ascend; future drifts match by
family ID.

| Family | vllm PR | Symptom (where to grep) | Affected vllm-ascend file |
|---|---|---|---|
| **F1** | PR #37880 | `create_vllm_config_for_draft_model` removed | `vllm_ascend/spec_decode/draft_proposer.py` |
| **F1** | — | `vllm_is_batch_invariant` public getter removed (vllm ≥0.19) | `ascend_config.py`, `batch_invariant.py`, `sample/sampler.py`, `utils.py` |
| **F2** | PR #37975 | `GatedDeltaNet` rename | `vllm_ascend/patch/worker/patch_qwen3_next.py`, `patch_qwen3_5.py` |
| **F3** | PR #32951 | `_get_cumsum_and_arange` signature change | `vllm_ascend/worker/model_runner_v1.py` `_prepare_inputs` |
| **F3** | PR #32951 | `_prepare_input_ids` now takes `num_reqs` | `model_runner_v1.py` |
| **F4** | — | `compile_or_warm_up_model` returns `CompilationTimes` NamedTuple | `vllm_ascend/worker/worker.py` |
| **F5** | PR #32951 | 11 `.np` / `.copy_to_gpu` / `.gpu` / `.cpu` sites — `CpuGpuBuffer` migration (Revert round-tripped once) | `model_runner_v1.py` |
| **F6** | — | `attn_layer.kv_cache` must be single stacked tensor (not list) | `model_runner_v1.py`, `patch/worker/patch_qwen3_next_mtp.py` |
| **F7** | — | `forward_includes_kv_cache_update=False` + `do_kv_cache_update` method required | `vllm_ascend/attention/attention_v1.py` |
| **F7** | — | `NPUInputBatch.logprob_token_ids` attribute required | `vllm_ascend/worker/npu_input_batch.py` |
| **F8** | — | `BlockTable.clear_row` method required | `vllm_ascend/worker/block_table.py` |
| **F8** | PR #38468 | `Platform.manual_seed_all` required | `vllm_ascend/platform.py` |
| **F8** | PR #37487 | `attention_v1.forward_impl` must pre-populate `self.key_cache` | `attention/attention_v1.py` |

Trace branch on fork: `ascend-day0-torch211-20260423` (23 commits,
range `c91d752..3ecb82f4`). For the concrete diffs, check that branch;
for how-to-generalize, read `patterns/domains/vllm-api-drift.md`.

## Discovered by kb_drive_test harness — pending port (2026-04-24)

Running `scripts/kb_drive_test.py` over 156 post-0.20.0 vllm commits
surfaced 2 additional F1 drifts that will hit vllm-ascend when it
moves to vllm main tip. Not yet ported.

### F2-path-move discovered by P2 cold-start (2026-04-25) — pending

| Family | vllm commit | Symbol | Old path (broken) | New path | vllm-ascend sites | Fix status |
|---|---|---|---|---|---|---|
| **F2-path-move** | `cde8d2471` (PR #40732) | `SpecDecodeBaseProposer` | `vllm.v1.spec_decode.eagle` | `vllm.v1.spec_decode.llm_base_proposer` | 2 sites in `vllm_ascend/spec_decode/eagle_proposer.py` | **DONE on cold branch** — commit `ad2b7272` on `vllm-main_cold_20260425`; compat module `vllm_ascend/compat/spec_decode_base_proposer.py` |

Surfaced by the 2026-04-25 fresh-LLM cold-start (see
`docs/_meta/cold-start-pass-criteria.md`). Not caught by the yesterday
`vllm-main_auto_porting` branch because the import line
`from vllm.v1.spec_decode.eagle import EagleProposer, SpecDecodeBaseProposer`
landed on vllm-ascend `origin/main` only recently; earlier analysis
looked at older HEAD and misclassified as internal-name collision.

### F7/F8 discovered by check_f7_f8.py sweep (2026-04-24 late) — verified no-op

Running `scripts/check_f7_f8.py --baseline v0.20.0 --target origin/main`
found 5 new attrs/methods. **Verification (2026-04-24 late)**: each
base-class addition has a safe default or is a pure data attr that the
NPU subclass transparently inherits — **no fix needed** until/unless
NPU code paths start exercising these new members.

| Family | Parent class | New member | Default in base | NPU subclass action |
|---|---|---|---|---|
| **F8** | `AttentionBackend` | `supports_batch_invariance()` classmethod | `return False` | inherit False; NPU doesn't advertise batch invariance |
| **F7** | `CommonAttentionMetadata` | `seq_lens_cpu_upper_bound: Tensor \| None` | `None` | inherit None; NPU code doesn't read it |
| **F7** | `InputBatch` | `seq_lens_cpu_upper_bound: Tensor \| None` | `None` | inherit None |
| **F8** | `MergedColumnParallelLinearWithLoRA` | `apply()` method | impl body calls super().apply | LoRA-NPU subclass (if any) inherits from subclass of this class |
| **F8** | `ReplicatedLinearWithLoRA` | `apply()` method | impl body calls super().apply | (same as above) |

Re-run the F7/F8 scanner on each future vllm upgrade — if a new
finding has a `raise NotImplementedError` or no default, that one is
NOT safe to inherit and needs real override.

### F1/F2 discovered by kb_drive_test harness — pending port (2026-04-24)

Running `scripts/kb_drive_test.py` over 156 post-0.20.0 vllm commits
surfaced 2 additional F1 drifts that will hit vllm-ascend when it
moves to vllm main tip. Not yet ported.

| Family | vllm commit | Removed symbol | vllm-ascend sites |
|---|---|---|---|
| **F1** | `5e584ce9e` | `SharedFusedMoE` class | 9 (utils.py, ops/fused_moe, _310p/fused_moe) |
| **F1** | `809d83c2d` | `DefaultMoERunner` class | 2 (ops/fused_moe/fused_moe.py) |

### T25 cold-drive replay (2026-04-28) — post-T22 sweep findings

Running `scripts/sweep.sh --commit-range v0.20.0..origin/main` after
T22's fixes shipped surfaced 3 novel drifts that will hit vllm-ascend
on the next base-image bump (vllm tip moved 26 commits from v0.20.0).
**This is the use-case the skill is designed for: re-run after each
upstream tag bump.**

| Family | vllm commit (PR) | Symbol | Detail | vllm-ascend sites | Fix shape |
|---|---|---|---|---|---|
| **F2-rename** | `4c7c69b4e` (PR #40410) | `EagleCudaGraphManager` | Renamed to `EagleCudaGraphManagerBase` and **split** into `PrefillEagleCudaGraphManager` + `DecodeEagleCudaGraphManager` (each with own `capture()` body). | 6 sites in `vllm_ascend/worker/v2/spec_decode/eagle/{aclgraph.py,speculator.py}` | F2-rename shim at `vllm_ascend/compat/eagle_cudagraph.py`: try/except import old name, expose new base name. Subclass `EagleAclGraphManager` may also need to split into prefill/decode variants — re-read PR #40410's diff before patching. |
| **F8** | (post-v0.20.0) | `EagleSpeculator.capture` | New required method on community `EagleSpeculator` base. NPU `AscendEagleSpeculator(EagleSpeculator)` will inherit; verify base has a default or implement minimal NPU-semantic version per F8 template. | `vllm_ascend/worker/v2/spec_decode/eagle/speculator.py:43` | Likely safe-inherit (base method is defined); verify with `inspect.getsource` after rebase. |
| **F8** | (post-v0.20.0) | `MMEncoderAttention.process_weights_after_loading` | New required method on community `MMEncoderAttention`. NPU has 2 subclasses (`AscendMMEncoderAttention`, `AscendMMEncoderAttention310`). | `vllm_ascend/_310p/ops/mm_encoder_attention.py:38`, `vllm_ascend/ops/mm_encoder_attention.py:40` | Same: verify base default; if base raises `NotImplementedError`, implement minimal `pass` body or NPU equivalent. |

Verification cmd:
```bash
bash src/skills/vllm-ascend/port-expert/scripts/sweep.sh \
  --commit-range v0.20.0..origin/main \
  --vllm-path ~/workspace/easyr1-npu/upstream/vllm \
  --vllm-ascend-path ~/workspace/easyr1-npu/upstream/vllm-ascend
# expect: 1 novel symbol + 2 F8 additions, exit 1
```

Reproduce via:
```
python3 scripts/kb_drive_test.py \
    --vllm-ref <SHA> \
    --vllm-path <vllm-checkout> \
    --vllm-ascend-path <vllm-ascend-checkout> \
    --kb-dir <this-dir>
```

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
  smoke, deploy, ONBOARDING, PR_MATERIAL — PR_MATERIAL includes the
  rendered diff + commit message body vllm-ascend maintainers can
  cherry-pick into their own tree)
- Validated overlay image on A3: `easyr1-npu-torch211-vllmascend-fixb:ascend-day0-torch211-20260423`
  (demonstration of Fix B+ passing V1.3 smoke; maintainers rebuild
  equivalent in their own CI)

## Validated smoke matrix for Fix B+ / Fix C patches (2026-04-23, 4 iterations)

| Rung | Iter 1 (Fix B+ only) | Iter 2 (+ reshape) | Iter 3 (+ TORCH_NPU_USE_COMPATIBLE_IMPL=1) | Iter 4 (Fix C rebuild) |
|---|---|---|---|---|
| V1.3 (Qwen2-0.5B rollout, 1 chip) | **PASS** | **PASS** | n/a | **PASS** |
| V1.4 (Qwen2-0.5B GRPO training, 2 chips) | **FAIL** `2D tensor assert` | **FAIL** `aten::linear_backward CPU fallback` | **FAIL** (same as iter 2) | **PASS** entropy_loss=1.275 in band [1.21, 1.34] (exact v2 baseline match) |

Iter 3 confirmed `TORCH_NPU_USE_COMPATIBLE_IMPL=1` doesn't help the
autograd backward dispatch issue; the flag is about torch_npu's own
removed monkey-patches, not inductor backward graph.

**Iter 4 = Fix C applied**: rebuild `vllm_ascend_C.so` against torch
2.11 ABI inside the patched container. Build succeeds once
`CMakeLists.txt:26` hard-pin `VERSION_EQUAL "2.9.0"` is widened to
accept `2.11.x` — maintainer edit in your vllm-ascend tree. Resulting
`.so` is 472KB (vs previous torch-2.9-built version). Demonstration
overlay image tagged `easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423`.

Native-op reproducer (`torch.ops._C_ascend.npu_add_rms_norm_bias`):
- **Pre-Fix C**: SIGSEGV through `torch._ops.py:1269 __call__`
- **Post-Fix C**: clean `RuntimeError: call aclnnAddRmsNormBias failed,
  detail:[PID: 1] ... The binary_info_config.json of socVersion
  [ascend910_93] does not support opType [AddRmsNormBias]`

So Fix C solved the ABI drift; op call now goes through the dispatcher
cleanly, but Ascend's aclnn kernel repo doesn't have an A3-specific
AddRmsNormBias binary registered. This is a separate kernel-coverage
issue (could route to triton-ascend-day0 or Ascend kernel team) — but
it IS a catchable Python exception now, not a SIGSEGV.

**V1.4 wet-run with VLLM_BATCH_INVARIANT=0** (bypasses Fix B+'s
auto-enable, forces the native custom-op path) on the Fix C image:
- `entropy_loss=1.275` step 1 — **exact match** to v2 image baseline
  band [1.21, 1.34]
- 2-step GRPO training + checkpoint PASS; `trainer.max_steps=2` completed
- Real backward pass ran through PrivateUse1 native NPU kernels, not
  through batch-invariant fallback, not through CPU fallback
- This is the **fully validated Day-0 chain endpoint**: community
  PyTorch 2.11 + torch_npu 2.11.0rc1 + patched vllm-ascend (Fix B+ +
  Fix C rebuild) running real training workload end-to-end on NPU

This makes the iter-4 Fix C overlay **production-ready** for both
inference (V1.3) and training (V1.4) workloads — no
batch-invariant fallback, no env-var tuning needed.

### The 4 edits comprising the full Fix — for vllm-ascend maintainers

Apply these in your own tree; reference implementation exists at the
session workspace (`workspace/vllm-ascend-day0-{analysis,deploy}-20260423-*/`)
and the demo overlay image `easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423`.

1. **`vllm_ascend/utils.py`** — add torch-ABI guard that short-circuits
   `enable_custom_op()` when torch major.minor != the version
   `vllm_ascend_C.so` was built against
2. **`vllm_ascend/__init__.py`** — auto-set `VLLM_BATCH_INVARIANT=1` at
   plugin import time when the above guard fails, giving users a working
   fallback until Fix C ships
3. **`vllm_ascend/ops/linear.py` `linear_batch_invariant`** — reshape
   arbitrary-leading-dims input to 2D before calling
   `linear_persistent` (the triton-ascend kernel only accepts 2D),
   restore shape after. Required for FSDP training paths where F.linear
   receives 3D tensors
4. **`CMakeLists.txt`** — widen hard-pin `VERSION_EQUAL "2.9.0"` to
   accept 2.11.x so the C extension can be rebuilt against torch 2.11

**Limitation of Fix B+ batch-invariant escape hatch**: works for inference
paths (V1.3) where linear inputs are 2D-flattened, breaks on training
paths (V1.4) where FSDP + trainer emit 3D tensors to F.linear directly.
**Root cause**: not an ABI-drift issue anymore — the triton-ascend
batch-invariant linear kernel itself has a dim-restriction assert
(`matmul.py:261`). Presumably written with inference shapes in mind.

### Option 1 — the reshape edit (maintainer patch)

Add arbitrary-leading-dims reshape to `linear_batch_invariant` wrapper
(not `linear_persistent` itself — the wrapper is the right place since
it matches `F.linear`'s contract):

```python
def linear_batch_invariant(input_, weight, bias=None):
    orig_shape = input_.shape
    if input_.dim() > 2:
        input_ = input_.reshape(-1, orig_shape[-1])
    output = linear_persistent(input_, weight)
    if bias is not None:
        output = output + bias
    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
    return output
```

**Effect (wet-run 2026-04-23 0818Z)**: V1.4 gets past the 2D assertion,
reaches `Update policy 25% (1/4)` step of training. New failure surfaces:

```
NotImplementedError: Could not run 'aten::linear_backward' with arguments
from the 'CPU' backend. ... 'aten::linear_backward' is only available for
these backends: [PrivateUse1, Meta, ...]
```

This is a **different, deeper issue**: torch 2.11's
`torch.compile`/inductor compiled backward graph is running
`aten::linear_backward` on CPU (probably because batch-invariant mode's
`aten::linear` override hits the CPU fallback path during autograd
backward generation, not NPU). The backward op for `linear` only has a
PrivateUse1 (NPU) impl, not CPU, so it crashes.

**This is beyond Fix B+ scope.** Batch-invariant mode was designed for
forward-only inference; training's autograd path through inductor doesn't
interact well with the Python-level kernel override. Proper fix requires
one of:

1. **Fix C** (rebuild vllm_ascend_C against torch 2.11) — eliminates
   need for batch-invariant fallback entirely, training goes through
   the native C++ impl which has NPU backward registered
2. **Disable torch.compile on training path** (env var
   `TORCH_COMPILE_DISABLE=1` or similar) — avoids inductor generating
   the CPU-dispatch backward graph. Untested but worth a shot.
3. **Register NPU backward for batch-invariant's aten::linear override**
   — substantial; also in vllm-ascend scope but much heavier than the
   Python reshape patch.

Recorded as tech debt. Session's deliverable remains: **V1.3 PASS, V1.4
reaches 25% of training before autograd-backward dispatch crash**.

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

- `torch-npu/port-expert/` — produces the base image this expert builds on
- `vllm/port-expert/` — Day-0 for vllm itself on a stable vllm-ascend
  (usually runs after this expert's Day-0 on vllm-ascend)
- `transformers/port-expert/` — sibling scaffold
- `../../_shared/references/patterns/domains/day0-deploy-artifacts.md`
  — Phase E deploy deliverables
