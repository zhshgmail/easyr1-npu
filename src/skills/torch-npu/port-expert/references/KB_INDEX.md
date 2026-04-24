# torch-day0-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL | Phase A |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | OL-03 + OL-08 + Outcome matrix + pre-probe | Phase A |
| [patterns/domains/torch-overlay-probe.md](patterns/domains/torch-overlay-probe.md) | torch-specific API probe + runtime smoke + API-drift classification | Phase A + B + C |
| [patterns/domains/overlay-image.md](patterns/domains/overlay-image.md) | Dockerfile.overlay-torch<M><m> template (pip overlay, no `import torch` at build time) | Phase B |
| [../../_shared/references/patterns/domains/day0-deploy-artifacts.md](../../_shared/references/patterns/domains/day0-deploy-artifacts.md) | 5 deploy deliverables mandatory for A / C-patch | Phase E |

## Quick symptoms → classification

| Symptom after pip-overlay target torch + torch_npu rc | Likely outcome |
|---|---|
| 6/6 runtime smoke PASS (metadata / import torch / import torch_npu / device count / NPU op / API presence) | **A** |
| Build-time `ImportError: libascend_hal.so` during RUN `import torch` (torch ≥ 2.11 only) | **Author error** — remove `import torch` from build; use `python3 -m py_compile` + runtime smoke |
| Runtime `import torch_npu` fails with `ImportError: _C` | Likely rc wheel / CANN mismatch; check version pair table |
| Runtime smoke 6/6 PASS but downstream vllm-ascend / third-party C++ ext SIGSEGV on op call | **A for torch layer** + **C-patch for downstream layer** (see vllm-ascend-day0 / transformers-day0) |
| `torch.npu.Stream.native_handle` missing (or similar surface hole) | **B** or **C-patch** depending on which side needs the fix |
| New PyTorch op has no torch_npu kernel AND consumer path exercises it | **C-patch** — add python fallback decomposition in `torch_npu/contrib/` or wait for kernel |
| ABI-blocker in `c10/core/DispatchKey.h` / tensor layout changes | **C-report** to Ascend/pytorch team (ABI changes require rc rebuild, not a python patch) |

## 2026-04-23 baseline snapshot

- v1 NPU image: torch 2.7.x / torch_npu 2.7.x / CANN 8.4
- v2 NPU image: torch 2.9.0+cpu / torch_npu 2.9.0 / CANN 8.5.1
- community PyTorch latest: **v2.11.0** (released ~2026-04 per release history)
- torch_npu latest stable: **2.9.0** (Jan 2026)
- torch_npu Day-0 rc on PyPI: **2.11.0rc1** (2026-03-24)
- CANN bundle matrix: `26.0.0-beta.1` pairs only with pytorch 2.10.0;
  no CANN bundle for pytorch 2.11 yet

Session torch-day0-manual-20260423-0537 results (torch 2.11 + torch_npu 2.11.0rc1 + CANN 8.5.1):
- 6/6 runtime smoke PASS (`torch 2.11.0+cpu`, `torch_npu 2.11.0rc1`, 16 NPUs, basic matmul on NPU OK, `torch.npu.Stream.native_handle` present)
- Classified: **A** (torch layer works)
- Downstream finding: vllm-ascend 0.17 (pre-2.11-built C++ extension) segfaults in RMSNorm custom op during vllm profile_run. That's vllm-ascend-day0 territory, not this session's scope.

## 2026-04-23 wet-run findings folded into KB

1. **PyTorch 2.11 `_import_device_backends()` auto-load** breaks build-time `import torch` inside docker build containers (no CANN libs mounted). Workaround: use py_compile + runtime smoke. Authored into `torch-overlay-probe.md` Phase B.

2. **Pre-2.11-built C++ extensions SIGSEGV on torch 2.11 dispatcher**. The `.so` loads and `TORCH_LIBRARY_IMPL` registers fine, but the first op call through `torch._ops.py:1269` segfaults because the ABI (IValue unpacking, DispatchKey layout) doesn't match. Use `TORCH_NPU_USE_COMPATIBLE_IMPL=1` or `VLLM_BATCH_INVARIANT=1` as downstream escape hatches.

3. **rc wheel version pin tight**: `torch-npu==2.11.0rc1` declares `requires_dist: ["torch==2.11.0"]` exact. Install combo must be `torch==2.11.0+cpu` from `download.pytorch.org`, not dev builds.

4. **CANN 8.5.1 works with torch_npu 2.11.0rc1** despite README pairing with 8.5.0. One patch-version ahead is fine (same pattern as torch_npu 2.9.0 + CANN 8.5.1 historically).

## Related KB (sibling / downstream experts)

- `vllm-ascend/port-expert/` (next layer) — when torch 2.11 overlay is
  deployed, vllm-ascend on top needs Fix B+ ABI guard. See
  `workspace/vllm-ascend-day0-*/analysis.md`.
- `vllm/port-expert/` — Day-0 vllm on NPU when vllm-ascend is already
  compatible. Usually runs AFTER torch-day0 has shipped the torch base.
- `transformers/port-expert/` — transformers Day-0 runs independently
  of torch-day0 usually (transformers doesn't pin torch ≥ X hard unless
  it uses a new torch API).
- `../../_shared/references/patterns/domains/day0-deploy-artifacts.md`
  — mandatory Phase E deliverables template.

## Concrete artifacts from 2026-04-23 session

- Analysis: `workspace/torch-day0-analysis-20260423-0531/analysis.md`
- Manual port log: `workspace/torch-day0-manual-20260423-0537/PROGRESS.md`
- Deploy artifacts: `workspace/torch-day0-deploy-20260423-0548/` (Dockerfile, smoke, deploy script, ONBOARDING, blocker-report)
- Overlay image on A3: `easyr1-npu-torch211:torch-day0-manual-20260423-0537`

---

## 2026-04-24 — torch 2.11 → 2.12 inductor path drift (F2-path-move family)

Separate problem class from "torch minor upgrade overlay" above: within
the *internal* torch API that torch_npu reaches into
(`torch._inductor.*`, `torch._dynamo.*`), symbols move between paths
each release. Detecting + patching these is the primary drift workload
for torch_npu going forward.

### Pattern family — F2-path-move

Symbol identity preserved, module path changed.

See the canonical definition in
[../../../vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md §F2-path-move](../../../vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md).

Fix shape: `torch_npu/compat/<symbol_group>.py` with try/except,
try old path first (keep old torch working), fall back to new.

### Concrete case registry

| # | Family | torch range | Symbol | Old path | New path | Fix status |
|---|---|---|---|---|---|---|
| 1 | F2-path-move | 2.11 → 2.12-rc3 | `FloorDiv` | `torch._inductor.utils` | `torch.utils._sympy.functions` | **DONE** — commit `2d81f06c8` on `gitcode.com/zhengshencn_hwca/pytorch` branch `torch-2.12_auto_porting` |
| 2 | F2-path-move | 2.11 → 2.12-rc3 | `ModularIndexing` | `torch._inductor.utils` | `torch.utils._sympy.functions` | **DONE** (same commit, 3 files) |
| 3 | F2-path-move | 2.11 → 2.12-rc3 | `FloorDiv` (also imported from `_inductor.ir`) | `torch._inductor.ir` | `torch.utils._sympy.functions` | **DONE** — commit `1ef8d845a` |
| 4 | F2-path-move | 2.11 → 2.12-rc3 | `ModularIndexing` (also from `_inductor.ir`) | `torch._inductor.ir` | `torch.utils._sympy.functions` | **DONE** — commit `1ef8d845a` |
| 5 | F2-path-move | 2.11 → 2.12-rc3 | `LoopBody` | `torch._inductor.ir` | `torch._inductor.loop_body` | **DONE** — commit `1ef8d845a` |
| 6 | F2-path-move | 2.11 → 2.12-rc3 | `Reduction` | `torch._inductor.ir` | `torch._decomp.decompositions` | **DONE** — commit `1ef8d845a` |
| 7 | F2-path-move | 2.11 → 2.12-rc3 | `ReductionHint` | `torch._inductor.ir` | `torch._inductor.runtime.hints` | **DONE** — commit `1ef8d845a` |
| 8 | F2-path-move | 2.11 → 2.12-rc3 | `IndentedBuffer` | `torch._inductor.codegen.common` | `torch._inductor.utils` | **DONE** — commit `1ef8d845a` |
| 9 | F2-path-move | 2.11 → 2.12-rc3 | `DisableReduction` | `torch._inductor.codegen.simd` | `torch._inductor.codegen.simd_kernel_features` | **DONE** — commit `1ef8d845a` |
| 10 | F2-path-move | 2.11 → 2.12-rc3 | `EnableReduction` | `torch._inductor.codegen.simd` | `torch._inductor.codegen.simd_kernel_features` | **DONE** — commit `1ef8d845a` |
| 11 | F2-path-move | 2.11 → 2.12-rc3 | `SIMDKernelFeatures` | `torch._inductor.codegen.simd` | `torch._inductor.codegen.simd_kernel_features` | **DONE** — commit `1ef8d845a` |
| 12 | F2-path-move | 2.11 → 2.12-rc3 | `UnsupportedFakeTensorException` | `torch._dynamo.utils` | `torch._subclasses.fake_tensor` | **DONE** — commit `1ef8d845a` |
| 13 | F2-path-move | 2.11 → 2.12-rc3 | `free_symbol_is_type` | `torch._inductor.codegen.common` | `torch.utils._sympy.symbol` | **DONE** — commit `1ef8d845a` |

**All rows 1-13 fix landed** (2026-04-24): compat modules
`torch_npu/compat/sympy_functions.py`, `inductor_ir.py`,
`inductor_codegen_common.py`, `inductor_codegen_simd.py`, `dynamo_utils.py`
on `gitcode.com/zhengshencn_hwca/pytorch` branch `torch-2.12_auto_porting`
(commits `2d81f06c8` + `1ef8d845a`). All 15 touched files py_compile clean.

### IMPORTANT: rows 1-13 are DEFENSIVE, not observed breakage (2026-04-24 late)

Cold-driven `torch-npu/port-expert` orchestrator run discovered that at
v2.12.0-rc3, the 13 old paths **still re-export** every one of the 13
symbols. The original claim ("moved from X to Y") described the
*canonical location* change, not an import breakage. `import torch._inductor.utils.FloorDiv`
works in 2.12-rc3 today. The compat shims therefore don't fix any
current bug; they are forward-compat protection for a future release
that eventually removes the re-exports. Keeping them is defensible
(every torch release removes some re-exports), but do **not** claim
these were needed to make torch 2.12 work — they weren't.

### Row 14: real observed v2.11→v2.12-rc3 breakage

| # | Family | torch range | Symbol | Old path | State at v2.12-rc3 | Fix status |
|---|---|---|---|---|---|---|
| 14 | **F1** (real) | 2.11 → 2.12-rc3 | `Union` (typing re-export) | `torch._inductor.codecache` | `Union` NOT imported into module namespace anymore (PEP 604 cleanup) | **DONE** — commit `5092fd54c` inline try/except in `torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/codecache.py` |

Evidence: `torch/_inductor/codecache.py:37` at v2.11.0 imports
`Any, ..., Union` from typing. Same line at v2.12.0-rc3 drops `Union`.
`torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/codecache.py:29-34`
was doing `from torch._inductor.codecache import ..., Union, ...`.

Affected torch_npu files for rows 1-2:
- `torch_npu/_inductor/lowering_fx.py:38`
- `torch_npu/_inductor/codegen/split_tiling.py:7`
- `torch_npu/_inductor/codegen/scheduling.py:27`

**Verification status (2026-04-24)**: rows 1-2 shim passed `/drift-port-validate` equivalent — 6/6 checks (3 OLD torch + 3 NEW torch paths) on local CPU stub-environment.

## F7/F8 discovered by check_f7_f8.py sweep (2026-04-24 late)

Running `scripts/check_f7_f8.py --baseline v2.11.0 --target v2.12.0-rc3`
found 6 new attrs/methods on torch parent classes. Verification:

| Class | New member | Verification outcome |
|---|---|---|
| `AsyncCompile` | `cutlass`, `metal`, `xpu` methods | **SAFE INHERIT** — new backend-specific methods (CUTLASS, XPU, Metal); NPU has own `CustomAsyncCompile` that overrides `process_pool`, doesn't call the new methods. torch_npu source grep shows 0 `.cutlass` / `.metal` / `.xpu` call sites. No fix needed. |
| `AutocastModeVariable` | `python_type` method | **SAFE INHERIT** — base returns a type, NPU doesn't override. |
| `GridExpr` | `from_meta_lazy`, `generate_lazy` | **SAFE INHERIT** — new lazy-compile variants; NPU grep shows 0 call sites to these methods. |
| `OutputAliasInfo` | `requires_grad_for_backward: bool` | **FALSE POSITIVE (scanner-filtered now)** — torch_npu defines its own shadow class at `npu/_graph_tree.py:706`. Scanner v2 filters this out via "must-be-imported-from-torch" check. |
| `PythonWrapperCodegen` | 6 `codegen_cuda_stream_*` / `codegen_deferred_*` / `generate_extern_kernel_multi_out` / `register_alignment_check_inputs` | **SAFE INHERIT** — NPU grep shows 0 call sites to any of the 6 new methods. |
| `StreamContextVariable` | `python_type` | **SAFE INHERIT** — same as AutocastModeVariable. |

**Conclusion**: All 6 candidates verified safe-inheritance or scanner false-positive. **No torch_npu changes needed for F7/F8 on torch 2.11 → 2.12-rc3.** This is the value of AST class-scope scanning + call-site grep verification: turns 6 "maybe" candidates into 6 "verified no-op" entries that won't waste future-maintainer time.

### High-risk surfaces to scan first on every torch upgrade

Ranked by torch_npu import-site count (2026-04-24 manual count):

| Private module | torch_npu import sites | Why high risk |
|---|---|---|
| `torch._inductor.utils` | ~29 | Utility grab-bag, symbols constantly moved out |
| `torch._inductor.virtualized` | ~20 | Experimental abstraction, reshuffled often |
| `torch._inductor.codegen.triton` | ~15 | Triton codegen API not stable |
| `torch._dynamo.utils` | ~12 | Private util module, symbols move |
| `torch._inductor.ir` | ~11 | IR node classes renamed |
| `torch._inductor.codegen.common` | ~10 | Codegen base |
| `torch._inductor.codecache` | ~10 | Caching layer rewritten |
| `torch._inductor.scheduler` | ~9 | Scheduler internals churn |
| `torch._inductor.codegen.simd` | ~9 | SIMD codegen — NPU-relevant |
| `torch._dynamo.device_interface` | ~9 | Device backend plugin API |
| `torch._C` (root) | ~9 | pybind layer — watch for C++ ABI |

### Scan procedure for a new torch upgrade

1. For each private module in the table above, grep torch_npu for every
   `from <private> import <SYMBOLS>` statement.
2. Extract the symbol names into a set.
3. Check target torch release: does each symbol still exist at the SAME
   path? Use `git grep -l "^class Symbol\|^def Symbol\|^Symbol = " <tag> -- torch/`.
4. If the symbol file is the same → no drift.
5. If the symbol moved to a different path → **F2-path-move** → add a
   compat entry.
6. If the symbol is gone from the tree → **F1** → file upstream issue.
7. If the symbol's signature changed → **F3**.

Scanner scripts for this workflow are at
`src/skills/torch-npu/port-expert/scripts/`:
- `extract_imports.py` — collect `from torch._<private> import X` pairs
- `check_drift.py` — F1 / F2-path-move detection (at-original / submodule / not-here / mod-gone)
- `check_sig_drift.py` — F3 signature change detection with PEP 604 + additive-default filters
- `check_f7_f8.py` — F7/F8 class-API additions on subclassed parents (AST)
- `sweep.sh` — one-command wrapper chaining all 4 scanners
