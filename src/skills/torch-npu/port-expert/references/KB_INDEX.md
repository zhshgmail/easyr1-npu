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

| Family | torch range | Symbol | Old path | New path | torch_npu sites | Fix landed |
|---|---|---|---|---|---|---|
| F2-path-move | 2.11 → 2.12-rc3 | `FloorDiv` | `torch._inductor.utils` | `torch.utils._sympy.functions` | 2 files | local commit `2d81f06c8` on branch `torch-2.12_auto_porting` |
| F2-path-move | 2.11 → 2.12-rc3 | `ModularIndexing` | `torch._inductor.utils` | `torch.utils._sympy.functions` | 3 files | (same commit) |

Affected files:
- `torch_npu/_inductor/lowering_fx.py:38`
- `torch_npu/_inductor/codegen/split_tiling.py:7`
- `torch_npu/_inductor/codegen/scheduling.py:27`

Fix compat module: `torch_npu/compat/sympy_functions.py`.

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

A scanner script for this (analog of `kb_drive_test.py` for torch) is
TBD — see the torch-npu port-expert `SKILL.md` for the manual workflow
in the meantime.
