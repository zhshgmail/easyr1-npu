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

- `vllm-ascend/day0-expert/` (next layer) — when torch 2.11 overlay is
  deployed, vllm-ascend on top needs Fix B+ ABI guard. See
  `workspace/vllm-ascend-day0-*/analysis.md`.
- `vllm/day0-expert/` — Day-0 vllm on NPU when vllm-ascend is already
  compatible. Usually runs AFTER torch-day0 has shipped the torch base.
- `transformers/day0-expert/` — transformers Day-0 runs independently
  of torch-day0 usually (transformers doesn't pin torch ≥ X hard unless
  it uses a new torch API).
- `../../_shared/references/patterns/domains/day0-deploy-artifacts.md`
  — mandatory Phase E deliverables template.

## Concrete artifacts from 2026-04-23 session

- Analysis: `workspace/torch-day0-analysis-20260423-0531/analysis.md`
- Manual port log: `workspace/torch-day0-manual-20260423-0537/PROGRESS.md`
- Deploy artifacts: `workspace/torch-day0-deploy-20260423-0548/` (Dockerfile, smoke, deploy script, ONBOARDING, blocker-report)
- Overlay image on A3: `easyr1-npu-torch211:torch-day0-manual-20260423-0537`
