# torch-day0-expert

**Product**: Day-0 NPU probe for a community PyTorch release whose
**matching stable `torch_npu` has not been published yet**. Pip-overlay
target torch + torch_npu rc wheel onto an existing NPU base image
(preserves CANN from base), validate import + device visibility + basic
NPU tensor ops, emit one of:

1. **A — works-as-is**: new torch + rc torch_npu + base image's CANN
   all cooperate, 6/6 runtime smoke PASS. Ship overlay + Phase 2.5
   deploy artifacts so downstream layers (vllm-ascend, triton-ascend,
   RL framework users) can port on top.
2. **A-with-note**: smoke PASS but dep tree has a known gap not covered
   by the smoke (e.g. a niche kernel path). Ship with ONBOARDING noting
   the caveat.
3. **B — forward-port feasible**: smoke fails but the fix is
   consumer-side (pin loosen, requirements.txt edit).
4. **C-patch — fix in Ascend/pytorch or triton-ascend**: smoke fails in
   a way that needs torch_npu / Huawei-owned integration code change.
   Apply minimal patch on `ascend-day0-torch<M><m>-<SESSION>` branch,
   rebuild overlay, re-smoke to PASS. Produce PR-ready diff.
5. **C-report — fix must ship in community PyTorch**: blocker-report
   with reproducer + suggested fix, session ends at the report
   (we don't patch community upstream).

## When to use

`dep-analysis-expert` probes the consumer's target torch version
against `knowledge/images/*.md` pip-freezes AND against published
`torch_npu` stable releases + CANN bundle matrix. If the target torch
has **no stable torch_npu release yet** (only a PyPI rc or a source
tag on Ascend/pytorch), route here.

## When NOT to use

- torch + torch_npu stable pair already shipped in an NPU image →
  `torch-npu-upgrade-expert` (Stage 2 shim-adapt; in-image combo).
- torch bumps alongside transformers / vllm as a whole-stack swap →
  chain the right Day-0 expert (usually transformers-day0 is first; see
  user's "depth-first deploy chain" memory).
- torch declared in consumer reqs but unused → cosmetic, no expert
  needed.

## Ground truth snapshot (2026-04-23)

| Who | torch | torch_npu | CANN |
|---|---|---|---|
| v1 NPU image | 2.7.x | 2.7.x | 8.4 |
| v2 NPU image | 2.9.0+cpu | 2.9.0 | 8.5.1 |
| community PyTorch latest | **v2.11.0** (2026-04) | — | — |
| torch_npu latest stable | — | **2.9.0** (Jan 2026) | — |
| torch_npu Day-0 rc on PyPI | — | **2.11.0rc1** (2026-03-24) | — |
| CANN bundle matrix | | | 26.0.0-beta.1 pairs only with pytorch 2.10.0 |

**Gap**: community PyTorch 2.11 has only a pre-release torch_npu
(2.11.0rc1) and no CANN bundle yet. Legit Day-0 target.

Orchestrator pre-probe (2026-04-23) of `pip install torch==2.11.0+cpu
+ torch_npu==2.11.0rc1` overlay on v2 image (CANN 8.5.1):

- native_functions.yaml delta: +8 ops, 6 CUDA/ROCm-only (safe), 2
  CompositeExplicitAutograd (cover NPU automatically). No new native
  kernel needed.
- DispatchKey.h: 1-line `noexcept` addition, no PrivateUse1 change
- CANN 8.5.1 one patch-level ahead of README-paired 8.5.0 (same delta
  as 2.9.0)
- `torch.npu.Stream.native_handle` present — no shim needed

Expected outcome: **A** (torch layer works); downstream C++ extensions
built for torch 2.9 will need their own C-patch (see vllm-ascend-day0
session 2026-04-23-0655 for an instance).

## Scope / boundaries

**In scope**:
- pip overlay target torch + torch_npu onto an NPU base image
- API-drift classification on torch N → N+1 (native_functions.yaml,
  DispatchKey.h, distributed/, torch/_ops.py)
- torch_npu rc coverage vs community torch gap
- CANN version window check (rc's README pairing vs image's CANN)
- Runtime 6-step smoke (metadata / imports / device / NPU op / API presence)
- C-patch on `upstream/torch-npu/` when fix belongs there (Huawei-owned)
- Phase 2.5 deploy artifacts (Dockerfile, smoke, deploy script,
  ONBOARDING, PR material if C-patch)

**Out of scope**:
- Editing community PyTorch source (community maintainer territory —
  C-report)
- Downstream C++ extensions breaking under new torch ABI (vllm-ascend,
  third-party — their own Day-0 expert handles that)
- CANN driver changes (outside this skill's edit scope; file against
  CANN team)

## Inputs

| Var | Meaning |
|---|---|
| `SESSION_TAG` | e.g. `torch-day0-20260501-0930` |
| `TARGET_TORCH_VERSION` | e.g. `2.11.0` |
| `TARGET_TORCH_NPU_VERSION` | e.g. `2.11.0rc1` |
| `BASE_IMAGE` | NPU image to overlay on (default: v2) |

## Deliverable

```json
{
  "session_tag": "torch-day0-...",
  "target_torch_version": "2.11.0",
  "target_torch_npu_version": "2.11.0rc1",
  "outcome": "A|A-with-note|B|C-patch|C-report",
  "base_image": "<v2 tag>",
  "overlay_image_tag": "easyr1-npu-torch<MM>:<SESSION_TAG>",
  "cann_version_used": "8.5.1",
  "smoke_results": {"runtime_6_step": "PASS", "api_presence": [...]},
  "api_drift_findings": [...],
  "patched_branch_if_C_patch": "ascend-day0-torch<MM>-<SESSION_TAG>",
  "blocker_diagnosis_if_C_report": null,
  "deploy_artifacts_dir": "workspace/torch-day0-deploy-<SESSION_TAG>/",
  "provenance": {"produced_by": "torch-day0-worker"},
  "cleanup": "partial (overlay image preserved; validation image deleted)"
}
```

## Downstream layer handoff

After A / A-with-note / C-patch outcome, the deploy artifacts ONBOARDING
enables downstream experts (`vllm-ascend-day0-expert`,
`transformers-day0-expert`, user RL framework) to start their own
Day-0 port on top of the deployed torch layer, per the
[upstream deploy chain pattern](../../memory/day0_upstream_deploy_chain.md).

See `_shared/references/patterns/domains/day0-deploy-artifacts.md` for
the 5-artifact template (Dockerfile, smoke, deploy, ONBOARDING, PR
material).
