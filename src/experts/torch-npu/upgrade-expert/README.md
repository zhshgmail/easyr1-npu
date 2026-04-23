# torch-npu-upgrade-expert

**Product**: single-dep NPU upgrade expert for **the torch / torch_npu
stack**. Given a target torch_npu version (e.g. 2.8 → 2.9 → 2.10) or a
base image shipping it, produce a validated image + port branch that:

- keeps the transformers NPU FA wiring working
- keeps vllm-ascend operational (torch_npu 2.x is often vllm-ascend's
  lower bound)
- re-applies NPU-BUG-001 (triton-ascend partial-install repair) and
  NPU-BUG-004 (upstream-triton amd/nvidia backend prune) per target
- catches NPU-BUG-003 (inductor crash on torch.compile'd log_probs
  paths) regressions
- passes V1.1 + V1.3 + V1.4 smokes against the target-image baseline

This is the **heaviest** of the three upgrade experts because the torch
stack drags along: triton-ascend (different wheel per torch major),
compressed_tensors (bound by vllm+transformers+torch+CANN consensus),
torch.compile codegen path on NPU (NPU-BUG-003 keeps recurring), and
FSDP internals.

## When to use

- `dep-analysis-expert` emits D on `torch`, `torch_npu`, or
  `triton-ascend` — i.e. the image ships a version that doesn't
  satisfy consumer reqs (or doesn't ship one at all).
- User explicitly asks for a torch_npu bump (CVE, bugfix, or new
  NPU feature required).
- A new Ascend/verl base image is released with a higher torch stack
  and the consumer needs to upgrade to it, but transformers / vllm stay
  put (else use `transformers-upgrade-expert`).

## When NOT to use

- Base-image atomic swap where transformers also moves (e.g. v1→v2
  moves transformers 4.57→5.3 AND torch_npu 2.8→2.9). That's
  `transformers-upgrade-expert`'s job — treats them as one atomic unit.
- torch is declared in reqs but `import torch` count is large (it
  always is) — this expert is ONLY about matching torch_npu / torch /
  triton-ascend versions, not about editing torch callsites.
- Pure `torch.cuda` → NPU routing changes. That's `easyr1-expert`'s
  NPU-CP-001 domain.

## Scope / boundaries

**In scope**:
- Pip-freeze diff source vs target, focused on torch-stack rows:
  `torch`, `torch_npu`, `triton`, `triton-ascend`, `compressed_tensors`,
  `torchdata`, `torchvision`.
- **Dockerfile**: most target images ship torch_npu pre-installed; any
  consumer Dockerfile.npu-* must keep the NPU-BUG-001 triton-ascend
  force-reinstall and (if target-image's upstream triton has them) the
  NPU-BUG-004 amd/nvidia backend prune.
- Verify `import torch_npu` cleanly loads inside the target image
  container (the #1 indicator NPU-BUG-001/004 are addressed).
- If torch.compile path regresses: add `use_torch_compile=false` to
  smoke configs (inherits from easyr1-expert canonical config, no
  change needed except possibly re-verifying the knob is still called
  `use_torch_compile`).
- V1.1 device smoke is the primary health check (any NPU-BUG in the
  torch stack fails V1.1 loudly). V1.3 and V1.4 validate downstream
  (vllm-ascend + FSDP training) still compose.

**Out of scope**:
- transformers API renames, vllm API renames — those are sibling
  experts' domains.
- New `torch.cuda.*` callsites in consumer source — that's NPU-CP-001
  (easyr1-expert).
- CANN version pinning in isolation (CANN comes with the base image;
  we consume, we don't drive).

## Inputs

| Var | Meaning |
|---|---|
| `SESSION_TAG` | e.g. `torch-upg-20260501-1100` |
| `SOURCE_IMAGE` | current working image |
| `TARGET_IMAGE` | target image shipping the target torch_npu |
| `TARGET_TORCH_NPU_VERSION` | advisory, e.g. `2.9.0` |
| `UPSTREAM_CONSUMER` | `EasyR1` |
| `UPSTREAM_REF` | baseline-working ref |
| `A3_HOST/PORT/USER/NPU_USER` | standard |

## Deliverable

```json
{
  "image_tag": "<target-image-tag or newly-built>",
  "image_id": "<sha256>",
  "branch": "ascend-torch-upg-<SESSION_TAG>",
  "torch_import_verified_in_container": true,
  "v11_pass": true,
  "v13_pass": true,
  "v14_step1_entropy_loss": <float>,
  "target_image_baseline_band": [<low>, <high>],
  "source_backcompat_verified": true,
  "platform_bugs_addressed": ["NPU-BUG-001", "NPU-BUG-004", ...],
  "dockerfile_adjustments": [...],
  "provenance": {"produced_by": "torch-npu-upgrade-worker"},
  "cleanup": "partial (image preserved for handoff)"
}
```

## How it relates to the broader harness

- Sibling to `transformers-upgrade-expert` and `vllm-upgrade-expert`.
- Distinguishing line: this expert is **dumb about transformers/vllm
  API**; it focuses on the torch-stack's own NPU plumbing (triton-ascend,
  inductor, FSDP, compressed_tensors).
- Pins `_shared/` via `SHARED_VERSION.txt`. Uses the same universal OL
  rules.
- Orchestrator routes `D on torch | torch_npu | triton-ascend` → this
  expert.
