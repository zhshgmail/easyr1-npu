# vllm-upgrade-expert

**Product**: single-dep NPU upgrade expert for vllm / vllm-ascend version
jumps (e.g. 0.13 → 0.18 → 0.20+, or a vllm-ascend branch bump). Given a
vllm/vllm-ascend target version or a base image that ships it, produce a
validated image + branch with consumer-side shims applied, V1.3 rollout
smoke PASS + V1.4 training smoke in-band.

## When to use

- `dep-analysis-expert` emits D or "version conflict" on `vllm` or
  `vllm-ascend`.
- User explicitly asks for a vllm bump (e.g. to pick up a CVE fix, a new
  model family supported only in vllm ≥ X).
- A new vllm-ascend release is available and the consumer needs it
  (typically when vllm upstream cut a new minor and vllm-ascend has
  published a matching version).

## When NOT to use

- The bump is coming as part of a **base-image swap** (e.g. v1→v2 moves
  vllm 0.13→0.18 along with transformers 4.57→5.3 along with CANN
  8.5.0→8.5.1). Use `transformers-upgrade-expert` — it treats the whole
  stack swap as one atomic upgrade.
- vllm is declared in consumer reqs but has **0 imports** in source
  (verify with `git grep 'import vllm\|from vllm' <ref>`). Then the bump
  is cosmetic and `dep-analysis` will classify accelerator-agnostic;
  no expert needed.

## Scope / boundaries

**In scope**:
- Pip-freeze diff source vs target image (focus on `vllm`, `vllm-ascend`,
  `compressed_tensors`, `xformers`, plus any transitively-changed vllm
  subpackages).
- Consumer-side shims for known rename/move classes:
  - `vllm.lora.models → vllm.lora.lora_model` (NPU-CP-002, at least
    vllm_utils.py:20-23 per dd71bbd)
  - `get_tensor_model_parallel_group → get_tp_group` (NPU-CP-004,
    sharding_manager/fsdp_vllm.py)
  - `SamplingParams.eos_token_id / stop_token_ids / output_kind`
    read-only property in vllm ≥ 0.18 (EC-03, vllm_rollout_spmd.py)
- New shim candidates surfaced by the target version's release notes
  (if any).
- If the target image doesn't already have vllm pre-installed (unusual):
  Dockerfile layer that installs the specific `vllm-ascend==<version>`
  wheel. Normally the base image ships it — confirm first.
- Consumer-side hijack points: `VLLMHijack` in `verl/utils/vllm_utils.py`
  (hijacks LoRA adapter load). Verify the hijack target method still
  exists on the target vllm; update signature if it moved.
- V1.3 rollout smoke (stresses vllm-ascend) MUST PASS; V1.4 must reach
  step-1 entropy_loss in target image's V1.4 band.

**Out of scope**:
- Consumer source porting beyond the 3-4 vllm-adjacent files listed
  above (that's `easyr1-expert`'s domain).
- vllm upstream development itself — we consume pre-built `vllm-ascend`
  wheels from the base image or pypi, never build vllm from source.
- Multi-dep bundled upgrade in one session (use `transformers-upgrade`
  for base-image swaps that move vllm+transformers+torch_npu together).

## Inputs

| Var | Meaning |
|---|---|
| `SESSION_TAG` | e.g. `vllm-upg-20260501-0900` |
| `SOURCE_IMAGE` | current working image shipping the current vllm |
| `TARGET_IMAGE` | candidate image shipping the target vllm |
| `TARGET_VLLM_VERSION` | canonical version tag (e.g. `0.18.0`), advisory — actual version comes from target image pip-freeze |
| `UPSTREAM_CONSUMER` | e.g. `EasyR1` |
| `UPSTREAM_REF` | baseline-working consumer ref |
| `A3_HOST`/`A3_PORT`/`A3_USER`/`NPU_USER` | standard |

## Deliverable

```json
{
  "image_tag": "easyr1-npu:<SESSION_TAG>",
  "image_id": "<sha256>",
  "branch": "ascend-vllm-upg-<SESSION_TAG>",
  "v13_rollout_pass": true,
  "v14_step1_entropy_loss": <float>,
  "target_image_baseline_band": [<low>, <high>],
  "source_backcompat_verified": true,
  "shims_applied": ["NPU-CP-002", "NPU-CP-004", "EC-03", ...],
  "hijack_points_verified": ["VLLMHijack._load_adapter"],
  "provenance": {"produced_by": "vllm-upgrade-worker"},
  "cleanup": "partial (image preserved for handoff)"
}
```

## How it relates to the broader harness

- Pins `_shared/` via `SHARED_VERSION.txt`. References
  `_shared/references/ALWAYS_LOADED_UNIVERSAL.md`.
- Sibling to `transformers-upgrade-expert`. Distinguishing lines:
  - `transformers-upgrade-expert` = v1→v2 atomic stack swap (transformers
    + vllm + torch_npu + CANN all at once). Used when the base image
    swap is the driver.
  - `vllm-upgrade-expert` = vllm-only bump within the same base image
    family (e.g. vllm-ascend 0.13 → 0.14 on CANN 8.5.0). Used when
    transformers / torch_npu / CANN stay put.
- Orchestrator's `task_plan` routes `D on vllm` → this expert.
