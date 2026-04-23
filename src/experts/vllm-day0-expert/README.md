# vllm-day0-expert

**Product**: Day-0 NPU probe for a community vllm release whose **matching
vllm-ascend build has not been published yet**. Overlay target vllm onto
existing NPU image (ships some older vllm-ascend), verify the mismatched
combination actually works at runtime, and emit one of:

1. **A — works-as-is**: existing vllm-ascend loads the new vllm, V1.3
   rollout + V1.4 training succeed. Emit overlay image + fresh baseline.
2. **B — forward-port feasible**: ship a patch to vllm-ascend's small
   surface (imports, API-rename shims). Rebuild wheel or overlay with
   patched source. Smoke to verify.
3. **C — blocked**: identify the specific kernel / attention backend /
   registry API the new vllm requires that vllm-ascend hasn't caught up
   on. Precise upstream-to-file-the-gap report.

## When to use

`dep-analysis-expert` probes the consumer's target vllm version against
`knowledge/images/*.md` pip-freezes AND against published `vllm-ascend`
releases. If the target vllm has **no matching vllm-ascend release yet**
(e.g. community vllm 0.19.1 released 2026-04-18 with vllm-ascend latest
at 0.18.0rc1), route here.

## When NOT to use

- vllm-ascend matching release already exists → `vllm-upgrade-expert`
  (Stage 2 shim-adapt; trivial API-rename only).
- target vllm moves alongside transformers / torch_npu (whole stack
  swap) → `transformers-day0-expert` or chain-of-day0.
- vllm is declared in consumer reqs but unused in source → dep-analysis
  classifies as cosmetic, no expert needed.

## Ground truth snapshot (2026-04-23)

| Who | vllm | vllm-ascend |
|---|---|---|
| v1 NPU image | 0.13.0+empty | 0.13.1.dev18 |
| v2 NPU image | 0.18.0+empty | 0.17.0rc2.dev109 |
| community vllm latest | **0.19.1** (2026-04-18) | — |
| vllm-ascend latest | — | **0.18.0rc1** (2026-04-01) |

**Gap**: community vllm 0.19.1 has no matching vllm-ascend. 2 minor
(0.17 → 0.19.1) between what v2 image ships and what community has.

Orchestrator pre-probe (2026-04-23) of pip overlay `vllm 0.19.1` into
v2 image:
- Import chain clean (vllm 0.19.1 imports; plugin group `vllm.platform_plugins`
  still finds `ascend` plugin from vllm-ascend 0.17)
- `vllm.lora.lora_model.LoRAModel` still accessible (CP-002 new arm
  continues)
- `vllm.distributed.parallel_state.get_tp_group` still present (CP-004
  baseline)
- SamplingParams: the 0.18 read-only property set `{eos_token_id,
  stop_token_ids, output_kind}` has grown to include at least
  `all_stop_token_ids, bad_words_token_ids` in 0.18 (probed on v2) — 0.19
  may add more; introspection-variant shim (EC-03) handles it without
  code change
- `SamplingParams()` constructor: API drift — positional args changed
  semantics between versions; any code calling `SamplingParams("")` will
  break. Consumer (EasyR1) uses kwargs so likely unaffected. Needs grep
  verification.

Expected outcome: likely **A or B**. Real value of this skill is the
verification + blocker report quality, not guessing.

## Scope / boundaries

**In scope**:
- pip overlay target vllm onto an existing NPU image
- Measure API drift focused on:
  - `vllm.lora.lora_model.LoRAModel` (or `vllm.lora.models`)
  - `vllm.distributed.parallel_state` group (TP fns, PP fns)
  - `SamplingParams` read-only property set + constructor signature
  - `vllm.LLM` + `LLM.generate` signature (prompts= / inputs= /
    prompt_token_ids= positional-vs-kwarg)
  - `VLLMHijack` target methods (LoRA worker manager,
    get_adapter_absolute_path, PEFTHelper)
  - `vllm.platform_plugins` group (does ascend plugin still register?)
- Consumer-side **shim application** for surfaced drift (same 3 files
  as vllm-upgrade-expert: vllm_utils.py, vllm_rollout_spmd.py,
  fsdp_vllm.py)
- V1.3 (primary vllm exercise) + V1.4 (secondary, catches weight-sync)
  smoke validation

**Out of scope**:
- Editing vllm-ascend source (a forward-port of vllm-ascend itself
  belongs to NPU team; if needed, this expert's outcome C lists what to
  patch there)
- transformers / torch_npu version changes (sibling day0 experts)

## Inputs

| Var | Meaning |
|---|---|
| `SESSION_TAG` | e.g. `vllm-day0-20260501-0930` |
| `TARGET_VLLM_VERSION` | e.g. `0.19.1` |
| `BASE_IMAGE` | NPU image to overlay on (default: v2) |
| `UPSTREAM_CONSUMER` / `UPSTREAM_REF` | standard |

## Deliverable

```json
{
  "session_tag": "vllm-day0-...",
  "target_vllm_version": "0.19.1",
  "outcome": "A|B|C",
  "base_image": "<v2 tag>",
  "overlay_image_tag": "easyr1-npu-vllm<MM>:<SESSION_TAG>",
  "api_drift_findings": [...],
  "smoke_results": {"V1.3": {...}, "V1.4": {...}},
  "blocker_diagnosis_if_C": null,
  "provenance": {"produced_by": "vllm-day0-worker"},
  "cleanup": "partial (overlay image preserved)"
}
```
