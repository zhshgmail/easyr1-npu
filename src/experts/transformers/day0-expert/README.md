# transformers-day0-expert

**Product**: the **real-problem** upgrade expert for transformers.
Invoked when upstream community releases a new transformers version
that the NPU ecosystem hasn't caught up on (target `TRANSFORMERS_VERSION`
not shipped in any existing NPU base image). Produces one of:

1. **A** — "It already works": pip-install the new version on top of
   existing NPU image + verify V1.1/V1.3/V1.4 still pass. Emit a
   validated image tag + fresh baseline numerics.
2. **B** — "Forward-port needed, feasible": cherry-pick NPU adaptation
   (npu_flash_attention signatures / ALL_ATTENTION_FUNCTIONS wiring /
   any new modeling_utils touch) to the new transformers tip; build
   wheel; verify.
3. **C** — "Blocked": precise diagnosis of which NPU ecosystem piece
   needs upstream help (specific torch_npu kernel / vllm_ascend version /
   transformers NPU FA hook). No forward-port in this session.

This is the **hardest** of the three day-0 experts (per user 2026-04-23):
transformers iterates fastest in community, API drift is large, and NPU
FA integration surface is wide (ALL_ATTENTION_FUNCTIONS, device_map,
modeling_utils private hooks).

## Distinguishing vs transformers-upgrade-expert (shim-adapt)

| Question | transformers-upgrade-expert (Stage 2) | transformers-day0-expert (Stage 3) |
|---|---|---|
| Does NPU image exist for target? | **Assumed yes** (consumes pre-built) | **Probes; often no** |
| Work output | shim branch + reuse image | patched image (via pip overlay or full rebuild) + possibly wheel patch |
| Typical scope | EC-02/03 rename handling | diagnose NPU ecosystem gap + decide fwd-port vs wait |
| Evidence chain | historical drill commits | probing target transformers AGAINST existing NPU bindings, measuring drift |

If dep-analysis finds a target transformers version is already shipped
in a known NPU image → route to transformers-upgrade-expert. If it's
**not** shipped anywhere → route here.

## Reality check — what Day-0 actually looks like (2026-04-23 snapshot)

Baseline established by probing real installed / published state:

| Who | transformers version | date / note |
|---|---|---|
| v1 NPU image | 4.57.6 | stable, known-good EasyR1 baseline (step-1 0.991) |
| v2 NPU image | **5.3.0.dev0** | NPU ecosystem front (drill 2026-04-19) |
| community 5.4 | 5.4.0 | 2026-03-27 |
| community 5.5 | 5.5.0 | 2026-04-02 |
| community **5.6.0** | 5.6.0 | **2026-04-22 — yesterday** |

NPU is **3 minor behind**. This is the real Day-0 trigger.

Probe results from `pip install transformers==5.6.0` overlay onto v2 image:

- Install succeeds (pip complains about consumer pins but still installs)
- `transformers.integrations.npu_flash_attention.npu_flash_attn_func` sig **identical**
  to 5.3 version: `(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs)`
- `npu_flash_attn_varlen_func` sig also identical
- `ALL_ATTENTION_FUNCTIONS` registry **expanded**: 5.3 had `flash_attention_2`,
  `sdpa`, `flex_attention`. 5.6 adds `flash_attention_3`, `flash_attention_4`,
  plus `paged|*` variants. **This is the new risk surface** — if a 5.6-era
  model defaults to `flash_attention_3` / `paged|*` and NPU adapter hasn't
  registered handlers for those keys, it silently falls through to a
  non-NPU path.
- vllm 0.18 (shipped in v2 image) **hard-pins** `transformers<5,>=4.56.0` —
  pip warns but installs. Runtime impact unknown until smoke. Possible
  vllm blocker at LLM init.
- Consumer (verl 0.3.3) pins `transformers<5.0.0` — requires a fixture
  commit to loosen the bound before testing; legitimate fixture per
  orchestrator-fixture convention.

## Scope / boundaries

**In scope**:
- Probe new community transformers version against existing NPU image
  (pip overlay, not rebuild from scratch — fast path)
- Identify drift: API rename, new ALL_ATTENTION_FUNCTIONS entries, sig
  changes to npu_flash_attention, modeling_utils private-hook moves
- Run V1.1/V1.3/V1.4 smokes with the overlaid version
- Emit ONE of: validated-works baseline, forward-port branch + wheel
  patch, blocker diagnosis

**Out of scope**:
- torch_npu / vllm_ascend upgrade (sibling day0 experts' domain)
- EasyR1 consumer source ports beyond transformers-adjacent backcompat
- Training any model beyond V1.4 canonical smoke (Qwen2-0.5B)
- Multi-version bump chain (one version at a time; chain via
  orchestrator task_plan)

## Inputs

| Var | Meaning |
|---|---|
| `SESSION_TAG` | e.g. `trans-day0-20260501-0830` |
| `TARGET_TRANSFORMERS_VERSION` | e.g. `5.6.0` — the community release under test |
| `BASE_IMAGE` | current NPU image to overlay on (default: v2 image) |
| `UPSTREAM_CONSUMER` | `EasyR1` |
| `UPSTREAM_REF` | baseline ref (usually a fixture branch that loosens the consumer's transformers pin) |
| `A3_HOST`/... | standard |

## Deliverable

```json
{
  "session_tag": "trans-day0-...",
  "target_transformers_version": "5.6.0",
  "outcome": "A_works_as_is | B_forward_port | C_blocked",
  "base_image": "<v2 image or newer>",
  "overlay_image_tag": "<new image built via pip overlay>",
  "api_drift_findings": [
    {"surface": "ALL_ATTENTION_FUNCTIONS", "delta": "5.6 adds flash_attention_3/4 + paged|*", "risk": "if model picks these, NPU FA adapter unrouted"},
    ...
  ],
  "smoke_results": {
    "V1.1": {"status": "PASS|FAIL", "..."},
    "V1.3": {...},
    "V1.4": {"status": "...", "entropy_loss": <float>, "baseline_band": "fresh|[1.21,1.34]|other"}
  },
  "blocker_diagnosis_if_C": "<precise which NPU piece is missing>",
  "provenance": {"produced_by": "transformers-day0-worker"},
  "cleanup": "partial (overlay image preserved for handoff)"
}
```

Note: for a Day-0 outcome A/B, V1.4 baseline_band may be "fresh" — i.e.
this is the first measurement at that transformers version, so the
assertion is "training runs 2 steps without crashing" not "entropy_loss
in known band". Post-run, the number is recorded as the new baseline
for that version.

## How it relates

Sibling to Stage 2 `transformers-upgrade-expert` (shim-adapt). Builds on
`_shared/` scaffolding. Feeds into the orchestrator via task_plan
scenario P2b (needs_day0).
