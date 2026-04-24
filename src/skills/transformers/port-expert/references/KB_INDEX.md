# transformers-day0-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL rules | Phase A |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | OL-03 + OL-08 | Phase A |
| [patterns/domains/api-drift-scan.md](patterns/domains/api-drift-scan.md) | How to measure transformers minor→minor drift impact on NPU adapter | Phase A + B |
| [patterns/domains/overlay-image.md](patterns/domains/overlay-image.md) | Dockerfile.overlay template (FROM base + pip install target + optional patch COPY) | Phase C |

## Quick symptoms → classification

| Symptom | Likely outcome |
|---|---|
| `pip install transformers==<target>` succeeds, npu_flash_attention sigs unchanged, ALL_ATTENTION_FUNCTIONS only expanded (not reshaped), consumer source grep shows no new symbols → **outcome A** |
| Same as A but `npu_flash_attention.npu_flash_attn_func` sig gained a kwarg (e.g. new `window_size` param) and consumer calls that slot → **outcome B**, 1-line wrapper |
| New ALL_ATTENTION_FUNCTIONS key becomes default for a model consumer uses AND NPU FA adapter has no handler → **outcome B**, add handler |
| vllm hard-pin `transformers<X` hits runtime (not just pip warning) → **outcome C** unless vllm-day0-expert runs first |
| npu_flash_attention module removed/renamed on target → **outcome C**, NPU team must republish |

## Relationship to torch_npu drift scanning (2026-04-24)

transformers' NPU integration surface is **much smaller than torch_npu's**:

| Upstream | NPU-touching files | Private torch API imports | Drift workload |
|---|---|---|---|
| torch_npu | 60+ | ~437 `from torch._* import` pairs | Heavy — run `scripts/extract_imports.py` + `scripts/check_drift.py` + `scripts/check_sig_drift.py` per torch minor bump |
| transformers | 1 (`src/transformers/integrations/npu_flash_attention.py`, 143 lines) | 1 (`from torch_npu import npu_fusion_attention`) | Light — manual grep is sufficient |

So there is **no need** for a transformers analog of
`src/skills/torch-npu/port-expert/scripts/check_drift.py`. When probing
a new transformers release:

1. Check that `npu_flash_attention.py` still has its current shape
   (one torch_npu import + one or two functions called against it).
2. Check that the upstream transformers attention-dispatcher still
   calls into that module (grep for `ALL_ATTENTION_FUNCTIONS` entries
   referencing `npu_flash_attention.*`).
3. If either check fails → that's the drift signal to work on, not
   a private-API drift. Outcome will be B (add handler) or C
   (module removed upstream, escalate to NPU team).

If torch_npu's `npu_fusion_attention` ITSELF changes signature, that is
a torch_npu F3 drift, not a transformers drift — route to
`src/skills/torch-npu/port-expert/`.

## Reference: today's baseline snapshot (2026-04-23)

- v1 NPU image: transformers 4.57.6
- v2 NPU image: transformers 5.3.0.dev0
- community latest: **transformers 5.6.0** (released 2026-04-22)

Gap: 3 minor (NPU 5.3.dev → community 5.6). First probe result (docker
run pip install 5.6 into v2):
- npu_flash_attention sig unchanged ✓
- ALL_ATTENTION_FUNCTIONS added flash_attention_3/4, paged|* variants
- vllm 0.18 pins transformers<5; pip warns but installs
- Consumer (verl) pins transformers<5; fixture-loosen required

Expected outcome: probably A (if none of EasyR1's touched models default
to `flash_attention_3` / `paged|*`), else B (add NPU FA handler for the
new key).

## Fixture protocol for day-0 probe

Day-0 probes often need a fixture branch on the consumer repo that:
- loosens the transformers upper pin (e.g. `transformers>=4.54.0,<6.0.0`)
- leaves the rest of reqs intact

Fixture commit provenance: `orchestrator-fixture` (same as
Day-0 post-mortem pattern from 2026-04-22 E2E).

## Outcome matrix (aligned with vllm-day0 / torch-day0 / vllm-ascend-day0)

| Outcome | Meaning | Action |
|---|---|---|
| **A** | pip overlay + probes all clean + consumer source unchanged → smoke PASS | Ship overlay + note |
| **A-with-note** | PASS but something observable changed (e.g. ALL_ATTENTION_FUNCTIONS grew but we don't use new default) | Ship overlay + ONBOARDING known-broken section |
| **B** | Consumer-side shim (1-line sig wrapper in `npu_flash_attention` or consumer rollout script) | Commit shim on fixture + smoke PASS |
| **C-patch** | Fix belongs in `upstream/transformers/src/transformers/integrations/npu_*` (Huawei-owned NPU integration files within the transformers repo) | Branch `ascend-day0-transformers-<SESSION>` on transformers fork + PR material |
| **C-report** | Fix needed in community transformers (non-NPU paths) | blocker report, we don't patch community code |

## Related KB (sibling experts — 3-layer Day-0 chain)

- `torch-npu/port-expert/` — when transformers 5.x depends on a torch version
  the NPU ecosystem hasn't shipped, chain torch-day0 first
- `vllm/port-expert/` — when transformers bump combines with new vllm
  (check dep-analysis routing)
- `vllm-ascend/port-expert/` — if transformers bump triggers vllm-ascend
  C++ extension rebuild need, route there
- `../../_shared/references/patterns/domains/day0-deploy-artifacts.md`
  — Phase E 5-artifact template

## 2026-04-23 validated result

Session `transformers-day0-trans-day0-wetrun-20260423-0109`:
- target: transformers 5.6.0 (community 2026-04-22)
- overlay image: `easyr1-npu-trans56:trans-day0-wetrun-20260423-0109`
- V1.1/V1.3/V1.4 all PASS; step-1 entropy_loss 1.310, in v2 baseline
  band [1.21, 1.34]
- outcome: A (no patch needed — all new ALL_ATTENTION_FUNCTIONS keys
  expanded without touching the defaults our models use)
