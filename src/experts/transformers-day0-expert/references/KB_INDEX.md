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
