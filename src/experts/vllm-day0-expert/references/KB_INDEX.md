# vllm-day0-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL | Phase A |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | OL-03 + OL-08 | Phase A |
| [patterns/domains/vllm-overlay-probe.md](patterns/domains/vllm-overlay-probe.md) | vllm-specific API probe protocol + known drift surfaces | Phase A + B |
| [patterns/domains/overlay-image.md](patterns/domains/overlay-image.md) | Dockerfile.overlay-vllm<MM> template (pip overlay, no build-time import of vllm — runtime only) | Phase C |

## Quick symptoms → classification

| Symptom after pip-overlay target vllm | Likely outcome |
|---|---|
| `vllm.platform_plugins` still finds `ascend` plugin + all probed APIs stable (lora_model, get_tp_group) + consumer source doesn't use changed surfaces | **A** |
| SamplingParams grew new RO property / LLM.generate positional-kwarg shifted AND consumer source uses it | **B** — introspection shim (EC-03 variant) + consumer-side kwarg fix |
| vllm-ascend's installed plugin fails to register with new vllm (e.g. registry API changed) | **C** — vllm-ascend needs NPU-team upstream fix |
| New vllm imports a kernel/symbol from an older dep that's pinned on NPU (e.g. `triton.runtime.jit.constexpr_function` missing) AND consumer path actually uses it | **C** — upstream triton-ascend needs the symbol, or consumer path must skip the new feature |
| Same new kernel import fails but ONLY for model types consumer doesn't use (e.g. gpt_oss MoE when consumer runs Qwen2-0.5B) | **A with informational note** — not a real blocker |

## 2026-04-23 baseline snapshot

- v1 NPU image: vllm 0.13.0+empty / vllm_ascend 0.13.1.dev18
- v2 NPU image: vllm 0.18.0+empty / vllm_ascend 0.17.0rc2.dev109
- community vllm latest: **0.19.1** (2026-04-18)
- vllm-ascend latest release: **0.18.0rc1** (2026-04-01; tracks vllm 0.18.x)

Orchestrator pre-probe of `pip install vllm==0.19.1 --no-deps` overlay on
v2 image (2026-04-23):
- vllm-ascend 0.17 plugin **still registers** against vllm 0.19.1 via
  `vllm.platform_plugins` group — plugin architecture is version-tolerant
- `lora.lora_model.LoRAModel` present; `parallel_state.get_tp_group`
  present
- `SamplingParams` RO properties: `{all_stop_token_ids, bad_words_token_ids,
  eos_token_id}` — same set as 0.18 (no drift)
- **NEW DAY-0 FINDING**: `gpt_oss_triton_kernels_moe.py` in 0.19 imports
  `triton.runtime.jit.constexpr_function` — triton-ascend 3.2.0 doesn't
  export that. BUT only fires for gpt_oss MoE model type; EasyR1's
  Qwen2-0.5B path doesn't hit it. **A with informational note** is the
  expected outcome.

## Related KB (sibling experts)

- `vllm-upgrade-expert/references/patterns/domains/vllm-rename-catalog.md`
  — per-version API ledger (CP-002/CP-004/EC-03). Load for shim patterns
  if outcome B.
- `transformers-day0-expert/references/patterns/domains/overlay-image.md`
  — Dockerfile.overlay template + build-time import trap warning (applies
  here too: don't `import vllm` at Dockerfile build time — it triggers
  torch_npu dlopen)
