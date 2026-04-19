# Codex proxy sign-off — v2 milestone

Date: 2026-04-19.
Context: v2 scope was enabling `padding_free=True` on NPU via `transformers.integrations.npu_flash_attention` (per the veRL reference). V2.1 smoke passed on A3 with numerical equivalence to V1.4.

---

## VERDICT: APPROVE WITH FOLLOW-UPS /s codex sign-off 2026-04-19

## RATIONALE

v2 clears the stated functional bar, but not the "standalone with no material debt" bar required for plain APPROVE. The shipped claim is narrow and substantiated: `padding_free=True` on NPU now runs through the `transformers.integrations.npu_flash_attention` varlen path, `V2.1` is a real branch state on `ascend-port` (`fbaa983` + `75bad74` + `9e971f0`), and the recorded result is strong enough to count as milestone evidence: 2/2 GRPO steps on 2 chips, checkpoints on both ranks, `reward_score: 0.016` matching prior smokes, and `entropy_loss: 0.991 → 1.264` numerically aligned with V1.4. The documentation side also passes: `NPU-CP-007` and `NPU-BUG-003` use the same 5-field schema as the existing catalog, the V1.6→V2.1 rename is reflected in branch history and in the new smoke-ladder convention doc, and the compile workaround is stated plainly in both `DELIVERABLE.md` and `npu-patterns.md`. Not BLOCK because nothing here contradicts the v2 promise. Not plain APPROVE because the shipped v2 still depends on a known platform workaround (`use_torch_compile=false`) and stops short of the logically next validation (`V2.2`).

## FOLLOW-UP LIST

- `MAJOR / M` — Treat `NPU-BUG-003` as owner-TBD stabilization debt: investigate triton-ascend inductor failure on `log_probs_from_logits` under varlen shapes, and determine whether v2 should keep compile permanently disabled on NPU or gain a narrower guard.
- `MEDIUM / S` — Run `V2.2` exactly as planned: 4-chip + ulysses-enabled validation for the padding-free path. Not required to sign off V2.1; it is the first missing smoke above the shipped envelope.
- `MINOR / S` — Clean up residual v1-era wording in `docs/DELIVERABLE.md` so sections describing the attention backend and deferred v2 work no longer preserve pre-v2 phrasing. **Addressed in this sign-off commit.**

## RESIDUAL RISKS

Week-2 failures are specific to the new varlen path, not the base v1 port. The main one is shape sensitivity: `padding_free=True` changes tensor geometry, and that already exposed `NPU-BUG-003`; similar shape- or mask-dependent failures may surface first under longer prompts, higher length variance, or 4-chip/ulysses topologies even if V2.1 is stable. Second risk is semantic: the NPU FA shim is now the adapter layer, so any upstream change in `transformers.integrations.npu_flash_attention`, causal-mask convention, or varlen argument handling could regress the path without touching EasyR1 code. Third risk is coverage: V2.1 proves 2-chip GRPO on this recipe, but not yet the multi-chip ulysses case v2 is aiming toward.

## DELTA VS V1 SIGN-OFF

Major change since 2026-04-18 v1 sign-off: `padding_free=True` has moved from explicit out-of-scope limitation to demonstrated functionality on NPU, documented across deliverable, journal, and pattern catalog. That materially narrows the earlier recommendation set: "implement NPU varlen attention" is no longer a recommendation because it has been done via the transformers shim pattern (`NPU-CP-007`). In its place, the primary recommendation shifts to stabilization of the compile interaction (`NPU-BUG-003`) and extension of coverage to the planned `V2.2` smoke. The catalog/convention hygiene concerns from v1 are also in better shape now: the stable-ID schema remains uniform, and the milestone-vs-level naming confusion has been corrected with a reusable convention doc (`knowledge/smoke-ladder-convention.md`).
