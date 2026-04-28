---
id: cross-layer-007
date: 2026-04-27
layer: cross-layer
title: agent walk-through reports do not validate end-to-end; only on-A3 import + run does
trigger:
  - "agent reports 5/5 walk-through PASS"
  - "cold-drive validates documentation coherence"
  - "agent confirms hypothetical scenario"
  - "without actually running the smoke / shim / kernel"
symptom_in_wild:
  - "session claims '4/4 skills cold-drive PASS' based on agent walk-through reports"
  - "later, real on-A3 smoke finds drift / crash / extra additive change that walk-through missed"
  - "PR_MATERIAL.md says 'validated' when only the docs were proofread, not the artifact"
root_cause: >
  An agent reading the docs and saying "this is what I would do, and I
  expect this to work" is verifying the **plan**, not the **artifact**.
  Real on-A3 import + run can find:
    - additional drift the static byte-compare missed (placeholder
      function added, new ALL_ATTENTION_FUNCTIONS keys)
    - shim plugin-init order issues that only show up in vllm's actual
      startup
    - base-image version-axis mismatches (target version pre-dates
      symbol introduction)
  None of those are visible to a walk-through.
mistake_pattern: "treating-walk-through-PASS-as-validation"
correction:
  - "Whenever an agent's report says 'walk-through PASS' or 'documentation coherent', mentally tag it as 'docs-only verification' and DO NOT treat it as 'end-to-end validated'."
  - "Before claiming a skill end-to-end PASS, run a real on-A3 smoke that imports the affected modules and runs at least one operational call (model.forward / kernel launch / shim.OLD-path-resolve)."
  - "When writing PR_MATERIAL.md or status-doc entries, distinguish 'validated by agent walk-through (docs only)' from 'validated by on-A3 import + run'. Different evidence weights."
  - "If an agent tells you a previously-claimed PASS reproduced cleanly, ask 'did you import + run, or did you read the doc and explain what you would do'. The latter is not a re-validation."
evidence:
  - "2026-04-27 T21 found vllm-ascend shim 3 fails to import on real on-A3 plugin-init phase, despite earlier 13/13 walk-through cold-drive that claimed PASS."
  - "2026-04-27 T21 found transformers v5.4 actually has 2 additive drifts vs v5.3 (npu_flash_attn_with_kvcache placeholder + flash_attention_4 keys) that the earlier outcome-A walk-through missed."
  - "User Discord 2026-04-27 series: insistent that real on-A3 smoke must precede claiming skill PASS, before allowing T22 RL integration. Caught me twice in this session inflating walk-through results into PASS claims."
mistake_pattern_relationship:
  - "Closely related to cross-layer-001 (pip-install-is-not-port) — same family of inflated-evidence mistakes."
  - "Companion to cross-layer-005 (no-conclusion-without-investigation) — both flag premature confidence."
---

# Why this matters

Walk-through agent reports are useful for one thing: **validating that
the documentation is internally coherent and that a fresh reader can
follow the recipe**. They do NOT validate that the recipe actually
works on hardware.

The 2026-04-27 cycle (T20 walk-through → T21 real run) demonstrated:
- 4 walk-through agents reported 4 skills PASS.
- 1 of 3 real on-A3 smokes found a real bug (vllm-ascend shim 3
  plugin-init order failure).
- 1 of 3 real on-A3 smokes found additional drift (transformers v5.4
  vs v5.3 has 2 additive changes the byte-compare-from-docs missed).

Net: 2/3 real smokes contradicted the walk-through's claims of clean.

# Skill / process implication

For any port-expert skill, the validation ladder should be:

1. **Walk-through agent**: verify docs let a fresh reader plan the
   work. Catches missing reproducer steps, broken links, ambiguous
   commands. **Output gates: documentation passes, NOT artifact
   passes.**
2. **On-A3 import smoke**: actually import the shim modules / source
   tree, verify plugin-init order is safe. Catches lazy-import
   ordering bugs.
3. **On-A3 operational smoke**: actually trigger the affected code
   path (kernel call, model forward, shim symbol resolution).
   Catches symbol-existence bugs, ABI mismatches, runtime crashes.
4. **End-to-end V1.4 / V1.5 etc.**: full RL loop or representative
   workload. Catches integration-level bugs.

A "PASS" claim must specify which rung of this ladder it represents.
"Cold-drive 13/13 PASS" was misleading because it bundled rung-1
results into language that sounded like rungs 1+2+3.

# Generalizable rule

The agent's job is to verify the **plan**. Yours is to verify the
**artifact**. Don't conflate the two — the plan can be coherent in
documentation and still fall apart against real upstream code.

When in doubt: prefer one real smoke over four walk-through reports.
