# PR material — transformers v5.4 outcome A-with-note

> Hand-off note for the **transformers maintainers** and for anyone
> evaluating whether transformers v5.4 has any NPU-integration impact.

## Outcome

After running the transformers-day0 stage-0 decision tree against
`v5.4` (validated against `v5.3.0` as baseline) on A3 host on
2026-04-27, the result is **outcome A-with-note**: no source edits
required for v5.3-era consumer code paths to keep working, but two
additive surface changes are worth flagging for forward-looking NPU
consumers.

Earlier this skill case was reported as plain "outcome A (no NPU
drift)" — the present file is the honest re-validation that landed
during T21 (real on-A3 smoke pass for the 3 day-0 skills).

## What was checked (validated 2026-04-27)

The stage-0 decision tree from the transformers port-expert skill:

### Step 1 — `npu_flash_attention.py` byte-compare

`src/transformers/integrations/npu_flash_attention.py` between v5.3.0
(138 lines) and v5.4.0 (143 lines): **DIFFERS**, additive.

```
138a139,143
> # This function is not implemented but should never be called because block table is not used on NPU
> def npu_flash_attn_with_kvcache():
>     raise NotImplementedError("npu_flash_attn_with_kvcache is not implemented")
```

v5.4 added a placeholder `npu_flash_attn_with_kvcache()` that raises
`NotImplementedError`. The comment says "should never be called
because block table is not used on NPU" — i.e. this is a guard for a
v5.4-era kvcache dispatch route that is intentionally not implemented
on NPU.

**Impact**: zero impact on v5.3-era consumer code paths (the symbol
didn't exist). v5.4-era code that explicitly routes through
`npu_flash_attn_with_kvcache` will hit `NotImplementedError`. Most
consumers (EasyR1, vllm-ascend) do not.

### Step 4 — `ALL_ATTENTION_FUNCTIONS` key set

| Version | Keys |
|---|---|
| v5.3.0 (8) | `flash_attention_2`, `flash_attention_3`, `flex_attention`, `paged\|eager`, `paged\|flash_attention_2`, `paged\|flash_attention_3`, `paged\|sdpa`, `sdpa` |
| v5.4.0 (10) | `flash_attention_2`, `flash_attention_3`, **`flash_attention_4`**, `flex_attention`, `paged\|eager`, `paged\|flash_attention_2`, `paged\|flash_attention_3`, **`paged\|flash_attention_4`**, `paged\|sdpa`, `sdpa` |

v5.4 adds `flash_attention_4` and `paged|flash_attention_4`. These
are additive (no removed key, no changed default).

**Impact**: zero impact on consumers using `flash_attention_2` /
`sdpa` / `flex_attention` (the typical NPU-friendly choices).
Consumers explicitly selecting `attn_implementation="flash_attention_4"`
would route to a community impl that may not run on NPU; those
consumers would need to pick a different `attn_implementation`.

### Steps 2 + 3 — import chain + spec

```
is_torch_npu_available(): True
transformers.integrations.npu_flash_attention spec found: True
```

Both unchanged from v5.3.

## Stage 0 decision

Per skill SKILL.md: "If 1-3 all YES and 4 shows only additive entries
(no new NPU-routing default), **outcome A with note** — skip the full
workflow, emit the classification, write the note in ONBOARDING."

Step 1 is "byte-differs but additive only", step 4 is "additive only,
no NPU-routing default changed". → **outcome A-with-note**.

## Notes for ONBOARDING

- v5.4 added `npu_flash_attn_with_kvcache` as a `NotImplementedError`
  placeholder. NPU paths through `transformers.integrations.npu_flash_attention`
  will not break; only direct calls into the new placeholder do.
- v5.4 added `flash_attention_4` family. NPU consumers should not
  select that `attn_implementation`; stay on `flash_attention_2` /
  `sdpa` / `flex_attention`.

## Independent re-verification

Anyone (including the transformers maintainers) can re-run this
check against any later transformers version with:

```bash
# Clone the harness repo:
git clone https://github.com/zhshgmail/easyr1-npu.git
cd easyr1-npu

# Stage-0 byte-compare + key-set check between two transformers tags:
python3 src/skills/transformers/port-expert/scripts/check_npu_integration_drift.py \
  --transformers-path /path/to/transformers-checkout \
  --base-ref v5.3 \
  --target-ref v5.4

# Expect output:
# == ALL_ATTENTION_FUNCTIONS key diff ==
# (empty)
# == NPU-relevant module byte-compare ==
# All NPU-relevant blocks identical or non-breaking.
# == Stage-0 decision: outcome A (no patch) ==
```

If outcome A: **no PR is needed**. Drop in the new transformers
version and rerun your existing tests.

If the script reports any of:
- new key in `ALL_ATTENTION_FUNCTIONS` that touches an NPU path
- byte-diff in a known NPU integration module
- new abstract method on a base class that NPU subclasses implement

then escalate to stage 1 (fork branch + shim) per the skill's full
workflow.

## Validated 2026-04-25

For target `v5.4`, stage-0 reported outcome A. Independent agent
confirmation: re-ran the script in a fresh agent context, same
result. No PR required for this transformers version.

## Reference

Skill definition + KB:
- [`SKILL.md`](https://github.com/zhshgmail/easyr1-npu/blob/main/src/skills/transformers/port-expert/SKILL.md)
- [`KB_INDEX.md`](https://github.com/zhshgmail/easyr1-npu/blob/main/src/skills/transformers/port-expert/references/KB_INDEX.md)
