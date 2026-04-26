# PR material — transformers v5.4 outcome A (no edits required)

> Hand-off note for the **transformers maintainers** and for anyone
> evaluating whether transformers v5.4 has any NPU-integration impact.

## Outcome

After running the transformers-day0 cold-drive against `v5.4`, the
result is **outcome A: no NPU-integration drift**. No fork branch was
created because no source edits were needed.

This document records what was checked, why it was negative, and how
to re-validate when the next transformers version ships.

## What was checked

The transformers port-expert skill performs a stage-0 quick decision
tree before committing any A3 hardware time:

1. **Byte-compare** the transformers files that NPU integrations
   typically touch. For v5.4 vs v5.3:
   - `src/transformers/modeling_utils.py` (PyTorch state-dict load /
     custom device hooks): no NPU-relevant block changed
   - `src/transformers/integrations/__init__.py`: no NPU-specific
     entry added or removed
   - `src/transformers/utils/import_utils.py` (`is_torch_npu_available`,
     `is_torch_xpu_available` etc.): no signature change

2. **`ALL_ATTENTION_FUNCTIONS` key set diff**: NPU-relevant keys
   (e.g. `"sdpa"`, `"flash_attention_2"`, `"npu_flash_attention"`)
   present in both versions with same callable shape.

3. **No new mandatory device hook** added to any base class that
   torch_npu/transformers users register against.

Stage 0 decision tree concluded: **outcome A** — no shim, no fork,
no patch.

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
