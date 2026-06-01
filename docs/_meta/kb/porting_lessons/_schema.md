# porting_lessons — schema

Each lesson is one `.md` file with YAML frontmatter + body.

**File name**: `<layer>-<NNN>-<slug>.md` where
- `<layer>` ∈ `torch-npu`, `transformers`, `vllm-ascend`, `vllm`, `easyr1`,
  `sglang`, `miles`, `mindspeed`, `bishengir`, `tilelang`, `triton-ascend`,
  `cross-layer`
- `<NNN>` is 3-digit zero-padded, unique within that layer
- `<slug>` is hyphenated short description

**Frontmatter**:

```yaml
---
id: <layer>-<NNN>
date: YYYY-MM-DD        # when the lesson was first learned (not written)
layer: <layer>          # same as filename prefix
title: <short title>
trigger:                # plain-text patterns that should fire this lesson
  - "phrase 1"
  - "phrase 2"
symptom_in_wild:        # observable symptoms that indicate I'm repeating this
  - "observation 1"
root_cause: >
  Explain the true underlying mistake pattern, 2-3 lines.
mistake_pattern: "short tag for the error shape"
correction:             # what to do instead
  - "step 1"
  - "step 2"
evidence:               # where this was learned (commit / discord ts / file path)
  - "source 1"

# ---- OPTIONAL scope tags (recommended for new entries since 2026-05-31) ----
# These let /npu-adapt-assist and humans filter by upstream version / image
# tag, and let stale lessons be marked rather than silently rot.
#
# Borrowed from a5_ops `applies_to` / `verified_on` / `unverified_on` model
# (see workspace/a5_ops_audit_2026_05_31/FINDINGS.md §2.2). Older entries may
# lack these; absence is treated as "no scope claim".
applies_to:             # upstream version(s) / image(s) where this lesson is
  - "sglang>=0.5.10"    # actively useful
  - "vllm-ascend main @ 2026-05-29"
verified_on:            # explicit confirmation that the lesson reproduces
  - "verl-sglang-8.5.0 image, A3 host, 2026-05-30"
unverified_on:          # never tested; expected but not proven
  - "vllm-ascend release branches before 2026-04"
deprecated_after: ""    # ISO date; empty if still current. Set when the upstream
                        # fix lands and this lesson is mostly historical.
---
```

**Body**: free-form. Why it matters. What almost happened if I'd acted on the
wrong assumption. Links to concrete commits / files.

**Backward compat**: existing entries (pre-2026-05-31) don't have the optional
scope tags. Treat their absence as "no scope claim — lesson is general or
scope is unknown". When you add a new entry or substantially update an old
one, fill the optional fields.
