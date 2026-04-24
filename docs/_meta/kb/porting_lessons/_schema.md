# porting_lessons — schema

Each lesson is one `.md` file with YAML frontmatter + body.

**File name**: `<layer>-<NNN>-<slug>.md` where
- `<layer>` ∈ `torch-npu`, `transformers`, `vllm-ascend`, `vllm`, `easyr1`,
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
---
```

**Body**: free-form. Why it matters. What almost happened if I'd acted on the
wrong assumption. Links to concrete commits / files.
