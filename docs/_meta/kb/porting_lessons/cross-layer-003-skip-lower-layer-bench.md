---
id: cross-layer-003
date: 2026-04-24
layer: cross-layer
title: Never move to upper-layer port while lower layer is not benchmark-verified
trigger:
  - "start port on layer N+1"
  - "claim layer N done based on layer N+1 success"
symptom_in_wild:
  - "working on vllm-ascend while torch-npu has no L1 op bench result"
  - "debugging transformers model loading while torch-npu layernorm numerics unverified"
  - "blaming upper layer's weird output on lower layer without evidence"
root_cause: >
  If layer N's numerics are silently wrong, layer N+1 will fail in ways that
  look like layer N+1 bugs. I will spend hours patching N+1 to work around
  layer N's quiet error, and the patches will be fragile even if they superficially
  pass smoke. Fix must go to layer N; any "fix" at layer N+1 that tolerates
  layer N's bug is technical debt and not a real port.
mistake_pattern: "upper-layer shim that secretly works around lower-layer fault"
correction:
  - "Before starting port on layer N+1, require layer N has a passing benchmark report (L<N>_benchmark_report.md with numeric pass/fail)."
  - "If a claimed upper-layer bug seems to disappear when a lower-layer op is swapped out, that is evidence the real bug is in the lower layer."
  - "Never ship a consumer-side patch that 'fixes' something until the layer-isolated bench at that level confirms the bug is there."
evidence:
  - "2026-04-23: 14-iter vllm-ascend drift patching while torch-npu ABI was silently wrong (guard returned False). Many of the 14 iter were fighting symptoms of the ABI guard issue."
  - "User 2026-04-24T02:47Z: 'torch npu 和 transformers 我都没法拿去和客户说啊'"
  - "User 2026-04-24T06:38Z: 'vllm 这个中部的模块...担心你做 skills 和完整测试的时发现 torch 和 Transformer 得改'"
---

# Decision rule

Before any action on layer N+1:

1. Is `docs/<N>/benchmark/L<N>_benchmark_report.md` present AND showing all-pass?
2. If no, stop. Go back to layer N, run/fix that benchmark first.
3. If yes, cite the report path in the first Discord message when starting
   layer N+1 work.
