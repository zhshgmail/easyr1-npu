---
id: cross-layer-002
date: 2026-04-24
layer: cross-layer
title: Using V1.4 training entropy_loss as THE port judgement confuses ownership
trigger:
  - "V1.4 entropy_loss"
  - "GRPO 2-step"
  - "training smoke as port criterion"
symptom_in_wild:
  - "claim 'vllm-ascend port done' based on V1.4 PASS"
  - "claim 'torch-npu port failed' because V1.4 entropy_loss 3.213"
  - "debugging a full RL training loop to diagnose a C++ ABI issue"
root_cause: >
  V1.4 is an end-to-end RL training indicator. Its failure can come from any of:
  torch_npu op numerics, vllm-ascend kernel, vllm rollout API drift, EasyR1
  actor code, FSDP, optimizer state, checkpoint resume bug. Using it as THE
  port judgement means when it fails, I cannot say which layer owns the failure,
  and I often misattribute the blame to whatever I was working on at the moment.
mistake_pattern: "one high-level metric picking up contamination from many layers"
correction:
  - "Each layer (torch-npu L1, transformers L2, vllm-ascend L3, EasyR1 L4) MUST have its own benchmark with its own number."
  - "V1.4 entropy_loss belongs to L4 (EasyR1). It is NOT a judgement for L1/L2/L3."
  - "When a lower layer is un-benchmarked, do not use upper-layer signals to infer lower-layer correctness."
  - "See docs/_meta/BENCHMARK-LAYERS.md for the per-layer criteria."
evidence:
  - "2026-04-23 iter 18 V1.4 entropy_loss = 3.213 (baseline 1.275). I wrongly blamed vllm-ascend; root cause later traced to stale checkpoint + ConsoleLogger + consumer-side vllm 0.20 API drift in EasyR1"
  - "2026-04-24T04:04Z user: '是否应该定义一个 inference 方面的 benchmark，而不是训练的？'"
---

# What to do instead

- For a vllm-ascend port claim: run L3 inference fidelity bench (100 prompts
  greedy vs reference). Do NOT run V1.4.
- For a torch-npu port claim: run L1 op numerical bench. Do NOT run V1.3/V1.4.
- V1.4 is the L4 (EasyR1) gate and only meaningful when L1/L2/L3 have all
  passed their own benchmarks independently.
