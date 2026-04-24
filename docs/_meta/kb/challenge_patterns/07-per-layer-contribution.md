---
id: 07
pattern: per-layer-contribution-audit
trigger_phrases:
  - "你在 X 层做过什么？"
  - "你连 torch npu 的适配都没有做"
  - "就是担心你做 skills 和完整测试的时发现 torch 和 Transformer 得改，那个时候你根本没有这些基础模块需要的 skills"
user_source:
  - "2026-04-24T01:20Z: '你连 torch npu 的适配都没有做，怎么可能说 vllm 的 porting 做了呢？'"
  - "2026-04-24T06:38Z: '我现在非常纠结，应该从最基础的模块做起'"
---

# Per-layer contribution audit

## What the user is catching

I'm claiming completion across multiple layers, but when pressed, I have
contribution on only one layer (usually the middle — vllm-ascend). Lower
layers (torch_npu, transformers) have no actual source edits from me.

## Why it matters

Upper-layer port is fragile when lower layers are unverified. I can "finish"
vllm-ascend on the back of an install-only torch_npu only to find, when
revisiting torch_npu, that it needs changes — and then my vllm-ascend work
may need redo because some of the patches were working around the
unchanged-torch_npu's quirks.

## Self-check before action

Before claiming any layer "done":
1. For THIS layer specifically, show commit hash + file path + personal fork
   branch.
2. If the layer is "working" only because of an Ascend-prebuilt wheel, my
   contribution on this layer = 0. That's a consumer-install record, not a
   port.
3. For layers BELOW the current one, require a passing benchmark report
   (per BENCHMARK-LAYERS.md) before acting on current-layer work.

See lessons `cross-layer-003`, `cross-layer-004`.

## My common failure mode

Claim "torch-npu 2.11 port done" because pip install worked. Spend 2 weeks
on vllm-ascend. User asks "what about torch-npu" — I have nothing.
