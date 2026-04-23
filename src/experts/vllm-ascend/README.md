# `vllm-ascend/` — skills + knowledge for the vllm-ascend upstream

**Upstream repo**: `github.com/vllm-project/vllm-ascend`
**Maintainer ownership**: Huawei Ascend team (C-patch allowed per `day0_patch_scope.md`)

This folder holds all experts, patterns, and references specific to the
**vllm-ascend** upstream. New sub-expert folders go here (e.g.
`upgrade-expert/`, `main2main-expert/`, etc.) as we add capability.

## Experts in this folder

| Expert | What it does | Audience |
|---|---|---|
| [`day0-expert/`](day0-expert/) | Probe Day-0 compatibility of a new vllm or torch release with the current vllm-ascend adapter cursor. Emits patches + reproducer + PR_MATERIAL. | vllm-ascend maintainers |

## Related folders (other upstream repos)

- [`../torch-npu/day0-expert/`](../torch-npu/day0-expert/) — Day-0 for community torch + Huawei torch_npu stack (not yet folder-organized)
- [`../vllm/day0-expert/`](../vllm/day0-expert/) — Day-0 probe for community vllm itself (C-report only since we're not vllm maintainers)
- [`../transformers/day0-expert/`](../transformers/day0-expert/) — Day-0 for community transformers incl. NPU integrations
- [`../_shared/`](../_shared/) — cross-expert templates + OL rules

## Why per-upstream folders

Previously experts were flat (`vllm-ascend-day0-expert/` `vllm/day0-expert/`) with
no hint which upstream each one targeted. User correction 2026-04-23T22:17Z:
"create folder using upstream repo name like vLLM-ascend so that people
know where to find necessary md or skills". Per-upstream folders = one
obvious place for each upstream repo's domain knowledge.

The other upstreams will be reorganised into sibling folders
(`torch-npu/`, `transformers/`, `vllm/`) in a follow-up pass. This is
the proof of structure for `vllm-ascend` first.
