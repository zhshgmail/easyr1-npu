# Archived: src/skills/vllm/ (obsoleted 2026-04-25)

## Why archived

`src/skills/vllm/port-expert/` was a skill for adapting **community vllm** (Huawei-external) to a newer version. At the time (pre-2026-04-24), this was framed as "Day-0 probe for vllm on NPU".

Per the 2026-04-24 rework, responsibilities got split clearly:

- **Huawei software owners** (vllm-ascend maintainers) patch `vllm-ascend` to adapt to whatever community vllm shipped. They use `src/skills/vllm-ascend/port-expert/` with F1-F8 scanners.
- **Community vllm** is not our code to patch. We file issues upstream when we find bugs, and work around them in `vllm-ascend`.

So `src/skills/vllm/` overlapped with `src/skills/vllm-ascend/` in intent but had a weaker / older framing. Keeping both was confusing. User flagged it 2026-04-25: "If vllm skills is obsolete, we should remove it to prevent confusing".

## Replacement

For anyone who used to think `src/skills/vllm/` was the entry point:

- **Porting vllm-ascend to a new vllm version**: use `src/skills/vllm-ascend/port-expert/`.
- **Checking if a new community vllm breaks NPU stack**: use the Mode-Sweep in `src/skills/vllm-ascend/port-expert/scripts/sweep.sh` against the commit range.

## What's preserved here

The full original content is kept under this directory for archaeological reference. Not to be shipped, not to be depended on.
