# dep-analysis-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL rules | Phase A first read |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | This expert's OL-03 + OL-08 | Phase A second read |
| [NPU_ECOSYSTEM_MAP.md](NPU_ECOSYSTEM_MAP.md) | Per-package classification rules (A/B/C/D/E) | Phase P3 classification |

## Tooling

| Tool | Purpose |
|---|---|
| `$REPO_ROOT/scripts/dep-gap-detect.sh` | Core classifier. Given reqs.txt + image-freeze, emits A/B/C/D/E markdown. |
| `$REPO_ROOT/scripts/inspect-ascend-image.sh` | Extracts pip freeze + apt list from a docker image; produces the knowledge/images/<slug>.md consumed here if cached |

## Classification shorthand

- **A — NPU native**: upstream supports NPU (e.g. `peft` rides on transformers+torch, no CUDA-specific code paths). Use as-is.
- **B — NPU-ported fork**: an NPU-specific drop-in exists (e.g. `vllm` → `vllm-ascend`). Use the fork.
- **C — CUDA-only, bypassable**: consumer needs a Python-layer shim (e.g. `flash-attn` → `transformers.integrations.npu_flash_attention`). Shim lives in consumer's port branch.
- **D — CUDA-only BLOCKER**: no bypass without new NPU adaptation work. This is a P2 scenario — route to an upgrade-expert or surface to user.
- **E — pure Python / CPU**: accelerator-agnostic (e.g. `numpy`, `pandas`, `datasets`). Install as-is.

## Task-plan output shape

For each classification outcome the worker emits:

- P1 (all A/B/C/E): `task_plan = [{step:1, expert:<consumer-port>, input:{--reuse-image: candidate}}]`
- P2 (any D, covered): `task_plan` lists upgrade-experts first, then consumer-port with `--reuse-image: <upgraded-image>`
- P2 (any D, not covered): `task_plan` records the unsupported dep + exits `stuck` — orchestrator surfaces to user
