---
name: npu-code-path-sweep
description: Scan a Python source tree for GPU-only / CUDA-specific call sites that will break on Ascend NPU. Produces a structured markdown report grouping hits by pattern (torch.cuda.*, init_device_mesh("cuda"), flash_attn.*, nccl backend, etc.) with file:line refs and suggested replacements. Use whenever beginning a new port of an RL / ML repo to NPU, and re-run after significant upstream updates.
---

# npu-code-path-sweep

## What it does

Greps a source tree for a **curated pattern set** of GPU-only code and emits `docs/code-path-sweep-<repo-name>.md`. Each hit becomes a row in a table under its pattern's section:

```
### NPU-CP-001 — torch.cuda.*

| File | Line | Source | Suggested replacement |
|---|---|---|---|
| verl/workers/fsdp_workers.py | 319 | device_id=torch.cuda.current_device(), | get_device_module().current_device() |
```

Sections are indexed by the stable IDs from `knowledge/npu-patterns.md` (`NPU-CP-NNN`, `NPU-BUG-NNN`). Hitting a section in the sweep report is a fix-this-with-that-pattern signal.

## When to use

- **Every new port target** (different repo or major upstream version bump): run this first, read the output, estimate the port size.
- After a sync/rebase from upstream origin (new GPU-only code may have landed).
- Before declaring a port "done": the sweep should be empty for patterns in scope, with only out-of-scope / intentionally-deferred exceptions.

## When not to use

- On already-ported code (expect false positives on the NPU-aware shims you added).
- Cross-language: this skill is Python-only (the patterns are all Python-API-specific). For C++/CUDA kernels, write a separate sweep.

## How to invoke

```bash
bash scripts/code-path-sweep.sh <source-tree>
# or with explicit output:
bash scripts/code-path-sweep.sh \
  "$HOME/workspace/easyr1-npu/upstream/EasyR1" \
  --out docs/easyr1/code-path-sweep-EasyR1.md
```

Default output path: `docs/code-path-sweep-<basename-of-tree>.md`.

Exit codes:
- 0: sweep completed (may have hits).
- 2: invalid arguments.
- 3: source tree doesn't exist or isn't readable.

## Pattern set (seeded from EasyR1 port, extensible)

The script uses a config array near its top; edit it to add patterns. Each pattern has:

- `id` — stable NPU-CP/NPU-BUG ID from `npu-patterns.md`.
- `title` — human-readable (used as markdown H3 header).
- `regex` — ripgrep-compatible pattern.
- `suggest` — suggested replacement text (rendered in the table).

Seeded patterns:

| ID | Title | Rough regex |
|---|---|---|
| NPU-CP-001 | `torch.cuda.*` | `torch\.cuda\.` |
| NPU-CP-001 | tensor `.cuda()` method | `\.cuda\(` |
| NPU-CP-001 | string `"cuda"` device | `device_map="cuda"\|init_device_mesh\("cuda"\|device="cuda"\|torch\.device\("cuda"` |
| NPU-CP-001 | `CUDA_VISIBLE_DEVICES` literal | `CUDA_VISIBLE_DEVICES` |
| NPU-CP-002 | `vllm.lora.models` import (pre-0.13) | `from vllm\.lora\.models` |
| NPU-CP-003 | Ray `num_gpus` / `"GPU"` resource | `num_gpus\b\|ray\.available_resources\(\)\[.*GPU\|\{"GPU": ` |
| NPU-CP-004 | vllm `get_tensor_model_parallel_group` | `get_tensor_model_parallel_group` |
| (misc) | `flash_attn` import | `from flash_attn\|import flash_attn` |
| (misc) | `liger_kernel` import | `from liger_kernel\|import liger_kernel` |
| (misc) | `nccl` backend | `backend=.?"nccl"` |

## Output schema

```markdown
# Code-path sweep for <repo>

Generated: <date>
Source tree: <path> (<N> python files scanned)
Total hits: <N>

## NPU-CP-001 — torch.cuda.*
<table>

## NPU-CP-001 — tensor .cuda() method
<table>

## NPU-CP-002 — vllm.lora.models
<table or "No hits.")
...
```

## Rules / gotchas

- **False positives are expected**. Example: a `.cuda()` hit in a comment or docstring. The skill doesn't filter those out — human review is part of the workflow. Don't auto-rewrite.
- **Add patterns over time**, don't try to enumerate all of them up front. New bugs → new pattern entry + new row in `npu-patterns.md`.
- The skill scans `.py` files only. For `.yaml` configs that hardcode `device: cuda`, grep separately.

## Related

- `knowledge/npu-patterns.md` — the catalog the IDs point to.
- `docs/easyr1/dep-matrix.md` §Code-path blockers — manual analog; eventually this skill should feed that section directly.
