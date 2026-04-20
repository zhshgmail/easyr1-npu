---
name: npu-image-inspect
description: Inspect an Ascend NPU docker image (e.g. quay.io/ascend/verl:verl-X.Y.Z-aN-*) and emit a structured knowledge doc about it — CANN/Python/torch versions, pip freeze, Ascend-specific package check, triton-ascend integrity (NPU-BUG-001 detector), apt package count. Use whenever targeting a new Ascend base image or verifying a suspected image-level bug.
---

# npu-image-inspect

## What it does

Given a docker image reference, produces `knowledge/images/<slug>.md` containing:

1. CANN version + install path (from `$ASCEND_TOOLKIT_HOME`).
2. Python version + install path.
3. Entrypoint and Cmd.
4. **Triton-ascend integrity check** (`NPU-BUG-001`): looks for `site-packages/triton/__init__.py`. If the dist-info says triton-ascend is installed but `__init__.py` is missing, flags it and tells the caller to add a `pip install --force-reinstall --no-deps triton-ascend==<ver>` layer in their downstream Dockerfile.
5. Core ML stack versions (torch, torch_npu, transformers, accelerate, tensordict, torchdata, ray, vllm, vllm_ascend, triton_ascend, peft, datasets, etc).
6. Totals: pip distribution count, apt package count.

## When to use

- Every time we target a new Ascend base image (new CANN, new vllm-ascend build, sglang variant, etc).
- To reproduce/verify a suspected image-level bug.
- Before editing `dep-matrix.md`: the inventory feeds the "image" columns.
- As input to a future `dep-diff` skill (compare two images or image vs source).

## When not to use

- To inspect a **running** container — use `docker exec` directly.
- To inspect a non-Ascend image — paths baked in (e.g. `/usr/local/Ascend`, `/usr/local/python3.11.14`) won't match.

## Prerequisites

- Docker available on the host running the skill.
- Image already pulled (or pullable); the script does not auto-pull.
- A few GB free for the temp site-packages copy.

## How to invoke

From the repo root:

```bash
bash scripts/inspect-ascend-image.sh \
  quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
```

Default output path: `knowledge/images/<slug>.md` where `<slug>` is the image ref with `/`, `:`, `+` replaced with `-`.

Override output: `--out path/to/file.md`.
Keep the temp dir (for debugging or running `grep` on pip-freeze): `--keep-tmp`.

## Output schema

The script auto-generates these sections with stable headers (other skills can parse them):

- `## Runtime environment` — Python prefix, ASCEND_HOME_PATH, Entrypoint, Cmd.
- `## Triton-ascend integrity check (NPU-BUG-001)` — emits `:warning:` block only when broken.
- `## Core ML stack (versions from dist-info)` — interesting packages filtered from pip-freeze.
- `## Totals` — python + apt package counts.
- `## Full pip freeze` — all dist-info-derived entries (collapsible).

**Hand-augmented sections** (not auto-generated; add manually after running the script if the doc is going to be cited elsewhere):

- `## Matching upstream refs` — which branch/tag of torch-npu / vllm-ascend / triton-ascend / transformers matches this image's pins. Derive from `knowledge/upstream-refs.md` and the image's own pip-freeze.
- `## Open questions` — anything we noticed inspecting the image but can't answer without running it on hardware.

The two hand-written examples in `knowledge/images/verl-8.5.0-a3.md` and `knowledge/images/verl-8.5.2-a3.md` include the hand-augmented sections. When you use `inspect-ascend-image.sh` to bootstrap a new image doc, consider adding at least the `Matching upstream refs` section before treating the doc as complete.

## Rules / gotchas

- **Don't re-run rapidly** without `--keep-tmp`: an interrupted docker cp can leave a stale container. The script traps on exit but SIGKILL misses that.
- **Python path fallback**: the script tries `/usr/local/python3.11.14`, then `/usr/local/python3.10`, then whatever exists under `/usr/local/python*`. If your image's python is somewhere else, pass `--python-prefix <path>`.
- **Editable installs** (`pip install -e .`) show up via their `.dist-info` but not the source dir — the skill reports the version string only. To see source paths, `pip show` inside the image.

## Related

- `knowledge/npu-patterns.md` — `NPU-BUG-001` is the triton-ascend integrity one.
- `knowledge/images/` — where outputs land.
- `knowledge/upstream-refs.md` — pairs image versions with matching upstream refs; this skill's output feeds it.
