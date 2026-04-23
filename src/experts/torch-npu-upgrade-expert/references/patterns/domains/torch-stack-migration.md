# Domain — torch-stack migration (Dockerfile + NPU-BUG recurrence)

**Load when**: Phase B writing `Dockerfile.npu-torch-<version>` or
verifying an existing one against the target image's pip-freeze.

## Canonical Dockerfile.npu-torch-<target> template

```dockerfile
# <consumer> on Ascend 910C, built on top of the target-torch-stack base image.
#
# Sole job: apply NPU-BUG-001 + NPU-BUG-004 workarounds on top of the target
# base image so `import torch_npu` works cleanly. Consumer code is bind-mounted
# at container runtime via run-npu-container.sh --live-source, so this image
# only needs the *platform* right.

ARG BASE_IMAGE=<target-image-tag>
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_DEFAULT_TIMEOUT=60

# NPU-BUG-001 — recurs on every Ascend/verl base image to date. The shipped
# triton-ascend wheel has partial file tree; force-reinstall to populate it.
# Version should match what the base image shipped (check Phase A pip-freeze).
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        --index-url ${PIP_INDEX_URL} \
        triton-ascend==<version-from-pip-freeze> \
 || pip install --no-cache-dir --force-reinstall --no-deps \
        --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
        triton-ascend==<version-from-pip-freeze>

# NPU-BUG-004 — present on v2 (verl-8.5.2-a3) and likely any base image that
# ships upstream `triton` with amd/nvidia backends. Only add this block if
# Phase A probe found them present.
RUN python3 -c '
import importlib.util, os, shutil, sys
spec = importlib.util.find_spec("triton")
if spec and spec.origin:
    triton_dir = os.path.dirname(spec.origin)
    for backend in ["amd", "nvidia"]:
        p = os.path.join(triton_dir, "backends", backend)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"pruned {p}")
'

# Consumer tree bind-mounted via --live-source at runtime.
WORKDIR /opt/<consumer>

# No CMD override — inherit base image's entrypoint (CANN env setup).
```

## Post-build verification

```bash
docker run --rm <new-image-tag> python3 -c '
import torch, torch_npu, triton
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("triton", triton.__version__)
import importlib.util, os
triton_dir = os.path.dirname(importlib.util.find_spec("triton").origin)
print("backends", sorted(os.listdir(os.path.join(triton_dir, "backends"))))
'
```

Expect:
- torch, torch_npu, triton all import without error
- `backends` list contains `npu`, does NOT contain `amd` / `nvidia` on v2+
- torch_npu version matches target pip-freeze

If `import torch_npu` fails:
- `cannot import name 'Config' from 'triton'` → NPU-BUG-001 didn't take.
  Check: did the `pip install --force-reinstall` line actually run? Did it
  use the right version? Is the aliyun fallback to huaweicloud path
  working?
- `cannot import name 'Language' from 'triton.backends.compiler'` →
  NPU-BUG-004 didn't take. Check: did the prune block execute after the
  triton-ascend reinstall (ordering matters)? Did `importlib.util.find_spec`
  actually resolve to the path where amd/nvidia live?

## NPU-BUG-003 (torch.compile + log_probs inductor crash)

**This expert does NOT flip `use_torch_compile` speculatively.**

Why: the canonical V1.4 smoke config already has `worker.actor.use_torch_compile=false`
(per easyr1-expert SMOKE_BASELINE.md `canonical config` table). Any
Phase D run inherits that setting. If NPU-BUG-003 still fires, something
ELSE is calling torch.compile — and that's out of scope for a torch
upgrade (it's a consumer code issue).

**When NPU-BUG-003 might be considered FIXED**: a target image's
torch_npu + triton_ascend combination runs V1.4 cleanly WITH
`use_torch_compile=true`. Positive finding to record in this expert's
PROGRESS.md — but require **two independent sessions** confirming before
proposing to flip the default.

## Per-version evidence table

Fill in rows as new target-image versions are probed. Never infer a row
without a live probe.

| torch / torch_npu | triton_ascend | CANN | NPU-BUG-001 | NPU-BUG-004 | NPU-BUG-003 (with use_torch_compile=true) | First validated |
|---|---|---|---|---|---|---|
| 2.8.0 / 2.8.0 (v1) | 3.2.0 | 8.5.0 | recurs (force-reinstall) | NOT present | not safe to enable | 2026-04-17 (rounds 1–4), E2E wet-run 2026-04-22 |
| 2.9.0 / 2.9.0 (v2) | 3.2.0 | 8.5.1 | recurs | recurs (prune) | not safe to enable | 2026-04-19 drill, E2E P2 2026-04-22 |
| 2.10+ / 2.10+ | ??? | ??? | ??? | ??? | ??? | TBD |

## Failure-mode decision tree

```
Phase A probe: `import torch_npu` fails in TARGET container
├── "cannot import name 'Config' from 'triton'" → NPU-BUG-001; add force-reinstall line
├── "cannot import name 'Language' from 'triton.backends.compiler'" → NPU-BUG-004; add prune block
├── "RuntimeError: CANN version mismatch" → torch_npu ↔ CANN mismatch at target;
│       escalate to user — target image itself is broken, not fixable here
└── segfault / hang → check dmesg; likely NPU-OPS-009 UDA namespace (not this
                      expert's domain; it's a container-runner/bind issue)

Phase D V1.1 fail after clean build
├── same ImportError as Phase A → Dockerfile workaround didn't take at build
│       time; rebuild with set -x in Dockerfile to trace
├── npu-smi reports chip busy → OL-05 violation; stop, don't co-schedule
└── new error not matching Phase A probes → record in PROGRESS "unclassified",
                      exit stuck; don't invent a workaround

Phase D V1.4 fail after V1.1 + V1.3 PASS
├── OOB entropy_loss → EC-12 (canonical config drift); unlikely in this
│       expert since no config edits; if hit, check that the image bind-mount
│       delivered the right smoke script
└── "no entropy_loss" → EC-11 (stdout vs jsonl) or EC-13 (stale checkpoint);
                      from easyr1-expert KB
```
