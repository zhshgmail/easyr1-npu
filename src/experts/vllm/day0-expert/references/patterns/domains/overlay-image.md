# Domain — Dockerfile.overlay-vllm<MM> template

**Load when**: Phase C building the Day-0 overlay image (outcome A or B).

## Why overlay

Same reason as `transformers/day0-expert/references/patterns/domains/overlay-image.md`:
adding one pip install on top of BASE_IMAGE is ~60s; rebuilding the full
v2 stack from scratch is 10-15min. For a vllm-only bump, overlay wins.

## Template

```dockerfile
# Overlay image: BASE_IMAGE + target community vllm.
# vllm-ascend from BASE_IMAGE is NOT modified — we're testing if its
# plugin registration survives the new vllm.

ARG BASE_IMAGE=<BASE_IMAGE tag>
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_DEFAULT_TIMEOUT=60

# Install target vllm. --no-deps because:
# - We don't want pip re-pulling torch / torch_npu (base is authoritative)
# - vllm 0.19 may declare new transitive deps (e.g. newer transformers);
#   we keep the transformers shipped in BASE_IMAGE unless outcome chain
#   explicitly bumps it via a separate transformers-day0 run
RUN pip install --no-cache-dir --no-deps vllm==<TARGET_VLLM_VERSION>

# Build-time sanity: VERSION-ONLY. Do NOT `import vllm` here — vllm import
# chain triggers torch_npu → libascend_hal.so dlopen which fails in the
# docker-build sandbox (no NPU device mount). Same trap as transformers-day0.
RUN python3 -c '
import importlib.metadata as md
print("overlaid vllm:", md.version("vllm"))
'

# Runtime verification happens in the FIRST smoke container (V1.3) where
# NPU devices are mounted. At that point, `import vllm` + plugin log
# message "Platform plugin ascend is activated" is the real signal.
```

Image tag convention: `easyr1-npu-vllm<major-tens><minor><patch>:<SESSION_TAG>`
(e.g. `easyr1-npu-vllm0191:vllm-day0-20260423-0223`).

## Outcome B extras

If consumer needs a shim (e.g. new SamplingParams RO property), the
shim is a commit on the `ascend-day0-vllm-$SESSION_TAG` branch of the
consumer repo; NOT a file in the image. `deploy_to_a3.sh` syncs the
branch, and the smoke script's `run-npu-container.sh --live-source`
bind-mount delivers the fresh code into the container.

So outcome B's Dockerfile is the same template as outcome A — the
shims aren't baked into the image.

## Build command

```
bash $VLLM_DAY0_EXPERT_ROOT/scripts/deploy_to_a3.sh \
    --branch ascend-day0-vllm-$SESSION_TAG \
    --image-tag easyr1-npu-vllm<NNN>:<SESSION_TAG> \
    --dockerfile Dockerfile.overlay-vllm<NNN> \
    --base-image <BASE_IMAGE> \
    --upstream-consumer $UPSTREAM_CONSUMER
```

(deploy_to_a3.sh has `--build-arg BASE_IMAGE` wiring per
commit 5f8e7e3 2026-04-23.)

## Post-build verification

Before Phase D smoke:

```
docker run --rm <overlay-image> python3 -c '
import importlib.metadata as md
print("vllm:", md.version("vllm"))
print("vllm-ascend:", md.version("vllm-ascend"))
print("transformers:", md.version("transformers"))
'
```

This doesn't trigger NPU device load. Real runtime verification is the
first smoke container starting.

## Cleanup

`cleanup_session.sh --session-tag <TAG> --preserve-image` keeps the
overlay image for the caller (orchestrator may hand it to easyr1-port
next).
