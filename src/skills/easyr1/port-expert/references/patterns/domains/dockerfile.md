# Domain — Dockerfile.npu (Stage 0 template)

**Load when**: Phase B/C, creating or modifying `Dockerfile.npu` for the
EasyR1 port.

---

## Stage 0 template — `Dockerfile.npu`

Drop this at the EasyR1 repo root as `Dockerfile.npu`. Tested on the
`verl-8.5.0-a3` base image and the ascend-port branch.

```dockerfile
# EasyR1 on Ascend 910C (A3) — built on top of veRL's A3 base image.
#
# Base image layout:
#   - Ubuntu 22.04, Python 3.11.14 at /usr/local/python3.11.14
#   - CANN 8.5.0 at /usr/local/Ascend/cann-8.5.0
#   - ATB (Ascend Transformer Boost) at /usr/local/Ascend/nnal/atb/...
#   - torch 2.8.0+cpu, torch_npu 2.8.0, triton_ascend 3.2.0,
#     vllm_ascend 0.13.1.dev18, transformers 4.57.6 pre-installed.
#
# Do NOT install flash-attn, liger-kernel, or upstream vllm here — those are
# in requirements-gpu.txt and break on NPU.

ARG BASE_IMAGE=quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Repair: base image ships triton-ascend 3.2.0 but site-packages/triton is
# missing __init__.py and several top-level modules that torch._inductor
# expects — symptom: `import torch_npu` raises ImportError.
# See NPU-BUG-001 in PLATFORM_BUGS.md.
#
# Use aliyun mirror (huaweicloud mirror is flaky — OL-07) with a timeout.
RUN pip install --no-cache-dir --default-timeout=60 \
        --force-reinstall --no-deps \
        --index-url https://mirrors.aliyun.com/pypi/simple/ \
        triton-ascend==3.2.0 \
 || pip install --no-cache-dir --default-timeout=60 \
        --force-reinstall --no-deps \
        --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
        triton-ascend==3.2.0

# EasyR1 source. Build from the EasyR1 repo root with:
#   docker build -t easyr1-npu:<SESSION_TAG> -f Dockerfile.npu .
WORKDIR /opt/easyr1
COPY . /opt/easyr1

# Install common requirements. Base image already satisfies most pins, so this
# is fast. Don't install requirements-gpu.txt on NPU.
RUN pip install --no-cache-dir --default-timeout=60 \
        --index-url https://mirrors.aliyun.com/pypi/simple/ \
        -r requirements.txt

# Install EasyR1 itself editable (no-deps — deps already resolved above).
RUN pip install --no-cache-dir --no-deps -e .

# Default entrypoint inherits from base (CANN / ATB env setup).
```

## Requirements split (required in EasyR1 repo root)

EasyR1 master has a monolithic `requirements.txt`. Stage 0 needs you to
split out flash-attn, liger-kernel, and any GPU-only pins to a new
`requirements-gpu.txt`, and add `requirements-npu.txt` (mostly empty —
comment that everything is already in the base image). Example:

```
# requirements-gpu.txt (new file, NOT used by Dockerfile.npu)
flash-attn
liger-kernel
```

```
# requirements-npu.txt (new file, NOT used by Dockerfile.npu)
# NPU-side pip deps come from the verl-8.5.0-a3 base image:
#   torch_npu==2.8.0, vllm_ascend==0.13.1.dev18, transformers==4.57.6
# This file is a placeholder so tooling that expects it doesn't error.
```

## Constraints — do not skip

- **`--default-timeout=60`** on every `pip install` line. The huaweicloud
  mirror is known to hang > 50 min without a timeout (OL-07 / EC-10).
- **aliyun mirror first, huaweicloud as fallback**. aliyun is strictly
  faster on this host; huaweicloud's Ascend mirror is only needed for
  the `triton-ascend` wheel (it's not on aliyun).
- **`--force-reinstall --no-deps` on the triton-ascend line**. `--no-deps`
  because the base image already has every triton-ascend dep, and reinstalling
  them pulls in upstream `triton` which conflicts (NPU-BUG-004).
- **No `apt-get install` without a good reason**. The base image has
  everything needed for Stage 0. Adding apt steps slows the build 5–10×.
- **Use `ARG BASE_IMAGE`** so orchestrator can pass v1 vs v2 image.
- **Do not `COPY .` before the triton repair step**. Putting the repair
  first keeps that layer cache-valid across source edits — your
  re-deploy-after-edit iteration stays fast.

## Verify

```bash
# On A3, after a successful build:
docker run --rm easyr1-npu:<SESSION_TAG> python3 -c '
import torch, torch_npu, triton
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("triton:", triton.__version__)
print("npu_available:", torch.npu.is_available())
'
# expect: torch 2.8.0+cpu, torch_npu 2.8.0, triton 3.2.0, npu_available True
```

If `import torch_npu` errors on "cannot import name 'Config' from 'triton'",
the triton-ascend repair didn't take — check the line made it past the
aliyun mirror and into the image.

## Evidence

Port-branch commits:
- `cbfe645` "add Dockerfile.npu layered on verl-8.5.0-a3"
- `cd16649` "Dockerfile.npu: force-reinstall triton-ascend to fix broken base install"

2026-04-22 round 2 deploy hung 50 min on `pip install triton-ascend` without
the timeout + aliyun-first pattern (this was the motivating incident for
OL-07 / EC-10). Current template avoids that failure mode.
