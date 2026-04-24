# Domain — Dockerfile.npu-\<target\> (new base-image swap)

**Load when**: Phase B writing `Dockerfile.npu-<target-short>` for the new
target image.

## Stage 0 template (parameterized)

Drop this at the consumer repo root as `Dockerfile.npu-<target-short>`
(e.g. `Dockerfile.npu-852` for the v2 drill image). Tested in the v1→v2
drill (2026-04-19, commits 318925f / df84212 / 16051e2).

```dockerfile
# <consumer> on Ascend 910C (A3) — built on top of the TARGET base image.
#
# Base image inventory confirmed in Phase A pip-freeze diff:
#   (e.g. Ubuntu 22.04, Python 3.11.14, CANN 8.5.1, torch 2.9.0+cpu,
#         torch_npu 2.9.0, triton_ascend 3.2.0, vllm_ascend 0.17.0rc2,
#         transformers 5.3.0.dev0)
#
# Do NOT install flash-attn, liger-kernel, or upstream vllm — those break on NPU.

ARG BASE_IMAGE=<target-image-tag>
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

# OL-07: aliyun-first mirror with --default-timeout=60. Huaweicloud mirror
# intermittently empty (2026-04-22 round 2 hang 50 min).
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_DEFAULT_TIMEOUT=60

# NPU-BUG-001 recurs on every base image: triton-ascend wheel unpacks
# incompletely. Must force-reinstall.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        --index-url ${PIP_INDEX_URL} \
        triton-ascend==<version-from-base-image> \
 || pip install --no-cache-dir --force-reinstall --no-deps \
        --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
        triton-ascend==<version-from-base-image>

# NPU-BUG-004 recurs on v2 image specifically: upstream triton wheel ships
# amd/ and nvidia/ backend subdirs in the site-packages/triton/ tree that
# clash with triton-ascend's registration. Remove them explicitly.
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

# Consumer source. Build from the consumer repo root with:
#   docker build -t easyr1-npu:<SESSION_TAG> -f Dockerfile.npu-<target-short> .
WORKDIR /opt/<consumer>
COPY . /opt/<consumer>

# Install consumer common requirements.
RUN pip install --no-cache-dir -r requirements.txt

# Install consumer itself editable (no-deps — base already has deps).
RUN pip install --no-cache-dir --no-deps -e .
```

## Parameters the worker fills in

| Token | v2 drill value | How to resolve |
|---|---|---|
| `<target-image-tag>` | `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5` | From `$TARGET_IMAGE` env |
| `<target-short>` | `852` | Last two digits of base-image version |
| `<consumer>` | `easyr1` | From `$UPSTREAM_CONSUMER` (lowercased) |
| `<version-from-base-image>` | `3.2.0` | From Phase A pip-freeze of target |

## NPU-BUG-004 — why the amd/nvidia prune is mandatory on v2

In the v2 base image the upstream `triton` wheel shipped with
`triton/backends/amd/` and `triton/backends/nvidia/` subdirs containing
dispatchers that reference CUDA/ROCm headers not present on A3. When
`triton-ascend` tries to register its `npu` backend, it walks the
`backends/` dir and blows up on the first broken backend's
`import_backends()`. The amd/nvidia subdirs have no use on NPU — prune
them.

Do NOT rely on `pip uninstall triton && pip install triton-ascend` —
triton-ascend wheel **reuses the same `triton/` directory** (by design);
uninstalling upstream triton breaks triton-ascend too. Prune the specific
subdirs, leave the rest.

Verified: v2 drill commit `15f9450`. Same fix works on any newer base
image that ships the amd/nvidia pollution.

## Verify after build

```bash
# sanity import chain
docker run --rm easyr1-npu:<SESSION_TAG> python3 -c '
import torch, torch_npu, triton, transformers, vllm
print(torch.__version__, torch_npu.__version__, triton.__version__,
      transformers.__version__, vllm.__version__)
'
# expect: the exact versions from Phase A target-image pip-freeze, no errors
```

Any import error here → match to ERROR_CORRECTIONS.md or PLATFORM_BUGS.md.
Don't advance to P4 until clean import.

## Related evidence

Drill branch `ascend-port-transformers-upgrade` commits:
- `df84212` initial Dockerfile.npu-852 + relax transformers pin
- `318925f` pin CN pip mirror
- `16051e2` aliyun triton-ascend + base-image notes
- `15f9450` NPU-BUG-004: prune upstream triton amd/nvidia backends
- `a18d1f8` NPU-BUG-004 fix: locate triton via importlib

V1.4 on v2 image after all of the above: step-1 entropy_loss=1.275 in v2
band [1.21, 1.34] (SMOKE_BASELINE.md).
