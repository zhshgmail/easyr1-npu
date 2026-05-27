# [BUG] sglang glm5-poc Ascend image ships incompatible triton (3.6.0) + triton-ascend (3.2.0) — torch_npu inference path unusable

> **FILED 2026-05-27**: <https://github.com/triton-lang/triton-ascend/issues/277> (overlaps with #234 on root cause)

## Summary

The published GLM-5 inference image

```
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-glm5
```

(image id `14ba6fb8385b`, 14.4 GB, built ~3 months ago) installs:

* `triton == 3.6.0` (vanilla wheel, only `amd` + `nvidia` backends)
* `triton-ascend == 3.2.0`

These two are ABI-incompatible. The triton-ascend `backends/ascend/compiler.py` imports
`AttrsDescriptor` from `triton.backends.compiler`, which no longer exists in triton 3.6.0.
Result: `triton.backends.backends` contains only `{'amd', 'nvidia'}` and `driver.active`
raises `RuntimeError: 0 active drivers ([])`.

As soon as sglang dispatches any triton kernel on NPU (e.g. the per-request
`alloc_extend_kernel` in `sglang/srt/hardware_backend/npu/allocator_npu.py`),
the request fails and the scheduler dies.

## Reproducer

```bash
docker run --rm \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  --entrypoint=/bin/bash swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-glm5 \
  -c '
python -c "
import triton
print(triton.__version__)                    # 3.6.0
print(list(triton.backends.backends))        # [amd, nvidia]
from triton.runtime import driver
driver.active                                # RuntimeError: 0 active drivers
"
'
```

Then attempt any sglang request → scheduler dies at `extract_slice` or `AttrsDescriptor`.

## Local workaround that partially works

`pip install triton==3.2.0` makes the ascend backend register (`driver.active = NPUDriver`
and backends list includes `ascend`). But sglang's own triton kernels (e.g.
`alloc_extend_kernel`) then trip:

```
AttributeError: module 'triton.language' has no attribute 'extract_slice'
```

…because `extract_slice` is a triton-ascend extension that exists on the
`release/3.6.x` branch of `Ascend/triton-ascend` but not on the published `v3.2.0`
wheel paired with vanilla triton 3.2.0.

## Asks

1. Publish a `triton-ascend` wheel from `release/3.6.x` that matches the image's
   triton 3.6.0, OR
2. Pin the image to `triton == 3.2.x` (vanilla) **and** use the matching
   `triton-ascend 3.2.x` ascend backend that ships `extract_slice` etc., OR
3. Document the supported triton/triton-ascend pin pair in the image's `README` /
   `requirements.txt` so downstream users know to fix it before launch.

## What this blocks

Any third-party validating GLM-5 on Ascend with the official image cannot run
inference without manually rebuilding `triton-ascend` from `release/3.6.x`. The
"poc" image is currently a non-functional black box for inference.

## Env

A3 host npu-smi 26.0.rc1 (driver 26.0.rc1), CANN 8.5.0 (baked into image),
torch 2.8.0+cpu / torch_npu 2.8.0.post2, sglang `0.1.dev9598+ga59cca5c0.d20260211`.
