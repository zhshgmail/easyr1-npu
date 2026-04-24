#!/usr/bin/env python3
"""Triton-ascend end-to-end smoke: compile+run @triton.jit vector_add on NPU.

Confirms: triton-ascend wheel built correctly + NPU codegen path works +
runtime dispatches to Ascend device. No external deps beyond torch_npu.

Pass: printed tensor elementwise equals torch reference within 1e-6.
Fail: any exception, or max-abs-diff > 1e-6.
"""

import torch
import torch_npu  # noqa: F401 — registers "npu" device
import triton
import triton.language as tl


@triton.jit
def _vec_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main() -> int:
    print(f"triton {triton.__version__} / torch {torch.__version__}")
    print(f"backends: {sorted(triton.backends.backends.keys())}")

    n = 4096
    BLOCK = 1024
    device = "npu"

    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, BLOCK),)
    _vec_add_kernel[grid](x, y, out, n, BLOCK=BLOCK)

    ref = x + y
    max_err = (out - ref).abs().max().item()
    print(f"max abs err: {max_err:.3e}")
    if max_err > 1e-6:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
