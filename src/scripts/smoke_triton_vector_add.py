#!/usr/bin/env python3
"""Triton-ascend NPU smoke suite — 6 representative kernel patterns.

Coverage:
  1. vector_add fp32 (aligned, n divisible by BLOCK)
  2. vector_add fp32 masked (n NOT divisible by BLOCK — exercises mask path)
  3. vector_add bf16 (low-precision elementwise)
  4. vector_add fp16 (low-precision elementwise)
  5. reduction sum along axis (1D reduce)
  6. dot 16x16x16 fp32 (matmul on 16x16 tiles)

Each kernel: compile + run + compare to torch reference; max abs err
threshold per kernel. Final output: PASS N/6 with per-kernel result.
"""

import sys
import torch
import torch_npu  # noqa: F401 — registers "npu" device
import triton
import triton.language as tl


# Kernel 1 + 2: vector_add (fp32)
@triton.jit
def _vec_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


# Kernel 3 + 4: vector_add lowprec (bf16 / fp16)
@triton.jit
def _vec_add_lowprec_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


# Kernel 5: reduction sum along axis 1
@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n_cols, BLOCK_COLS: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = row * n_cols + tl.arange(0, BLOCK_COLS)
    mask = tl.arange(0, BLOCK_COLS) < n_cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + row, s)


# Kernel 6: dot 16x16x16 fp32
@triton.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr,
                M, N, K,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.dot(a, b)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def run_kernel(name, fn, ref_fn, atol, rtol=0.0):
    """Returns (passed: bool, max_err: float, msg: str)."""
    try:
        out, ref = fn()
        max_err = (out.float() - ref.float()).abs().max().item()
        ok = max_err <= atol + rtol * ref.float().abs().max().item()
        return ok, max_err, ""
    except Exception as e:
        return False, float("inf"), f"{type(e).__name__}: {e}"


def k1_vec_add_aligned():
    n, BLOCK = 4096, 1024
    x = torch.randn(n, device="npu", dtype=torch.float32)
    y = torch.randn(n, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    _vec_add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    return out, x + y


def k2_vec_add_masked():
    n, BLOCK = 4097, 1024  # not divisible -> last block masked
    x = torch.randn(n, device="npu", dtype=torch.float32)
    y = torch.randn(n, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    _vec_add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    return out, x + y


def k3_vec_add_bf16():
    n, BLOCK = 4096, 1024
    x = torch.randn(n, device="npu", dtype=torch.bfloat16)
    y = torch.randn(n, device="npu", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    _vec_add_lowprec_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    return out, x + y


def k4_vec_add_fp16():
    n, BLOCK = 4096, 1024
    x = torch.randn(n, device="npu", dtype=torch.float16)
    y = torch.randn(n, device="npu", dtype=torch.float16)
    out = torch.empty_like(x)
    _vec_add_lowprec_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    return out, x + y


def k5_reduction_sum():
    rows, n_cols, BLOCK_COLS = 32, 256, 256
    x = torch.randn(rows, n_cols, device="npu", dtype=torch.float32)
    out = torch.empty(rows, device="npu", dtype=torch.float32)
    _reduce_sum_kernel[(rows,)](x, out, n_cols, BLOCK_COLS=BLOCK_COLS)
    return out, x.sum(dim=1)


def k6_dot_16x16x16():
    M, N, K = 16, 16, 16
    a = torch.randn(M, K, device="npu", dtype=torch.float32)
    b = torch.randn(K, N, device="npu", dtype=torch.float32)
    c = torch.empty(M, N, device="npu", dtype=torch.float32)
    _dot_kernel[(1, 1)](a, b, c, M, N, K,
                        BLOCK_M=M, BLOCK_N=N, BLOCK_K=K)
    return c, a @ b


KERNELS = [
    ("k1_vec_add_aligned", k1_vec_add_aligned, 1e-6),
    ("k2_vec_add_masked", k2_vec_add_masked, 1e-6),
    ("k3_vec_add_bf16", k3_vec_add_bf16, 1e-2),
    ("k4_vec_add_fp16", k4_vec_add_fp16, 1e-2),
    ("k5_reduction_sum", k5_reduction_sum, 1e-3),
    ("k6_dot_16x16x16", k6_dot_16x16x16, 1e-3),
]


def main() -> int:
    print(f"triton {triton.__version__} / torch {torch.__version__}")
    backends = sorted(triton.backends.backends.keys()) if hasattr(triton.backends, "backends") else []
    print(f"backends: {backends}")
    print()

    results = []
    for name, fn, atol in KERNELS:
        ok, err, msg = run_kernel(name, fn, None, atol)
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {name}: max_abs_err={err:.3e} (atol={atol:.0e})"
        if msg:
            line += f"  -- {msg}"
        print(line)
        results.append(ok)

    passed = sum(results)
    total = len(results)
    print()
    print(f"=== {passed}/{total} PASS ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
