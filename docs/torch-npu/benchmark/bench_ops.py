#!/usr/bin/env python3
"""
Layer 1 benchmark: torch op numerical correctness on Ascend NPU.

Per docs/_meta/BENCHMARK-LAYERS.md L1.1.

Each test:
  - generates a fixed-seed input on CPU (fp32 reference)
  - clones input to NPU and runs op in target dtype (bf16 / fp16)
  - cast NPU result back to fp32 and compares to CPU fp32 reference
  - PASS if max abs diff <= atol AND max rel diff <= rtol

Usage (inside container with torch_npu + Ascend driver mounted):
  python3 bench_ops.py [--json /path/out.json] [--md /path/out.md]

Exit 0 if 100% PASS; non-zero otherwise.
"""
import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import torch


@dataclass
class OpResult:
    op_name: str
    shape: str
    dtype: str
    max_abs_diff: float
    max_rel_diff: float
    atol: float
    rtol: float
    passed: bool
    wall_sec: float
    error: Optional[str] = None


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


def _diff(a_npu: torch.Tensor, a_cpu: torch.Tensor) -> tuple[float, float]:
    """Return (max abs diff, max rel diff) between NPU result (cast to fp32) and CPU fp32 reference."""
    a = a_npu.detach().to("cpu", dtype=torch.float32)
    b = a_cpu.detach().to(torch.float32)
    abs_diff = (a - b).abs()
    rel_diff = abs_diff / (b.abs() + 1e-9)
    return abs_diff.max().item(), rel_diff.max().item()


def run_test(
    name: str,
    shape: tuple,
    dtype: torch.dtype,
    make_inputs_cpu: Callable[[tuple], tuple],
    run_op: Callable[..., torch.Tensor],
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> OpResult:
    """Run one op test. make_inputs_cpu returns a tuple of CPU fp32 tensors.
    run_op(tensors) -> result tensor. Called once on CPU (fp32) and once on NPU (dtype)."""
    start = time.time()
    try:
        cpu_inputs = make_inputs_cpu(shape)
        # CPU reference (fp32)
        cpu_out = run_op(*cpu_inputs)

        # NPU version: clone, cast to target dtype, move to npu
        npu_inputs = tuple(t.to(dtype=dtype, device="npu") for t in cpu_inputs)
        npu_out = run_op(*npu_inputs)

        abs_d, rel_d = _diff(npu_out, cpu_out)
        passed = (abs_d <= atol) and (rel_d <= rtol)
        elapsed = time.time() - start
        return OpResult(
            op_name=name, shape=str(shape), dtype=str(dtype).replace("torch.", ""),
            max_abs_diff=abs_d, max_rel_diff=rel_d, atol=atol, rtol=rtol,
            passed=passed, wall_sec=elapsed,
        )
    except Exception as e:
        return OpResult(
            op_name=name, shape=str(shape), dtype=str(dtype).replace("torch.", ""),
            max_abs_diff=float("nan"), max_rel_diff=float("nan"), atol=atol, rtol=rtol,
            passed=False, wall_sec=time.time() - start,
            error=f"{type(e).__name__}: {e}",
        )


def make_mat_mul_inputs(shape: tuple):
    """shape = (B, N, K, M); returns (A[B,N,K], B[K,M])"""
    B, N, K, M = shape
    torch.manual_seed(42)
    a = torch.randn(B, N, K)
    b = torch.randn(K, M)
    return a, b


def make_linear_inputs(shape: tuple):
    """shape = (B, N, in, out); returns (input, weight, bias)"""
    B, N, I, O = shape
    torch.manual_seed(42)
    x = torch.randn(B, N, I)
    w = torch.randn(O, I)
    b = torch.randn(O)
    return x, w, b


def make_layernorm_inputs(shape: tuple):
    """shape = (B, N, D); returns (input, weight, bias)"""
    B, N, D = shape
    torch.manual_seed(42)
    x = torch.randn(B, N, D)
    w = torch.ones(D)
    b = torch.zeros(D)
    return x, w, b


def make_softmax_inputs(shape: tuple):
    B, H, N1, N2 = shape
    torch.manual_seed(42)
    return (torch.randn(B, H, N1, N2),)


def make_sdpa_inputs(shape: tuple):
    """shape = (B, H, N, D); returns (q, k, v) all same shape"""
    B, H, N, D = shape
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    return q, k, v


def make_cross_entropy_inputs(shape: tuple):
    """shape = (B, N, V); returns (logits, labels)"""
    B, N, V = shape
    torch.manual_seed(42)
    logits = torch.randn(B * N, V)
    labels = torch.randint(0, V, (B * N,))
    return logits, labels


def op_matmul(a, b):
    return torch.matmul(a, b)


def op_linear(x, w, b):
    return torch.nn.functional.linear(x, w, b)


def op_layernorm(x, w, b):
    return torch.nn.functional.layer_norm(x, w.shape, weight=w, bias=b)


def op_softmax(x):
    return torch.nn.functional.softmax(x, dim=-1)


def op_sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def op_cross_entropy(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels, reduction="mean")


def op_add(a, b):
    return a + b


def op_mul(a, b):
    return a * b


def op_gelu(x):
    return torch.nn.functional.gelu(x)


def op_silu(x):
    return torch.nn.functional.silu(x)


def make_unary_inputs(shape: tuple):
    B, N, D = shape
    torch.manual_seed(42)
    return (torch.randn(B, N, D),)


def make_binary_inputs(shape: tuple):
    B, N, D = shape
    torch.manual_seed(42)
    return torch.randn(B, N, D), torch.randn(B, N, D)


TESTS = [
    # name, shape, dtype, make_inputs, op, atol, rtol
    ("matmul", (1, 32, 64, 128), torch.bfloat16, make_mat_mul_inputs, op_matmul, 0.05, 0.05),
    ("matmul", (4, 32, 64, 128), torch.bfloat16, make_mat_mul_inputs, op_matmul, 0.05, 0.05),
    ("matmul", (16, 32, 64, 128), torch.bfloat16, make_mat_mul_inputs, op_matmul, 0.05, 0.05),
    ("matmul", (1, 32, 64, 128), torch.float16, make_mat_mul_inputs, op_matmul, 0.01, 0.02),

    ("linear", (1, 32, 64, 128), torch.bfloat16, make_linear_inputs, op_linear, 0.1, 0.1),
    ("linear", (4, 32, 64, 128), torch.bfloat16, make_linear_inputs, op_linear, 0.1, 0.1),
    ("linear", (1, 32, 64, 128), torch.float16, make_linear_inputs, op_linear, 0.05, 0.05),

    ("layer_norm", (1, 32, 128), torch.bfloat16, make_layernorm_inputs, op_layernorm, 0.05, 0.05),
    ("layer_norm", (4, 32, 128), torch.bfloat16, make_layernorm_inputs, op_layernorm, 0.05, 0.05),
    ("layer_norm", (1, 32, 128), torch.float16, make_layernorm_inputs, op_layernorm, 0.02, 0.02),

    ("softmax", (1, 8, 32, 32), torch.bfloat16, make_softmax_inputs, op_softmax, 0.02, 0.05),
    ("softmax", (4, 8, 32, 32), torch.bfloat16, make_softmax_inputs, op_softmax, 0.02, 0.05),
    ("softmax", (1, 8, 32, 32), torch.float16, make_softmax_inputs, op_softmax, 0.01, 0.02),

    ("sdpa_causal", (1, 8, 32, 64), torch.bfloat16, make_sdpa_inputs, op_sdpa, 0.1, 0.1),
    ("sdpa_causal", (4, 8, 32, 64), torch.bfloat16, make_sdpa_inputs, op_sdpa, 0.1, 0.1),

    ("cross_entropy", (1, 32, 1024), torch.float32, make_cross_entropy_inputs, op_cross_entropy, 1e-3, 1e-3),
    ("cross_entropy", (4, 32, 1024), torch.float32, make_cross_entropy_inputs, op_cross_entropy, 1e-3, 1e-3),

    ("add", (1, 32, 128), torch.bfloat16, make_binary_inputs, op_add, 0.01, 0.01),
    ("mul", (1, 32, 128), torch.bfloat16, make_binary_inputs, op_mul, 0.01, 0.01),
    ("gelu", (1, 32, 128), torch.bfloat16, make_unary_inputs, op_gelu, 0.02, 0.02),
    ("silu", (1, 32, 128), torch.bfloat16, make_unary_inputs, op_silu, 0.02, 0.02),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="JSON output path")
    ap.add_argument("--md", help="Markdown output path")
    args = ap.parse_args()

    if not _npu_available():
        print("ERROR: NPU not available (torch_npu import failed or torch.npu.is_available() False)", file=sys.stderr)
        sys.exit(2)

    print(f"torch={torch.__version__}")
    import torch_npu
    print(f"torch_npu={torch_npu.__version__}")
    print(f"device count={torch.npu.device_count()}")
    print()

    results = []
    for name, shape, dtype, mk, op, atol, rtol in TESTS:
        r = run_test(name, shape, dtype, mk, op, atol, rtol)
        status = "PASS" if r.passed else "FAIL"
        err = f" [{r.error}]" if r.error else ""
        print(f"[{status}] {r.op_name} shape={r.shape} dtype={r.dtype} "
              f"abs={r.max_abs_diff:.4f} rel={r.max_rel_diff:.4f} "
              f"(atol={r.atol}, rtol={r.rtol}) {r.wall_sec:.2f}s{err}")
        results.append(asdict(r))

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print()
    print(f"===== {passed_count}/{total} PASS =====")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "total": total, "passed": passed_count,
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "results": results,
            }, f, indent=2)
        print(f"wrote {args.json}")

    if args.md:
        with open(args.md, "w") as f:
            f.write(f"# L1 torch_npu benchmark report\n\n")
            f.write(f"- torch: `{torch.__version__}`\n")
            f.write(f"- torch_npu: `{torch_npu.__version__}`\n")
            f.write(f"- device count: {torch.npu.device_count()}\n")
            f.write(f"- **{passed_count}/{total} PASS**\n\n")
            f.write("| op | shape | dtype | abs diff | rel diff | atol | rtol | status | error |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in results:
                status = "PASS" if r["passed"] else "FAIL"
                err = r["error"] if r["error"] else ""
                f.write(f"| {r['op_name']} | {r['shape']} | {r['dtype']} | "
                        f"{r['max_abs_diff']:.4f} | {r['max_rel_diff']:.4f} | "
                        f"{r['atol']} | {r['rtol']} | {status} | {err} |\n")
        print(f"wrote {args.md}")

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
