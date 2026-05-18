"""Workaround: fp8_gemm with static M (no T.symbolic).

T32.11+ chain-fix kept hitting deeper codegen bugs. Test: does the kernel
work if we drop T.symbolic and hardcode M? This isolates whether the
GEMM algorithm itself is OK on MLIR vs the symbolic-shape codegen path
is the bug surface.
"""
import os
import torch
import tilelang as tl
import tilelang.language as T


def _gen_fp8_e4m3_like_tensor(shape, device: torch.device) -> torch.Tensor:
    x = torch.randn(shape, dtype=torch.float32) * 96.0
    x = torch.clamp(x, -448.0, 448.0)
    if hasattr(torch, "float8_e4m3fn"):
        x = x.to(torch.float8_e4m3fn).to(torch.float16)
    else:
        x = x.to(torch.float16)
    return x.to(device).contiguous()


def _ceildiv(a, b):
    return (a + b - 1) // b


@tl.jit(target="npuir")
def fp8_gemm_static(
    M, N, K, out_dtype="float16", in_dtype="float16", accum_dtype="float32"
):
    """Same as fp8_gemm_kernel but M is a Python int, not T.symbolic."""
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    assert M % block_M == 0
    assert N % block_N == 0
    assert K % block_K == 0

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
        scales_a: T.Tensor((M, T.ceildiv(K, group_size)), "float32"),
        scales_b: T.Tensor(
            (T.ceildiv(N, group_size), T.ceildiv(K, group_size)), "float32"
        ),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True
        ) as (cid, _):
            bx = cid % T.ceildiv(N, block_N)
            by = cid // T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            Scale_C_shared = T.alloc_shared((block_M), "float32")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local_accum)
            k_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(k_iters, num_stages=2):
                k_start = k * block_K
                T.copy(
                    A[by * block_M : (by + 1) * block_M, k_start : k_start + block_K],
                    A_shared,
                )
                T.copy(
                    B[bx * block_N : (bx + 1) * block_N, k_start : k_start + block_K],
                    B_shared,
                )
                scale_b = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * scale_b
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, init=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] = (
                        C_local_accum[i, j] + C_local[i, j] * Scale_C_shared[i]
                    )
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_local_accum[i, j].astype(
                    out_dtype
                )

    return fp8_gemm_kernel_


def run_test_case_static(m, n, k, out_dtype="float16"):
    group_size = 128
    k_groups = _ceildiv(k, group_size)
    n_groups = _ceildiv(n, group_size)
    npu_device = torch.device("npu")
    a = _gen_fp8_e4m3_like_tensor((m, k), npu_device)
    b = _gen_fp8_e4m3_like_tensor((n, k), npu_device)
    a_s = (torch.rand((m, k_groups), dtype=torch.float32).npu() * 0.2 + 0.9).contiguous()
    b_s = (torch.rand((n_groups, k_groups), dtype=torch.float32).npu() * 0.2 + 0.9).contiguous()

    kernel = fp8_gemm_static(m, n, k, out_dtype=out_dtype)
    print(f"Static-M kernel compiled for ({m},{n},{k}); running...")

    out = kernel(a, a_s, b, b_s)
    # CPU reference: a (m, k) fp16 quantized * scales, b (n, k) fp16 quantized * scales
    a_dq = a.to(torch.float32)
    b_dq = b.to(torch.float32)
    # Reproduce same dequant as the kernel
    a_block = a_dq.reshape(m, k_groups, group_size) * a_s.unsqueeze(-1)
    a_block = a_block.reshape(m, k)
    b_block = b_dq.reshape(n, k_groups, group_size) * b_s.reshape(n_groups, k_groups, 1).repeat_interleave(group_size, dim=-1)
    b_block = b_block.reshape(n, k)
    ref = a_block @ b_block.t()
    if out_dtype != "float32":
        ref = ref.to({"float16": torch.float16, "bfloat16": torch.bfloat16}[out_dtype])

    print(f"  out[:2,:4] = {out[:2,:4].cpu()}")
    print(f"  ref[:2,:4] = {ref[:2,:4].cpu()}")
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    print(f"  PASS at ({m},{n},{k}) {out_dtype}")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.npu.set_device(0)
    tl.cache.clear_cache()
    run_test_case_static(m=128, n=256, k=256, out_dtype="float16")
    print("\033[92mFP8 GEMM static-M test passed.\033[0m")
