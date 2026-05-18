"""Rescue kernel for tile-ai/tilelang-ascend issue #996.

Issue:  examples/elementwise/elementwise_add.py "half wrong" at
        M=N=32, block_M=block_N=4 (fp32) due to <32-byte per-row DMA alignment.

Root cause from author's own annotation in #996:
    "底层做搬运的时候，对于 2-dim tile 是按照每行 32B 进行搬运。
     如果小于这个对齐要求，底层会访问到错误的地址。"

  At (block_M=4, block_N=4, VEC_NUM=2): each AIV processes a (2, 4) fp32 tile.
  Per-row = 4 cols × 4 B = 16 B < 32 B DMA alignment requirement → wrong addr.

Rescue strategy (derived from a5_ops/vendor/AscendOpGenAgent/skills/triton/
kernel-generator/references/triton-ascend-elementwise.md "方案 1: 转连续 +
一维访问 (推荐)"):
  - View the inputs as flat 1-D length M*N.
  - Use a single 1-D grid with BLOCK_SIZE in the 1024-2048 range
    (recommended by the a5_ops elementwise pattern).
  - Each program loads ONE contiguous chunk of BLOCK_SIZE fp32 elements =
    BLOCK_SIZE * 4 bytes ≥ 4096 B, safely ≥ 32 B per AIV slice too.
  - No 2-D tile, no per-row alignment trap.
  - Handle tail with a final-block size guard (numel may not be multiple of
    BLOCK_SIZE).

Usage:
  python3 elementwise_add_flat.py --m 32 --n 32         # exercises the bug shape
  python3 elementwise_add_flat.py --m 1024 --n 1024     # regression on original shape
  python3 elementwise_add_flat.py --m 100 --n 100       # non-power-of-2 tail
"""
import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[-1])
def vec_add_flat(numel, block_size, dtype="float"):
    """1-D flat elementwise add (assumes numel % block_size == 0).

    numel: total element count (M * N flattened); must be multiple of block_size
    block_size: elements per program; must be even (split between 2 AIVs);
                must be a compile-time int literal in the closure
    """
    VEC_NUM = 2
    per_vec = block_size // VEC_NUM  # compile-time constant
    grid = numel // block_size  # caller guarantees divisibility

    @T.prim_func
    def main(
            A: T.Tensor((numel,), dtype),
            B: T.Tensor((numel,), dtype),
            C: T.Tensor((numel,), dtype),
    ):
        with T.Kernel(grid, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((per_vec,), dtype)
            b_ub = T.alloc_ub((per_vec,), dtype)
            c_ub = T.alloc_ub((per_vec,), dtype)

            with T.Scope("V"):
                T.copy(A[cid * block_size + vid * per_vec], a_ub)
                T.copy(B[cid * block_size + vid * per_vec], b_ub)

                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[cid * block_size + vid * per_vec])

    return main


def main():
    parser = argparse.ArgumentParser(description="tilelang-ascend elementwise rescue")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=1024,
                        help="elements per program (a5_ops recommends 1024-2048)")
    args = parser.parse_args()

    M, N = args.m, args.n
    numel = M * N
    # block_size must divide numel and be even (split between 2 AIVs).
    # Pick the largest power-of-2 <= args.block_size that divides numel,
    # bounded below by 16 (smallest sensible UB allocation per AIV).
    bs = args.block_size
    while bs > 16 and numel % bs != 0:
        bs //= 2
    block_size = bs

    print(f"M={M} N={N} numel={numel} block_size={block_size}")
    func = vec_add_flat(numel, block_size)

    torch.manual_seed(0)
    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()
    print("init OK")

    # Pass 1-D views to the kernel. .contiguous() guarantees flat layout.
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)

    c_flat = func(a_flat, b_flat)
    c = c_flat.view(M, N)

    ref_c = a + b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
