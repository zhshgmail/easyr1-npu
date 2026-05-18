"""Reproducer for tile-ai/tilelang-ascend issue #996.

Same kernel as upstream examples/elementwise/elementwise_add.py, just with
block_M/block_N exposed as CLI flags so we can hit the small-block case.

Expected: at --m 32 --n 32 --block-m 4 --block-n 4 (fp32), the assertion
fails with ~half wrong elements per #996.
"""
import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--block-m", type=int, default=4)
    parser.add_argument("--block-n", type=int, default=4)
    args = parser.parse_args()
    M, N, bm, bn = args.m, args.n, args.block_m, args.block_n
    print(f"M={M} N={N} block_M={bm} block_N={bn} "
          f"per_row_bytes_per_aiv={bn * 4}")

    func = vec_add(M, N, bm, bn)
    torch.manual_seed(0)
    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()
    torch.npu.synchronize()
    print("init OK")
    c = func(a, b)
    ref_c = a + b
    try:
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
        print(f"Match at ({M},{N})/({bm},{bn})")
    except AssertionError as e:
        diff = (c - ref_c).abs()
        wrong = (diff > 1e-2).sum().item()
        total = M * N
        print(f"MISMATCH at ({M},{N})/({bm},{bn}): {wrong}/{total} "
              f"({100*wrong/total:.1f}%) elements differ; "
              f"max_abs_diff={diff.max().item():.4g}")
        # print first few diffs
        flat_c = c.flatten().cpu()
        flat_r = ref_c.flatten().cpu()
        for i in range(min(8, total)):
            print(f"  [{i}] got={flat_c[i].item():.4f} expected={flat_r[i].item():.4f}")


if __name__ == "__main__":
    main()
