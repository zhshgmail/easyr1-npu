#!/usr/bin/env python3
"""R-KA-16 minimal reproducer — sparse_mla_fwd online-softmax NaN at NS>=2
on Ascend bishengir HIVM backend.

Run inside the tlrescue (or any tilelang-mlir-ascend dev) container:
    python repro_rka16.py
"""
import os
os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[-2, -1],
    target="npuir",
    pass_configs={"npuir.enable_auto_multi_buffer": False},
)
def sparse_mla_fwd_minimal(batch, seq, skv, h, d, dt, topk, block_M, block_N):
    DQK = d + dt
    dtype = "float16"
    accum_dtype = "float32"
    idx_dtype = "int32"
    sm_scale = (1.0 / DQK) ** 0.5

    @T.prim_func
    def main(
        Q: T.Tensor([batch, seq, h, DQK], dtype),
        KV: T.Tensor([batch, skv, 1, DQK], dtype),
        Indices: T.Tensor([batch, seq, 1, topk], idx_dtype),
        Output: T.Tensor([batch, seq, h, d], dtype),
        Lse: T.Tensor([batch, seq, h, 1], accum_dtype),
    ):
        with T.Kernel(batch * seq, is_npu=True) as (cid, _):
            Q_shared = T.alloc_shared([block_M, d], dtype)
            Q_tail_shared = T.alloc_shared([block_M, dt], dtype)
            KV_shared = T.alloc_shared([block_N, d], dtype)
            K_tail_shared = T.alloc_shared([block_N, dt], dtype)
            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_cast = T.alloc_fragment([block_M, block_N], dtype)
            correction = T.alloc_fragment([block_M, 1], accum_dtype)
            local_max = T.alloc_fragment([block_M, 1], accum_dtype)
            local_sum = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_m = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_l = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_o = T.alloc_fragment([block_M, d], accum_dtype)
            tmp = T.alloc_fragment([block_M, block_N], accum_dtype)
            tmp1 = T.alloc_fragment([block_M, 1], accum_dtype)
            new_max = T.alloc_fragment([block_M, 1], accum_dtype)
            scales = T.alloc_fragment([block_M, block_N], accum_dtype)
            idx_buf = T.alloc_fragment([block_N], idx_dtype)

            zero_val = 0
            min_val = -T.infinity(accum_dtype)
            scale_val = sm_scale  # bind to local Var (R-KA-5 trap: bare scalar fails rank check)
            T.vbrc(zero_val, acc_o)
            T.vbrc(zero_val, acc_l)
            T.vbrc(min_val, acc_m)
            T.vbrc(scale_val, scales)
            T.copy(Q[0, 0, 0:block_M, 0:d], Q_shared)
            T.copy(Q[0, 0, 0:block_M, d:d + dt], Q_tail_shared)

            for k in T.Pipelined(T.ceildiv(topk, block_N), num_stages=1):
                T.copy(Indices[0, 0, 0, k * block_N], idx_buf)
                for bi_i in T.serial(block_N):
                    cur = idx_buf[bi_i]
                    T.copy(KV[0, cur, 0, 0:d], KV_shared[bi_i, 0:d])
                    T.copy(KV[0, cur, 0, d:d + dt], K_tail_shared[bi_i, 0:dt])

                T.gemm(Q_shared, KV_shared, scores, initC=True, b_transpose=True)
                T.gemm(Q_tail_shared, K_tail_shared, scores, initC=False, b_transpose=True)
                T.vmul(scores, scales, scores)
                T.reduce_max(scores, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max, tmp1)
                T.vexp(tmp1, correction)
                T.vsub(scores, new_max, tmp)
                T.vexp(tmp, scores)
                T.reduce_sum(scores, local_sum, dim=1)
                T.vmul(acc_l, correction, acc_l)
                T.vadd(acc_l, local_sum, acc_l)
                # === BUG SURFACE: cross-iter broadcast vmul on acc_o ===
                T.vmul(acc_o, correction, acc_o)
                T.vcast(scores, scores_cast, round_mode="rint")
                T.vbrc(zero_val, tmp1)
                T.vadd(tmp1, new_max, acc_m)
                T.gemm(scores_cast, KV_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_M, d], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            T.copy(O_cast, Output[0, 0, 0:block_M, 0:d])
            Lse_shared = T.alloc_shared([block_M, 1], accum_dtype)
            tmp_lse = T.alloc_fragment([block_M, 1], accum_dtype)
            T.vln(acc_l, tmp_lse)
            T.vadd(tmp_lse, acc_m, tmp_lse)
            T.copy(tmp_lse, Lse_shared)
            T.copy(Lse_shared, Lse[0, 0, 0:block_M, 0:1])

    return main


def main():
    torch.npu.set_device(0)
    h, d, dt = 64, 64, 16
    DQK = d + dt
    block_M, block_N = 64, 64
    print(f"R-KA-16 minimal repro — h={h} d={d} dt={dt} block_M={block_M} block_N={block_N}")
    print("Expected: all NaN ratios should be 0%%. Actual: NaN appears at NS>=2.")
    print()

    for NS_target in [1, 2, 4, 8]:
        skv = block_N * NS_target
        topk = skv
        kernel = sparse_mla_fwd_minimal(
            batch=1, seq=1, skv=skv, h=h, d=d, dt=dt, topk=topk,
            block_M=block_M, block_N=block_N,
        )
        torch.manual_seed(0)
        q = (torch.randn(1, 1, h, DQK) * 0.5).to(torch.float16).npu().contiguous()
        kv = (torch.randn(1, skv, 1, DQK) * 0.5).to(torch.float16).npu().contiguous()
        idx = torch.randint(0, skv, (1, 1, 1, topk), dtype=torch.int32).npu()

        # Two runs for run-to-run determinism check
        out1, _ = kernel(q, kv, idx)
        out2, _ = kernel(q, kv, idx)
        nan1 = (~torch.isfinite(out1)).sum().item()
        nan2 = (~torch.isfinite(out2)).sum().item()
        tot = out1.numel()
        same = (torch.isfinite(out1) == torch.isfinite(out2)).float().mean().item()
        print(f"NS={NS_target:2d} (skv={skv}, topk={topk}): "
              f"run1 nan={nan1:4d}/{tot} ({100*nan1/tot:5.1f}%)  "
              f"run2 nan={nan2:4d}/{tot} ({100*nan2/tot:5.1f}%)  "
              f"same_finite_pattern={100*same:.1f}%")


if __name__ == "__main__":
    main()
