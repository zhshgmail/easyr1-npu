#!/usr/bin/env python3
"""Dump the npuir tilelang generates for the R-KA-16 minimal repro at NS=4,
without actually invoking bishengir-compile (we'll run it separately with
--print-after-all).

Output: /tmp/rka16_ns4.npuir
"""
import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

NPUIR_OUT = "/tmp/rka16_ns4.npuir"


def _patched_npuir_to_bin(self):
    """Replacement for JITKernel._npuir_to_bin_enable_npu_compile.

    Writes self.mlir_content to a known path and returns a stub so the
    rest of JIT init doesn't crash before we can exit.
    """
    with open(NPUIR_OUT, "w") as f:
        f.write(self.mlir_content)
    print(f"[dump_rka16_ir] wrote npuir to {NPUIR_OUT} ({len(self.mlir_content)} bytes)")
    # Raise so we abort before generating .o / .so we don't need.
    raise SystemExit(0)


def main():
    # Monkey-patch BEFORE we instantiate the kernel
    import tilelang
    import tilelang.language as T
    from tilelang.jit import jit_npu

    jit_npu.compiler_npu._npuir_to_bin_enable_npu_compile = _patched_npuir_to_bin

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
                scale_val = sm_scale
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

    h, d, dt = 64, 64, 16
    block_M, block_N = 64, 64
    NS_target = 4  # the failing case per R-KA-16
    skv = block_N * NS_target
    topk = skv

    print(f"[dump_rka16_ir] building NS={NS_target} kernel (skv={skv}, topk={topk})...")
    try:
        sparse_mla_fwd_minimal(
            batch=1, seq=1, skv=skv, h=h, d=d, dt=dt, topk=topk,
            block_M=block_M, block_N=block_N,
        )
    except SystemExit as e:
        # Expected — we patched _npuir_to_bin to raise after dumping.
        sys.exit(int(e.code) if e.code is not None else 0)
    print("[dump_rka16_ir] unexpected: kernel built without raising")
    sys.exit(1)


if __name__ == "__main__":
    main()
