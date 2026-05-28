# [Bug-Report|缺陷反馈]: bishengir HIVM 多迭代 pipelined 循环中,online-softmax 跨迭代累加器在 NS≥4 出现 NaN(非确定性)

> **FILED 2026-05-28**: <https://gitcode.com/Ascend/AscendNPU-IR/issues/251>

### Describe the current behavior / 问题描述 (Mandatory / 必填)

A tilelang kernel that implements a standard **flash-attention-style online softmax** (cross-iter persistent accumulators `acc_o [block_M, D] fp32`, `acc_m [block_M, 1] fp32`, `acc_l [block_M, 1] fp32`) inside a `T.Pipelined` (`scf.for`) loop with `NS = ceil(topk / block_N)` iterations produces NaN output when `NS >= 4`, regardless of `num_stages`.

Quantitative measurements at block_M=64, D=64 (small enough that UB-overflow does not interfere):

| topk | block_N | NS = topk / block_N | NaN ratio (run 1) | Run-to-run determinism |
|------|---------|---------------------|-------------------|-------------------------|
| 64   | 64      | 1                   | **0%** ✅          | deterministic           |
| 128  | 64      | 2                   | ~32–82% 🐛        | varies across runs      |
| 256  | 64      | 4                   | ~67%             | varies across runs      |
| 512  | 64      | 8                   | ~73%             | varies across runs      |
| 1024 | 64      | 16                  | ~69%             | varies across runs      |

Two-run example at NS=2 / SKV=512 / topk=128 (same input tensors, same seed):

```
run 1: nan = 1344 / 4096 elements
run 2: nan = 1535 / 4096 elements  (different)
```

The non-determinism implies the failing pattern is **not** pure pass mis-compilation but interacts with NPU runtime ordering (UB layout / pipeline scheduling / sync barrier insertion).

### Environment / 环境信息 (Mandatory / 必填)

* Chip: Ascend 910C (A3, dav-c220), npu-smi 26.0.rc1
* CANN: 8.5.0 (also reproduced on 8.5.1 with hot-swapped bishengir-compile built from `Ascend/AscendNPU-IR` master HEAD `31f690369d` from 2026-05-18)
* tilelang-mlir-ascend fork: `github.com/zhshgmail/tilelang-mlir-ascend`, branch `npu-tilelang-dispatch`, commit `4cdfc1f`
* torch_npu: 2.9.0
* Container: `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`

### Steps to reproduce the issue / 重现步骤 (Mandatory / 必填)

Full kernel: <https://github.com/zhshgmail/miles/blob/npu-tilelang-dispatch/miles_plugins/models/glm5/ops/_npu/_sparse_mla_fwd_kernel.py>

Minimal Python repro (no miles, no Megatron; pure tilelang):

```python
import os
os.environ["TILELANG_ASCEND_MODE"] = "Developer"
import torch, torch_npu
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-2, -1], target="npuir",
              pass_configs={"npuir.enable_auto_multi_buffer": False})
def sparse_mla_fwd_minimal(NS_IS_4: bool = True):
    BATCH = 1
    SEQ = 1
    SKV = 256 if NS_IS_4 else 64
    H = 64
    D = 64
    DT = 16
    TOPK = SKV
    BLOCK_M = 64
    BLOCK_N = 64

    dtype = "float16"
    accum_dtype = "float32"
    idx_dtype = "int32"
    sm_scale = (1.0 / (D + DT)) ** 0.5

    @T.prim_func
    def main(
        Q: T.Tensor([BATCH, SEQ, H, D + DT], dtype),
        KV: T.Tensor([BATCH, SKV, 1, D + DT], dtype),
        Indices: T.Tensor([BATCH, SEQ, 1, TOPK], idx_dtype),
        Output: T.Tensor([BATCH, SEQ, H, D], dtype),
        Lse: T.Tensor([BATCH, SEQ, H, 1], accum_dtype),
    ):
        with T.Kernel(BATCH * SEQ, is_npu=True) as (cid, _):
            Q_shared = T.alloc_shared([BLOCK_M, D], dtype)
            Q_tail_shared = T.alloc_shared([BLOCK_M, DT], dtype)
            KV_shared = T.alloc_shared([BLOCK_N, D], dtype)
            K_tail_shared = T.alloc_shared([BLOCK_N, DT], dtype)
            scores = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            scores_cast = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
            correction = T.alloc_fragment([BLOCK_M, 1], accum_dtype)
            local_max = T.alloc_fragment([BLOCK_M, 1], accum_dtype)
            local_sum = T.alloc_fragment([BLOCK_M, 1], accum_dtype)
            acc_m = T.alloc_fragment([BLOCK_M, 1], accum_dtype)
            acc_l = T.alloc_fragment([BLOCK_M, 1], accum_dtype)
            acc_o = T.alloc_fragment([BLOCK_M, D], accum_dtype)
            tmp = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            tmp1 = T.alloc_fragment([BLOCK_M, 1], accum_dtype)
            new_max = T.alloc_fragment([BLOCK_M, 1], accum_dtype)
            scales = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            idx_buf = T.alloc_fragment([BLOCK_N], idx_dtype)

            zero_val = 0
            min_val = -T.infinity(accum_dtype)
            T.vbrc(zero_val, acc_o)
            T.vbrc(zero_val, acc_l)
            T.vbrc(min_val, acc_m)
            T.vbrc(sm_scale, scales)
            T.copy(Q[0, 0, 0:BLOCK_M, 0:D], Q_shared)
            T.copy(Q[0, 0, 0:BLOCK_M, D:D + DT], Q_tail_shared)

            for k in T.Pipelined(T.ceildiv(TOPK, BLOCK_N), num_stages=1):
                T.copy(Indices[0, 0, 0, k * BLOCK_N], idx_buf)
                for bi_i in T.serial(BLOCK_N):
                    cur = idx_buf[bi_i]
                    T.copy(KV[0, cur, 0, 0:D], KV_shared[bi_i, 0:D])
                    T.copy(KV[0, cur, 0, D:D + DT], K_tail_shared[bi_i, 0:DT])

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
                # *** The cross-iter broadcast vmul on acc_o is the bug surface ***
                T.vmul(acc_o, correction, acc_o)
                T.vcast(scores, scores_cast, round_mode="rint")
                T.vbrc(zero_val, tmp1)
                T.vadd(tmp1, new_max, acc_m)
                T.gemm(scores_cast, KV_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([BLOCK_M, D], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            T.copy(O_cast, Output[0, 0, 0:BLOCK_M, 0:D])
            Lse_shared = T.alloc_shared([BLOCK_M, 1], accum_dtype)
            tmp_lse = T.alloc_fragment([BLOCK_M, 1], accum_dtype)
            T.vln(acc_l, tmp_lse)
            T.vadd(tmp_lse, acc_m, tmp_lse)
            T.copy(tmp_lse, Lse_shared)
            T.copy(Lse_shared, Lse[0, 0, 0:BLOCK_M, 0:1])

    return main


torch.npu.set_device(0)
torch.manual_seed(0)
H, D, DT = 64, 64, 16

for NS_target in [1, 2, 4, 8]:
    SKV = 64 * NS_target
    TOPK = SKV
    k = sparse_mla_fwd_minimal()
    q = (torch.randn(1, 1, H, D + DT) * 0.5).to(torch.float16).npu().contiguous()
    kv = (torch.randn(1, SKV, 1, D + DT) * 0.5).to(torch.float16).npu().contiguous()
    idx = torch.randint(0, SKV, (1, 1, 1, TOPK), dtype=torch.int32).npu()
    out, _ = k(q, kv, idx)
    nan = (~torch.isfinite(out)).sum().item()
    print(f"NS={NS_target}: nan={nan}/{out.numel()} ({100*nan/out.numel():.1f}%)")
```

Expected output (deterministic, all finite):
```
NS=1: nan=0/4096 (0.0%)
NS=2: nan=0/4096 (0.0%)
NS=4: nan=0/4096 (0.0%)
NS=8: nan=0/4096 (0.0%)
```

Actual output (from running `repro_rka16.py`, attached):
```
NS= 1 (skv= 64, topk= 64): run1 nan=   0/4096 (0.0%)  run2 nan=   0/4096 (0.0%)  same=100.0%
NS= 2 (skv=128, topk=128): run1 nan=   0/4096 (0.0%)  run2 nan=   0/4096 (0.0%)  same=100.0%
NS= 4 (skv=256, topk=256): run1 nan=2816/4096 (68.8%) run2 nan=1664/4096 (40.6%) same= 34.4%
NS= 8 (skv=512, topk=512): run1 nan=2496/4096 (60.9%) run2 nan=3136/4096 (76.6%) same= 65.6%
```

NS=1 and NS=2 are deterministic and clean. **NS=4 and NS=8 are non-deterministic** — same input tensors, same seeds, different NaN counts and different finite/NaN patterns. Run-to-run finite-pattern overlap is only 34% at NS=4, indicating a race condition or NPU cache-coherency interaction with the pipelined loop scheduling.

### Diagnostic data so far

| Hypothesis | Test | Result |
|---|---|---|
| Software pipelining (`num_stages=2`) interaction | `num_stages=1` | Reduces NaN but does **not** eliminate at NS ≥ 4 |
| Broadcast vsub at line `T.vsub(scores, new_max, tmp)` (R-KA-13 family) | Replace with scalar-fill `new_max_expanded` | **Increases** NaN at NS=2 (73% vs 33%) — not the right intervention |
| Broadcast vmul at line `T.vmul(acc_o, correction, acc_o)` (R-KA-13 family applied to vmul) | Replace with scalar-fill `correction_expanded` | **Eliminates** NaN at NS=2 (1.6% → 0%); does **not** help at NS ≥ 4 |
| Head dim D | Vary D ∈ {64, 128, 256, 384, 512} at NS=2 | D does not affect NaN ratio when NS ≥ 2 |
| Head count H | Vary H ∈ {16, 32, 64} at NS=2 | H does not affect NaN ratio when NS ≥ 2 |
| Run-to-run determinism | Run twice with same input | Different NaN counts (1344 vs 1535) at NS=2 |

### What likely interacts

Suspect bishengir HIVM passes (from `bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp`):

1. `scf::createCanonicalizeIterArgPass` — could be elimination of `acc_o` iter_arg if some pattern misidentifies it as loop-invariant
2. `mlir::scf::createRemoveRedundantLoopInitPass`
3. `createEnableMultiBufferPass` — cross-iter buffer doubling
4. `createCreatePreloadPass` / `createCVPipeliningPass` — pipeline scheduling

The R-KA-13 (bwd vsub schedule-locality) fix in tilelang has a similar flavour: the broadcast-operand construction inside the inner pipelined iter changes the bishengir lowering result.

### Asks / 诉求

1. Confirm reproducible on Huawei test farm at NS ≥ 4. The repro script above runs in ~60 seconds and produces unambiguous output.
2. Identify which HIVM pass introduces the issue. The most efficient bisect is `bishengir-compile -print-after-all` and look for where `acc_o` either disappears from iter_args or has its initial value written wrong.
3. Provide a fix. Until upstream is fixed, downstream tilelang kernels with flash-attention-style online-softmax accumulation cannot run on Ascend NPU at any production topk (DeepSeek-V4-Flash topk=512 → NS=8; DeepSeek-V3.2 topk=2048 → NS=32).
