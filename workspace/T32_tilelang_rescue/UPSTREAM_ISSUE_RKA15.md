# [BUG] tilelang `T.atomic_addx4` with all-zero src buffer writes 6e37 magic value instead of being a no-op

## Summary

When `T.atomic_addx4(dst, src, size=[4])` is called with `src` being an all-zero buffer, the destination is updated with the magic floating-point value **60405883577979550923416661670064291840.0** (≈6.04e37) per element, instead of remaining unchanged (which would be the correct no-op behavior since adding 0 to anything should leave it unchanged).

The garbage value is **deterministic**: across multiple runs with `q = torch.zeros(...)` or `q = torch.ones(...) * -0.1`, the produced dKV values are the same bit pattern `60405883577979550923416661670064291840`. So this is not RNG / scratchpad uninitialized memory — the kernel is doing some specific computation that misbehaves on all-zero source.

## Environment

* `bishengir-compile version 19.1.7` built from `Ascend/AscendNPU-IR` master HEAD `31f690369d1247fbd5529a3f88b758f7d470ae4f` (2026-05-18)
* CANN 8.5.1 + torch_npu 2.1.0
* `tile-ai/tilelang-mlir-ascend` master + R-KA-9-fixed bwd kernel
* Ascend A3 (910C / V220 dual-die, npu_arch=2201)

## Reproducer

Full reproducer in this commit on our fork: <https://github.com/zhshgmail/tilelang-mlir-ascend/blob/t33-sparse-mla-fwd-port-and-tdynamic/_p2_dk_minimal4.py>

Minimal extract:

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(target="npuir", pass_configs={"npuir.enable_auto_multi_buffer": False})
def kernel():
    @T.prim_func
    def f(
        src: T.Tensor([4], "float32"),
        dst: T.Tensor([16, 4], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, _u):
            zero_buf = T.alloc_shared([4], "float32")
            value_zero = 0.0
            T.vbrc(value_zero, zero_buf)
            T.atomic_addx4(dst[0, 0], zero_buf[0], size=[4])

    return f
```

When called:
- expected: `dst[0, 0:4]` should be unchanged (e.g. zeros after `torch.zeros(16, 4)`)
- actual: `dst[0, 0:4]` becomes ~`6e37` magnitude

## Full diagnostic chain

See <https://github.com/zhshgmail/tilelang-mlir-ascend/commit/d8527d8> commit message for the 4-stage bisect that isolated this. Key facts:

| Q input | dk max abs |
|---------|------------|
| `all zeros` | **6.04e37 EXPLODES** |
| `all 0.1` | 0.011 (correct) |
| `all -0.1` | **6.04e37 EXPLODES (same magic value as zeros)** |
| `randn` (any seed) | reasonable values |

The Q value distribution affects whether the kernel internal `scores_relu` is all-zero. Q=0 or Q=all-negative → `Q @ K^T ≤ 0` → `relu(scores) = 0` → `gated = grad * w * 0 = 0` → `d_k = gated @ Q = 0` → atomic_addx4 should be a no-op but isn't.

## Workaround currently in use

Wrapper-side: skip kernel call when `grad_scores.abs().max() < 1e-30` (i.e. when we know the kernel would compute all-zero gradients anyway). This works around the bug but doesn't protect against real training scenarios where some queries naturally produce all-negative scores.

## Severity

Medium. Real training data with mixed-sign embeddings rarely triggers, but downstream training stability is risked when (a) early-training Q embeddings have specific initializations, (b) certain attention layers have all-suppressed neighborhoods, or (c) any topk-style sparse-grad path produces empty per-query selections.

## Asks

1. Confirm reproducer fires for you locally.
2. Identify whether this is in `bishengir-compile`'s lowering of `tl.npuir_atomic_add` op, or in the downstream `bisheng` clang code generator for AICore atomic instructions.
3. Fix or provide a kernel-side annotation we can use to guarantee no-op behavior.
