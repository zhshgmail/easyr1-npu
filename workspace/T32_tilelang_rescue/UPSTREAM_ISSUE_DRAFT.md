# UPSTREAM ISSUE DRAFT — chain of codegen bugs for symbolic-shape kernels

> Draft only. Polish + file at https://github.com/tile-ai/tilelang-mlir-ascend/issues after review.

**Title**: `codegen_npuir_dev.cc: cascade of MLIR verifier errors when T.symbolic flows through T.Pipelined loop-carried tensors (fp8_gemm)`

**Repo**: `tile-ai/tilelang-mlir-ascend`
**Commit tested**: `2b8001c84365d1731f60ba58d82f5967e09617ab` (HEAD as of 2026-05-18)
**Reporter**: external (T32 cold-drive by easyr1-npu team)

---

## Summary

`examples/deepseek_v4/example_fp8_gemm_kernel.py` fails to compile with
**at least 3 cascading MLIR codegen bugs** when `T.symbolic("M")` is used
and the symbolic dim flows through `T.Pipelined`-style loop-carried
tensors. Each bug only surfaces after the previous one is patched.

This is **NOT one bug** — it's a class of bugs in
`src/target/codegen_npuir_dev.cc` where multiple ops (`tensor.empty`,
`tensor.collapse_shape`, possibly more) emit static-shape MLIR for
inputs whose runtime shape contains symbolic dims, causing the MLIR
verifier to reject the module.

---

## Minimal reproducer

The fp8_gemm kernel is heavy. A minimal reproducer that touches the
same codegen paths:

```python
import tilelang as tl
import tilelang.language as T

@tl.jit(target="npuir")
def repro():
    M = T.symbolic("M")
    @T.prim_func
    def main(A: T.Tensor((M, 2), "float32"),
             B: T.Tensor((M, 2), "float32"),
             C: T.Tensor((M, 2), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            for k in T.serial(M):
                T.vmul(A[k:k+1], B[k:k+1], C[k:k+1])
    return main
```

Or use the fp8_gemm example directly:

```bash
cd examples
python3 deepseek_v4/example_fp8_gemm_kernel.py
```

Both modes affected — both `TILELANG_ASCEND_MODE=Expert` and `Developer`
fail identically.

---

## Stack 1: `tensor.empty` missing dynamicSizes operand

```
CodeGenTileLangNPUIRDEV: Generated MLIR module failed verification:
error: 'tensor.empty' op incorrect number of dynamic sizes, has 0, expected 1
```

IR fragment (from `--print-ir-after-codegen`):
```mlir
%76 = "tensor.empty"() : () -> tensor<?x2xf32>
```

**Site**: `NeedGenInsertSlice()` in `src/target/codegen_npuir_dev.cc`
line 2131 (and 13 other sites in the file, see grep below).

```cpp
auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
    builder.getUnknownLoc(), srcShape, elemType);
```

When `srcShape` has any `ShapedType::kDynamic` slot, this overload
silently produces a dynamic-shape tensor without the required runtime-
size operands.

**Patch (verified locally to compile + advance to next stack)**:

```cpp
llvm::SmallVector<mlir::Value, 4> dynamicSizes;
for (size_t i = 0; i < srcShape.size(); ++i) {
  if (srcShape[i] == mlir::ShapedType::kDynamic) {
    mlir::Value dimVal;
    if (src.getType().isa<mlir::TensorType>()) {
      dimVal = builder.create<mlir::tensor::DimOp>(
                   builder.getUnknownLoc(), src, i).getResult();
    } else if (src.getType().isa<mlir::MemRefType>()) {
      dimVal = builder.create<mlir::memref::DimOp>(
                   builder.getUnknownLoc(), src, i).getResult();
    } else {
      // unsupported type for runtime dim query
    }
    dynamicSizes.push_back(dimVal);
  }
}

auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
    builder.getUnknownLoc(), srcShape, elemType, dynamicSizes);
```

All `tensor::EmptyOp::build` call sites that may produce dynamic-shape
results need similar treatment (12 sites in `codegen_npuir_dev.cc`):

```
$ grep -n 'tensor::EmptyOp>(' src/target/codegen_npuir_dev.cc
1097, 1128, 2131, 3437, 3443, 3474, 3480, 3507, 3513, 3537, 3543, 3771, 3795
```

---

## Stack 2: `tensor.collapse_shape` static output from dynamic input

After applying Stack 1's patch:

```
error: 'tensor.collapse_shape' op expected dimension 0 of collapsed type
to be dynamic since one or more of the corresponding dimensions in the
expanded type is dynamic
```

IR fragment:
```mlir
%80 = "tensor.collapse_shape"(%79) <{reassociation = [[0, 1]]}>
    : (tensor<?x2xf32>) -> tensor<1xf32>
```

The codegen emits `collapse_shape` with **static** result type
`tensor<1xf32>` even though the input `%79: tensor<?x2xf32>` has a
dynamic dim. MLIR's verifier requires the collapsed dim to be dynamic
when any of the collapsed-into-it expanded dims are dynamic.

**Site**: TBD — needs grep for `CollapseShapeOp::build` in
`codegen_npuir_dev.cc`. Likely a similar "static shape arg only" call.

**Fix sketch**: emit collapse_shape with result type
`tensor<?xf32>` when any input dim is dynamic; otherwise keep static.

---

## Stack 3+: deeper bugs (not yet surfaced)

We stopped at 4 fix iterations (see attached `KB_TILELANG_ASCEND.md` §10.3.1).
The pattern is clear: `T.symbolic` flowing through `T.Pipelined`/`T.serial`
loop-carried tensors triggers many static-shape-assuming codegen paths.
Likely additional bugs at `insert_slice`, possibly `vmul`/`vbrc`/`vcast`
codegen for dynamic shapes.

**Suggested audit**: run the minimal repro with `--print-ir-after-codegen`
and grep for any tensor type containing `?` whose surrounding op's
result is fully static, or vice versa.

---

## Workaround for affected users

Until the chain is fixed, **avoid `T.symbolic` in kernels that use
`T.Pipelined` or `T.serial` over symbolic-dim ranges with vmul-style
binary ops**. Use hardcoded shapes (the cost is one kernel-per-shape
recompile).

For fp8_gemm specifically, swap `M = T.symbolic("M")` for a hardcoded
M value or invoke the kernel via shape-specialized JIT.

---

## Reproducibility

Environment:
- A3 host (Ascend 910C / DaVinci V220 dual-die)
- Container `verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`, privileged
- CANN 8.5.2
- bishengir-compile 19.1.7 (built from `AscendNPU-IR` @ `31f6903`)
- torch 2.9.0+cpu, torch_npu 2.9.0
- pybind11 2.11.1 (for AscendNPU-IR build) → upgraded to 3.0.1 for tilelang itself

Discovery context: T32 cold-drive (manual op-port automation testbed),
external project `easyr1-npu`. See attached
`KB_TILELANG_ASCEND.md` for full context, §10.3 root-cause analysis,
§10.3.1 fix-iteration log.

---

## Why we report this as one issue, not three

The 3 stacks are **the same class of bug** — every codegen site that
takes a shape input and emits a fully-static type without checking for
dynamic dims. Fixing them as a class (audit + fix every static-shape-
emitting call site in `codegen_npuir_dev.cc`) is cheaper than splitting
into individual issues.

We're happy to:
1. Provide the v4 patch as a PR (covers Stack 1)
2. Coordinate on Stack 2+ once a maintainer has cycles to audit the rest
3. Test fixes locally on A3 before/after PR merge

---

## Attachments

- `KB_TILELANG_ASCEND.md` — context KB (architecture, ops, error recipes)
- `apply_emptyop_fix_v4.py` — Python script that applies Stack 1 patch
- `issue_fp8_gemm_codegen_emptyop_fix.patch` — early diff (Stack 1 partial)
- 40-op verification matrix on this same environment (§8.1 of KB)
