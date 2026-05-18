#!/usr/bin/env python3
"""Fix: when src0 is scalar and src1 is tensor in CreateHIVMBinaryVectorOp,
swap them BEFORE constructing the vmul/vadd op. The bishengir verifier
has VectorOnlyTrait<0> meaning operand 0 must be a vector.

For commutative ops (Add, Mul, Max, Min, Or, And, Xor) the swap is
mathematically safe. For non-commutative (Sub, Div, Pow, Shl, Shr, Cmp,
ShR) we cannot swap; fall through to existing path which will error if
unsupported.

Triggered by 10 test failures in testing/npuir/arith_ops:
- test_tensor_extract_dev.py (8 fails: add/mul/sub/div × fp16/fp32)
- test_vadd_1x1_dev.py (2 fails: vadd_1x1 × fp16/fp32)

Error: 'hivm.hir.vadd' op failed to verify that operand at index 0 is
vector-only (was f32 scalar after our F1/v6/v9 patches let scalars
through).
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
t = CODEGEN.read_text()

# Find the v9 scalar-scalar dispatch we added; insert a new swap-when-
# only-src0-is-scalar branch just BEFORE the scalar-scalar dispatch.
# The dispatch block starts with the v9 comment.

OLD = """  // T32 v6+F1+scalar-dispatch: if both src0/src1 are scalars (came from
  // per-element loads via is_scalar_load path), emit arith.mulf instead
  // of vmul. The dst is also a single-element write to the destination
  // buffer."""

NEW = """  // T32 vmul-scalar-swap: if only src0 is scalar but src1 is tensor,
  // swap them (for commutative ops) so operand 0 is vector. The
  // bishengir VAddOp/VMulOp/VMaxOp/etc all have VectorOnlyTrait<0>
  // requiring operand 0 to be vector.
  if (!src0.getType().isa<mlir::TensorType>() &&
      src1.getType().isa<mlir::TensorType>()) {
    constexpr bool is_commutative =
        std::is_same_v<T, mlir::hivm::VAddOp> ||
        std::is_same_v<T, mlir::hivm::VMulOp> ||
        std::is_same_v<T, mlir::hivm::VMaxOp> ||
        std::is_same_v<T, mlir::hivm::VMinOp> ||
        std::is_same_v<T, mlir::hivm::VOrOp> ||
        std::is_same_v<T, mlir::hivm::VAndOp> ||
        std::is_same_v<T, mlir::hivm::VXorOp>;
    if constexpr (is_commutative) {
      std::swap(src0, src1);
      std::swap(buffer_shape0, buffer_shape1);
    }
  }

  // T32 v6+F1+scalar-dispatch: if both src0/src1 are scalars (came from
  // per-element loads via is_scalar_load path), emit arith.mulf instead
  // of vmul. The dst is also a single-element write to the destination
  // buffer."""

count = t.count(OLD)
print(f"Found {count} matches (expect 1)")
if count == 0:
    exit(1)
t = t.replace(OLD, NEW)
CODEGEN.write_text(t)
print("vmul-scalar-swap applied")
