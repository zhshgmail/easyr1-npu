#!/usr/bin/env python3
"""Fix for Stack 2: tensor.collapse_shape static output from dynamic input.

In ReshapeTensorImpl, the srcRank > dstRank branch builds dstTensorTy from
dstShapeStatic — if src has dynamic dims that get collapsed into a dst dim,
that dst dim's static size is wrong (or kDynamic, but only if any of the
input dims to that group were dynamic).

The CORRECT dst static shape: for each dst dim group, if ANY input dim in
that group is dynamic, the dst dim must be kDynamic too. Otherwise, the
product of the static input dims.
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
text = CODEGEN.read_text()

OLD = """  if (srcRank > dstRank) {
    // Try collapse_shape: [1, 1, M, N] -> [M, N]
    // Reassociation maps each dst dim to a group of src dims
    llvm::SmallVector<mlir::ReassociationIndices> reassoc;
    int64_t extraDims = srcRank - dstRank;

    for (int64_t dstIdx = 0; dstIdx < dstRank; ++dstIdx) {
      mlir::ReassociationIndices group;
      if (dstIdx == 0) {
        // First dst dimension absorbs all the leading extra dimensions
        for (int64_t i = 0; i <= extraDims; ++i) {
          group.push_back(i);
        }
      } else {
        // Other dimensions map 1-to-1
        group.push_back(extraDims + dstIdx);
      }
      reassoc.push_back(group);
    }

    return builder.create<mlir::tensor::CollapseShapeOp>(loc, dstTensorTy, src,
                                                         reassoc);
  }"""

NEW = """  if (srcRank > dstRank) {
    // Try collapse_shape: [1, 1, M, N] -> [M, N]
    // Reassociation maps each dst dim to a group of src dims
    llvm::SmallVector<mlir::ReassociationIndices> reassoc;
    int64_t extraDims = srcRank - dstRank;

    for (int64_t dstIdx = 0; dstIdx < dstRank; ++dstIdx) {
      mlir::ReassociationIndices group;
      if (dstIdx == 0) {
        // First dst dimension absorbs all the leading extra dimensions
        for (int64_t i = 0; i <= extraDims; ++i) {
          group.push_back(i);
        }
      } else {
        // Other dimensions map 1-to-1
        group.push_back(extraDims + dstIdx);
      }
      reassoc.push_back(group);
    }

    // T32.9 stack-2 fix: when src has dynamic dims, the corresponding
    // dst dim group must be dynamic. Rebuild dstTensorTy honoring src's
    // dynamic dims.
    llvm::SmallVector<int64_t> adjustedDstShape(dstShapeStatic.begin(),
                                                dstShapeStatic.end());
    for (int64_t dstIdx = 0; dstIdx < dstRank; ++dstIdx) {
      for (int64_t srcIdx : reassoc[dstIdx]) {
        if (mlir::ShapedType::isDynamic(srcShape[srcIdx])) {
          adjustedDstShape[dstIdx] = mlir::ShapedType::kDynamic;
          break;
        }
      }
    }
    auto adjustedDstTy = mlir::RankedTensorType::get(
        adjustedDstShape, srcTensorTy.getElementType());

    return builder.create<mlir::tensor::CollapseShapeOp>(loc, adjustedDstTy,
                                                         src, reassoc);
  }"""

count = text.count(OLD)
print(f"Found {count} matches (expect 1)")
if count == 0:
    exit(1)
text = text.replace(OLD, NEW)
CODEGEN.write_text(text)
print("collapse_shape fix applied")
