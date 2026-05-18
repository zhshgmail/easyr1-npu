#!/usr/bin/env python3
"""V3: use tensor::DimOp(src, i) to derive dynamicSizes from src directly.

V2 was wrong: it tried to pull dynamic sizes from shape_val, but shape_val
is built from `range` (the destination subview shape), which doesn't
necessarily align with srcShape's dynamic dim positions.

V3 correctly queries src's actual runtime dims via tensor::DimOp.
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
text = CODEGEN.read_text()

OLD = """  auto srcType = src.getType().cast<mlir::TensorType>();
  auto dstType = GetVarValue(buffer_data).getType().cast<mlir::TensorType>();
  auto elemType = dstType.getElementType();
  auto srcShape = srcType.getShape();

  // T32.9 fix: extract dynamic-size Values from shape_val (mixed
  // OpFoldResult). Each ShapedType::kDynamic slot in srcShape needs a
  // corresponding mlir::Value passed via dynamicSizes operand.
  llvm::SmallVector<mlir::Value, 4> dynamicSizes;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] == mlir::ShapedType::kDynamic) {
      // shape_val[i] is OpFoldResult; for dynamic shape it must be a Value
      auto v = shape_val[i].dyn_cast<mlir::Value>();
      ICHECK(v) << "expected mlir::Value for dynamic shape dim " << i;
      dynamicSizes.push_back(v);
    }
  }

  auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
      builder.getUnknownLoc(), srcShape, elemType, dynamicSizes);"""

NEW = """  auto srcType = src.getType().cast<mlir::TensorType>();
  auto dstType = GetVarValue(buffer_data).getType().cast<mlir::TensorType>();
  auto elemType = dstType.getElementType();
  auto srcShape = srcType.getShape();

  // T32.9 fix (v3): derive dynamic-size Values directly from src via
  // tensor::DimOp. For each ShapedType::kDynamic slot, emit a dim query.
  llvm::SmallVector<mlir::Value, 4> dynamicSizes;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] == mlir::ShapedType::kDynamic) {
      dynamicSizes.push_back(
          builder.create<mlir::tensor::DimOp>(builder.getUnknownLoc(), src, i)
              .getResult());
    }
  }

  auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
      builder.getUnknownLoc(), srcShape, elemType, dynamicSizes);"""

count = text.count(OLD)
print(f"Found {count} matches for v2 patch (expect 1)")
if count == 0:
    print("ERROR: pattern not found — v2 may not have been applied")
    exit(1)

text = text.replace(OLD, NEW)
CODEGEN.write_text(text)
print("v3 patch applied. Marker present:", "v3" in text)
