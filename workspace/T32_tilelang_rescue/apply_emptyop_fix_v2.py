#!/usr/bin/env python3
"""Apply EmptyOp dynamic-shape fix to codegen_npuir_dev.cc — round 2.

Round 1 (apply_emptyop_fix.py) patched the AllocateNode visitor (lines
3771, 3783). But the actual %76 = tensor.empty() in fp8_gemm IR comes
from a DIFFERENT site: NeedGenInsertSlice() at line 2131.

This patch fixes site 2131 by extracting the dynamic-size Values from
the already-computed shape_val SmallVector (which has OpFoldResult of
mixed integer attrs + index Values).
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
text = CODEGEN.read_text()

OLD = """  auto srcType = src.getType().cast<mlir::TensorType>();
  auto dstType = GetVarValue(buffer_data).getType().cast<mlir::TensorType>();
  auto elemType = dstType.getElementType();
  auto srcShape = srcType.getShape();

  auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
      builder.getUnknownLoc(), srcShape, elemType);"""

NEW = """  auto srcType = src.getType().cast<mlir::TensorType>();
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

count = text.count(OLD)
print(f"Found {count} matches for NeedGenInsertSlice EmptyOp site (expect 1)")
if count == 0:
    print("ERROR: pattern not found")
    exit(1)

text = text.replace(OLD, NEW)
CODEGEN.write_text(text)

verified = "T32.9 fix" in text
print(f"Patched. Marker present: {verified}")
