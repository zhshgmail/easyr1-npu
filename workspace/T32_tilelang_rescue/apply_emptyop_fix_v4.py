#!/usr/bin/env python3
"""V4: type-dispatch between tensor::DimOp and memref::DimOp.

V3 used tensor::DimOp unconditionally, but when src is a memref (e.g.
loop-carried memref<?x2xf32>), tensor::DimOp rejects it. Pattern at
line 1141-1145 in same file already shows the right dispatch.
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
text = CODEGEN.read_text()

OLD = """  // T32.9 fix (v3): derive dynamic-size Values directly from src via
  // tensor::DimOp. For each ShapedType::kDynamic slot, emit a dim query.
  llvm::SmallVector<mlir::Value, 4> dynamicSizes;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] == mlir::ShapedType::kDynamic) {
      dynamicSizes.push_back(
          builder.create<mlir::tensor::DimOp>(builder.getUnknownLoc(), src, i)
              .getResult());
    }
  }"""

NEW = """  // T32.9 fix (v4): derive dynamic-size Values from src. Use tensor::DimOp
  // when src is a tensor, memref::DimOp when src is a memref (the loop-
  // carried args can be memref).
  llvm::SmallVector<mlir::Value, 4> dynamicSizes;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] == mlir::ShapedType::kDynamic) {
      mlir::Value dimVal;
      if (src.getType().isa<mlir::TensorType>()) {
        dimVal = builder.create<mlir::tensor::DimOp>(
                     builder.getUnknownLoc(), src, i)
                     .getResult();
      } else if (src.getType().isa<mlir::MemRefType>()) {
        dimVal = builder.create<mlir::memref::DimOp>(
                     builder.getUnknownLoc(), src, i)
                     .getResult();
      } else {
        ICHECK(false) << "unsupported src type for dynamic-shape dim query";
      }
      dynamicSizes.push_back(dimVal);
    }
  }"""

count = text.count(OLD)
print(f"Found {count} matches for v3 patch (expect 1)")
if count == 0:
    print("ERROR: pattern not found")
    exit(1)

text = text.replace(OLD, NEW)
CODEGEN.write_text(text)
print("v4 patch applied. Marker:", "v4" in text)
