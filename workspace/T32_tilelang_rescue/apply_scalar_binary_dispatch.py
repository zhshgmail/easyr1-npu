#!/usr/bin/env python3
"""Final fix: in CreateHIVMBinaryVectorOp, dispatch on src type.
- If both src0 and src1 are scalars (not tensor/memref): emit arith.mulf
  (or appropriate scalar arith op) + scalar insert into destination.
- Else: keep existing vmul-based codegen.

This handles the case where v6 + F1 both apply, producing scalar src0
and src1 from per-element scalar loads.
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
text = CODEGEN.read_text()

# Find the section after processImm calls — insert dispatch.
OLD = """  mlir::Value src0, src1;
  Array<PrimExpr> buffer_shape0, buffer_shape1;
  processImm(src0, 0, buffer_shape0);
  processImm(src1, 1, buffer_shape1);
  // dst
  const CallNode *region_node_dst = op->args[2].as<CallNode>();

  tvm::tl::RegionOp region_dst_tmp(region_node_dst->args, vmap);
  Array<Range> dst_range = region_dst_tmp.GetRanges();

  auto srcTensorTy = src0.getType().cast<mlir::TensorType>();
  auto srcShape = srcTensorTy.getShape();"""

NEW = """  mlir::Value src0, src1;
  Array<PrimExpr> buffer_shape0, buffer_shape1;
  processImm(src0, 0, buffer_shape0);
  processImm(src1, 1, buffer_shape1);
  // dst
  const CallNode *region_node_dst = op->args[2].as<CallNode>();

  tvm::tl::RegionOp region_dst_tmp(region_node_dst->args, vmap);
  Array<Range> dst_range = region_dst_tmp.GetRanges();

  // T32 v6+F1+scalar-dispatch: if both src0/src1 are scalars (came from
  // per-element loads via is_scalar_load path), emit arith.mulf instead
  // of vmul. The dst is also a single-element write to the destination
  // buffer.
  if (!src0.getType().isa<mlir::TensorType>() &&
      !src1.getType().isa<mlir::TensorType>()) {
    // Scalar-scalar path. Emit scalar arith op + memref.store / tensor.insert.
    mlir::Value scalarResult;
    auto loc = builder.getUnknownLoc();
    if constexpr (std::is_same_v<T, mlir::hivm::VMulOp>) {
      scalarResult = builder.create<mlir::arith::MulFOp>(loc, src0, src1).getResult();
    } else if constexpr (std::is_same_v<T, mlir::hivm::VAddOp>) {
      scalarResult = builder.create<mlir::arith::AddFOp>(loc, src0, src1).getResult();
    } else if constexpr (std::is_same_v<T, mlir::hivm::VSubOp>) {
      scalarResult = builder.create<mlir::arith::SubFOp>(loc, src0, src1).getResult();
    } else if constexpr (std::is_same_v<T, mlir::hivm::VDivOp>) {
      scalarResult = builder.create<mlir::arith::DivFOp>(loc, src0, src1).getResult();
    } else {
      // Fallback for ops without scalar arith equivalent — bail to vmul path
      ICHECK(false) << "scalar-scalar fallback only supports VMul/Add/Sub/Div";
    }
    // Store to destination at the indexed location
    mlir::Value dst = GetVarValue(region_node_dst);
    // Collect destination indices from dst_range
    llvm::SmallVector<mlir::Value, 4> dstIndices;
    for (const auto &r : dst_range) {
      if (auto s_int = as_const_int(r.get()->min)) {
        // const index — make arith.constant
        mlir::Value c = builder.create<mlir::arith::ConstantIndexOp>(loc, *s_int);
        dstIndices.push_back(c);
      } else {
        dstIndices.push_back(CreateIndexCastOp(MakeValue(r.get()->min)));
      }
    }
    if (dst.getType().isa<mlir::MemRefType>()) {
      builder.create<mlir::memref::StoreOp>(loc, scalarResult, dst, dstIndices);
    } else {
      // tensor type — insert via tensor.insert
      mlir::Value updated = builder.create<mlir::tensor::InsertOp>(
          loc, scalarResult, dst, dstIndices);
      SetVarValue(region_node_dst, updated);
    }
    return;
  }

  auto srcTensorTy = src0.getType().cast<mlir::TensorType>();
  auto srcShape = srcTensorTy.getShape();"""

count = text.count(OLD)
print(f"Found {count} matches (expect 1)")
if count == 0:
    exit(1)
text = text.replace(OLD, NEW)
CODEGEN.write_text(text)
print("scalar-binary dispatch added")
