#!/usr/bin/env python3
"""Fix for the real bug: asymmetric arg_id==0 vs arg_id==1 handling for
scalar loads in CreateHIVMBinaryVectorOp::processImm.

In line ~2752-2755:
  if (is_scalar_load) {
    if (arg_id == 0) {
      src = GetVarValue(region_node);  // ← returns WHOLE buffer
    } else {
      src = VisitExpr_(buffer_node);   // ← correctly extracts scalar
    }

For 1×1 region loads, BOTH arg_id branches should extract the scalar.
The arg_id==0 branch returning the full buffer causes downstream vmul
to operate on tensor<128x2xf32> instead of scalar f32.

Fix: make arg_id==0 also call VisitExpr_(buffer_node).
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
text = CODEGEN.read_text()

OLD = """      if (is_scalar_load) {
        if (arg_id == 0) {
          src = GetVarValue(region_node);
          auto region = tvm::tl::RegionOp(region_node->args, vmap);
          auto region_buffer = region.GetBuffer();
          buffer_shape.clear();
          for (auto dim : region_buffer->shape) {
            buffer_shape.push_back(dim);
          }
        } else {
          src = VisitExpr_(buffer_node);
        }
      } else {"""

NEW = """      if (is_scalar_load) {
        // T32: both arg_id branches should extract the scalar. Previously
        // arg_id==0 returned the whole buffer which caused vmul to operate
        // on full-tensor instead of scalar, then collapse_shape would try
        // to reduce 128x2 to 1 element — verifier reject. Use scalar load
        // for both args.
        src = VisitExpr_(buffer_node);
        auto region = tvm::tl::RegionOp(region_node->args, vmap);
        auto region_buffer = region.GetBuffer();
        buffer_shape.clear();
        for (auto dim : region_buffer->shape) {
          buffer_shape.push_back(dim);
        }
      } else {"""

count = text.count(OLD)
print(f"Found {count} matches (expect 1)")
if count == 0:
    exit(1)
text = text.replace(OLD, NEW)
CODEGEN.write_text(text)
print("scalar load asymmetry fix applied")
