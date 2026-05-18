#!/usr/bin/env python3
"""Revert the v6 (scalar_load) patch. v6 broke too many downstream paths.
The right fix needs a larger refactor (transform scalar*scalar mul into
arith.mulf instead of vmul) — beyond patch-level intervention.
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
text = CODEGEN.read_text()

NEW_BAD = """      if (is_scalar_load) {
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

ORIG = """      if (is_scalar_load) {
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

if NEW_BAD in text:
    text = text.replace(NEW_BAD, ORIG)
    CODEGEN.write_text(text)
    print("reverted v6 patch")
else:
    print("v6 marker not found — already reverted?")
