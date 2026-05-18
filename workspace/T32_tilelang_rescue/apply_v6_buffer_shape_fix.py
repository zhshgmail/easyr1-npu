#!/usr/bin/env python3
"""Fix v6's buffer_shape population.

v6 unified the is_scalar_load branches and always populates buffer_shape
with the full buffer shape. This breaks getBroadcastDim later when the
two operand shapes don't match (e.g. B_ub[0] is (3,) but A_ub is (1, N)).

The original behavior: arg_id==0 populated buffer_shape with buffer shape;
arg_id==1 didn't populate (left empty). After v6 unified to "extract
scalar", buffer_shape should be EMPTY (consistent with arg_id==1's
previous behavior) — because the operand is effectively a scalar f32,
not a tensor with shape.

Fix: clear buffer_shape AFTER VisitExpr_, don't repopulate.
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
t = CODEGEN.read_text()

OLD = """      if (is_scalar_load) {
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

NEW = """      if (is_scalar_load) {
        // T32 v6 + buffer_shape fix: scalar load extracts an f32 (or other
        // scalar) — the operand is effectively a scalar with no tensor shape.
        // Leave buffer_shape EMPTY so downstream getBroadcastDim() short-
        // circuits via its `if (...empty()) return dims;` early-return.
        src = VisitExpr_(buffer_node);
        buffer_shape.clear();
      } else {"""

count = t.count(OLD)
print(f"Found {count} matches (expect 1)")
if count == 0:
    exit(1)
t = t.replace(OLD, NEW)
CODEGEN.write_text(t)
print("buffer_shape population fix applied")
