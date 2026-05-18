#!/usr/bin/env python3
"""Option F1: in HandleBinaryExpression's !resolved_b broadcast branch,
detect loop-invariant BufferLoad operand and KEEP it as a runtime scalar
(skip the broadcast-to-tmp-buffer path) so vmul codegen treats it as a
scalar via the existing Scalar case in processImm.

Idea: change `IsScalar(operands[1]) || operands[1].as<VarNode>()` to also
accept `operands[1].as<BufferLoadNode>()` when the buffer load is
loop-invariant. Then region_b = operands[1] carries the BufferLoad
PrimExpr; codegen's processImm sees BufferLoad/VarNode-like operand and
takes the scalar path (which calls MakeValue → memref.load → scalar f32).

Sites: lines ~835 and ~857 (both arg_id=0 and arg_id=1 broadcast branches).

Caveat: codegen processImm only accepts `IntImm/FloatImm/VarNode` in
the Scalar branch. BufferLoad will fall through to the Vector branch.
So this fix alone isn't enough — also need to widen processImm Scalar
case.

Combined patch: F1 widening here + extend processImm Scalar branch to
accept BufferLoadNode.
"""
from pathlib import Path

NPU_LV = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/transform/npu_loop_vectorize.cc")
CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")

# Step 1: widen IsScalar check in HandleBinaryExpression's !resolved_b branch
lv_text = NPU_LV.read_text()
OLD_LV = """    if (!resolved_b) {
      if (IsScalar(operands[1]) || operands[1].as<VarNode>()) {
        region_b = operands[1];
      } else {
        // BufferLoad or compound expression: broadcast to tmp buffer.
        const BufferAccessInfo &ref = *resolved_a;
        auto scalar_buf = CreateTempBuffer(ref, binary_op_name);
        tmp_bufs->push_back(scalar_buf);
        stmts->push_back(BuildNpuirCall(
            "Broadcast", {operands[1], BuildRegionCall(scalar_buf, 2, 1)}));
        region_b = BuildRegionCall(scalar_buf, 1, 1);
      }
    }"""

NEW_LV = """    if (!resolved_b) {
      // T32 F1: loop-invariant BufferLoad is a runtime-scalar — keep it as
      // the PrimExpr (don't broadcast). The codegen Scalar branch handles
      // runtime BufferLoad via MakeValue→memref.load.
      bool is_loop_invariant_bufload =
          operands[1].as<BufferLoadNode>() &&
          IsLoopInvariantExpr(operands[1], loop_vars);
      if (IsScalar(operands[1]) || operands[1].as<VarNode>() ||
          is_loop_invariant_bufload) {
        region_b = operands[1];
      } else {
        // BufferLoad or compound expression: broadcast to tmp buffer.
        const BufferAccessInfo &ref = *resolved_a;
        auto scalar_buf = CreateTempBuffer(ref, binary_op_name);
        tmp_bufs->push_back(scalar_buf);
        stmts->push_back(BuildNpuirCall(
            "Broadcast", {operands[1], BuildRegionCall(scalar_buf, 2, 1)}));
        region_b = BuildRegionCall(scalar_buf, 1, 1);
      }
    }"""

if OLD_LV not in lv_text:
    print("FAIL: pattern not found in npu_loop_vectorize.cc")
    exit(1)
lv_text = lv_text.replace(OLD_LV, NEW_LV)
NPU_LV.write_text(lv_text)
print("Step 1: patched npu_loop_vectorize.cc HandleBinaryExpression")

# Step 2: extend codegen processImm Scalar branch to accept BufferLoadNode
cg_text = CODEGEN.read_text()

OLD_CG = """  auto processImm = [&](mlir::Value &src, int arg_id,
                        Array<PrimExpr> &buffer_shape) {
    if (op->args[arg_id].as<IntImm>() || op->args[arg_id].as<FloatImm>() ||
        op->args[arg_id].as<tir::VarNode>()) {
      // Scalar case
      const CallNode *region_node = op->args[1 - arg_id].as<CallNode>();
      const BufferLoadNode *buffer_load_node =
          region_node->args[0].as<BufferLoadNode>();
      if (op->args[arg_id]->dtype != buffer_load_node->buffer->dtype) {
        src = ScalarConvertType(op->args[arg_id],
                                buffer_load_node->buffer->dtype);
      } else {
        src = MakeValue(op->args[arg_id]);
      }
    } else {"""

NEW_CG = """  auto processImm = [&](mlir::Value &src, int arg_id,
                        Array<PrimExpr> &buffer_shape) {
    // T32 F1: include BufferLoadNode as a Scalar case input — when the
    // upstream pass detects a loop-invariant scalar BufferLoad and keeps
    // it as PrimExpr (rather than broadcasting), it reaches us as a
    // BufferLoadNode. Handle it as a runtime scalar via MakeValue
    // (which lowers to memref.load returning a scalar mlir::Value).
    if (op->args[arg_id].as<IntImm>() || op->args[arg_id].as<FloatImm>() ||
        op->args[arg_id].as<tir::VarNode>() ||
        op->args[arg_id].as<BufferLoadNode>()) {
      // Scalar case
      const CallNode *region_node = op->args[1 - arg_id].as<CallNode>();
      const BufferLoadNode *buffer_load_node =
          region_node->args[0].as<BufferLoadNode>();
      if (op->args[arg_id]->dtype != buffer_load_node->buffer->dtype) {
        src = ScalarConvertType(op->args[arg_id],
                                buffer_load_node->buffer->dtype);
      } else {
        src = MakeValue(op->args[arg_id]);
      }
    } else {"""

if OLD_CG not in cg_text:
    print("FAIL: pattern not found in codegen_npuir_dev.cc")
    exit(1)
cg_text = cg_text.replace(OLD_CG, NEW_CG)
CODEGEN.write_text(cg_text)
print("Step 2: patched codegen_npuir_dev.cc processImm Scalar branch")

print("F1 patch applied to both files")
