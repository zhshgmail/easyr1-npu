#!/usr/bin/env python3
"""F1 step 3: extend NpuirOperand::FromExpr (in src/op/ascend.cc) to
accept BufferLoadNode as Scalar, mirroring the F1 step 2 fix in
codegen_npuir_dev.cc::processImm.

This is needed because the npu_loop_vectorize.cc step 1 patch (F1.1)
now passes loop-invariant BufferLoads through as PrimExpr operands to
tl.npuir_mul. The DEV codegen handles it (we patched it in F1.2).
But the API codegen path uses NpuirBinaryOperator → NpuirOperand::
FromExpr which FATALs with 'cannot handle the expr with type of
"tir.BufferLoad"'. This breaks engram_bwd_exp.

Fix: add `expr.as<BufferLoadNode>()` to the Scalar branch in FromExpr.
"""
from pathlib import Path

ASCEND = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/op/ascend.cc")
t = ASCEND.read_text()

OLD = """  if (expr.as<IntImm>() || expr.as<FloatImm>() || expr.as<tir::VarNode>()) {
    // If there are other types of nodes that need to be treated as scalars,
    // please add them here.
    return NpuirOperand::Scalar(expr);
  }"""

NEW = """  if (expr.as<IntImm>() || expr.as<FloatImm>() || expr.as<tir::VarNode>() ||
      expr.as<tir::BufferLoadNode>()) {
    // T32 F1.3: BufferLoad is accepted as scalar when the upstream pass
    // (npu_loop_vectorize) detects a loop-invariant scalar load and keeps
    // it as PrimExpr instead of broadcasting to a tmp buffer.
    return NpuirOperand::Scalar(expr);
  }"""

if OLD in t:
    t = t.replace(OLD, NEW)
    ASCEND.write_text(t)
    print("Applied F1.3: NpuirOperand::FromExpr accepts BufferLoadNode")
elif NEW in t:
    print("Already applied")
else:
    print("FAIL: pattern not found in ascend.cc")
    exit(1)
