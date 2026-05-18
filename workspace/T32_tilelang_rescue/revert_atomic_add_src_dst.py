#!/usr/bin/env python3
"""Revert v13's atomic_add src/dst swap — broke example/elementwise/atomic_add.py.
Need to revisit the diagnostic; the bug isn't this simple."""
from pathlib import Path

ASCEND_CC = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/op/ascend.cc")
t = ASCEND_CC.read_text()

NEW = """NpuirAtomicAdd::NpuirAtomicAdd(Array<PrimExpr> args, BufferMap vmap) {
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  // T32: args[0] is the source (UB), args[1] is the dest (GM).
  // Verifier requires hivm.hir.store(src_ub, dst_gm) order; previously
  // this ctor had src/dst swapped, causing "only support store ub to
  // gm currently!" verifier reject.
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
}"""

OLD = """NpuirAtomicAdd::NpuirAtomicAdd(Array<PrimExpr> args, BufferMap vmap) {
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->dst, this->src) = std::tie(bf[0], bf[1]);
  std::tie(this->dst_range, this->src_range) = std::tie(rgs[0], rgs[1]);
}"""

if NEW in t:
    t = t.replace(NEW, OLD)
    ASCEND_CC.write_text(t)
    print("Reverted v13 src/dst swap to original")
else:
    print("v13 already reverted or pattern not found")
