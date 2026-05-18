#!/usr/bin/env python3
"""Fix: NpuirAtomicAdd parses args in wrong src/dst order.

Test kernels call: T.npuir_atomic_add(UB_source, GM_dest, size)
   - args[0] = UB source
   - args[1] = GM dest

But NpuirAtomicAdd ctor does:
   std::tie(this->dst, this->src) = std::tie(bf[0], bf[1]);

So dst=UB, src=GM, then codegen emits hivm.hir.store(src=GM, dst=UB)
which violates "only support store ub to gm" verifier check.

Fix: swap the std::tie order.
"""
from pathlib import Path

ASCEND_CC = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/op/ascend.cc")
t = ASCEND_CC.read_text()

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

if OLD in t:
    t = t.replace(OLD, NEW)
    ASCEND_CC.write_text(t)
    print("Applied: NpuirAtomicAdd src/dst order swap")
elif NEW in t:
    print("Already applied")
else:
    print("FAIL: pattern not found")
    exit(1)
