#!/usr/bin/env python3
"""Apply EmptyOp dynamic-shape fix to codegen_npuir_dev.cc.

Targets the AllocateNode visitor (~line 3771, 3783) where the codegen
emits `tensor::EmptyOp::build(loc, shape, elementType)` without dynamic
sizes. When op->extents contains a symbolic dim, this fails MLIR verifier
with "incorrect number of dynamic sizes, has 0, expected 1".

Fix: gather SSA values for non-const extents via MakeValue(), index-cast
if needed, pass as dynamicSizes to the correct EmptyOp::build overload.
"""
from pathlib import Path

CODEGEN = Path("/home/z00637938/workspace/tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc")
text = CODEGEN.read_text()

# Backup
backup = CODEGEN.with_suffix(".cc.orig.t32_emptyop")
if not backup.exists():
    backup.write_text(text)
    print(f"Backup written: {backup}")

OLD_PATTERN = """    auto tensorEmptyOp = builder.create<mlir::tensor::EmptyOp>(
        builder.getUnknownLoc(), shape, DTypetoMLIRType(op->dtype));"""

NEW_PATTERN = """    // Issue T32.9: gather dynamic-shape SSA values for symbolic extents.
    llvm::SmallVector<mlir::Value, 4> dynamicSizes;
    for (const auto &extent : op->extents) {
      if (!as_const_int(extent)) {
        dynamicSizes.push_back(this->CreateIndexCastOp(this->MakeValue(extent)));
      }
    }
    auto tensorEmptyOp = builder.create<mlir::tensor::EmptyOp>(
        builder.getUnknownLoc(), shape, DTypetoMLIRType(op->dtype),
        dynamicSizes);"""

count = text.count(OLD_PATTERN)
print(f"Found {count} matches in AllocateNode visitor (expect 2)")

if count == 0:
    print("ERROR: no matches; pattern may have already been patched or source diverged")
    exit(1)

new_text = text.replace(OLD_PATTERN, NEW_PATTERN)
CODEGEN.write_text(new_text)

# Verify
verified = new_text.count("dynamicSizes")
print(f"Patched. 'dynamicSizes' occurrences in new file: {verified} (expect >= {count * 4})")

# Make sure include for arith::IndexCastOp is present
includes_present = "mlir/Dialect/Arith/IR/Arith.h" in new_text or "arith::IndexCastOp" in text[: text.find("namespace tvm")]
if not includes_present:
    print("WARN: may need to ensure #include for mlir::arith::IndexCastOp; check build for missing-symbol errors")
