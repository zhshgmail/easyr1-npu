#!/usr/bin/env python3
"""Strip T32/T32.9/v6/F1/etc session-internal tags from comments before
opening upstream PR. Keep the technical explanation intact, remove the
session-bookkeeping noise that's meaningless to upstream maintainers.
"""
from pathlib import Path
import re

ROOT = Path("/home/z00637938/workspace/tilelang-mlir-ascend")

# Replacements: pattern -> replacement
# Goal: keep the technical sentence, drop the "T32 XXX:" preface.
replacements = [
    # "T32 F1.3: BufferLoad is accepted as scalar" -> "BufferLoad is accepted as scalar"
    (re.compile(r"// T32[\w.]* ?(\w+\.?\w*)?[ :]*"), "// "),
    (re.compile(r"// Issue T32[\w.]*[ :]*"), "// "),
]

for f in [
    ROOT / "src/op/ascend.cc",
    ROOT / "src/target/codegen_npuir_dev.cc",
    ROOT / "src/transform/npu_loop_vectorize.cc",
]:
    t = f.read_text()
    orig = t
    for pat, rep in replacements:
        t = pat.sub(rep, t)
    if t != orig:
        f.write_text(t)
        # Count T32 occurrences remaining
        rem = t.count("T32")
        print(f"{f.name}: cleaned, {rem} T32 mentions remain")
    else:
        print(f"{f.name}: no change")
