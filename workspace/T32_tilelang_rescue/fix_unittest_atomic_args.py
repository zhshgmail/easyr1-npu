#!/usr/bin/env python3
"""Fix 6 unittest atomic_add tests: swap args to match wrapper signature.

Wrapper `npuir_atomic_add(dst, src, size)` expects dst first, src second.
The 6 tests pass (UB_src, GM_dst, size) - wrong order. The reference IR
in mlir_files/ shows the CORRECT (UB→GM atomic) semantics, meaning the
tests were authored when something else was different OR the test
authors just made a swap mistake.

Fix: swap the args to (GM_dst, UB_src, size).
"""
from pathlib import Path

TESTS = [
    "test_atomic_add_1d_dev.py",
    "test_atomic_add_1d_exp.py",
    "test_atomic_add_2d_dev.py",
    "test_atomic_add_2d_exp.py",
    "test_atomic_addx4_dev.py",
    "test_atomic_addx4_exp.py",
]

base = Path("/home/z00637938/workspace/tilelang-mlir-ascend/unittest/npuir")

for t in TESTS:
    p = base / t
    src = p.read_text()
    # Swap the args in the npuir_atomic_add* call
    # Pattern: T.npuir_atomic_add[x4](A_VEC, B[...], [...])
    # Want:    T.npuir_atomic_add[x4](B[...], A_VEC, [...])
    import re
    # Match `T.npuir_atomic_add(A_VEC, B[...], [...])` or `T.npuir_atomic_addx4(...)`
    pattern = re.compile(r'T\.npuir_atomic_add(x4)?\(A_VEC, (B\[[^\]]+\]), \[([^\]]+)\]\)')
    new_src = pattern.sub(r'T.npuir_atomic_add\1(\2, A_VEC, [\3])  # T32: swapped to match (dst, src) wrapper signature', src)
    if new_src != src:
        p.write_text(new_src)
        print(f"Patched {t}")
    else:
        print(f"No change to {t}")
