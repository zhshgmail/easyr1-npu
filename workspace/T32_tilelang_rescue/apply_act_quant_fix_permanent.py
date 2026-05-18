#!/usr/bin/env python3
"""Apply act_quant fixes permanently:
1. round_mode "round" → "rint" (match torch.round tie-to-even)
2. scale tensor shape: s = x.new_empty(N, 1) → x.new_empty(x.size(0), 1)
   (was using cols when it should be rows)
"""
from pathlib import Path

KERNEL = Path("/home/z00637938/workspace/tilelang-mlir-ascend/examples/deepseek_v4/example_act_quant_kernel.py")
t = KERNEL.read_text()

OLD1 = 'T.vcast(x_ub_fp, x_ub_fp, round_mode="round")'
NEW1 = 'T.vcast(x_ub_fp, x_ub_fp, round_mode="rint")  # T32: match torch.round (tie-to-even)'

OLD2 = 's = x.new_empty(N, 1, dtype=torch.float32)'
NEW2 = 's = x.new_empty(x.size(0), 1, dtype=torch.float32)  # T32: was N (cols), fixed to M (rows)'

if OLD1 in t:
    t = t.replace(OLD1, NEW1)
    print("Applied: round_mode round → rint")
elif NEW1 in t:
    print("Already applied: round_mode")

if OLD2 in t:
    t = t.replace(OLD2, NEW2)
    print("Applied: scale shape N → x.size(0)")
elif NEW2 in t:
    print("Already applied: scale shape")

KERNEL.write_text(t)
print("Done.")
