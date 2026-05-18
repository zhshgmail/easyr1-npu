#!/bin/bash
# Full fix for act_quant:
# 1. Change vcast round_mode from "round" (tie-away-from-zero) to "rint"
#    (tie-to-even) to match torch.round semantics.
# 2. Fix kernel host-side scale tensor allocation: s = x.new_empty(N, 1)
#    is WRONG (uses N=cols instead of M=rows). Should be M (rows).

TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

KERNEL=$TLPATH/examples/deepseek_v4/example_act_quant_kernel.py
BACKUP=/tmp/act_quant.orig.py
[ -f "$BACKUP" ] || cp "$KERNEL" "$BACKUP"
cp "$BACKUP" "$KERNEL"

python3 << 'EOF'
from pathlib import Path
p = Path("/home/z00637938/workspace/tilelang-mlir-ascend/examples/deepseek_v4/example_act_quant_kernel.py")
t = p.read_text()

# Fix 1: round_mode
OLD1 = 'T.vcast(x_ub_fp, x_ub_fp, round_mode="round")'
NEW1 = 'T.vcast(x_ub_fp, x_ub_fp, round_mode="rint")  # T32: match torch.round (tie-to-even)'
assert OLD1 in t, "round_mode pattern not found"
t = t.replace(OLD1, NEW1)

# Fix 2: scale shape allocation. The current `s = x.new_empty(N, 1, ...)` is
# wrong (uses N=#cols instead of M=#rows). Should match torch ref's [M, 1].
OLD2 = 's = x.new_empty(N, 1, dtype=torch.float32)'
NEW2 = 's = x.new_empty(x.size(0), 1, dtype=torch.float32)  # T32: was N (cols), fixed to M (rows)'
assert OLD2 in t, "scale alloc pattern not found"
t = t.replace(OLD2, NEW2)

p.write_text(t)
print("kernel patched: round_mode + scale alloc shape")
EOF

cd $TLPATH/examples
echo "== act_quant after FULL fix =="
timeout 600 python3 deepseek_v4/example_act_quant_kernel.py 2>&1 | tail -15

cp "$BACKUP" "$KERNEL"
echo "(restored)"
