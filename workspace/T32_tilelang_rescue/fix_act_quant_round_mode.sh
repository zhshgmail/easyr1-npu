#!/bin/bash
# Fix: change vcast round_mode from "round" (tie-away-from-zero) to "rint"
# (tie-to-even) to match torch.round's default. This fixes the int8 mismatch
# at exactly 0.5-valued elements where the two rounding modes diverge.

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
OLD = 'T.vcast(x_ub_fp, x_ub_fp, round_mode="round")'
NEW = 'T.vcast(x_ub_fp, x_ub_fp, round_mode="rint")  # T32: match torch.round (tie-to-even)'
if OLD not in t:
    print("FAIL: pattern not found")
    exit(1)
t = t.replace(OLD, NEW)
p.write_text(t)
print("kernel patched: round_mode round → rint")
EOF

cd $TLPATH/examples
echo "== act_quant after round_mode=rint =="
timeout 600 python3 deepseek_v4/example_act_quant_kernel.py 2>&1 | tail -15

cp "$BACKUP" "$KERNEL"
echo "(restored)"
