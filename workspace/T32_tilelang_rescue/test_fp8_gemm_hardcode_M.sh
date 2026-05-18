#!/bin/bash
# Test: does fp8_gemm pass on MLIR if we hardcode M instead of T.symbolic?
# If yes → confirms symbolic-shape codegen path is the bug surface.

set -e

TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

KERNEL=$TLPATH/examples/deepseek_v4/example_fp8_gemm_kernel.py
BACKUP=/tmp/fp8_gemm_kernel.orig.py

# Backup once
[ -f "$BACKUP" ] || cp "$KERNEL" "$BACKUP"

# Restore from backup, then patch
cp "$BACKUP" "$KERNEL"

python3 << 'EOF'
from pathlib import Path
p = Path("/home/z00637938/workspace/tilelang-mlir-ascend/examples/deepseek_v4/example_fp8_gemm_kernel.py")
t = p.read_text()
t = t.replace('M = T.symbolic("M")', 'M = 128  # T32: hardcoded static M for symbolic-codegen workaround')
# Only test m=128 cases since M is now baked
t = t.replace('run_test_case(m=96, n=128, k=256, out_dtype="float32")',
              '# run_test_case(m=96, n=128, k=256, out_dtype="float32")  # M hardcoded')
p.write_text(t)
EOF

cd $TLPATH/examples

echo "--- M definition after patch ---"
grep -E '^\s*M = ' $KERNEL | head -3

echo "--- run ---"
timeout 600 python3 deepseek_v4/example_fp8_gemm_kernel.py 2>&1

# Always restore at end
cp "$BACKUP" "$KERNEL"
echo "--- restored ---"
