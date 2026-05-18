#!/bin/bash
# Probe step §10.1.2: relax tolerance and see if it passes.

TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

KERNEL=$TLPATH/examples/fp8_lighting_indexer.py
BACKUP=/tmp/fp8_indexer.orig.py
[ -f "$BACKUP" ] || cp "$KERNEL" "$BACKUP"
cp "$BACKUP" "$KERNEL"

python3 << 'EOF'
from pathlib import Path
p = Path("/home/z00637938/workspace/tilelang-mlir-ascend/examples/fp8_lighting_indexer.py")
t = p.read_text()
t = t.replace(
    'torch.testing.assert_close(o.cpu().reshape(B, M, N), o_torch, rtol=1e-2, atol=1e-2)',
    'torch.testing.assert_close(o.cpu().reshape(B, M, N), o_torch, rtol=5e-2, atol=3e-2)'
)
p.write_text(t)
EOF

cd $TLPATH/examples
echo "== fp8_indexer with relaxed atol=0.03 rtol=0.05 =="
timeout 600 python3 fp8_lighting_indexer.py 2>&1 | grep -iE 'passed|pass!|failed|mismatch|error:|assertion' | tail -5

cp "$BACKUP" "$KERNEL"
echo "(restored)"
