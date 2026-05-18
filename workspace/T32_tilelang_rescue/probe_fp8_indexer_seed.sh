#!/bin/bash
# Try different random seeds to see if 5/16M is deterministic or seed-dependent.

TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

KERNEL=$TLPATH/examples/fp8_lighting_indexer.py
BACKUP=/tmp/fp8_indexer.orig.py
[ -f "$BACKUP" ] || cp "$KERNEL" "$BACKUP"
cp "$BACKUP" "$KERNEL"

for SEED in 1 2 3 42; do
  python3 - << EOF
from pathlib import Path
p = Path("/home/z00637938/workspace/tilelang-mlir-ascend/examples/fp8_lighting_indexer.py")
t = p.read_text()
t = t.replace("torch.manual_seed(0)", "torch.manual_seed($SEED)")
p.write_text(t)
EOF
  cd $TLPATH/examples
  echo "=== seed $SEED ==="
  timeout 600 python3 fp8_lighting_indexer.py 2>&1 | grep -iE 'mismatch|passed|assertion' | tail -3
  cp "$BACKUP" "$KERNEL"
done
