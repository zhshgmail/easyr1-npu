#!/bin/bash
TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd $TLPATH
echo "=== testing/npuir suite (excluding 'broken' dir) ==="
# Run all; don't stop on first failure; line-level traceback for failures
timeout 5400 python3 -m pytest testing/npuir --ignore=testing/npuir/broken --tb=line -q --no-header 2>&1 | tail -80
