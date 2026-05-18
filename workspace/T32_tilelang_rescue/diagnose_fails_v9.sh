#!/bin/bash
TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd $TLPATH/examples
echo "=== verify our tolerance fix is in place ==="
grep -n "rtol=" fp8_lighting_indexer.py | head -3
echo
echo "=== fp8_lighting_indexer full output ==="
timeout 600 python3 fp8_lighting_indexer.py 2>&1 | tail -20
echo
echo "=== engram/engram_bwd_exp.py full output ==="
timeout 600 python3 engram/engram_bwd_exp.py 2>&1 | tail -25
