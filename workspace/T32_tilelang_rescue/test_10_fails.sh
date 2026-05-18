#!/bin/bash
TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd $TLPATH
echo "=== Run the 10 previously failing tests ==="
python3 -m pytest testing/npuir/arith_ops/test_tensor_extract_dev.py testing/npuir/arith_ops/test_vadd_1x1_dev.py --tb=line -q 2>&1 | tail -25
