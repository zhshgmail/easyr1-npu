#!/bin/bash
TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd $TLPATH/examples
for op in elementwise/vec_add_1d.py norm/example_rms_norm.py flash_attn_npuir.py gemm/matmul.py atomic_add.py; do
  echo "== $op =="
  timeout 300 python3 $op 2>&1 | grep -iE 'passed|pass!|failed|mismatch|error:|assertion' | tail -3
done
