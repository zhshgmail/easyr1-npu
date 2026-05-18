#!/bin/bash
TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd $TLPATH/examples
for op in elementwise/atomic_add.py elementwise/vec_add_2d.py exp2.py log2.py gemm/example_gemm.py gemv/example_gemv.py norm/layer_norm.py deepseek_v32/sparse_mla_fwd.py deepseek_v4/example_sparse_attn_kernel.py engram/engram_fwd.py; do
  echo "== $op =="
  timeout 600 python3 $op 2>&1 | grep -iE 'passed|pass!|failed|mismatch|error:|assertion' | tail -3
done
