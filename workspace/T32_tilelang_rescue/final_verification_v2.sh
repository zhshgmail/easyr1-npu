#!/bin/bash
# v2: detect silent-pass (returncode 0 with no AssertionError) too.
TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd $TLPATH/examples
ops=(
  elementwise/vec_add_1d.py
  elementwise/vec_add_2d.py
  elementwise/vec_add_2d_dynamic_shape.py
  elementwise/vec_add_2d_multi_buffer.py
  elementwise/vec_add_auto_brc.py
  elementwise/atomic_add.py
  elementwise/atomic_add_dev.py
  elementwise/example_elementwise_add.py
  exp2.py
  log2.py
  vectorization_in_parallel.py
  mixcv_mixkernel.py
  gemm/example_gemm.py
  gemm/example_gemm_int82int32.py
  gemm/matmul.py
  gemm/matmul_dynamic_shape.py
  gemv/example_gemv.py
  norm/example_rms_norm.py
  norm/layer_norm.py
  flash_attn_npuir.py
  flash_attn_npuir_dev.py
  sparse_mla_fwd.py
  sparse_mla_fwd_dynamic_shape.py
  fp8_lighting_indexer.py
  deepseek_v32/sparse_mla_bwd.py
  deepseek_v32/sparse_mla_fwd.py
  deepseek_v4/example_act_quant_kernel.py
  deepseek_v4/example_fp8_gemm_kernel.py
  deepseek_v4/example_hc_split_sinkhorn_kernel.py
  deepseek_v4/example_mhc_post.py
  deepseek_v4/example_sparse_attn_kernel.py
  deepseek_v4/example_sparse_attn_kernel_highperf.py
  engram/engram_fwd.py
  engram/engram_bwd.py
  engram/engram_bwd_exp.py
  engram/engram_decode.py
)
pass=0
fail=0
fails=""
for op in "${ops[@]}"; do
  out=$(timeout 600 python3 $op 2>&1)
  rc=$?
  if [ $rc -eq 0 ] && ! echo "$out" | grep -qiE 'assertionerror|traceback|runtimeerror|error:|fatal'; then
    pass=$((pass+1))
  else
    # Show last-line of error
    last=$(echo "$out" | grep -iE 'assertionerror|error:|fatal|mismatch' | tail -1)
    echo "FAIL  $op  (rc=$rc) : $last"
    fail=$((fail+1))
    fails="$fails $op"
  fi
done
echo
echo "=========="
echo "Summary: $pass PASS, $fail FAIL of ${#ops[@]} total"
if [ $fail -gt 0 ]; then echo "Failing ops:$fails"; fi
