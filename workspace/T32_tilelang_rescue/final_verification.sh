#!/bin/bash
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
  result=$(timeout 600 python3 $op 2>&1 | grep -iE 'passed|pass!|failed|mismatch|error:|AssertionError|Comparison passed' | tail -3)
  if echo "$result" | grep -iqE 'passed|pass!|comparison passed'; then
    if echo "$result" | grep -iqE 'failed|assertion|mismatch'; then
      # contains both pass and fail strings — likely got past one assertion then failed another
      echo "FAIL  $op : $(echo $result | head -c 200)"
      fail=$((fail+1))
      fails="$fails\n$op"
    else
      pass=$((pass+1))
    fi
  else
    echo "FAIL  $op : $(echo $result | head -c 200)"
    fail=$((fail+1))
    fails="$fails\n$op"
  fi
done

echo
echo "=========="
echo "Summary: $pass PASS, $fail FAIL of ${#ops[@]} total"
if [ $fail -gt 0 ]; then
  echo -e "Failing ops:$fails"
fi
