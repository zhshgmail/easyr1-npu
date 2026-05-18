#!/bin/bash
# Make torch reference use fp32 throughout (matching NPU kernel's accum_dtype="float32").
# NPU kernel: gemm(fp16→fp32 L0C) → fp16 store → fp32 vcast → fp32 vmul/reduce/add → fp16 vcast → fp16 store.
# Torch ref BEFORE: fp16 matmul → fp16 relu → fp32 multiply/sum → fp16 final cast.
# Torch ref AFTER: fp32 matmul → fp32 relu → fp32 multiply/sum → fp16 final cast.

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
OLD = '''    q_reshaped = q_ptr.view(B, M * H, K)
    q_reshaped.to(dtype=torch.float16)
    temp = torch.matmul(q_reshaped, k_ptr)  # (B, M * H, N)
    temp = temp.view(B, M, H, N)
    temp_relu = torch.relu(temp)
    temp_relu = temp_relu.to(dtype=torch.float32)'''
NEW = '''    q_reshaped = q_ptr.view(B, M * H, K)
    # T32 fix: NPU kernel uses fp32 accum in L0C and fp32 vmul/reduce.
    # Match that precision in the reference by promoting to fp32 BEFORE matmul.
    q_reshaped_32 = q_reshaped.to(dtype=torch.float32)
    k_ptr_32 = k_ptr.to(dtype=torch.float32)
    temp = torch.matmul(q_reshaped_32, k_ptr_32)  # (B, M * H, N) in fp32
    temp = temp.view(B, M, H, N)
    temp_relu = torch.relu(temp)
    # Already fp32 from matmul promotion'''
if OLD not in t:
    print("FAIL: pattern not found")
    exit(1)
t = t.replace(OLD, NEW)
p.write_text(t)
print("ref upgraded to fp32 throughout")
EOF

cd $TLPATH/examples
echo "== fp8_indexer with fp32 ref throughout =="
timeout 600 python3 fp8_lighting_indexer.py 2>&1 | tail -8

cp "$BACKUP" "$KERNEL"
echo "(restored)"
