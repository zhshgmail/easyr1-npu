#!/bin/bash
# Try making the torch reference use fp32 accumulator for matmul, matching NPU's accum_dtype.

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
# Force fp32 accumulator in torch reference matmul, matching NPU's L0C accum_dtype="float32".
t = t.replace(
    "q_reshaped.to(dtype=torch.float16)\n    temp = torch.matmul(q_reshaped, k_ptr)",
    "q_reshaped32 = q_reshaped.to(dtype=torch.float32)\n    k_ptr32 = k_ptr.to(dtype=torch.float32)\n    temp = torch.matmul(q_reshaped32, k_ptr32).to(dtype=torch.float16)  # T32: fp32 accum to match NPU L0C"
)
p.write_text(t)
EOF

cd $TLPATH/examples
echo "== fp8_indexer with fp32-accum reference =="
timeout 600 python3 fp8_lighting_indexer.py 2>&1 | grep -iE 'passed|pass!|failed|mismatch|error:|assertion' | tail -5

cp "$BACKUP" "$KERNEL"
echo "(restored)"
