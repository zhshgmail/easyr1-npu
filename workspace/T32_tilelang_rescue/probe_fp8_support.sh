#!/bin/bash
# Probe: does A3 (910C) and its tooling actually support fp8?

set +e
echo "=== TORCH FP8 TYPES ==="
python3 -c 'import torch; print("float8_e4m3fn:", hasattr(torch, "float8_e4m3fn")); print("float8_e5m2:", hasattr(torch, "float8_e5m2"))'

echo
echo "=== TORCH_NPU FP8 CAST ==="
python3 << 'EOF'
import torch, torch_npu
try:
    if hasattr(torch, "float8_e4m3fn"):
        x = torch.randn(8, dtype=torch.float32, device="npu")
        print("Trying x.to(float8_e4m3fn) on NPU...")
        try:
            y = x.to(torch.float8_e4m3fn)
            print(f"  OK: dtype={y.dtype} device={y.device}")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
    else:
        print("torch has no float8_e4m3fn type")
except Exception as e:
    print(f"NPU test setup failed: {e}")
EOF

echo
echo "=== CANN HEADERS / TOOLKIT FP8 REFS ==="
find /usr/local/Ascend -name '*fp8*' 2>/dev/null | head -5
find /usr/local/Ascend -name '*float8*' 2>/dev/null | head -5

echo
echo "=== ASCENDC TYPES IN HEADERS ==="
grep -rE 'float8|fp8_e4m3|fp8_e5m2|f8E4M3|f8E5M2' /usr/local/Ascend/cann-8.5.2/x86_64-linux/include 2>/dev/null | head -5
grep -rE 'float8|fp8' /usr/local/Ascend/cann-8.5.2/x86_64-linux/include/ascendc 2>/dev/null | head -5

echo
echo "=== BISHENGIR FP8 SUPPORT ==="
/home/z00637938/workspace/tilelang-mlir-ascend/3rdparty/AscendNPU-IR/build/install/bin/bishengir-compile --help 2>&1 | grep -iE 'fp8|float8|f8' | head -5

echo
echo "=== ASCENDNPU-IR FP8 DIALECT ==="
grep -rE 'fp8|float8|F8E4M3|F8E5M2' /home/z00637938/workspace/tilelang-mlir-ascend/3rdparty/AscendNPU-IR/bishengir 2>/dev/null | head -10

echo
echo "=== SOC SUPPORT QUERY ==="
python3 << 'EOF'
import torch, torch_npu
# Try to find soc info
import acl
try:
    name = acl.get_soc_name()
    print(f"SOC name: {name}")
except Exception as e:
    print(f"acl.get_soc_name failed: {e}")
EOF
