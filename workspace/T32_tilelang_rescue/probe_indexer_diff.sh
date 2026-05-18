#!/bin/bash
# Find WHICH 5 elements differ and inspect them in detail.

TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

KERNEL=$TLPATH/examples/fp8_lighting_indexer.py
BACKUP=/tmp/fp8_indexer.orig.py
[ -f "$BACKUP" ] || cp "$KERNEL" "$BACKUP"
cp "$BACKUP" "$KERNEL"

# Replace the assert_close with detailed diff printing
python3 << 'EOF'
from pathlib import Path
p = Path("/home/z00637938/workspace/tilelang-mlir-ascend/examples/fp8_lighting_indexer.py")
t = p.read_text()
OLD = 'torch.testing.assert_close(o.cpu().reshape(B, M, N), o_torch, rtol=1e-2, atol=1e-2)'
NEW = '''o_cpu = o.cpu().reshape(B, M, N)
    diff = (o_cpu - o_torch).abs()
    rdiff = diff / (o_torch.abs() + 1e-9)
    mismatched = (diff > 1e-2) & (rdiff > 1e-2)
    nfail = mismatched.sum().item()
    print(f"Mismatched: {nfail} / {o_cpu.numel()}")
    if nfail > 0:
        idx = mismatched.nonzero()
        for i in range(min(10, nfail)):
            b, m, n = idx[i].tolist()
            print(f"  [{b},{m},{n}]: o={o_cpu[b,m,n].item():.6f} ref={o_torch[b,m,n].item():.6f} diff={diff[b,m,n].item():.6f} rdiff={rdiff[b,m,n].item():.4f}")
            # Check if input value is at fp16 boundary
            print(f"    o_bits={o_cpu[b,m,n].view(torch.int16).item():#06x}, ref_bits={o_torch[b,m,n].view(torch.int16).item():#06x}")
'''
t = t.replace(OLD, NEW)
p.write_text(t)
EOF

cd $TLPATH/examples
echo "== Inspect 5 mismatches =="
timeout 600 python3 fp8_lighting_indexer.py 2>&1 | tail -25

cp "$BACKUP" "$KERNEL"
echo "(restored)"
