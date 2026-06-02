#!/usr/bin/env python3
"""Sweep fp_attn_npuir — data is args-driven (shape from seq_len/dim, dtype from --dtype) => valid sweep.
Covers the bf16 dimension (flash_attn exposes --dtype, unlike sparse_mla)."""
import subprocess, sys, os, re
EX = "/home/z00637938/workspace/tilelang-mlir-ascend/examples/flash_attn_npuir.py"
ENV = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0",
           PYTHONPATH="/home/z00637938/workspace/tilelang-mlir-ascend:"
                      "/home/z00637938/workspace/bishengir/python_packages/mlir_core:"
                      "/home/z00637938/workspace/bishengir/python_packages/bishengir:"+os.environ.get("PYTHONPATH",""))
def run(args, timeout=120):
    try:
        r = subprocess.run([sys.executable, EX]+args, capture_output=True, text=True, timeout=timeout, env=ENV)
        out = r.stdout + r.stderr
        if "All check passed" in out: return "PASS"
        if "Mismatched elements" in out:
            m = re.search(r"\(([\d.]+)%\)", out); return f"FAIL({m.group(1)}%)" if m else "FAIL"
        if "AssertionError" in out: return "FAIL(assert)"
        m = re.search(r"(\w*Error)", out); return f"ERR({m.group(1)})" if m else f"ERR(rc={r.returncode})"
    except subprocess.TimeoutExpired: return "TIMEOUT"
COMBOS = [
    (["--dtype","bfloat16"], "bf16 (default shape)"),
    (["--seq_len","1024"], "seq_len=1024"),
    (["--seq_len","2048","--dim","64"], "seq2048 dim64"),
    (["--dim","256"], "dim=256"),
    (["--block_m","128","--block_n","128"], "block_m128 n128"),
    (["--dtype","bfloat16","--seq_len","1024"], "bf16 seq1024"),
]
print("=== flash_attn_npuir sweep (incl bf16) ===", flush=True)
for args, label in COMBOS:
    print(f"[fsweep] {label:24s} {' '.join(args):34s} -> {run(args)}", flush=True)
