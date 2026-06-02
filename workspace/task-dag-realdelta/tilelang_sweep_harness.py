#!/usr/bin/env python3
"""Systematic sweep harness for tilelang-ascend sparse_mla_fwd — find operator issues across
param combinations. Runs the (fixed) example as subprocess per combo, classifies verdict.

Discipline (verify_harness_propagates_swept_param): sparse_mla_fwd's __main__ builds ALL data
shapes from args.* (verified line 484-498), so CLI-arg sweeps are valid. dtype is hardcoded fp16
(line 70+469) — swept separately via a patched copy, labeled.
"""
import subprocess, sys, itertools, os

EX = "/home/z00637938/workspace/tilelang-mlir-ascend/examples/sparse_mla_fwd.py"
ENV = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0",
           PYTHONPATH="/home/z00637938/workspace/tilelang-mlir-ascend:"
                      "/home/z00637938/workspace/bishengir/python_packages/mlir_core:"
                      "/home/z00637938/workspace/bishengir/python_packages/bishengir:" + os.environ.get("PYTHONPATH",""))

def run(args, exfile=EX, timeout=90):
    try:
        r = subprocess.run([sys.executable, exfile] + args, capture_output=True, text=True, timeout=timeout, env=ENV)
        out = r.stdout + r.stderr
        if "All check passed" in out: return "PASS"
        if "Mismatched elements" in out:
            import re; m = re.search(r"Mismatched elements: \d+ / \d+ \(([\d.]+)%\)", out)
            return f"FAIL({m.group(1)}%)" if m else "FAIL"
        if "assert" in out.lower() and "AssertionError" in out: return "FAIL(assert)"
        if "Error" in out or "error" in out: 
            import re; m = re.search(r"(\w*Error): (.{0,60})", out)
            return f"ERR({m.group(1)})" if m else "ERR"
        return "NO-VERDICT"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

# Sweep grid — combos that exercise distinct code paths. Keep modest to fit wall-clock.
COMBOS = [
    (["--heads","4"], "heads=4 (fixed-path)"),
    (["--heads","16"], "heads=16"),
    (["--heads","64"], "heads=64 (V4-Flash real)"),
    (["--top_k","256"], "top_k=256"),
    (["--block_i","128","--top_k","128"], "block_i=128"),
    (["--seq_len","256"], "seq_len=256"),
    (["--seq_len_kv","512","--top_k","256"], "seq_kv=512"),
    (["--dim","128","--tail_dim","64"], "dim=128"),
    (["--batch_size","2"], "batch=2"),
    (["--num_kernels","48"], "num_kernels=48"),
]
print("=== sparse_mla_fwd systematic sweep ===", flush=True)
for args, label in COMBOS:
    print(f"[sweep] {label:28s} {' '.join(args):30s} -> {run(args)}", flush=True)
