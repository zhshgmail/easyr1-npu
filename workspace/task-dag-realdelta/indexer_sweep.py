#!/usr/bin/env python3
"""Sweep harness for fp8_lighting_indexer — using the HARNESS-CORRECTED copy (honors --h for data).
Verifies operator issues across h/m/n/k/bs combos. The corrected copy makes __main__ use args.* (the
upstream example hardcodes H=32, which caused my earlier false 'bug')."""
import subprocess, sys, os, re
EX = "/home/z00637938/workspace/task-dag-realdelta/idx_harness_correct.py"
ENV = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0",
           PYTHONPATH="/home/z00637938/workspace/tilelang-mlir-ascend:"
                      "/home/z00637938/workspace/bishengir/python_packages/mlir_core:"
                      "/home/z00637938/workspace/bishengir/python_packages/bishengir:"+os.environ.get("PYTHONPATH",""))
def run(args, timeout=95):
    try:
        r = subprocess.run([sys.executable, EX]+args, capture_output=True, text=True, timeout=timeout, env=ENV)
        out = r.stdout + r.stderr
        if "Mismatched elements" in out:
            m = re.search(r"\(([\d.]+)%\)", out); return f"FAIL({m.group(1)}%)" if m else "FAIL"
        if "AssertionError" in out: return "FAIL(assert)"
        if r.returncode == 0: return "PASS"   # indexer has no success banner; exit 0 + no mismatch = pass
        m = re.search(r"(\w*Error)", out); return f"ERR({m.group(1)})" if m else f"ERR(rc={r.returncode})"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
COMBOS = [
    (["--h","4"], "h=4"), (["--h","8"], "h=8"), (["--h","16"], "h=16"),
    (["--h","32"], "h=32 (default)"), (["--h","64"], "h=64 (V4-Flash idx)"),
    (["--m","512","--n","1024"], "smaller m/n"),
    (["--bs","128"], "bs=128"), (["--k","128"], "k=128"),
]
print("=== fp8_lighting_indexer sweep (harness-corrected, honors --h) ===", flush=True)
for args, label in COMBOS:
    print(f"[isweep] {label:24s} {' '.join(args):22s} -> {run(args)}", flush=True)
