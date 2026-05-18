#!/usr/bin/env python3
"""tlfix-sweep — Phase 1 of /tilelang-fix loop.

Walk tilelang-mlir-ascend examples/, testing/npuir/, unittest/npuir/;
classify each op against KB §11 status enum; emit results.json for
/tlfix-triage to consume.

Usage:
    python3 sweep.py <tilelang_mlir_ascend_dir> <out_dir>
        [--include-snapshot]  (default skip 63 known-stale snapshot tests)
        [--timeout SEC]       (per-op timeout, default 300)
        [--filter PATTERN]    (only ops whose rel path contains PATTERN)
        [--mode {expert,developer,both}]  (TILELANG_ASCEND_MODE)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

STATUS_PATTERNS = [
    ("COMPILE_FAIL_MLIR_VERIFIER",
     re.compile(r"Generated MLIR module failed verification", re.IGNORECASE)),
    ("COMPILE_FAIL_TIR_PASS",
     re.compile(r"InternalError:\s*Check failed.*ascend_", re.IGNORECASE)),
    ("SNAPSHOT_DIFF",
     re.compile(r"are not identical", re.IGNORECASE)),
    ("PRECISION_FAIL_TOLERANCE",
     re.compile(r"Mismatched elements:\s*\d+", re.IGNORECASE)),
    ("PRECISION_FAIL_STRICT",
     re.compile(r"AssertionError.*torch\.all", re.IGNORECASE | re.DOTALL)),
    ("RUNTIME_FAIL_NPU",
     re.compile(r"aclrt\w+\s+execution failed|EL\d+_\w+", re.IGNORECASE)),
    ("IMPORT_FAIL",
     re.compile(r"^(ModuleNotFoundError|ImportError)", re.MULTILINE)),
]

PASS_PATTERNS = [
    re.compile(r"all\s*check(s)?\s*passed", re.IGNORECASE),
    re.compile(r"\bPASSED\b"),
    re.compile(r"Pass!"),
    re.compile(r"check\s+passed", re.IGNORECASE),
    re.compile(r"Demo check passed", re.IGNORECASE),
    re.compile(r"Comparison passed", re.IGNORECASE),
    re.compile(r"\baccuracy check passed\b", re.IGNORECASE),
    re.compile(r"^\d+\s+passed\b", re.MULTILINE),
]

SIGNATURE_EXTRACTORS = {
    "COMPILE_FAIL_MLIR_VERIFIER":
        re.compile(r"error:\s*['\"]?([\w.]+\.[\w.]+)['\"]?\s*op[^\n]*?(?:requires|expected|incorrect|failed)\s+[^\n]+", re.IGNORECASE),
    "COMPILE_FAIL_TIR_PASS":
        re.compile(r"InternalError:\s*Check failed:\s*\(([^)]+)\)\s*[^\n]*", re.IGNORECASE),
    "PRECISION_FAIL_TOLERANCE":
        re.compile(r"Mismatched elements:\s*(\d+)\s*/\s*(\d+)\s*\(([^)]+)\)"),
    "SNAPSHOT_DIFF":
        re.compile(r"'([^']+)' and '([^']+)' are not identical"),
}


def classify(stdout: str, stderr: str, returncode: int, elapsed: float, timeout: int):
    if elapsed >= timeout and returncode != 0:
        return ("TIMEOUT", f"exceeded {timeout}s")
    combined = stdout + "\n" + stderr
    is_likely_fail = returncode != 0 or any(
        m in combined for m in ["AssertionError", "Traceback", "error:", "FAIL"]
    )
    if is_likely_fail:
        for status, pat in STATUS_PATTERNS:
            if pat.search(combined):
                sig = ""
                if status in SIGNATURE_EXTRACTORS:
                    m = SIGNATURE_EXTRACTORS[status].search(combined)
                    if m:
                        sig = " | ".join(g for g in m.groups() if g)[:200]
                return (status, sig)
    for pat in PASS_PATTERNS:
        if pat.search(combined):
            return ("PASS", "")
    if returncode == 0:
        return ("PASS", "")
    return ("UNKNOWN", combined[-200:].replace("\n", " "))


def sweep_dir(root: Path, rel_glob: str, env: dict, args, results: dict):
    pattern = root / rel_glob
    for path_str in sorted(glob.glob(str(pattern), recursive=True)):
        path = Path(path_str)
        rel = str(path.relative_to(root))
        if args.filter and args.filter not in rel:
            continue
        if "unittest/npuir/" in rel and not args.include_snapshot:
            if "test_atomic" not in path.name:
                continue
        cwd = path.parent if path.parts[-2] == "npuir" else root
        t0 = time.time()
        try:
            p = subprocess.run(
                ["python3", str(path)],
                env=env,
                capture_output=True, text=True, timeout=args.timeout,
                cwd=cwd,
            )
            elapsed = time.time() - t0
            status, sig = classify(p.stdout, p.stderr, p.returncode, elapsed, args.timeout)
            results[rel] = {
                "status": status,
                "signature": sig,
                "returncode": p.returncode,
                "elapsed_s": round(elapsed, 1),
                "stdout_tail": p.stdout[-500:],
                "stderr_tail": p.stderr[-500:],
            }
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            results[rel] = {
                "status": "TIMEOUT",
                "signature": f"exceeded {args.timeout}s",
                "returncode": -1,
                "elapsed_s": round(elapsed, 1),
                "stdout_tail": "", "stderr_tail": "",
            }
        print(f"  {results[rel]['status']:35s}  {rel}  ({results[rel]['elapsed_s']}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tilelang_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--include-snapshot", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--mode", choices=["expert", "developer", "both"], default="expert")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.tilelang_dir.resolve()

    bishengir_pkgs = root / "3rdparty/AscendNPU-IR/build/install/python_packages"
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(root),
        str(bishengir_pkgs / "mlir_core"),
        str(bishengir_pkgs / "bishengir"),
        env.get("PYTHONPATH", "")
    ])
    env["PATH"] = str(root / "3rdparty/AscendNPU-IR/build/install/bin") + ":" + env.get("PATH", "")
    if args.mode != "both":
        env["TILELANG_ASCEND_MODE"] = "Expert" if args.mode == "expert" else "Developer"

    results = {}
    print("=== examples/ ===", flush=True)
    sweep_dir(root, "examples/**/*.py", env, args, results)
    print("=== testing/npuir/ ===", flush=True)
    sweep_dir(root, "testing/npuir/**/*.py", env, args, results)
    print("=== unittest/npuir/ ===", flush=True)
    sweep_dir(root, "unittest/npuir/test_*.py", env, args, results)

    summary = {"by_status": {}}
    for r in results.values():
        s = r["status"]
        summary["by_status"][s] = summary["by_status"].get(s, 0) + 1
    print(f"\n=== Summary ===")
    for s, n in sorted(summary["by_status"].items(), key=lambda x: -x[1]):
        print(f"  {s:35s}  {n}")
    print(f"  TOTAL: {len(results)}")

    out_json = args.out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
