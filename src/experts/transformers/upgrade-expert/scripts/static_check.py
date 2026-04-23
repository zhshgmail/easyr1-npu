#!/usr/bin/env python3
"""Static Python check — py_compile every file, dry-import the top-level package.

Purpose: catch the class of failure where an agent edits source files (e.g. via
code-path-sweep replacements), the Dockerfile's `pip install -e .` runs fine
(it doesn't actually import the package), but the package is broken because
the edits produced invalid Python. Runtime smoke would then fail at first
import.

2026-04-22 round 2 agent specifically produced `from ..utils.device import ...`
inserted inside an unclosed `from ..utils.torch_functional import (` block in
fsdp_workers.py. py_compile would have caught it immediately.

This script is called:
  - By port-worker agent's Stop hook (check_port_worker.sh)
  - By docs/workflow/port_state_machine.yaml P5 invariant
  - Can be run manually too

Exit codes:
  0  = all clean (py_compile OK + dry-import OK)
  1  = py_compile failed on at least one file
  2  = py_compile OK but dry-import fails (ImportError, name errors)
  3  = usage error
"""
from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path


def py_compile_all(sources: list[Path]) -> tuple[bool, list[str]]:
    """Compile each .py file. Return (all_ok, failure_messages)."""
    failures: list[str] = []
    for src in sources:
        if not src.exists():
            failures.append(f"{src}: file not found")
            continue
        if not src.suffix == ".py":
            continue
        try:
            py_compile.compile(str(src), doraise=True, quiet=1)
        except py_compile.PyCompileError as e:
            failures.append(f"{src}: {e.msg.strip()}")
        except Exception as e:  # SyntaxError wrapper edge cases
            failures.append(f"{src}: {type(e).__name__}: {e}")
    return (len(failures) == 0, failures)


def dry_import(package: str, python_bin: str = sys.executable) -> tuple[bool, str]:
    """Try `python -c 'import <package>'`. Return (ok, stderr_on_fail)."""
    try:
        result = subprocess.run(
            [python_bin, "-c", f"import {package}; print('OK')"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return (True, "")
        return (False, result.stderr or result.stdout)
    except subprocess.TimeoutExpired:
        return (False, f"import timed out after 60s — circular import or heavy module-load side effect?")


def find_edited_py_files(edited_paths: list[str]) -> list[Path]:
    """Expand arg list: files are used as-is, dirs are walked for *.py."""
    out: list[Path] = []
    for arg in edited_paths:
        p = Path(arg)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(p.rglob("*.py"))
        else:
            print(f"WARN: path not found: {p}", file=sys.stderr)
    # dedup, sort
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="Static Python check — py_compile + dry-import")
    ap.add_argument(
        "--files", nargs="+", default=[],
        help="Specific files (or dirs — walked recursively) to py_compile"
    )
    ap.add_argument(
        "--import-package", default=None,
        help="Package name to `python -c 'import <pkg>'` as dry-import test"
    )
    ap.add_argument(
        "--python-bin", default=sys.executable,
        help="Python binary to use for dry-import (default: current interpreter)"
    )
    ap.add_argument(
        "--report", default=None,
        help="Write JSON report to path (for PROGRESS.md citation)"
    )
    ap.add_argument(
        "--edited-list", default=None,
        help="Path to a text file listing edited files, one per line (alt to --files)"
    )
    args = ap.parse_args()

    files: list[str] = list(args.files)
    if args.edited_list:
        try:
            with open(args.edited_list) as f:
                files.extend([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"ERROR: --edited-list file not found: {args.edited_list}", file=sys.stderr)
            return 3

    if not files and not args.import_package:
        print("ERROR: must pass at least one of --files, --edited-list, --import-package", file=sys.stderr)
        return 3

    report = {
        "py_compile": {"checked": 0, "failures": []},
        "dry_import": {"package": args.import_package, "ok": None, "error": None},
    }
    exit_code = 0

    if files:
        py_sources = find_edited_py_files(files)
        print(f"py_compile: checking {len(py_sources)} Python files...")
        ok, failures = py_compile_all(py_sources)
        report["py_compile"]["checked"] = len(py_sources)
        report["py_compile"]["failures"] = failures
        if not ok:
            print(f"\n❌ py_compile FAILED for {len(failures)} files:")
            for f in failures[:10]:
                print(f"  {f}")
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more")
            exit_code = 1

    if args.import_package and exit_code == 0:
        print(f"\ndry-import: `python -c 'import {args.import_package}'`...")
        ok, err = dry_import(args.import_package, python_bin=args.python_bin)
        report["dry_import"]["ok"] = ok
        report["dry_import"]["error"] = err.strip() if not ok else None
        if not ok:
            print(f"\n❌ dry-import FAILED for package '{args.import_package}':")
            # trim error to last ~20 lines
            err_tail = "\n".join(err.strip().splitlines()[-20:])
            print(err_tail)
            exit_code = 2

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    if exit_code == 0:
        print(f"\n✅ static_check PASS")
    else:
        print(f"\n❌ static_check FAIL (exit {exit_code})")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
