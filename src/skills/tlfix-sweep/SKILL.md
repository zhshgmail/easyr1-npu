---
name: tlfix-sweep
description: Phase 1 of /tilelang-fix loop. Walk tilelang-mlir-ascend examples/, testing/npuir/, unittest/npuir/ subdirs; classify each op against KB §11 status enum (PASS / COMPILE_FAIL_MLIR_VERIFIER / PRECISION_FAIL_* / SNAPSHOT_DIFF / RUNTIME_FAIL_NPU / TIMEOUT / UNKNOWN). Emits results.json for /tlfix-triage to consume.
---

# /tlfix-sweep — Phase 1 (auto-discover)

Standalone runnable; called by `/tilelang-fix` orchestrator.

## Inputs

- `tilelang_dir`: path to tilelang-mlir-ascend checkout (required)
- `out_dir`: where to write results.json (required)
- `--include-snapshot`: include the ~63 stale-snapshot tests in
  `unittest/npuir/` (default skip; they're known-not-real-bugs)
- `--timeout SEC`: per-op timeout (default 300)
- `--filter PATTERN`: only ops whose path contains PATTERN
- `--mode {expert,developer,both}`: which TILELANG_ASCEND_MODE to use;
  `both` runs each test twice. Default `expert` (the upstream default).

## Status enum (per KB §11)

| Status | Trigger |
|--------|---------|
| `PASS` | rc==0 AND no AssertionError/Traceback/error in output |
| `PRECISION_FAIL_STRICT` | `assert torch.all(==)` AssertionError |
| `PRECISION_FAIL_TOLERANCE` | `assert_close` Mismatch with element count |
| `COMPILE_FAIL_MLIR_VERIFIER` | "Generated MLIR module failed verification" |
| `COMPILE_FAIL_TIR_PASS` | "InternalError: Check failed" inside ascend_* pass |
| `RUNTIME_FAIL_NPU` | aclrtLaunchKernel / EL* error code |
| `SNAPSHOT_DIFF` | "are not identical" (IR fixture comparison) |
| `IMPORT_FAIL` | ModuleNotFoundError / ImportError early |
| `TIMEOUT` | wall-clock > --timeout |
| `UNKNOWN` | anything else |

## Output schema

`results.json`:
```jsonc
{
  "summary": {"by_status": {"PASS": 326, "SNAPSHOT_DIFF": 63, ...}},
  "results": {
    "examples/elementwise/vec_add_2d.py": {
      "status": "PASS",
      "signature": "",
      "returncode": 0,
      "elapsed_s": 38.2,
      "stdout_tail": "...All check passed!...",
      "stderr_tail": ""
    },
    ...
  }
}
```

## How to invoke

```bash
bash run.sh <tilelang_dir> <out_dir>
# or
python3 sweep.py <tilelang_dir> <out_dir> [--include-snapshot] [--timeout SEC] [--filter PATTERN]
```

## See also

- `tilelang-fix/SKILL.md` — orchestrator
- `tlfix-triage/SKILL.md` — Phase 2 (consumes our results.json)
