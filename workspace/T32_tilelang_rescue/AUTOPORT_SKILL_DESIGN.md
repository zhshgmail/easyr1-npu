# Design — `/tilelang-auto-port` skill (auto-discover + auto-fix MLIR bugs)

> User vision (2026-05-18 03:27): 「我们之所以做这个自动化，就是想实现自动发现
> mlir 的问题，并自动修复。记住这个这样才能实现基于 tilelang 的三方件迁移
> 到 npu 的全自动化.」

This file is the **design** of a future `/tilelang-auto-port` skill that
turns the manual T32 cold-drive into a one-command workflow. Status:
draft — not implemented yet. KB content in `KB_TILELANG_ASCEND.md` is
the input substrate.

---

## 1. The 5-step cold-loop

```
┌─── auto-discover ───┐    ┌── auto-classify ──┐    ┌─── auto-fix ────┐
│  examples/ sweep:    │    │  triage tree:      │    │  KB recipe →     │
│  run every .py       │ ─> │  precision FAIL?    │ ─> │  apply patch     │
│  collect PASS/FAIL   │    │  compile FAIL?      │    │  (template /     │
│  classify output     │    │  HW assumption?     │    │   TIR pass /     │
└──────────────────────┘    └────────────────────┘    │   tol adjust)    │
         ↑                                              └────────┬─────────┘
         │                                                       │
         │           ┌─── KB feedback ────┐                       ↓
         └─────────  │  new fail type →    │  <─────  ┌─ auto-verify ─┐
                    │  new triage rule    │          │  rerun + diff  │
                    │  new fix recipe     │          │  confirm PASS  │
                    │  new HW probe Q     │          └────────────────┘
                    └─────────────────────┘
```

Each iteration **shrinks the FAIL list** and **grows the KB**. After
N iterations, FAIL list reaches steady-state (only fundamentally
hardware-unsupported ops remain) and KB contains a complete bug class
taxonomy.

---

## 2. Auto-discover (step 1)

**Input**: a tilelang-mlir-ascend checkout (or PTO checkout — both
supported).

**Output**: a JSON with one entry per `examples/**/*.py`:

```jsonc
{
  "examples/elementwise/vec_add_2d.py": {
    "argsets": [{}, {"--M": 32, "--N": 32, "--block_M": 4, "--block_N": 4}],
    "results": [
      {"args": {}, "status": "PASS", "stdout_tail": "All check passed!", "elapsed_s": 38},
      {"args": {...}, "status": "PASS", "stdout_tail": "All check passed!", "elapsed_s": 12}
    ]
  },
  "examples/deepseek_v4/example_fp8_gemm_kernel.py": {
    "argsets": [{}],
    "results": [
      {"args": {}, "status": "COMPILE_FAIL",
       "error_class": "MLIR_VERIFIER",
       "error_signature": "tensor.empty op incorrect number of dynamic sizes, has 0, expected 1",
       "stdout_tail": "...",
       "elapsed_s": 14}
    ]
  }
}
```

**Implementation** (Python script):
- Walk `examples/` for `*.py`
- For each, decide arg set: default args + bug-shape variants from KB §8.4
- Subprocess-spawn each run with timeout 600s
- Parse stdout: look for "All check passed" / "Check passed" / "PASSED" /
  "AssertionError" / "RuntimeError" / "error: " etc.
- Classify into status enum (see §3)

**KB feedback hook**: when a new error signature is encountered (not
matched by any existing recipe in §7), prompt user to triage and add a
new row.

---

## 3. Auto-classify (step 2)

**Status enum**:

| Status | Trigger |
|--------|---------|
| `PASS` | Output contains `(check passed|Passed|PASSED|All checks passed)` |
| `PRECISION_FAIL_STRICT` | `assert torch.all(==)` AssertionError with values visually matching (§10.2 family) |
| `PRECISION_FAIL_TOLERANCE` | `assert_close` fail with absolute tol < expected (§10.1 family) |
| `COMPILE_FAIL_MLIR_VERIFIER` | Stderr contains "Generated MLIR module failed verification" |
| `COMPILE_FAIL_TIR_PASS` | Stderr contains "InternalError: Check failed" inside `ascend_*` pass |
| `RUNTIME_FAIL_NPU` | Stderr contains "RuntimeError: ... NPU" or "aclrtLaunchKernel failed" |
| `TIMEOUT` | Wall-clock > 600s |
| `IMPORT_FAIL` | Process exits before kernel compile (missing dep) |
| `UNKNOWN` | None of the above match |

**Triage tree** for each non-PASS status:

```
COMPILE_FAIL_MLIR_VERIFIER
├── signature "tensor.empty op incorrect number of dynamic sizes"
│   → KB §10.3 → fix: codegen_npuir_dev.cc EmptyOp build sites
├── signature "incorrect type of operand"
│   → TBD (new bug class — add row)
└── ...

PRECISION_FAIL_TOLERANCE
├── max_abs_diff ~ 1 ULP of dtype
│   → KB §10.1, §10.2 family → likely Cast round_mode mismatch
│   → fix: probe HW round_mode (KB §9b) + adjust codegen
├── max_abs_diff >> 1 ULP, % wrong > 1%
│   → real arithmetic bug → bisect lowering pipeline
└── ...

COMPILE_FAIL_TIR_PASS
├── signature "Extent must be an integer constant"
│   → KB §7 → user kernel uses non-constexpr UB size — kernel-side fix
└── ...
```

The triage tree IS the KB §10.x sections, formalized as decision rules.

---

## 4. Auto-fix (step 3)

For each `(status, signature)` pair that matches a KB recipe, the auto-
fix step produces a **patch** + **proof-of-fix predicate**.

| Recipe class | Patch format | Verify how |
|--------------|--------------|------------|
| Codegen C++ fix (e.g. §10.3 EmptyOp) | `.patch` against `tilelang-mlir-ascend/src/target/...` | Rebuild bishengir+tilelang-mlir; rerun the failing op + 3 regression ops; all must PASS |
| Template C++ fix (e.g. PTO §10 Option A) | `.patch` against `src/tl_templates/ascend/common.h` | Copy patched header to installed location; rerun bug case (expect compile error w/ message) + baseline (expect still PASS) |
| TIR pass fix | New `.cc` file + CMakeLists update | Rebuild; verify failing kernel now compiles + passes |
| Tolerance adjustment | `.patch` against `examples/.../test_xxx.py` | Rerun; verify both old-failing AND new-passing |
| Kernel-side workaround | `.py` template generator | Generate fixed `.py`; verify it passes |

**Authority gate**: auto-fix CAN propose, but **MUST NOT auto-commit** to
upstream. Output is a "fix bundle" directory containing the patch +
verification log + draft PR description. User then chooses to push.

---

## 5. Auto-verify (step 4)

For each proposed fix:
1. Apply patch
2. Rebuild affected component (cmake/ninja)
3. Re-run the failing op — expect new status `PASS`
4. Re-run a regression set of N=10 randomly-sampled PASSING ops —
   confirm none regress
5. If both 3+4 hold, mark fix `VERIFIED`. Otherwise rollback patch and
   re-classify as `FIX_FAILED` with diff.

The fix bundle then contains:
- `patch.diff`
- `verification.log` (before/after run logs)
- `regression.log` (the N=10 sample's PASS status)
- `pr_description.md` (auto-generated from signature + recipe)

---

## 6. KB feedback (step 5)

The loop CLOSES here. Anything that exits triage as `UNKNOWN` (no
matching signature) MUST trigger a KB write:

1. Append new row to §8.1 in `KB_TILELANG_ASCEND.md`
2. Open new §10.X subsection (skeleton): error signature, stdout
   excerpt, HW context relevance pointer to §9b
3. If a fix is found, append fix recipe to §7 + §10.X
4. If a HW assumption is unverified (e.g. "what's the Cast round mode?"),
   add a probe to a "open probes" queue (TBD section in KB)

After M iterations, the KB §7 / §10 grow until the failure rate
plateaus.

---

## 7. Mode selection (which backend to target)

`/tilelang-auto-port` accepts `--backend pto|mlir|both`:

- `pto`: targets `tilelang-ascend@ascendc_pto` (the wheel users have
  TODAY). Bug surface = #996-style template bugs.
- `mlir`: targets `tilelang-mlir-ascend` (future). Bug surface = codegen
  bugs like §10.3, plus shared TIR pass bugs.
- `both`: runs PTO first, then MLIR, then **diff the results** — any op
  PASSes on one but FAILs on the other is a high-value bug (suggests
  the surviving backend's mechanism is the missing thing on the other).

---

## 8. Implementation order

Phase 1 — `auto-discover` only (script that runs all examples and
classifies into the status enum). 1 day of work. Output: KB §8.1
auto-generated daily.

Phase 2 — `triage tree` (signature matching to KB §10.x). 2-3 days.

Phase 3 — `auto-fix` for known recipes (start with EmptyOp + cast
round_mode). 3-5 days.

Phase 4 — `auto-verify` + bundle output. 2-3 days.

Phase 5 — `KB feedback` automation (new-row prompts, probe queue). 1-2
days.

**Total**: ~2 weeks for a working v1.

---

## 9. What this skill IS NOT

- Not a compiler from scratch — uses bishengir / tilelang as black box
- Not a new DSL — runs existing `examples/`
- Not a CI replacement — runs alongside, complementary
- Not auto-merge to upstream — only proposes patches; human gates push

---

## 10. Bootstrapping today (Phase 1 sketch)

The minimum viable Phase 1 is a 200-line Python script:

```python
# auto_port_discover.py
import json, subprocess, glob, time
from pathlib import Path

EXAMPLES_DIR = Path("/home/z00637938/workspace/tilelang-mlir-ascend/examples")
PYTHONPATH_PREFIX = f"{EXAMPLES_DIR.parent}:{EXAMPLES_DIR.parent}/3rdparty/AscendNPU-IR/build/install/python_packages/mlir_core:{EXAMPLES_DIR.parent}/3rdparty/AscendNPU-IR/build/install/python_packages/bishengir"

def classify(stdout: str, stderr: str, returncode: int) -> tuple[str, str]:
    """Return (status, signature)."""
    combined = stdout + stderr
    if any(m in combined for m in ["All check passed", "Check passed", "All checks passed", "PASSED", "Pass!", "accuracy check passed", "post check passed", "out check passed"]):
        return ("PASS", "")
    if "Generated MLIR module failed verification" in combined:
        # Extract the verifier message
        sig = next((line for line in combined.split("\n") if "error:" in line and "verif" in line.lower()), "MLIR_VERIFIER_UNKNOWN")
        return ("COMPILE_FAIL_MLIR_VERIFIER", sig.strip()[:200])
    if "InternalError: Check failed" in combined and "ascend_" in combined:
        return ("COMPILE_FAIL_TIR_PASS", "Check failed in ascend pass")
    if "AssertionError" in combined and "Mismatched elements" in combined:
        # Tolerance-based fail
        return ("PRECISION_FAIL_TOLERANCE", "assert_close mismatch")
    if "AssertionError" in combined and "torch.all" in combined:
        return ("PRECISION_FAIL_STRICT", "strict equality fail")
    if "TimeoutExpired" in combined:
        return ("TIMEOUT", "")
    return ("UNKNOWN", combined[-200:])

results = {}
for py in glob.glob(f"{EXAMPLES_DIR}/**/*.py", recursive=True):
    if "torch_tl_ops" in py:  # has its own infra
        continue
    rel = Path(py).relative_to(EXAMPLES_DIR.parent)
    print(f"running {rel}")
    t0 = time.time()
    p = subprocess.run(
        ["python3", str(py)],
        env={"PYTHONPATH": PYTHONPATH_PREFIX, "PATH": "...", "ASCEND_RT_VISIBLE_DEVICES": "0", ...},
        capture_output=True, text=True, timeout=400)
    status, sig = classify(p.stdout, p.stderr, p.returncode)
    results[str(rel)] = {
        "status": status,
        "signature": sig,
        "elapsed_s": time.time() - t0,
        "stdout_tail": p.stdout[-500:],
        "stderr_tail": p.stderr[-500:],
    }
    print(f"  → {status} {sig[:80]}")

with open("auto_port_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

Run this every time tilelang-mlir-ascend updates; diff vs previous run
to see regressions. This is the foundation.

---

## 11. KB ⇆ skill round-trip

The skill **reads** `KB_TILELANG_ASCEND.md` §7 (recipes) and §10.x
(deep-dives) as triage knowledge.

The skill **writes** to:
- `KB_TILELANG_ASCEND.md` §8.1 (new rows after sweep)
- `KB_TILELANG_ASCEND.md` §10.X (new sections for UNKNOWN failures)
- A new `auto_port_runs/` directory for raw JSON snapshots (one per run)

Result: knowledge accumulates monotonically. Each run is a contribution.

---

## See also

- `KB_TILELANG_ASCEND.md` — current KB (substrate)
- `REPORT.md` — first manual cold-drive (T32.3 PTO rescue)
- `OPTION_A_RESULT.md` — first manual fix (T32.6 PTO static_assert)
- `ARCHITECTURE_EXPLAINED.md` — backend pipeline reference
