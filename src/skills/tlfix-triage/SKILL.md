---
name: tlfix-triage
description: Phase 2 of /tilelang-fix. Consume sweep results.json; for each FAIL row, match its (status, signature) against KB §11 taxonomy and §10.x deep-dive patterns. Output triage.json with hypothesis + candidate-patch-recipe pointer (or UNKNOWN → ask for KB extension).
---

# /tlfix-triage — Phase 2 (classify → KB hypothesis)

Standalone runnable; called by `/tilelang-fix` orchestrator.

## Inputs

- `results_json`: path to sweep's `results.json`
- `kb_path`: path to `KB_TILELANG_ASCEND.md` (default:
  `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md`)
- `out_dir`: where to write `triage.json`

## Triage tree (KB §11 mapping)

For each FAIL row in results.json, walk this decision tree:

```
status = COMPILE_FAIL_MLIR_VERIFIER
├── signature contains "tensor.empty op incorrect number of dynamic sizes"
│   → KB §11.1 / §10.3.1 → recipe: apply_emptyop_fix_v*.py family
├── signature contains "tensor.dim op operand must be tensor"
│   → KB §11.1 / §10.3.1 v4 → recipe: tensor/memref DimOp dispatch
├── signature contains "tensor.collapse_shape op expected dim N to be dynamic"
│   → KB §11.1 v5 → recipe: ReshapeTensorImpl reassoc-aware result type
├── signature contains "hivm.hir.vadd|vmul|... operand at index 0 is vector-only"
│   → KB §11.2 v11 → recipe: commutative-op operand swap
├── signature contains "hivm.hir.store op only support store ub to gm"
│   → KB §11.4 → recipe: check test args order vs wrapper signature
└── ... (extend per new findings)

status = PRECISION_FAIL_STRICT (assert torch.all(==))
└── KB §11.3.1 → likely Cast op round_mode mismatch
    → recipe: change round_mode="round" → "rint" in kernel

status = PRECISION_FAIL_TOLERANCE
├── if mismatch_count / total < 0.001% AND values match to N digits
│   → KB §11.3.2 → fp16 cross-impl ULP noise
│   → recipe: relax atol/rtol (apply_fp8_indexer_tol_fix.py pattern)
├── if values are systematically zero (acc-buffer never accumulated)
│   → KB §12 R-KA-13 (E5 workaround verified) — broadcast-operand schedule-locality
│   → recipe: find a working T.vsub elsewhere in the same kernel; check that
│     its broadcast operand is built via Python `T.serial` scalar-fill INSIDE
│     the same inner pipelined iter immediately before the vsub. Replicate
│     that construction for the failing vsub. See challenge_patterns/12.
├── if cosine ~0.5 vs autograd / kernel direction half-right
│   → same as above (R-KA-13 family) — vsub silently no-op'd
│   → recipe: try E5 first; only file upstream after E5 also fails
├── else
│   → real arithmetic bug → bisect lowering pipeline (manual)

status = SNAPSHOT_DIFF (unittest/npuir)
└── KB §11.X → stale mlir_files/ reference vs newer compiler output
    → recipe: NOT A REAL BUG; either regenerate fixtures OR defer

status = UNKNOWN
└── prompt user/agent to: (a) classify manually, (b) add new KB §11 row
```

## Output schema

`triage.json`:
```jsonc
{
  "results": [
    {
      "op": "examples/deepseek_v4/example_fp8_gemm_kernel.py",
      "status": "COMPILE_FAIL_MLIR_VERIFIER",
      "signature": "tensor.empty op | incorrect number of dynamic sizes",
      "hypothesis": "KB §10.3.1 family — missing dynamicSizes operand for symbolic-shape EmptyOp",
      "recipe_candidates": [
        "workspace/T32_tilelang_rescue/apply_emptyop_fix.py",
        "workspace/T32_tilelang_rescue/apply_emptyop_fix_v4.py"
      ],
      "confidence": "high"
    },
    ...
  ]
}
```

## How to invoke

```bash
python3 triage.py <results.json> <out_dir> --kb <kb_path>
```

## See also

- `tlfix-sweep/SKILL.md` — produces our input
- `tlfix-patch/SKILL.md` — consumes our triage.json
- `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md` §11 — taxonomy
