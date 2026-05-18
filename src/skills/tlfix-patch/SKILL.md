---
name: tlfix-patch
description: Phase 3 of /tilelang-fix. Apply a recipe from the KB patch library (workspace/T32_tilelang_rescue/apply_*.py) to a tilelang-mlir-ascend tree; rebuild and stage for /tlfix-verify. If no matching recipe, generate a draft patch proposal.
---

# /tlfix-patch — Phase 3 (apply fix from recipe library)

Standalone runnable; called by `/tilelang-fix` orchestrator.

## Inputs

- `tilelang_dir`: tilelang-mlir-ascend checkout
- `recipe`: path to an `apply_*.py` script from KB recipe library
  OR `recipe_name` (e.g. `emptyop_fix_v4`) — resolves to
  `workspace/T32_tilelang_rescue/apply_<recipe_name>.py`
- `out_dir`: where to write diff + verification stub
- `--rebuild` (default true): incremental `make -j` after patch

## Behavior

1. Backup affected source files (`.orig.tlfix_<recipe_name>`)
2. Run the recipe script (idempotent — checks marker present)
3. Run `format.sh` (clang-format) on changed files
4. If `--rebuild`: incremental rebuild; report compile pass/fail
5. Write `diff.patch` (git diff of changes) + `patch_meta.json`
   (recipe used, files changed, rebuild status)

## Output schema

`out_dir/patch_meta.json`:
```jsonc
{
  "recipe": "emptyop_fix_v4",
  "applied": true,
  "files_changed": ["src/target/codegen_npuir_dev.cc"],
  "rebuild": {"status": "OK", "make_exit": 0, "elapsed_s": 240},
  "diff_summary": "+24 -3 lines"
}
```

## Recipe library (current KB)

| Recipe | Bug class | Files affected |
|--------|-----------|----------------|
| `apply_emptyop_fix.py` v1 | EmptyOp AllocateNode dyn sizes | codegen_npuir_dev.cc |
| `apply_emptyop_fix_v2.py` | EmptyOp NeedGenInsertSlice | codegen_npuir_dev.cc |
| `apply_emptyop_fix_v3.py` | DimOp tensor | codegen_npuir_dev.cc |
| `apply_emptyop_fix_v4.py` | DimOp tensor/memref dispatch | codegen_npuir_dev.cc |
| `apply_collapse_shape_fix.py` v5 | CollapseShape reassoc-aware result | codegen_npuir_dev.cc |
| `apply_F1_loop_invariant_scalar.py` F1.1+F1.2 | scalar-as-PrimExpr threading | npu_loop_vectorize.cc + codegen_npuir_dev.cc |
| `apply_F1_step3_api_codegen.py` F1.3 | API codegen scalar parity | src/op/ascend.cc |
| `apply_scalar_load_fix.py` v6 | is_scalar_load arg symmetry | codegen_npuir_dev.cc |
| `apply_v6_buffer_shape_fix.py` v12 | scalar-load buffer_shape stay empty | codegen_npuir_dev.cc |
| `apply_scalar_binary_dispatch.py` v9 | scalar-scalar arith.mulf fallback | codegen_npuir_dev.cc |
| `apply_vmul_scalar_swap.py` v11 | commutative-op operand swap | codegen_npuir_dev.cc |
| `apply_act_quant_fix_permanent.py` | round_mode + scale shape | examples/deepseek_v4/example_act_quant_kernel.py |
| `apply_fp8_indexer_tol_fix.py` | fp16 tolerance bump | examples/fp8_lighting_indexer.py |
| `fix_unittest_atomic_args.py` | unittest args order | unittest/npuir/test_atomic_*.py |

## How to invoke

```bash
bash run.sh <tilelang_dir> <recipe_name> <out_dir> [--no-rebuild]
```

## See also

- `tlfix-triage/SKILL.md` — picks recipe for us
- `tlfix-verify/SKILL.md` — confirms our patch doesn't regress
- `workspace/T32_tilelang_rescue/apply_*.py` — recipe library
