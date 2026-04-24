---
id: vllm-ascend-001
date: 2026-04-24
layer: vllm-ascend
title: `_TORCH_VERSION_BUILT_FOR` guard has two ends — Python read + C++ inject — check BOTH exist before trusting the guard
trigger:
  - "_torch_abi_safe_for_custom_ops()"
  - "VLLM_BATCH_INVARIANT auto-enable"
  - "Fix B+ / Fix C ABI guard"
symptom_in_wild:
  - "guard always returns False even after rebuilding the .so against the new torch"
  - "training path silently routed through batch-invariant fallback without anyone noticing"
  - "V1.4 entropy_loss mysterious divergence vs baseline"
root_cause: >
  In 2026-04-23 Fix B+ work I (session author) wrote the Python side of the
  guard to read `vllm_ascend.vllm_ascend_C._TORCH_VERSION_BUILT_FOR` and fall
  back to a conservative heuristic if the constant is missing. But I never
  wrote the C++ side that actually sets the constant. So on every image built
  since then, the constant was missing, the fallback fired, and the guard
  concluded the .so is unsafe — forcing batch-invariant fallback even when the
  .so had been properly rebuilt against the running torch.
mistake_pattern: "one-sided contract: wrote only the read side without ever verifying the write side existed"
correction:
  - "When writing a guard that reads a C++ attribute, write AND push the C++ attribute injection in the same commit"
  - "Verify post-build: `python3 -c 'import vllm_ascend.vllm_ascend_C as _C; print(_C._TORCH_VERSION_BUILT_FOR)'` prints the expected string"
  - "Verify post-build: `_torch_abi_safe_for_custom_ops()` returns True"
  - "Run one native custom op call (e.g. `torch.ops._C_ascend.npu_add_rms_norm_bias`) and check it does not SIGSEGV"
  - "Only THEN claim Fix C rebuild completed"
evidence:
  - "2026-04-23 Fix B+ commit 7c2078e7 (`[BugFix] Guard custom ops against torch ABI mismatch`)"
  - "2026-04-24T05:xxZ probe: `_TORCH_VERSION_BUILT_FOR: <not set>` on both Fix C image and iter 18 image"
  - "2026-04-24 commits 551cdb91 + 3ecb82f4: softened the Python heuristic to trust import-success. STILL no C++ injection yet — that is a follow-up."
---

# Follow-up work still owed on this lesson

The Python side was softened (`returns True when constant absent`). But the
proper fix is to add C++ injection in `csrc/torch_binding.cpp` (or the right
PYBIND11_MODULE entry) so `_TORCH_VERSION_BUILT_FOR` is set to `TORCH_VERSION`
at compile time. Without that, a user with a torch-2.9-built .so running under
torch 2.11 will still silently pass the softened guard (import-success doesn't
catch ABI mismatch).

This follow-up is tracked in `src/skills/vllm-ascend/port-expert/references/ALWAYS_LOADED_RULES.md`
under Level 4 checklist; not yet implemented.
