---
id: vllm-ascend-002
date: 2026-04-24
layer: vllm-ascend
title: Image tag `fixc` does NOT prove the `.so` was rebuilt against the running torch
trigger:
  - "easyr1-npu-torch211-fixc:..."
  - "Fix C image"
  - "rebuilt vllm_ascend_C.so against torch 2.11"
symptom_in_wild:
  - "claim 'Fix C image has native custom op working' solely based on the image tag name"
  - "assume `.so` in fixc image is torch-2.11 ABI simply because we labeled it so"
root_cause: >
  When I tagged the image `fixc`, what I actually did was relax the CMakeLists
  `VERSION_EQUAL "2.9.0"` check and re-run `setup.py build_ext --inplace` in the
  container. I didn't verify post-build whether the resulting .so linked to
  torch-2.11's libc10 / libtorch or still had the previous copy lying around,
  whether the compile-time constant (if any) reflected 2.11, or whether a
  native op call actually succeeded. The tag became a statement of intent
  (`this image is the one where I tried to Fix C`), not a verified state
  (`the .so in this image is rebuilt and working`).
mistake_pattern: "naming artifact to describe intended state, then treating the name as evidence of the state"
correction:
  - "Before tagging an image with a fix/build status label, verify the expected runtime state (ldd, nm, constant-probe, one real op call)"
  - "Prefer tags that describe WHAT was done, not the OUTCOME — e.g. `fixc-attempt-20260423-0929` not `fixc`"
  - "When consuming an image tagged with an outcome label, re-verify the outcome at first use"
  - "Never cite an image tag as evidence of correctness"
evidence:
  - "2026-04-23 09:29Z: `easyr1-npu-torch211-fixc:ascend-day0-torch211-20260423` built, `.so` size 472KB"
  - "2026-04-24T05:xxZ probe on that same image: `_TORCH_VERSION_BUILT_FOR: <not set>`, `_torch_abi_safe_for_custom_ops()` returns False → confirmed the Fix C image's .so was NOT verified rebuilt-for-2.11, despite the tag"
---

# Procedure for tagging a build image

1. Pick a descriptive but non-outcome tag (e.g. `<date>-rebuild-for-torch211-attempt`).
2. Run post-build verification:
   - `docker run <tag> python3 -c 'import <module>; ...'` (import ok)
   - `docker run <tag> ldd /path/to/.so | grep libtorch` (links to right version)
   - `docker run <tag> nm -D /path/to/.so | grep <key_symbol>` (new symbols present)
   - Run one representative op call that should only work with the new build
3. If ALL verification steps pass, THEN create a second tag that asserts the
   outcome (e.g. `<date>-torch211-abi-verified`).
4. Downstream users reference only the outcome-tag.
