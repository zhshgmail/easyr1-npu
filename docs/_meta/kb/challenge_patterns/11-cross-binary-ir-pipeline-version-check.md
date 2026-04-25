---
id: 11
pattern: cross-binary-ir-pipeline-version-check
trigger_phrases:
  - any plan that builds two binaries from different LLVM/MLIR sources and pipes IR between them
  - "use prebuilt LLVM for triton-ascend" + "build bishengir-compile from submodule"
  - "smoke this NPU triton kernel"
  - "let triton emit MLIR and bishengir parse it"
user_source:
  - "2026-04-25T18:18Z: '记住你在过程中遇到的各种问题最好在泛化后保存在知识库里，以后遇到类似的问题可以参考'"
  - "2026-04-25T18:23Z: '好' (agreed to make this a critic auto-trigger)"
---

# Cross-binary IR pipeline LLVM-version pre-check

## What the user is catching

Any time I'm about to run an end-to-end smoke that flows IR text between two
LLVM-backed binaries (frontend emits → middle binary parses/transforms →
backend lowers), I should verify both binaries were built against the same
LLVM source — BEFORE running smoke, not when it fails halfway through.

## Why it matters

MLIR text format is not stable across major LLVM versions. Op spellings,
attribute syntax, and dialect canonical names diverge. When two binaries
disagree on format, the failure mode is misleading: looks like a
"custom op X is unknown" or "Failed to parse input file" — sounds like a
dialect bug, is actually format-version drift.

Concrete cost paid 2026-04-25: built triton-ascend's libtriton.so against
the prebuilt LLVM 22 (fad32722) from triton-ascend's OBS, built
bishengir-compile from AscendNPU-IR submodule pinned at LLVM 19
(cd708029e0). Pipeline went all the way through JIT compile + ttadapter
emission + bishengir invocation, ate ~10 hours of build time, and finally
errored on `custom op 'to' is unknown` from format mismatch. Re-installed
LLVM 19 to a unified prefix and rebuilt libtriton.so against it. Could have
been caught by a 5-second hash compare before smoke ran.

## Self-check before action

Before invoking any NPU kernel smoke that crosses triton-ascend ↔ bishengir
or any similar two-LLVM-binary pipeline, check both:

```bash
# Frontend (libtriton.so) LLVM hash
cat <triton-ascend>/cmake/llvm-hash.txt | head -c 8
# What the actually-built binary uses (from LLVM_SYSPATH or OBS prebuilt)
echo $LLVM_SYSPATH
ls /root/.triton/llvm/

# Backend (bishengir-compile) LLVM hash
$BISHENGIR_PATH/bishengir-compile --version | grep llvm
```

If the two hashes differ in major LLVM version (or differ at all if minor
also differs), STOP. Either:
- rebuild backend against the frontend's LLVM source, or
- override frontend's LLVM_SYSPATH to match the backend's source, then
  rebuild frontend.

DO NOT run the smoke and hope it works.

## My common failure mode

I treat "both binaries built successfully" as "both binaries are compatible".
The build step has no LLVM-version cross-check; nothing fails until runtime
IR parsing. I fall into "the build's green so it should work" pattern.

The fix is: never trust two-binary IR pipelines until you've matched the
LLVM hashes. Add this as a one-line precondition before invoking smoke
in any NPU compile-driven workflow.

## Generalizable rule

For any IR-text pipeline crossing binary boundaries (not just triton ↔
bishengir; also any future torch.compile ↔ bishengir or vllm-compile ↔
bishengir scenarios): verify version alignment first, smoke second. The
version pre-check is cheap (10 seconds), the wrong-version smoke debug is
expensive (hours).
