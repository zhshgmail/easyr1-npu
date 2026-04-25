---
id: triton-ascend-001
date: 2026-04-25
layer: triton-ascend
title: triton-ascend libtriton.so and bishengir-compile must build against the SAME LLVM source
trigger:
  - "build triton-ascend with LLVM_SYSPATH override"
  - "use prebuilt LLVM from triton-ascend OBS while bishengir built from AscendNPU-IR submodule"
  - "any time you build the triton-ascend Python C extension and the bishengir kernel compiler from different LLVM trees"
symptom_in_wild:
  - "MLIRCompilationError: custom op 'to' is unknown (tried 'func.to' as well)"
  - "[ConvertLinalgIRToBinary] encounters error: ... Failed to parse input file: kernel.ttadapter.mlir"
  - "triton kernel JIT reaches bishengir invocation, bishengir prints a generic MLIR parse error"
root_cause: >
  triton-ascend's libtriton.so EMITS MLIR text against the LLVM/MLIR major
  version it was compiled against. bishengir-compile PARSES MLIR text against
  the LLVM/MLIR major version IT was compiled against. MLIR's text format is
  not stable across major LLVM versions: ops, operand syntax, attribute
  syntax, and even keywords like `func.to` (vs older equivalents) shift.
  When the two binaries are built against different LLVM sources, the parser
  errors are misleading — they look like dialect/op problems but are actually
  format-version mismatches.

  Concrete example 2026-04-25: built triton-ascend libtriton.so with the
  prebuilt `llvm-fad32722-ubuntu-x64.tar.gz` (LLVM 22) from triton-ascend's
  OBS; built bishengir-compile from AscendNPU-IR submodule pinned to
  `cd708029e0` (LLVM 19.1.7). Pipeline ran end-to-end through `bishengir-compile`
  invocation; bishengir errored "custom op 'to' is unknown" on the
  `kernel.ttadapter.mlir` triton-ascend produced. Root cause: format
  divergence between LLVM 19 and LLVM 22 MLIR text.
mistake_pattern: "two-LLVM-builds in one pipeline → silent text-format mismatch at the boundary"
correction:
  - "Pick ONE LLVM source. Either: (a) the OBS prebuilt LLVM that triton-ascend's setup.py downloads (LLVM 22 fad32722), and build bishengir-compile from a matching LLVM source — but Huawei doesn't publish bishengir source for that hash. Or (b) the LLVM source pinned in AscendNPU-IR submodule (LLVM 19 cd708029e0), build BOTH bishengir-compile AND libtriton.so against it."
  - "For path (b): run `cmake --install` from the AscendNPU-IR build dir to a unified prefix (e.g. `/workspace/cann9-llvm19`) so it has the standard `include/{llvm,mlir,lld}` + `lib/cmake/{llvm,mlir,lld}` layout. Then set `LLVM_SYSPATH=/workspace/cann9-llvm19` before running `pip install -e .` in triton-ascend. setup.py's get_thirdparty_packages reads LLVM_SYSPATH first and skips the OBS download."
  - "The 5 LLVM API drift fixes done for fad32722 (FileSystem.h include, NVVMMemorySpace::kSharedMemorySpace, triple.getTriple(), getStaticTripCount inline) all happen to apply correctly on LLVM 19 too — verify with `grep -rE` in LLVM 19 source headers before assuming."
  - "Validation gate: bishengir-compile binary's --version output should print an LLVM commit hash matching the one libtriton.so was built against. If they differ in major version, expect the 'custom op' parse error and stop early — do not run smoke."
evidence:
  - "2026-04-25 triton-ascend v3.6.0 NPU smoke: error 'custom op to is unknown' from bishengir on /tmp/.../kernel.ttadapter.mlir after a clean LLVM-22-libtriton + LLVM-19-bishengir pairing."
  - "User Discord 2026-04-25T18:18Z: '记住你在过程中遇到的各种问题最好在泛化后保存在知识库里'"
---

# Why this matters

This is the kind of bug where everything builds fine, every binary's `--help`
looks correct, the pipeline runs to within one stage of completion, and the
error message points at a "dialect" or "op" problem. None of those are wrong
in the literal sense — but the underlying cause is the `.mlir` text format
itself diverging across LLVM versions.

If you see "custom op X is unknown" or "Failed to parse input file" out of
bishengir-compile, **check LLVM version alignment first**, before inspecting
the `.mlir` content or chasing a bishengir bug.

# Validation procedure

Before invoking the full smoke:

```bash
# bishengir's LLVM version
$BISHENGIR/bishengir-compile --version | grep llvm
# Should print: llvm 19.1.7 64a154cd9a19 (or whatever AscendNPU-IR pinned)

# libtriton.so's LLVM version (best-effort: read from setup.py download or LLVM_SYSPATH)
cat /workspace/upstream/triton-ascend/cmake/llvm-hash.txt
# Should print: cd708029e0... if you set LLVM_SYSPATH at the AscendNPU-IR LLVM 19 build,
# or fad32722... if you let setup.py download the OBS prebuilt LLVM 22.
```

If those two hashes are different — or even just different major versions —
stop and rebuild one of the two against the matching source.

# Why "you'd think they should be compatible"

LLVM/MLIR's IR text format is informally documented as "stable" within the
major version (e.g., 22.0.0 → 22.1.0) but the project explicitly does NOT
guarantee cross-major-version text compatibility. Most ops keep the same
text spelling, but enough don't that any non-trivial kernel will hit at
least one. The `to` op the kernel.ttadapter.mlir uses is one such case
(bufferization.to_tensor → func.to in newer LLVM, etc.).

# Generalizable rule

Any time you have a multi-binary IR pipeline (frontend emits → middle binary
reads/transforms → backend reads/lowers), and the binaries are built from
different LLVM source trees, the boundary IR will eventually diverge. Pick
one LLVM source per pipeline, or build all components against it.
