---
id: cross-layer-006
date: 2026-04-26
layer: cross-layer
title: never give "30 minutes" or "1 hour" estimates for LLVM/MLIR/bishengir-style compiler-stack rebuilds — they almost always take multiples of that
trigger:
  - "rebuild llvm"
  - "rebuild bishengir"
  - "rebuild triton-ascend C++"
  - "ninja -jN" against an LLVM tree
  - "after applying patches, build with -j2 should take ~30 min"
  - any time I'm tempted to give a single-digit-hour estimate for a compiler stack
symptom_in_wild:
  - "I told the user 'ETA 30 min', actual time 90 min"
  - "I told the user '1-2 hours', actual time 8-12 hours"
  - "I told the user 'ninja resume should be a few minutes', actual time 30 min because of one big translation unit"
  - "user asks 'update?' multiple times within the predicted window because the binary isn't out yet"
root_cause: >
  LLVM/MLIR/bishengir builds at -j2 on a shared host have wildly variable
  per-translation-unit cost (some files take 30 seconds, some take 5+
  minutes for heavy templates). Total target count (e.g. 5240 ninja
  targets) does NOT divide evenly into per-target time. Plus configure
  steps, link steps for big static libs, and tablegen passes are
  serialized.

  My prediction model has been "fits-in-an-hour"-shaped because that's
  how Python builds feel. Compiler stacks don't fit that model. Single
  data points from this 2026-04-25 session:
    - First triton-ascend C++ build (-j8): not measured, but multiple hours
    - bishengir-compile from AscendNPU-IR submodule (-j2 on shared host,
      5240 ninja targets, full LLVM 19 + MLIR build from source +
      bishengir tools): roughly 10 wall-clock hours
    - lld libs + binary, delta build (-j2, 71 targets): ~80 minutes
    - triton-ascend rebuild against LLVM 19 install (-j2, ~2000 targets):
      not yet completed; previous estimate "30-50 min" was way off
mistake_pattern: "minutes-hours conflation for compiler stacks"
correction:
  - "Default budget for any LLVM/MLIR rebuild from source: 6-12 hours at -j2 on a shared host. NOT 30 min, NOT 2 hours."
  - "Default budget for a delta build (re-link after one source edit) on a fresh build dir: still 1-3 hours because of cmake config + tablegen passes + link steps."
  - "When predicting, multiply your gut estimate by 4 and add 30 min for cmake reconfigure overhead. That's still optimistic but closer."
  - "When building, drop the prediction entirely and tell the user 'I'll notify when it's done; rough order of magnitude is hours, not minutes'. Update at major milestones (target N/M crossed 25%, 50%, 75%) instead of giving an ETA."
  - "Compute estimated remaining via current ninja [X/Y] count + observed wall-clock-per-target rate, NOT a global gut feel. ninja gives you the data, use it."
  - "If the user asks 'update?' before you'd planned to check, that's a signal your last prediction was wrong by a noticeable margin."
evidence:
  - "Discord 2026-04-26T01:35Z user: 'Can you see how you missed every prediction duration in this session? You should update your knowledge about compiler level compilation'"
  - "Multiple wrong predictions this session: triton-ascend C++ first build said 30-50min, took ~10h. lld build said 'short', took 80min. AscendNPU-IR full build no prediction given, took ~10h. triton-ascend rebuild said 30-50min, blew past."
---

# When in doubt, do not give a duration

The honest user-facing default for any LLVM/MLIR-stack work:

> "This is a compiler-stack rebuild. I'll let it run and notify on
> completion. Order of magnitude is hours."

Then:
- post a status update at every ninja milestone (25/50/75%, link phase entry, install phase entry)
- if the user asks before then, give the current ninja [X/Y] count and the
  observed elapsed time, not a re-prediction

## Why this matters for skill design

The triton-ascend port-expert skill SHOULD warn the consumer (a real
upstream maintainer running it) that P4 (full C++ build) and P4'
(rebuild-against-different-LLVM) are multi-hour operations on a 2-core
budget. Currently the SKILL.md says "rebuild + run smoke" without a time
disclaimer. Update to set expectation correctly.

## Generalizable rule

For any compiler stack (LLVM/MLIR/bishengir/triton-C++/clang from source):
- Stop saying "minutes" unless the stack is fully cached and you're
  re-linking ONE translation unit.
- The default unit of time is "hours" plural.
- A "delta rebuild after one source edit" is NOT a quick operation — cmake
  reconfigure alone can be 30+ seconds; tablegen passes can be 5+ min.
