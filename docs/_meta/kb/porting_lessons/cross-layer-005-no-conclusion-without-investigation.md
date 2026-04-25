---
id: cross-layer-005
date: 2026-04-25
layer: cross-layer
title: do not conclude root cause / scope before fully investigating, especially for OSS dep changes
trigger:
  - "this is out of skill scope"
  - "AIL team territory"
  - "needs CANN/driver bump"
  - "we should give up here"
  - "blocked on upstream API change"
  - "version X has flag Y therefore version X+1 has it too"
  - any "I think the cause is …" said before reading docs / repo / release notes
symptom_in_wild:
  - "session declares a hard wall and writes a 'BLOCKED' note in the KB on a single error message"
  - "session installs a new version of a dep without first checking what that version's layout is, then is surprised by missing binaries"
  - "session attributes an error to the wrong layer (e.g. CANN runtime vs OSS submodule build)"
  - "session marks a task COMPLETED with status 'out of scope' when one more 30-minute search would change the answer"
root_cause: >
  Reasoning from a single dmesg / build error to a root cause without
  investigating the OSS dependency graph. Software stacks involving multiple
  upstreams (triton-ascend → AscendNPU-IR → CANN; vllm-ascend → torch_npu
  → CANN; etc.) place flags / binaries / APIs in places that are not
  obvious from the error text. "I will declare scope and move on" feels
  like discipline but is often premature surrender that costs the user
  the actual fix.
mistake_pattern: "stop-investigating-and-declare-scope-boundary"
correction:
  - "When tempted to write 'BLOCKED on X / out of scope': pause. Ask one more layer of WHY. Read the actual upstream that owns X."
  - "For a missing CLI flag / symbol: grep the OSS source tree (the project's own repo + every submodule it pins) before concluding the binary is unfit. The flag may be defined IN a submodule that just hasn't been built into a binary yet."
  - "For a 'this is in CANN / driver / kernel-team scope' claim: check the project's own README / docs / Makefile / Dockerfile pin first. The project's own pinned version is the executable truth; out-of-tree assertions about scope often lag behind."
  - "Verify a version-rename hypothesis by inspecting the new version's layout (find / ls) before installing it. Don't assume rename direction from a single comment."
  - "When delegating research, spend the time. A focused web-search agent run takes 3 min and saves 30 min of wrong-direction shell churn."
evidence:
  - "2026-04-25 triton-ascend session: declared 'CANN/bishengir = AIL team scope' on one error from bishengir-compile. Real cause: --link-aicore-bitcode was defined in the AscendNPU-IR submodule's Options.td:585 (open source, in-repo), not in any CANN release. The fix is to build bishengir-compile from that submodule, not bump CANN. Cost ~30 min of CANN 9.0 install + smoke retry rounds before research caught it."
  - "Same session, hour later: assumed CANN 9.0 'must' ship bishengir-compile because triton-ascend uses it. Actual: CANN 9.0 dev build does NOT ship it. Cost another ~20 min of installer-extract + find before discovering the binary just isn't there."
  - "User Discord 2026-04-25T05:11Z: '没有充分调查前不要过早下结论，特别是开源软件的依赖调整方面' — user explicitly flagged the pattern after watching me make it twice in one session."
---

# What to check before declaring a hard wall

If your next sentence is going to be "this is out of skill scope" or "this
is blocked on an upstream we can't fix", run this checklist first:

## 1. Where does the offending flag / symbol / API actually live?

Don't trust the error text's framing. `bishengir-compile: Unknown command
line argument '--link-aicore-bitcode'` looks like a CANN/binary issue.
Real answer: the flag is **defined in source** in
`third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Tools/bishengir-compile/Options.td:585`,
which is an open-source submodule. The binary just hasn't been built from that
source yet. The correct response is "build from submodule", not "wait for
AIL team".

## 2. What does the project's OWN docs / Dockerfile / Makefile pin?

Project-internal pins are the executable truth. If `Dockerfile` pins CANN
8.5.0 but the docs say CANN 8.3.RC1, the Dockerfile is what's actually
tested. If a new CANN version was released last week but the project
hasn't updated, no amount of installing the new CANN will help.

## 3. Is the error in the layer I think it's in?

triton-ascend → AscendNPU-IR → LLVM → CANN runtime forms a dependency
graph. An error that prints "bishengir-compile: ..." may be in:
- triton-ascend's invocation logic (Python)
- AscendNPU-IR's compile pipeline (C++ in submodule)
- The bishengir-compile binary itself (built from submodule)
- LLVM/MLIR underneath (transitive submodule)
- CANN runtime that the binary links against

Don't pick "CANN" by default just because CANN is closed-source and feels
like a dead end. Check each layer.

## 4. Did I delegate web research before deciding?

Three minutes of WebSearch / WebFetch (via the general-purpose agent or
direct WebFetch tool) often turns a "blocked / out of scope" into a
specific, fixable instruction. The cost of NOT delegating is wasted shell
churn AND user time waiting on that churn AND wrong KB entries (which
poison future sessions).

## 5. Am I confusing "version X+1" with "version X+1 in my hand"?

A pre-GA / dev build of version X+1 may have a totally different layout
from the final GA. The CANN 9.0 dev artifactory build does not ship
`bishengir-compile`. The future CANN 9.0 GA might. They are NOT the same
artifact even though they share a version number.

## When in doubt, the correction is "spend ten more minutes investigating", not "declare scope and move on"

Premature scope boundaries are MORE costly than premature actions.
Premature actions can be reverted (if I haven't violated the
container-rm rule). Premature scope boundaries land in the KB as
authoritative docs and mislead the next session.
