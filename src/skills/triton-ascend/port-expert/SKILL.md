---
name: triton-ascend-port
description: >
  triton-ascend is a FORK of triton, not a plugin that imports from it.
  Upgrading triton-ascend to a newer community triton release is a
  rebase/merge problem, not a drift-patch problem. This skill documents
  the pattern and points at the right tools (git rebase / merge), not
  at the F1-F8 scanners used for vllm-ascend / torch_npu.
argument-hint: >
  target-triton-version: community triton release to rebase onto
  (e.g. 3.4.0)
context: inline
---

# /triton-ascend-day0 — triton fork upgrade

## Why this is different from vllm-ascend / torch_npu

| Dimension | vllm-ascend / torch_npu | triton-ascend |
|---|---|---|
| Source structure | Plugin that imports from upstream | Fork of upstream with NPU-specific patches baked in |
| Drift shape | F1-F8 (symbol additions / removals / signature changes) | Merge conflicts on rebase |
| Fix pattern | Compat shim under `<project>/compat/` | Resolve merge conflicts, re-apply NPU-specific diffs on new base |
| Scanner | `kb_drive_test.py`, `check_f4.py`, `check_f7_f8.py` | `git rebase` / `git merge` |
| Validation | `/drift-port-validate` (shim branch check) | rebuild + run triton-ascend's test suite |

## Workflow (this is git-flow, not scan-flow)

```
P0  identify target triton community tag
P1  on fork (zhengshencn_hwca/triton-ascend), create branch
    `<target-version>_auto_porting` from the current main
P2  git fetch upstream community triton; git rebase --onto <community-tag>
    <current-base> <target-branch>
P3  resolve conflicts; re-apply NPU-specific patches that didn't merge
    cleanly. Common conflict surfaces:
    - `python/triton/backends/` — NPU backend added as sibling of
      `cuda/` / `amd/`. Conflicts when upstream adds a new backend
      or refactors the backend plugin protocol.
    - `python/triton/language/` — NPU lang extensions
    - `python/triton/runtime/` — NPU-specific compile cache, jit config
    - `third_party/ascend/` (if present) — NPU kernels
P4  rebuild triton-ascend wheel; run triton-ascend's test suite
    (python/test/unit/ascend/ or similar)
P5  smoke NPU end-to-end: run a tiny NPU triton kernel via torch_npu
    inductor path (see torch-npu port-expert deploy artifacts for the
    smoke harness)
P6  push to fork branch; file PR against Ascend/triton-ascend master
```

## When NOT to use this skill

- If upstream triton has NOT released a new version — no work needed.
- If upstream triton only added GPU-specific code in a new commit
  (e.g. new CUDA backend path, new AMD-specific feature) — NPU backend
  isn't affected; just rebase and resolve trivial conflicts.
- If the target triton is a PATCH version (x.y.z → x.y.z+1 typically
  doesn't touch backend plugin protocols) — shallow rebase, fast review.

Use **only** when minor or major version jump introduces backend-protocol
changes that the NPU backend needs to adapt to.

## Interaction with other port-experts

- **Does a triton-ascend rebase force a torch_npu rebuild?** Usually
  yes if the triton-ascend wheel's C++ ABI (pybind layer) changed.
  Open a follow-up task against `torch-npu/port-expert` with scope
  "rebuild torch_npu against new triton-ascend wheel" — that's a C
  extension rebuild, not an F1-F8 drift, so the torch_npu expert's
  Mode A (overlay) is the right skill, not Mode B (drift scan).
- **Does torch_npu upgrade force triton-ascend?** Only when the new
  torch bumps a triton-version constraint that the pinned
  triton-ascend wheel doesn't satisfy. Typically: torch ≥ 2.N bumps
  triton ≥ 3.M, you check if triton-ascend's `version.txt` is ≥ 3.M
  already; if not, trigger this skill in parallel with the torch_npu
  upgrade.
- **vllm-ascend → triton-ascend**: no direct dep path. vllm-ascend's
  custom ops use triton-ascend only when triton-based fused ops are
  compiled on NPU; if triton-ascend changes the triton.compile() API,
  F3 scanner on vllm-ascend catches it (sig change on
  `triton.compile`). No rebase of vllm-ascend needed.
- **transformers → triton-ascend**: no path. transformers' NPU
  integration uses torch_npu's fused-attention op, not triton.

## Knowledge query paths

- `memory/a3_server.md` — A3 host access for rebuild + smoke
- `memory/fork_branch_naming.md` — `<target-version>_auto_porting` convention
- `docs/torch-npu/PORTING-GUIDE.md` — for how triton-ascend integrates
  with torch_npu (triton-ascend ships the Ascend triton backend that
  torch_npu's inductor path targets)

## What this skill is NOT

- Not an F1-F8 symbol-drift scanner target. Use `git rebase` mental model.
- Not a compat-shim target. There's nothing to shim — triton-ascend
  IS triton + NPU backend, so a conflict is a merge conflict, not
  a missing symbol.
- Not a runtime install-issue fixer. A separate memory
  (`a3_server.md`) notes an install-time wheel-completeness issue
  with a specific triton-ascend rc wheel that is unrelated to version
  upgrades. Don't confuse "rebase to new community triton" (this skill)
  with "reinstall triton-ascend wheel on A3" (operational, not porting).
