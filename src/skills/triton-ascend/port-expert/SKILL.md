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
    `<target>_auto_porting` from the current main
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

## Knowledge query paths

- `memory/a3_server.md` — A3 host access for rebuild + smoke
- `memory/fork_branch_naming.md` — `<target>_auto_porting` convention
- `docs/torch-npu/PORTING-GUIDE.md` — for how triton-ascend integrates
  with torch_npu (triton-ascend ships the Ascend triton backend that
  torch_npu's inductor path targets)

## What this skill is NOT

- Not an F1-F8 symbol-drift scanner target. Use `git rebase` mental model.
- Not a compat-shim target. There's nothing to shim — triton-ascend
  IS triton + NPU backend, so a conflict is a merge conflict, not
  a missing symbol.
- Not currently a hot-path. 2026-04-24 state: user's memory
  `a3_server.md` notes "triton-ascend 3.2.0 is partially installed —
  triton/__init__.py and sibling files are missing. Symptom:
  `import torch_npu` fails with `ImportError: cannot import name
  'Config' from 'triton'`. Fix: `pip install --force-reinstall
  --no-deps triton-ascend==3.2.0`". That's install-time packaging,
  not version upgrade.
