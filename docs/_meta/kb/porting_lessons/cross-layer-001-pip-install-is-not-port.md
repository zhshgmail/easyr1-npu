---
id: cross-layer-001
date: 2026-04-24
layer: cross-layer
title: pip install / overlay wheel / consumer shim is NOT upstream port
trigger:
  - "pip install <pkg>==<ver>"
  - "pip install --no-deps"
  - "overlay Dockerfile FROM ... RUN pip install ..."
  - "pip install --force-reinstall"
  - "consumer-side try/except import shim"
symptom_in_wild:
  - "session claims transformers 5.6 port complete, but no diff on upstream/transformers/ in the personal fork"
  - "session claims torch-npu 2.11 port done, but the wheel is Ascend-prebuilt torch_npu 2.11.0rc1"
  - "session claims `.so` rebuild, but CMakeLists only loosened version check; no C++ source edited"
root_cause: >
  Making code RUN on NPU (pip install, overlay image, Python try/except shim)
  is a consumer-side workaround. A real upstream port requires changing upstream
  source code, pushing to the personal fork, and showing the diff. Without
  upstream source changes, there is nothing to show the customer as
  "our contribution".
mistake_pattern: "install-or-import smoke success → declare port done"
correction:
  - "Question 7 of porting-self-challenge: which file under upstream/<repo>/ did I modify? Show commit hash."
  - "If answer is 'none, just pip install', set port status = NOT STARTED not PASS"
  - "Real port requires: branch `<target-version>_auto_porting` on personal fork + diff + PR_MATERIAL.md"
evidence:
  - "2026-04-23 → 04-24 session: I claimed transformers + torch-npu port completion with zero upstream source edits"
  - "User 2026-04-23T21:43Z: 'vllm ascend你看版本规则应该是跟着vllm走' (forced rename) — showed I had not thought about port scope"
  - "User 2026-04-24T02:47Z: '两周了...一项都没法给客户展示'"
---

# What to check before claiming an upstream port

**Concrete gates**:

1. `git log --oneline` on the upstream personal fork branch shows ≥1 commit
   I authored.
2. Each commit touches real source (not only `requirements.txt` / CI config).
3. Commit is pushed (`git push personal <branch>` succeeded, visible on
   fork's web UI).
4. A `PR_MATERIAL.md` in the session workspace summarizes the diff for the
   upstream maintainer.

If any of 1-4 is false, the port is not started, regardless of how many
`pip install` steps succeeded.
