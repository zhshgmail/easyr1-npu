---
id: cross-layer-008
date: 2026-05-29
layer: cross-layer
title: sys.path containing '/' or '' shadows editable installs via namespace package resolution
trigger:
  - "ImportError on a package known to be installed (vllm, sglang, easyr1) inside a container"
  - "pip show <pkg> succeeds but `python -c 'import <pkg>'` fails"
  - "Working directory matters — running from / fails, running from /tmp succeeds"
  - "Bash launcher / docker entrypoint sets cwd to /"
symptom_in_wild:
  - "ImportError: cannot import name 'LLM' from 'vllm'"
  - "ImportError on sglang despite pip install -e ."
  - "Same import works in /tmp but fails in /"
  - "`python -c 'import sys; print(sys.path)'` shows '' or '/' at the front"
root_cause: >
  Python's `PathFinder` walks `sys.path` in order. When `''` (cwd) or `'/'` is
  present and the search finds a directory named like the package
  (e.g. `/vllm/` exists as a root-level directory because of how the container
  mounts), it resolves to an *implicit namespace package* — an empty
  shell with no `__init__.py` — and stops searching.

  This shadows the real package installed via `pip install -e` (which
  registers as a `_distutils_hack`-style editable finder later in sys.path).
  Result: `from vllm import LLM` finds the namespace shell, not the editable
  module.

  vllm's `+empty` editable hack is specifically vulnerable; sglang's
  setup.py-based editable install is also.
mistake_pattern: "implicit namespace package silently outranks editable finder due to sys.path ordering"
correction:
  - "Fix at consumer script: `sys.path = [p for p in sys.path if p not in ('', '/')]` at the very top, before any third-party import"
  - "Also ensure cwd is not `/` when launching: `cd /tmp && python script.py`"
  - "If a container ENTRYPOINT runs the script from /, add cd to script wrapper"
  - "Verify: `python -c \"import sys; print(sys.path); import vllm; print(vllm.__file__)\"` should show the real path under site-packages or /opt/<editable-dir>, not /vllm"
  - "Don't ask vllm or sglang to fix this — it's a Python path-resolution semantics issue, and stripping '/' from sys.path is the consumer-side hygiene the runtime expects"
evidence:
  - "tlrescue empirical 2026-05-29: `import vllm` from /workspace fails; from /tmp succeeds. sys.path included '/' due to docker entrypoint."
  - "Same pattern for sglang in glm5 image 2026-05-30: smoke script `workspace/T32_tilelang_rescue/sglang_npu_smoke.py` requires sys.path strip + non-root cwd"
  - "Memory: feedback_vllm_editable_sys_path_root_shadow.md (vllm); feedback_sglang_npu_smoke_recipe.md (sglang)"
  - "Affected packages confirmed: vllm (+empty), sglang (editable), easyr1 (editable)"
---

# cross-layer-008 — sys.path root namespace shadow

## Why this matters

When this bites, the failure looks like a broken install — but everything is installed correctly. The diagnosis can take an hour if you don't recognize the pattern. With this lesson memorized: 60-second fix.

## Quick diagnostic

```bash
# In the failing environment:
python -c "import sys; print(sys.path)"
# Look for '' or '/' anywhere. Each is suspicious.

# Confirm the shadow:
ls /vllm /sglang /easyr1 2>/dev/null
# If any of these exist as plain directories, namespace package resolution fires.
```

## Standard fix template

```python
# At the very top of any script that imports the affected package
import sys
sys.path = [p for p in sys.path if p not in ("", "/")]

# Now safe to import
import vllm  # or sglang, easyr1
```

## Container-side fix (cleaner but requires image rebuild)

In the Dockerfile or container entrypoint:
```
WORKDIR /workspace
ENTRYPOINT ["python", "-c", "import sys; sys.path = [p for p in sys.path if p not in ('', '/')]; ..."]
```

Or simply don't create root-level directories that collide with package names. `/vllm` and `/sglang` were created by a container author who wanted convenient cd targets; the cost was this bug.
