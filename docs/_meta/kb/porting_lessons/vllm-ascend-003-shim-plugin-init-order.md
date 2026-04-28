---
id: vllm-ascend-003
date: 2026-04-27
layer: vllm-ascend
title: shims targeting vllm.v1.sample.* / vllm.v1.spec_decode.* must use find_spec + lazy __getattr__ to survive plugin-init
trigger:
  - "writing a shim under vllm_ascend/compat/"
  - "shim target lives in vllm.v1.sample.*"
  - "shim target lives in vllm.v1.spec_decode.*"
  - "ImportError: cannot import name 'SamplingParams' from 'vllm'"
  - "shim works in REPL but fails when vllm-ascend's plugin loads it"
symptom_in_wild:
  - "shim 3 (SpecDecodeBaseProposer) imports clean in a fresh python interpreter but raises ImportError during vllm-ascend plugin-init"
  - "stack trace shows 'from vllm import SamplingParams' fails with 'cannot import name SamplingParams'"
  - "shim's try-old/except fallback both fail despite the symbol clearly existing in vllm at runtime"
root_cause: >
  vllm-ascend is a vllm "platform plugin". When vllm starts, it triggers
  plugin discovery via `entry_points`, which `import vllm_ascend`. Any
  module imported by vllm-ascend at this point runs BEFORE
  `vllm/__init__.py` has finished its own setup — specifically before
  the lazy `from vllm.sampling_params import SamplingParams` re-export
  has populated `vllm`'s namespace.

  An eager shim like
  `from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer`
  transitively loads
  `vllm.v1.sample.metadata` -> `vllm.v1.sample.logits_processor` ->
  `from vllm import SamplingParams`. That last line raises because
  `vllm` is mid-init.

  This is NOT specific to spec_decode — any vllm-ascend shim whose
  target module transitively imports `vllm.v1.sample.*` will hit it.
mistake_pattern: "shim-eagerly-imports-during-plugin-init"
correction:
  - "Use `importlib.util.find_spec(target_path)` to detect the path WITHOUT loading the module. Set the `_SOURCE` flag at shim import time."
  - "Use module-level `__getattr__` to defer the actual `from target import Symbol` to first attribute access. By then vllm has finished its own startup."
  - "Cache the resolved class in `globals()` after first __getattr__ call — subsequent accesses skip the lazy path."
  - "Test the shim by importing it during plugin-init phase (in vllm-ascend Python session), not just in a clean REPL — those have different vllm-init states."
evidence:
  - "2026-04-27 T22a: vllm-ascend ascend-port/vllm-main shim 3 hit this exact pattern. Fix shipped at commit 6bac1f5e on github.com/zhshgmail/vllm-ascend."
  - "Fix file: vllm_ascend/compat/spec_decode_base_proposer.py — full implementation in branch ascend-port/vllm-main."
  - "PR_MATERIAL.md (same branch) §'Notes for the maintainer' explicitly warns to use lazy pattern for any future shim under vllm.v1.sample.* / vllm.v1.spec_decode.*."
mistake_pattern_relationship:
  - "Adjacent to triton-ascend-001 (LLVM-version-must-match) — both are 'shim correct in concept, wrong in implementation detail' bugs."
  - "Doesn't apply to torch_npu/compat/ shims — torch's __init__.py is fully eager, no init-order trap."
---

# When this rule fires

You're writing a shim file at `vllm_ascend/compat/<symbol>.py` and the
upstream symbol's module path starts with:

- `vllm.v1.sample.*`
- `vllm.v1.spec_decode.*`
- (likely also) `vllm.v1.engine.*`, `vllm.v1.worker.*` — anything that
  transitively imports `vllm.v1.sample.metadata` or top-level `vllm`
  re-exports

Use the lazy-attribute pattern. Otherwise the simpler eager
try/except is fine (faster import, simpler code).

# The pattern

```python
# vllm_ascend/compat/<symbol>.py
import importlib
import importlib.util

_OLD_PATH = "<old.module.path>"
_NEW_PATH = "<new.module.path>"  # if F2-path-move

if importlib.util.find_spec(_OLD_PATH) is not None:
    _SOURCE = _OLD_PATH
elif importlib.util.find_spec(_NEW_PATH) is not None:
    _SOURCE = _NEW_PATH
else:
    raise ImportError(f"shim: neither {_OLD_PATH!r} nor {_NEW_PATH!r} present")


def _resolve_target():
    return getattr(importlib.import_module(_SOURCE), "<SymbolName>")


def __getattr__(name: str):
    if name == "<SymbolName>":
        cls = _resolve_target()
        globals()[name] = cls   # cache so subsequent access skips here
        return cls
    raise AttributeError(name)


__all__ = ["<SymbolName>", "_SOURCE"]
```

The eager pattern (used by the other 2 shims at the same fork
branch) is fine when the target module is in
`vllm.model_executor.layers.fused_moe.*` or similar — those don't
trip the SamplingParams chain.

# Generalizable rule

For any plugin-style integration where the consumer (here:
vllm-ascend) loads BEFORE the host (here: vllm) finishes its own
setup, treat the shim's import-time as "host is in mid-init,
nothing in host namespace is reliable yet". Eager imports of host
symbols are dangerous; use `find_spec` + lazy `__getattr__`.
