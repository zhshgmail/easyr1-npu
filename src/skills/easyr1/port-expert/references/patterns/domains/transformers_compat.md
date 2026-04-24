# Domain — transformers compatibility (EC-02, drill shims)

**Load when**: Phase B or Phase C, hitting `ImportError` from transformers
internals — especially `no_init_weights`, `PreTrainedModel._init_weights`,
or initialization helpers.

---

## Scope

Stage 0 targets `transformers 4.57.6` (what the `verl-8.5.0-a3` v1 image
ships). You shouldn't need any shim at Stage 0. This file exists mainly
to document the drill-path shims so you recognize the symptom if a future
port bumps to transformers 5.x/6.x.

## Known breaking change — `no_init_weights`

`transformers.modeling_utils.no_init_weights` moved to
`transformers.initialization.no_init_weights` in transformers ≥ 5.0.

### Symptom

```
ImportError: cannot import name 'no_init_weights' from 'transformers.modeling_utils'
```

### Fix

```python
try:
    from transformers.initialization import no_init_weights  # transformers >= 5.0
except ImportError:
    from transformers.modeling_utils import no_init_weights  # transformers <= 4.x
```

Apply at every callsite — grep `from transformers.modeling_utils import.*no_init_weights`.

## Version detection helper (optional — only if you do lots of compat)

```python
from importlib.metadata import version as _pkg_version
from packaging import version as _v

def _transformers_ge(target: str) -> bool:
    return _v.parse(_pkg_version("transformers")) >= _v.parse(target)
```

## Files typically touched

- `verl/utils/model_utils.py` (if EasyR1 uses `no_init_weights`)
- Stage 0 on transformers 4.57.6: likely NO edits needed.

## Evidence

Port-branch commit `1f716ea` "[drill] transformers 5.x: no_init_weights moved
to transformers.initialization" — this is the drill image's shim, not Stage
0's. At Stage 0, skip this file unless you see the ImportError.
