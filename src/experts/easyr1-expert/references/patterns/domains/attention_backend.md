# Domain — Attention backend (NPU-CP-005, NPU-CP-007)

**Load when**: Phase B, fixing `from flash_attn import ...`,
`from flash_attn.bert_padding import ...`, `attn_implementation="flash_attention_2"`,
or any padding_free/ulysses varlen path.

---

## Why this is non-trivial

`flash_attn` is a CUDA-only package. It doesn't install on NPU. But EasyR1's
code imports both (a) kernels (`flash_attn_func`, `flash_attn_varlen_func`)
and (b) helpers (`pad_input`, `unpad_input`, `index_first_axis` from
`flash_attn.bert_padding`). The kernel side has an NPU equivalent shipped by
transformers 4.57+; the helpers do not — they need to be re-implemented in
pure torch for NPU.

## Fix — three pieces

### Piece 1: kernel swap (NPU-CP-005)

In `verl/models/transformers/flash_attention_utils.py`, replace:

```python
from flash_attn import flash_attn_func, flash_attn_varlen_func
```

With a backend-dispatching import:

```python
from ...utils.device import is_npu_available

if is_npu_available():
    # transformers ≥ 4.57 ships NPU-aware adapters under
    # transformers.integrations.npu_flash_attention that wrap torch_npu's
    # npu_fusion_attention behind flash-attn-style signatures.
    from transformers.integrations.npu_flash_attention import (
        npu_flash_attn_func as flash_attn_func,
        npu_flash_attn_varlen_func as flash_attn_varlen_func,
    )
    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_use_top_left_mask = False  # NPU FA2 is bottom-right causal by default
elif is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
```

Downstream code (`prepare_fa2_from_position_ids`, `flash_attention_forward`,
etc.) stays backend-agnostic — it just calls the names `flash_attn_func` /
`flash_attn_varlen_func` which are bound correctly at import time.

### Piece 2: vendored `bert_padding` helpers (pure torch, no flash_attn dep)

Create `verl/utils/npu_flash_attn_utils.py` with pure-torch reimplementations
of `pad_input`, `unpad_input`, `index_first_axis`. See the port branch
commit `da2487f` "decouple padding helpers from the flash-attn package".

Key function signatures:

```python
def pad_input(hidden_states, indices, batch, seqlen):
    """Inverse of unpad_input. Takes (total_tokens, ...) back to (B, S, ...)."""
    ...

def unpad_input(hidden_states, attention_mask):
    """Flatten a (B, S, ...) tensor to (total_tokens, ...) using mask."""
    ...

def index_first_axis(tensor, indices):
    return tensor[indices]  # but with shape/stride handling for varlen
```

Then callsites (grep `from flash_attn.bert_padding`) switch to:

```python
from ...utils.device import is_npu_available
if is_npu_available():
    from ..utils.npu_flash_attn_utils import pad_input, unpad_input, index_first_axis
else:
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
```

### Piece 3: default attn_implementation (NPU-CP-005)

Anywhere `attn_implementation="flash_attention_2"` is passed to
`from_pretrained` or `AutoModelForCausalLM`, replace with:

```python
from ...utils.device import get_default_attn_implementation

model = AutoModelForCausalLM.from_pretrained(
    ...,
    attn_implementation=get_default_attn_implementation(),
    # returns "sdpa" on NPU, "flash_attention_2" on CUDA
)
```

## NPU-CP-007 — padding_free + ulysses varlen on NPU

EasyR1's `padding_free=True` path calls `flash_attn_varlen_func`. After
Piece 1 above, that name resolves to `npu_flash_attn_varlen_func` on NPU.
But `verl/workers/actor/dp_actor.py` and `dp_critic.py` have explicit
`raise NotImplementedError("padding_free on NPU")`-style guards left over
from older transformers. Remove those guards:

```python
# verl/workers/actor/dp_actor.py, _build_model_optimizer()
# old:
if is_npu_available() and self.config.padding_free:
    raise RuntimeError("padding_free not supported on NPU")
# new: (just delete it — the backend dispatch in flash_attention_utils handles it)
```

Also in `verl/models/monkey_patch.py` `apply_ulysses_patch()`: drop the
`if is_npu_available(): raise` guard for the same reason.

## Files typically touched

- `verl/models/transformers/flash_attention_utils.py` — kernel dispatch
- `verl/utils/npu_flash_attn_utils.py` — NEW, pure-torch helpers (~100 lines)
- `verl/utils/attention_utils.py` — may exist in EasyR1 master, route helpers
- `verl/workers/actor/dp_actor.py` — drop NPU padding_free guard (CP-007)
- `verl/workers/critic/dp_critic.py` — same
- `verl/models/monkey_patch.py` — drop NPU ulysses guard (CP-007)
- Anywhere `attn_implementation="flash_attention_2"` appears hard-coded

## Verify

```bash
# kernel swap working?
python3 -c '
from verl.models.transformers.flash_attention_utils import flash_attn_func
print(flash_attn_func)
# On NPU expect: <function npu_flash_attn_func at 0x...>
# On CUDA expect: <function flash_attn_func at 0x...>
'

# attn_implementation default?
python3 -c 'from verl.utils.device import get_default_attn_implementation; print(get_default_attn_implementation())'
# NPU → "sdpa"; CUDA → "flash_attention_2"
```

Full verification is V1.4 smoke + V2.1 smoke (padding_free=True path):
- V1.4 PASS → Piece 1/3 work (non-padding_free path uses SDPA)
- V2.1 PASS → Piece 1 + NPU-CP-007 work (padding_free path uses npu_flash_attn_varlen_func)

## Evidence

Port-branch commits:
- `6701a50` "make attention backend NPU-aware"
- `da2487f` "decouple padding helpers from the flash-attn package"
- `fbaa983` "NPU-CP-007: enable padding_free + ulysses on NPU via transformers FA shim"

V1.4 step-1 `entropy_loss=0.991` and V2.1 step-1 `≈0.991` both measured
on the ascend-port branch with this template.
