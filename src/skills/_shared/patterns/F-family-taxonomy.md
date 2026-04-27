# F-family API-drift taxonomy (cross-upstream)

> **Shared across all 4 NPU upstream skills** (vllm-ascend / torch-npu /
> transformers / triton-ascend). Each skill's per-upstream
> `patterns/domains/<upstream>-api-drift.md` is a thin pointer to this
> file plus per-upstream concrete examples.
>
> **Load when**: Phase B classification points at `C-patch` and the root
> cause is an upstream community API change (not an NPU/torch ABI issue).
> Use to match the symptom to one of 8 families (F1–F8), then apply the
> corresponding fix template. Families are abstract — **they do not name
> specific upstream versions, SHAs, or file paths**; those belong in the
> per-upstream `KB_INDEX.md §"Known API removals / adds"`.
>
> Examples in this file use vllm-ascend syntax (the taxonomy was first
> abstracted out of vllm work). Substitute upstream names for your own;
> the pattern shapes are identical.

## How to use this file

1. Identify the failure *shape* (import error? attribute error? type error?
   assertion? silent wrong-output?). Pick the matching family below.
2. Read the family's `Detect` signal — confirm the match with one probe.
3. Apply the `Fix template`. Fix templates are written to be
   version-forward-compatible: the vllm-ascend code should keep working
   on the *old* vllm too, so the same tree can be released against both.
4. After the fix lands, record the **new concrete case** in
   `KB_INDEX.md §"Known vllm API removals / adds"` with (a) vllm PR
   number, (b) affected vllm-ascend file, (c) the matched family ID.

## F1 — Removed symbol (import)

**Shape**: `ImportError: cannot import name 'X' from vllm.Y.Z` where a
public helper/class vllm-ascend depended on was deleted upstream.

**Detect**: grep vllm-ascend source for `from vllm... import X`; verify
in the target vllm tree that `X` is absent but an equivalent symbol /
environment variable exists.

**Fix template** (forward-compat; keeps working on older vllm):

```python
# In vllm-ascend module that needs the symbol:
try:
    from vllm.<old.path> import <old_symbol>
    def _get_X():
        return <old_symbol>()
except ImportError:
    # Upstream removed it — map to new mechanism
    from vllm import envs as _vllm_envs
    def _get_X():
        return bool(getattr(_vllm_envs, "<NEW_ENV_VAR>", False))
```

Then use `_get_X()` at the original call sites. **Do NOT** silently
default to a hard-coded value — that hides upstream breakage from CI.

**Variant — optional import**: if the missing symbol is only used in
one optional code path (e.g. speculative decoding), skip the shim and
make the import optional with a `None` fallback that disables the path.

## F2-path-move — Symbol preserved, module path moved

**Shape**: `ImportError: cannot import name 'X' from '<old.module>'`.
The symbol still exists in the upstream tree, just at a different
import path. Very common on torch upstreams where private modules
(`torch._inductor`, `torch._dynamo`) get reorganized each minor
release. Same symbol identity, different home.

**Detect**: grep upstream source tree at the target version for the
symbol name. If it exists somewhere but NOT at the path your caller
uses, this is F2-path-move (not F1 removal, not F2-rename).

**Fix template** (forward-compat; keeps working on older upstream):

```python
# In <your-project>/compat/<symbol_group>.py
try:
    from <old.path> import <Symbol>
    _SOURCE = "<old.path>"
except ImportError:
    from <new.path> import <Symbol>  # moved here
    _SOURCE = "<new.path>"

__all__ = ["<Symbol>"]
```

Then callers import from `<your-project>.compat.<module>` instead of
the original upstream path.

**Concrete example** (torch_npu 2.11 → 2.12):
`FloorDiv` and `ModularIndexing` moved from `torch._inductor.utils`
to `torch.utils._sympy.functions`. See
`torch_npu/compat/sympy_functions.py` on branch `torch-2.12_auto_porting`
of the personal fork.

**Why separate from F2 (rename)?** F2-rename changes the symbol name
(caller must change both path AND name); F2-path-move changes only
the path. The shim shape is slightly different:
- F2-rename: `from new.path import NewName as OldName`
- F2-path-move: `from new.path import OldName` (name preserved)

## F2 — Renamed type / class

**Shape**: `AttributeError: module 'vllm.X' has no attribute 'OldName'`
or `ImportError: cannot import name 'OldName'`. Frequent on model
classes (`GatedDeltaNet → GatedDeltaNetV2` type renames) or registry
keys.

**Detect**: the symbol exists in old vllm, absent in new. A new symbol
with similar semantics exists in the new version.

**Fix template**:

```python
try:
    from vllm.<path> import <NewName> as _ResolvedName
except ImportError:
    from vllm.<path> import <OldName> as _ResolvedName
# or via getattr on the module:
_mod = importlib.import_module("vllm.<path>")
_ResolvedName = getattr(_mod, "<NewName>", None) or getattr(_mod, "<OldName>")
```

Use `_ResolvedName` at call sites. For multiple renames in the same
subsystem (e.g. `patch_qwen3_next.py` + `patch_qwen3_5.py` both touch
the same renamed attention class), extract the resolver into one
shared helper in `vllm_ascend/compat/` and reuse.

## F3 — Signature change (args added / removed / reordered)

**Shape**: `TypeError: func() missing 1 required positional argument:
'new_arg'` or `unexpected keyword argument` or silent wrong output
when arg order changed.

**Detect**: compare the vllm function's signature in the target version
vs the one your vllm-ascend call site was written against. Use
`inspect.signature(func)` at runtime if uncertain.

**Fix template** — runtime signature-sniffing when the same vllm-ascend
code must work across vllm versions:

```python
import inspect
_SIG = inspect.signature(vllm_func)
if "new_arg" in _SIG.parameters:
    vllm_func(old_arg, new_arg=<compute_new>)
else:
    vllm_func(old_arg)
```

Cache the detection at import time, not per-call, if the call is hot.
Prefer keyword args at call sites so signature reorders don't silently
miscall the function.

## F4 — Return-type migration

**Shape**: previously returned `scalar` (float/int/tensor); now returns
a `NamedTuple` / `dataclass` / dict. Call sites that did arithmetic on
the return silently break (`TypeError: unsupported operand type(s) for +`)
or get wrong values (indexing the tuple as the scalar).

**Detect**: `type()` of the return. If migrating from scalar → tuple,
every site that used the return must unpack the correct field.

**Fix template**:

```python
_ret = vllm_func(...)
# Tuple / NamedTuple case:
if hasattr(_ret, "<primary_field>"):
    value = _ret.<primary_field>
else:
    value = _ret  # old-vllm scalar fallback
```

Or construct a shim `NamedTuple` from a scalar return to unify downstream.

**Pitfall**: do not do `_ret[0]` to unpack — NamedTuple indexing may
land you on the wrong field if fields are reordered later. Use attribute
access.

## F5 — Buffer API migration (most invasive)

**Shape**: vllm changes how internal buffers are represented. Common
migration: `CpuGpuBuffer` (dataclass with `.np / .cpu / .gpu /
.copy_to_gpu()`) ↔ plain `torch.Tensor`. Every access point on the
vllm-ascend side that touched `.np` / `.copy_to_gpu(...)` breaks.

**Detect**: grep vllm-ascend for `\.copy_to_gpu(`, `\.np\b`,
`CpuGpuBuffer`. Count sites — typically **10+** across
`model_runner_v1.py` + `npu_input_batch.py`. This family is the most
expensive because the patches touch hot paths.

**Fix template** — prefer a **thin helper** over 11 scattered edits:

```python
# In vllm_ascend/compat/buffer.py:
try:
    from vllm.v1.utils import CpuGpuBuffer  # new vllm path
    _USE_BUFFER = True
except ImportError:
    _USE_BUFFER = False

def to_gpu(buf_or_tensor):
    """Return the NPU-resident tensor regardless of buffer/plain form."""
    if _USE_BUFFER and hasattr(buf_or_tensor, "copy_to_gpu"):
        return buf_or_tensor.copy_to_gpu()
    return buf_or_tensor  # already a plain tensor on device

def np_view(buf_or_tensor):
    """Return the numpy-backed CPU view."""
    if _USE_BUFFER and hasattr(buf_or_tensor, "np"):
        return buf_or_tensor.np
    return buf_or_tensor.cpu().numpy()
```

Then migrate call sites to `to_gpu(x)` / `np_view(x)` in one PR.
**Warning**: F5 has a known Revert-and-retry sub-pattern — the first
attempt often introduces a regression (identity tensor shared, shape
mismatch on first batch). Always V1.3 + V1.4 smoke after F5 fixes.

## F6 — kv_cache tensor-vs-list contract

**Shape**: `AssertionError: attn_layer.kv_cache must be single tensor`
or silent wrong generation because vllm-ascend registered the cache in
the wrong shape. Failures are intermittent and depend on batch/seq len.

**Detect**: vllm's attention contract expects `layer.kv_cache` to be
either a list-of-tensors OR a single stacked tensor, version-dependent.
Check `isinstance(layer.kv_cache, list)` vs `torch.Tensor` in the
target vllm version.

**Fix template**:

```python
# Version-detect at import time:
_EXPECTS_STACKED_TENSOR = None
def _get_kv_cache_expects_stacked():
    global _EXPECTS_STACKED_TENSOR
    if _EXPECTS_STACKED_TENSOR is None:
        # inspect vllm attention backend class signature / docstring
        # or check a version constant
        from vllm._version import __version__ as _vv
        _EXPECTS_STACKED_TENSOR = _parse_version(_vv) >= (0, 20, 0)
    return _EXPECTS_STACKED_TENSOR

def bind_kv_cache(attn_layer, cache):
    if _get_kv_cache_expects_stacked():
        # Stack the 2-element [k, v] list into one tensor of shape [2, ...]
        attn_layer.kv_cache = torch.stack(cache) if isinstance(cache, list) else cache
    else:
        attn_layer.kv_cache = cache if isinstance(cache, list) else [cache[0], cache[1]]
```

Always V1.3 PASS **plus** bit-exact token diff vs a GPU reference for
1–2 prompts. Silent kv_cache corruption often passes the smoke marker
but produces garbage tokens — matches `no-concrete-number` pitfall.

## F7 — New required attribute on NPU integration class

**Shape**: `AttributeError: 'NPUX' object has no attribute 'new_flag'`
when vllm's scheduler / runner probes the attribute. vllm added a new
attr expected on every backend/platform/input-batch class; NPU subclass
didn't get the update.

**Detect**: compare vllm's base class (`AttentionBackendImpl`,
`InputBatch`, `Platform`) in the target version vs what vllm-ascend
subclasses. New fields on the base class may be required.

**Fix template** — add the attribute with a sensible default that
matches NPU semantics. Do NOT default to `True` just to silence the
error; figure out the correct NPU behavior:

```python
class AscendAttentionBackendImpl(AttentionBackendImpl):
    # vllm 0.20+ probes this to decide whether it or the backend
    # drives the kv_cache update. NPU does it in the backend's
    # forward, so we say False here.
    forward_includes_kv_cache_update = False

    def do_kv_cache_update(self, ...):
        # Implement if base class made it required
        ...
```

Be explicit in a comment why the default was chosen, because "it
silenced the AttributeError" is not a semantic justification.

## F8 — New required method on NPU integration class

**Shape**: `NotImplementedError` or silent skip of feature. vllm added
a new method to a base class; NPU subclass needs to implement (or
explicitly opt-out).

**Detect**: same as F7 but for methods. Look for `raise NotImplementedError`
in the new base class method, or a base-class docstring "Subclasses
must implement X".

**Fix template**:

```python
def clear_row(self, row_idx: int):  # vllm 0.20+ block_table API
    # NPU block-table is logically identical; just zero the row
    self.block_table[row_idx, :] = 0

def manual_seed_all(self, seed: int):  # vllm 0.20+ Platform API
    # NPU: route to torch_npu.npu.manual_seed_all
    torch_npu.npu.manual_seed_all(seed)
```

Keep the method **minimal** — only what the base contract requires.
Extended NPU-specific behavior belongs elsewhere.

## Cross-family meta-rule

**Every F1–F8 fix must land with a forward-compat try/except or
sig-sniff** so the same NPU-fork tree can be built against **both**
the old and new upstream (policy: one NPU-fork tree serves both the
stable and nightly upstream). Hard version gates like
`if upstream.__version__ >= "X"` are **last resort** — they force a
release fork. Prefer duck-typing / try-import / hasattr.

## What this file is NOT

- Not a list of "here's the fix for upstream-PR-NNN". That belongs
  in the per-upstream `KB_INDEX.md §"Known API removals / adds"` as
  a dated entry pointing back to the family.
- Not a replacement for the per-upstream failure-minimization probe
  doc (`patterns/domains/<upstream>-probe.md`). You still need to
  isolate the smallest repro before classifying; classify too early
  and you may match the wrong family.
