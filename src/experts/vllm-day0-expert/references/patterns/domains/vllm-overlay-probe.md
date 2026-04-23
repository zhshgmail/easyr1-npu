# Domain — vllm overlay probe (new vllm × older vllm-ascend)

**Load when**: Phase A probing a target vllm version against an NPU
image that ships an older vllm-ascend.

## What makes vllm Day-0 different from transformers Day-0

- vllm releases **faster** than vllm-ascend (vllm 0.19 exists while
  vllm-ascend caps at 0.18.0rc1 as of 2026-04-23).
- vllm-ascend is a **plugin** registered via
  `vllm.platform_plugins` entry_point group. Plugin architecture is
  version-tolerant: vllm-ascend 0.17 can register into vllm 0.19 as long
  as the plugin registration API itself is stable.
- This means the "outcome A works-as-is" likelihood is **higher** than
  for transformers Day-0 — a matching vllm-ascend isn't required if
  plugin registration still works.

## Scan protocol

One-shot docker run into a throwaway container based on BASE_IMAGE with
the target vllm pip-installed (`--no-deps` so we don't perturb torch
stack):

```
docker run --rm $BASE_IMAGE bash -c '
pip install --quiet --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --default-timeout=60 --no-deps vllm==$TARGET_VLLM_VERSION 2>&1 | tail -3
echo ===
python3 <<PYEOF
import vllm, inspect, json

# 1. Basic import + plugin load
print("vllm:", vllm.__version__)

# 2. vllm-ascend plugin registration status
# (vllm prints "Platform plugin ascend is activated" on plugin load — check
#  the log. If the plugin import fails, vllm-ascend's entry_point isn\t
#  compatible with new vllm.)

# 3. lora.lora_model module path (CP-002)
try:
    from vllm.lora.lora_model import LoRAModel
    print("lora_model: OK")
except Exception as e:
    print("lora_model FAIL:", type(e).__name__, str(e)[:200])

# 4. parallel_state (CP-004)
try:
    from vllm.distributed import parallel_state as p
    print("get_tp_group:", hasattr(p, "get_tp_group"))
except Exception as e:
    print("parallel_state FAIL:", type(e).__name__, str(e)[:200])

# 5. SamplingParams read-only property set (EC-03 surface)
from vllm import SamplingParams
sp = SamplingParams(n=1)  # kwargs — positional ctor signature may drift
cls = type(sp)
ro = []; mutable_props = []
for name in dir(cls):
    if name.startswith("_"): continue
    d = getattr(cls, name, None)
    if isinstance(d, property):
        (ro if d.fset is None else mutable_props).append(name)
print("SP RO props:", sorted(ro))

# 6. LLM.generate signature drift (prompts= vs inputs= vs prompt_token_ids=)
from vllm import LLM
try:
    sig = inspect.signature(LLM.generate)
    # Report param names only
    print("LLM.generate params:", list(sig.parameters.keys()))
except Exception as e:
    print("LLM.generate sig FAIL:", type(e).__name__, str(e)[:200])

# 7. VLLMHijack targets
for target in [
    ("vllm.lora.worker_manager", "LRUCacheWorkerLoRAManager"),
    ("vllm.lora.utils", "get_adapter_absolute_path"),
    ("vllm.lora.peft_helper", "PEFTHelper"),
]:
    mod, name = target
    try:
        m = __import__(mod, fromlist=[name])
        ok = hasattr(m, name)
        print(f"HIJACK {mod}.{name}: {ok}")
    except Exception as e:
        print(f"HIJACK {mod}.{name}: FAIL {type(e).__name__}")
PYEOF
'
```

Persist to `$WORKSPACE/api-drift-scan.raw.txt`.

## Interpret the scan

### vllm-ascend plugin load

If the `docker run` output contains
`Platform plugin ascend is activated` during vllm init, the plugin
registered successfully. This is the #1 gate — if it fails, nothing
else matters.

If plugin fails to load (message like `plugin failed to register` or
silent absence of the activation log), it's **outcome C**:
vllm-ascend's `vllm_ascend/__init__.py` or its `register()` function
references vllm API that changed. Specific file to cite in blocker
report: whichever vllm module the traceback complains about.

### lora_model / get_tp_group

Both are stable across vllm 0.13 → 0.19 per today's probe. If they
break, it's likely a deeper refactor and **outcome C**.

### SamplingParams RO property set

v2 image (vllm 0.18): `{all_stop_token_ids, bad_words_token_ids,
eos_token_id}`.
vllm 0.19 probe (2026-04-23): same set, no growth.

If target version adds new RO properties AND consumer code does
`setattr` on them → **outcome B** with EC-03 introspection shim (already
version-proof; just needs to land in consumer source if not already).

### LLM.generate signature

Historical drift:
- vllm 0.13: `generate(prompts, sampling_params, ...)` — positional
- vllm 0.18+: `generate(prompts=None, sampling_params=None, ...)` —
  mostly kwargs
- Consumer (vllm_rollout_spmd.py:225 in EasyR1) uses `prompts=` kwarg,
  safe

If target introduces a new positional or removes `prompts=`, that's
**outcome B** with consumer kwarg-fix.

### VLLMHijack targets

All three imports must resolve. If any fails → VLLMHijack needs
re-targeting → **outcome B** (update hijack file) or **outcome C**
(method was removed entirely).

### New-kernel imports that fail (2026-04-23 finding)

vllm 0.19's `gpt_oss_triton_kernels_moe.py` imports
`triton.runtime.jit.constexpr_function`, which triton-ascend 3.2.0
doesn't export. The import ERRORs at vllm init time but **doesn't kill
vllm** — it's a module that only gets used if consumer requests a
gpt_oss MoE model.

**Decision rule**: grep consumer source for any model type that would
trigger the failing import path. If consumer doesn't use it →
**outcome A with informational note** (document the conditional
failure mode so users who might later enable gpt_oss know to expect
a blocker then). If consumer uses it → **outcome C** with specific
triton-ascend symbol to upstream.

## Impact grep on consumer source

```
cd upstream/<consumer>
git show $UPSTREAM_REF:verl/ | grep -rE '
  vllm.lora.models|
  get_tensor_model_parallel_group|
  SamplingParams\(.*\).*=|
  LLM\.generate\(|
  gpt_oss|
  moe
' .
```

Map hits to risk table above.

## Output

`$WORKSPACE/api-drift-scan.md`:

```markdown
# vllm API drift scan: base {base_vllm_ver} (vllm-ascend {base_ascend_ver}) → target {target_vllm_ver}

## Plugin registration status
- vllm-ascend activates on target: {YES | NO — err: ...}

## API surfaces
- lora.lora_model.LoRAModel: {OK | ...}
- parallel_state.get_tp_group: {present | ...}
- SamplingParams RO set: {same as base | added: [...]}
- LLM.generate params: [{list}] (compare to base: {same | drifted: ...})
- VLLMHijack targets: {all resolve | failed: [...]}

## New-kernel import failures (non-fatal)
- failing_module: {path in site-packages}
- missing symbol: {name}
- triggered by: {consumer model types that hit this path}
- actually used by consumer? {yes → outcome C | no → outcome A w/ note}

## Consumer-source impact
- {symbol}: {file}:{line}: {safe | shim | handler}

## Recommendation
{A | B | C} — rationale: ...
```
