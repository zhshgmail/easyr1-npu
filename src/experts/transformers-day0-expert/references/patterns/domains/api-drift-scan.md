# Domain — API drift scan (transformers minor→minor)

**Load when**: Phase A probing what changed between the base image's
transformers version and the target community release.

## Inputs

- `BASE_IMAGE_TRANSFORMERS_VER` — from pip-freeze of BASE_IMAGE
- `TARGET_TRANSFORMERS_VERSION` — the new community release under test
- `UPSTREAM_REF` — consumer source ref (for impact-grep)

## Scan protocol

All probes run inside a throwaway container based on BASE_IMAGE with
the target version pip-installed:

```
docker run --rm $BASE_IMAGE bash -c '
pip install --quiet --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --default-timeout=60 transformers==$TARGET_TRANSFORMERS_VERSION 2>&1 | tail -5
python3 <<EOF
import transformers, inspect, json
print("transformers:", transformers.__version__)

# 1. NPU FA sigs
try:
    from transformers.integrations import npu_flash_attention as npu_fa
    sigs = {
        "npu_flash_attn_func": str(inspect.signature(npu_fa.npu_flash_attn_func)),
        "npu_flash_attn_varlen_func": str(inspect.signature(npu_fa.npu_flash_attn_varlen_func)),
    }
    print("NPU_FA_SIGS:", json.dumps(sigs))
except Exception as e:
    print("NPU_FA_FAIL:", type(e).__name__, str(e)[:400])

# 2. ALL_ATTENTION_FUNCTIONS
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    print("ATTN_KEYS:", json.dumps(sorted(ALL_ATTENTION_FUNCTIONS.keys())))
except Exception as e:
    print("ATTN_FAIL:", type(e).__name__, str(e)[:400])

# 3. Known-touched APIs (from consumer source patterns)
touched = [
    ("transformers.modeling_utils", "no_init_weights"),
    ("transformers.initialization", "no_init_weights"),
    ("transformers", "PreTrainedModel"),
]
for mod, name in touched:
    try:
        m = __import__(mod, fromlist=[name])
        ok = hasattr(m, name)
        print(f"API_PROBE {mod}.{name}: {ok}")
    except Exception as e:
        print(f"API_PROBE {mod}.{name}: FAIL {type(e).__name__}")
EOF
'
```

Persist raw output to `$WORKSPACE/api-drift-scan.raw.txt`.

## Interpret the scan

For each probed surface, compare to known-good (BASE_IMAGE) values:

### NPU FA sigs

Known-good v2 image (transformers 5.3.0.dev0, observed 2026-04-23):
- `npu_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs)`
- `npu_flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs)`

Decision:
- **Identical sig** → no blocker from this surface
- **New required positional/kwarg w/ no default** → outcome B (need wrapper
  that fills the missing arg sensibly)
- **Removed kwarg consumer was passing** → outcome B (drop the arg in
  consumer path) or outcome C if semantically required
- **Module moved/renamed** → outcome C (NPU team's job)

### ALL_ATTENTION_FUNCTIONS registry

Known-good v2 keys (transformers 5.3.0.dev0, 2026-04-23):
```
['flash_attention_2', 'sdpa', 'flex_attention']  # approx
```

Known 5.6.0 keys (observed 2026-04-23):
```
['flash_attention_2', 'flash_attention_3', 'flash_attention_4',
 'flex_attention', 'paged|eager', 'paged|flash_attention_2',
 'paged|flash_attention_3', 'paged|flash_attention_4',
 'paged|sdpa', 'sdpa']
```

**Risk logic**:
- New keys **alone** don't break anything (registry entries are only used
  when a model explicitly requests them)
- **Break scenario**: a 5.6-era model sets `attn_implementation="flash_attention_3"`
  by default AND the NPU FA adapter hasn't registered a handler for
  `flash_attention_3` → silent fallthrough
- **Mitigation probe**: for each model the consumer instantiates in V1.1/V1.3/V1.4
  smokes (Qwen2-0.5B for EasyR1), inspect its default
  `attn_implementation` at target transformers and confirm it's a key the
  NPU adapter registers
- If risky: outcome B (add NPU handler for the new key in a patched
  `npu_flash_attention.py`)

### Known-touched APIs

Surface-level: if `no_init_weights` moved location AGAIN between 5.3 and
target, either shim gets updated (easyr1-expert task, not this one) or
outcome C with escalation.

## Impact grep on consumer source

```
cd /home/z00637938/workspace/easyr1-npu/upstream/<consumer>
git show $UPSTREAM_REF:verl/ | grep -rE 'ALL_ATTENTION_FUNCTIONS|npu_flash_attn|attn_implementation' .
```

Map each hit to the scan's per-surface risk assessment. Consumer hits on
a key that the target version removes or reshapes = outcome B or C.

## Output

`$WORKSPACE/api-drift-scan.md`:
```markdown
# API drift scan: base {ver_base} → target {ver_target}

## NPU FA sigs
- npu_flash_attn_func: {SAME | DIFF: base=... target=...}
- npu_flash_attn_varlen_func: {SAME | DIFF: ...}

## ALL_ATTENTION_FUNCTIONS
- new keys: [...]
- removed keys: [...]
- risk: {NONE | consumer uses removed | consumer may default to new unhandled}

## Known-touched APIs
- no_init_weights @ transformers.initialization: {OK | MOVED to ...}
- PreTrainedModel: {OK | signature-changed-at ...}

## Consumer impact
- {symbol} @ {file}:{line}: {safe | needs-shim | needs-handler}

## Recommendation for P2_decide
{A_works_as_is | B_forward_port | C_blocked} — rationale: ...
```

---

## Expected platform-bug carryovers across transformers minor bumps

When overlaying a newer transformers onto an existing NPU image, platform
bugs from the **image stack** generally carry through regardless of which
transformers minor sits on top. This table is informational (don't block
on it; just expect it).

| Platform bug | Trigger | Mitigation (inherited) |
|---|---|---|
| NPU-BUG-001 | triton-ascend partial install | already in base image's Dockerfile.npu-* |
| NPU-BUG-003 | torch.compile'd log_probs path crash | easyr1-expert canonical V1.4 smoke sets `worker.actor.use_torch_compile=false` |
| NPU-BUG-004 | upstream triton amd/nvidia backend pollution | already pruned in v2 base image |

Verified on 2026-04-23 transformers-day0 wet-run (5.6.0 on v2): NPU-BUG-003
still fires on V1.4 **unless** `use_torch_compile=false` is in the smoke
config. Agent cherry-picked the existing workaround (ea08078) and recovered
in 3 iterations. Not a Day-0 finding — expected platform debt.

**Takeaway for future day0 agents**: running the canonical V1.4 smoke config
verbatim (per `easyr1-expert/references/SMOKE_BASELINE.md §"canonical
config"`) already avoids this. If the smoke config you're using is a stripped
version, add `use_torch_compile=false` explicitly before the first V1.4
iteration.

