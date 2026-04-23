# Domain — Image diff + backcompat shim application

**Load when**: Phase A diffing SOURCE_IMAGE vs TARGET_IMAGE pip-freeze;
Phase B applying known shims from EC catalog.

## The workflow

1. **Extract pip-freeze from both images** (A3-side):
   ```bash
   docker run --rm $SOURCE_IMAGE pip freeze > /tmp/source-freeze.txt
   docker run --rm $TARGET_IMAGE pip freeze > /tmp/target-freeze.txt
   diff /tmp/source-freeze.txt /tmp/target-freeze.txt > workspace/.../pip-diff.txt
   ```
   (If `docker run` on A3 is slow: run once and keep the freeze files cached
   in `knowledge/images/<slug>.md` — they're cheap to store, expensive to
   regenerate.)

2. **Focus on these dep classes** in the diff (ordered by "most-likely to
   need a shim"):
   - `transformers` — top-level public APIs move between majors
   - `vllm` / `vllm-ascend` — TP group helpers, SamplingParams, LoRA
   - `torch` / `torch_npu` — rarely renames public API within a major
   - `triton-ascend` — integrity issues, not API issues (see dockerfile-target.md)
   - `accelerate` / `peft` — usually stable
   - `transformers` indirect deps (safetensors, tokenizers) — usually stable

3. **For each candidate-breaking version delta, grep the consumer codebase
   for the symbol**:
   ```bash
   # target ships transformers 5.x; source had 4.x
   git grep -n 'from transformers.modeling_utils import no_init_weights' \
     upstream/EasyR1/verl/
   # → any hit is a candidate shim site
   ```

4. **Apply the minimal try/except shim** from `ERROR_CORRECTIONS.md`
   EC-NN. Keep the source-image import path as the fallback so v1 smoke
   still PASSes (Phase C backcompat verification depends on this):
   ```python
   try:
       from transformers.initialization import no_init_weights  # >= 5.0
   except ImportError:
       from transformers.modeling_utils import no_init_weights  # <= 4.x
   ```

## Known shim sites on EasyR1 for v1→v2 (transformers 4.57→5.3 + vllm 0.13→0.18)

Drill evidence: only **2 shim edits** total for this upgrade. Both are
one-line try/except (full code in EC-02 / EC-03):

| Callsite | Shim type | EC |
|---|---|---|
| `verl/workers/fsdp_workers.py` (top import of `no_init_weights`) | transformers 5.x import path | EC-02 |
| `verl/workers/rollout/vllm_rollout_spmd.py` (`update_sampling_params`) | vllm 0.18 read-only property skip | EC-03 |

**That's it.** Everything else in the transformers 4.57→5.3 jump was verified
drift-free in Qwen2-0.5B forward pass (see drill report §"what broke vs what
we predicted"):

- `ALL_ATTENTION_FUNCTIONS` stays at `transformers.modeling_utils` in 5.3
  (not `transformers.initialization`)
- `transformers.integrations.npu_flash_attention` API signatures are
  identical (verified via `inspect.signature` pre-drill)
- vllm_ascend LoRAModel + TP group renames were already handled by
  NPU-CP-002 / NPU-CP-004 via `hasattr` gates in the easyr1-expert domain

## Verify after shim application

```bash
# G2: static_check in the consumer checkout
python3 $SHARED/scripts/static_check.py \
  --files $(git diff --name-only $UPSTREAM_REF... | grep '\.py$') \
  --import-package verl
# must exit 0

# Backcompat verify: re-run source-image V1.4 smoke with the shims applied.
# Step-1 entropy_loss must still land in source image's band [0.94, 1.04].
# This proves the shims don't break v1 users.
bash ../_shared/scripts/deploy_to_a3.sh --reuse-image $SOURCE_IMAGE ...
bash scripts/smoke_validate.sh --rung V1.4 --image-family v1 ...
# expect: PASS in source band
```

Backcompat verification is **mandatory** before P4_validation_smoke. Skipping
it = you might ship a "v2 works" image that silently breaks v1 users.
