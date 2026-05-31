---
id: cross-layer-010
date: 2026-05-29
layer: cross-layer
title: Single-process RL: import vllm BEFORE mindspeed; mindspeed's stub flash_attn corrupts vllm's find_spec
trigger:
  - "single-process RL driver doing rollout (vllm) + train (mindspeed+Megatron) in one Python interpreter"
  - "vllm rotary embedding crash mid-generate() in the same process where mindspeed was imported"
  - "import order ambiguity between vllm and mindspeed"
symptom_in_wild:
  - "Top-level mindspeed import succeeds"
  - "Top-level vllm import succeeds"
  - "First vllm.generate() call crashes in rotary embedding with flash_attn-related error"
  - "Same code in two separate processes (rollout proc + train proc) works fine"
root_cause: >
  MindSpeed `create_dummy=True` registers stub modules in `sys.modules` for
  `flash_attn` (which doesn't exist on NPU). The stub is a placeholder that
  lets `import flash_attn` not crash; it does not implement the actual API.

  vllm uses `importlib.util.find_spec("flash_attn")` to detect flash-attn
  availability. find_spec returns truthy for sys.modules entries — INCLUDING
  stubs. vllm then enables its flash-attn fast path; first call into the
  stub fails because the API isn't implemented.

  The collision is symmetric in principle but asymmetric in practice: vllm
  must register its capability detection at import time, before mindspeed
  can install the stub. Once vllm has decided "no flash-attn", further stub
  installation doesn't affect it.

  See also mindspeed-001 for the MindSpeed-side view of the same incident.
mistake_pattern: "two libraries' independently-correct namespace tricks collide; import order decides who wins"
correction:
  - "Import order in driver: vllm FIRST, then defer mindspeed imports to inside the actor-train function. Pattern in _e2e_rl_step_mindspeed.py:"
  - "  # top of module"
  - "  from vllm import LLM, SamplingParams  # auto-discovery runs now"
  - "  ..."
  - "  def _actor_train(...):"
  - "      import mindspeed; mindspeed.megatron_adaptor.apply()"
  - "      # safe — vllm already decided"
  - "Do NOT try to `del sys.modules['flash_attn']` after mindspeed import — mindspeed may re-install in lazy path"
  - "Do NOT fork a subprocess to isolate just to avoid this — overhead is enormous; ordering is enough"
  - "If both must import at top level for some external reason, use multiprocessing.spawn for one of them"
evidence:
  - "tlrescue 2026-05-29 RL step driver: vllm-first PASS, mindspeed-first crash at rotary embedding"
  - "vllm source uses find_spec in `vllm/_custom_ops.py` and rotary path"
  - "mindspeed source: `mindspeed/features_manager/megatron_basic/requirements_basic.py` register_patch with create_dummy=True"
  - "Memory: feedback_vllm_mindspeed_flash_attn_collision.md"
---

# cross-layer-010 — vllm + MindSpeed flash_attn stub collision

## Why this matters

This is the canonical "two upstream libraries are each correct in isolation, collision happens at consumer" pattern. Filed cleanly as "use deferred-import pattern". The temptation to file at vllm asking for find_spec hardening (or at mindspeed asking it to not register stubs) is wrong — both are doing what they're meant to do.

## Diagnostic decision tree

```
crash mentions flash_attn?
  └─ NO → not this bug
  └─ YES
      ├─ NPU host with flash-attn truly installed? → different bug (check wheel build)
      ├─ NPU host with no flash-attn wheel?
      │   ├─ mindspeed imported before vllm? → THIS BUG (reorder)
      │   ├─ vllm in subprocess from mindspeed? → not this bug (isolated procs)
      │   └─ both in same process?
      │       └─ flip import order to vllm-first → fixes
```

## Why the natural fix doesn't work

You'd think: "just delete the stub after mindspeed import". The problem is mindspeed's stub registration is lazy in some paths — it can re-add the stub at first call. The robust fix is import order, not stub cleanup.
