---
id: mindspeed-001
date: 2026-05-29
layer: mindspeed
title: MindSpeed create_dummy=True installs stub flash_attn in sys.modules; tricks vllm find_spec into believing flash-attn is available
trigger:
  - "import vllm followed by import mindspeed in same Python process"
  - "vllm rotary embedding crashes with cryptic 'flash_attn' related error"
  - "find_spec('flash_attn') returns truthy on NPU but `import flash_attn` then fails"
  - "Single-process RL on NPU (rollout + train in one Python interpreter)"
symptom_in_wild:
  - "vllm-ascend rotary_embedding crash mentioning flash_attn"
  - "ImportError mid-call when vllm tries to use flash-attn fast path that was advertised as available"
  - "Works fine if mindspeed is imported in a subprocess but breaks in single-process driver"
root_cause: >
  MindSpeed's `create_dummy=True` registration pattern installs stub modules in
  `sys.modules` for packages that may not be present on the target system
  (flash_attn, apex, etc.) so that downstream `import flash_attn` doesn't
  crash and MindSpeed's patched/native code path can intercept.

  vllm uses `importlib.util.find_spec("flash_attn")` to decide whether to enable
  its flash-attn fast path. `find_spec` returns truthy for sys.modules entries
  including stubs. vllm then calls into what it believes is real flash_attn;
  the stub does not implement the API; downstream crash.

  Neither package is wrong. MindSpeed `create_dummy` is intentional;
  vllm `find_spec` is the standard Python idiom. The collision happens at the
  consumer layer.
mistake_pattern: "two libraries' independently-correct namespace tricks collide in consumer's import order"
correction:
  - "**Import vllm BEFORE mindspeed.** In the consumer driver: defer all mindspeed/megatron/miles imports to inside the actor-train function. vllm's auto-discovery runs to completion first, finds no flash_attn (correctly), takes the non-flash path."
  - "Concrete pattern in `_e2e_rl_step_mindspeed.py`: top-level `from vllm import LLM, SamplingParams` + rollout phase; then in `_actor_train()`, `import mindspeed; mindspeed.megatron_adaptor.apply()`"
  - "Do NOT try to del sys.modules['flash_attn'] after mindspeed import — mindspeed may re-add it later in its lazy path"
  - "Do NOT try to upstream this — both libs are behaving correctly. The fix is consumer-side import ordering."
  - "If both must be imported at top level (e.g. because of module-level decorators that need both), spawn a subprocess for one of them."
evidence:
  - "Empirical: tlrescue 2026-05-29, vllm rollout (Qwen2-0.5B) + mindspeed actor train PASS only when import order is vllm-first"
  - "vllm source: `vllm/_custom_ops.py` and rotary path use `find_spec` not `try: import` -- the bug is reproducible by any project that depends on `find_spec` semantics with stub modules"
  - "MindSpeed source: `mindspeed/megatron_adaptor.py` invokes patch_features() which uses `create_dummy=True` for flash_attn"
  - "User Discord 2026-05-29: confirmed deferred-import pattern works without modifying either upstream"
---

# mindspeed-001 — create_dummy=True flash_attn stub trips vllm find_spec

## Why this matters

Single-process RL on NPU is the validation pattern for "rollout + train share the same process". Without it, every weight sync goes through disk or P2P RPC. The collision above was the single largest blocker for the miles PoC RL step; once import order was fixed, everything else fell into place.

## Symptom signatures

You'll know it's this bug when:
- vllm initialization itself succeeds (no top-level crash)
- vllm rotary_embedding fails inside a generate() call
- the same code in two processes (rollout proc + train proc) works fine
- `python -c "from importlib.util import find_spec; print(find_spec('flash_attn'))"` returns truthy on a system where no real flash_attn wheel is installed

## Anti-fix: don't try to "make flash_attn detection robust"

Tempting fixes that DON'T work:
- `try: import flash_attn; except: pass` inside vllm — would need an upstream PR, slow path
- Custom `find_spec` shim that filters stubs — fragile, breaks honest stubs
- `del sys.modules['flash_attn']` before mindspeed import — mindspeed re-adds it later

The simple, correct fix is import order in the consumer. Memorize it.
