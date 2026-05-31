---
id: sglang-001
date: 2026-05-30
layer: sglang
title: sglang.Engine uses multiprocessing spawn; top-level driver must guard with if __name__ == "__main__"
trigger:
  - "calling sglang.Engine(...) in a Python script without __main__ guard"
  - "RL driver instantiating sglang Engine for rollout inside a single Python process"
  - "RuntimeError, fork bomb, or hang at Engine init on Ascend A3"
symptom_in_wild:
  - "Engine init never returns / hangs"
  - "Process tree shows N copies of the driver script each re-executing top-level code"
  - "RuntimeError: 'context has already been set' on Engine init"
  - "Memory grows unboundedly during Engine startup"
root_cause: >
  sglang's `Engine.__init__` spawns child workers via Python `multiprocessing`
  with `spawn` start method (required on Linux for CUDA/NPU compatibility). The
  spawned children re-execute the top-level module body. Without
  `if __name__ == "__main__":` guarding the Engine construction, each spawned
  child also constructs an Engine, which spawns more children, ad infinitum.
mistake_pattern: "Python standard etiquette for multiprocessing-spawn modules violated"
correction:
  - "Always wrap sglang.Engine usage in `if __name__ == '__main__':`"
  - "Pattern:"
  - "  def main():"
  - "      llm = sgl.Engine(model_path=..., device='npu', ...)"
  - "      outs = llm.generate([...], sampling_params={...})"
  - "  if __name__ == '__main__': main()"
  - "Don't move the Engine into a module-level constant — that defeats the guard"
  - "If you must use Engine across multiple modules: instantiate in main() and pass the handle, never reach for it from globals"
evidence:
  - "sglang source: `sglang/srt/server.py` uses `multiprocessing.get_context('spawn')`"
  - "Tested 2026-05-30 in `quay.io/ascend/verl:verl-sglang-8.5.0-a3-...` container"
  - "Cold-drive smoke script `workspace/T32_tilelang_rescue/sglang_npu_smoke.py` -- without guard fork bomb; with guard 20.1s init + 16.9s generate"
  - "Memory: feedback_sglang_npu_smoke_recipe.md -- 3 must-avoid pitfalls for sglang baseline on NPU"
---

# sglang-001 — Engine spawn requires __main__ guard

## Why this matters

This is the most common first-time stumble for sglang-on-NPU users. The error mode is opaque (fork bomb or context error), not the obvious-when-you-know-it "multiprocessing module re-imports itself" message Python documents.

## Not worth filing upstream

This is standard Python `multiprocessing` etiquette. sglang's docs could mention it in the Engine page (single sentence), but a PR adding "if __name__" guards into examples is enough. We don't file an issue because:
- sglang follows standard Python multiprocessing semantics
- the guard requirement is a Python language fact, not a sglang bug
- adding runtime checks for it inside sglang would be invasive and miss other valid use cases (e.g. notebooks)

## Variant: jupyter / IPython

In notebooks, the guard is unnecessary because `__name__ == '__main__'` is always true. But spawning sglang Engine inside a notebook on Linux can still hit "context has already been set" if a prior cell already set the multiprocessing context to fork. Restart the kernel before instantiating Engine.
