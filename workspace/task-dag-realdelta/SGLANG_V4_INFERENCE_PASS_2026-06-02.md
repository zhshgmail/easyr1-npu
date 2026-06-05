# sglang V4 inference — re-verified THIS session 2026-06-02 ~23:56

**Captured raw stdout (witnessed inline; Engine in-proc shutdown's kill_process_tree hang
prevents tee/redirect from flushing a file, so the evidence is the captured stdout below):**

```
[v4-min] importing
[v4-min] sgl 0.5.12.post2.dev434+gb13d3d18c
[v4-min] dsv4 jit kernels monkey-patched to torch fallbacks (elw + dv4 namespaces)
[v4-min] Engine starting
[v4-min] Engine init OK in 20.1s
[v4-min] CALLING generate (timeout 120s expected)
[v4-min] generate done in 0.9s
[v4-min] output: [{'text': '醺报废', 'output_ids': [122081, 112435],
   'finish_reason': {'type':'length','length':2}, 'prompt_tokens': 3,
   'completion_tokens': 2, 'e2e_latency': 0.8726s}]
```

## What this proves (and its limits)
- sglang V4 `DeepseekV4ForCausalLM` `generate()` RUNS on Ascend A3 NPU (bf16) and produces real
  tokens (output_ids=[122081,112435], 2 completion tokens, 0.9s). Re-verified this session.
- IDENTICAL output text "醺报废" to the 2026-06-01 log `v4_PASS_log_2026_06_01.txt` (temp=0 → deterministic),
  corroborating the prior result was real.
- **Recipe** (`workspace/v4_attempt_2026_06_01/_sglang_v4_minimal_PASS.py`, copied to sgl_probe `/tmp/v4_min_pass.py`):
  torch-fallback — set ALL `SGLANG_OPT_USE_TILELANG_*=0` + `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` etc., and
  monkey-patch `sglang.srt.layers.mhc` + `sglang.jit_kernel.dsv4` to pure-torch. **tilelang is NOT used.**
- Container `sgl_probe` (sglang 0.5.12.post2.dev434, torch 2.8, 4 NPU devices); model
  `/host-models/dsv4_REAL_1layer_fab` (reduced-layer random-weight fab → gibberish text is EXPECTED;
  this validates loop runnability, not text quality).

## Limits (honest)
- Reduced-layer (1-layer) random-weight fab; gibberish output is expected, not a defect.
- This is `generate()` runnability via torch-fallback, NOT the tilelang-kernel path, NOT real-weight quality.
- The "rollout→weight-sync→rollout closed loop" prior claim was attention-only (5-tensor) — that limit stands;
  not re-verified this session (only generate() re-verified).
