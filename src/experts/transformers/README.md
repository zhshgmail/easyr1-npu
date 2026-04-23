# `transformers/` — Hugging Face transformers

**Upstream repo**: `github.com/huggingface/transformers`
**Edit scope**:
- Community core files — **C-report only**
- `src/transformers/integrations/npu_*` — Ascend-contributed NPU integrations, C-patch OK

## Experts in this folder

| Expert | What it does | Audience |
|---|---|---|
| [`day0-expert/`](day0-expert/) | Day-0 probe of a new transformers release before NPU integration catches up. | transformers NPU integration maintainers + downstream |
| [`upgrade-expert/`](upgrade-expert/) | Shim-adapt transformers to a new release on a known-good NPU image. | EasyR1 / RL framework porters |

## Related folders

- [`../torch-npu/`](../torch-npu/) — torch/torch_npu dep
- [`../vllm-ascend/`](../vllm-ascend/) — vllm-ascend sometimes depends on transformers API
