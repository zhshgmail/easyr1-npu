# `torch-npu/` — PyTorch + Ascend torch_npu

**Upstream repos**:
- `github.com/pytorch/pytorch` (community torch — C-report only)
- `gitcode.com/Ascend/pytorch` (Huawei `torch_npu` — C-patch OK)

## Experts in this folder

| Expert | What it does | Audience |
|---|---|---|
| [`port-expert/`](port-expert/) | Day-0 probe when a new torch/torch_npu pair lands and NPU ecosystem hasn't caught up. Produces overlay image + deploy artifacts for downstream experts. | torch_npu maintainers + downstream experts (vllm-ascend-day0 etc.) |
| [`port-expert/_legacy-upgrade/`](port-expert/_legacy-upgrade/) | Shim-adapt when both torch and torch_npu have NPU-validated releases. | EasyR1 / RL framework porters |

## Related folders

- [`../vllm-ascend/`](../vllm-ascend/) — common downstream consumer (vllm-ascend_C.so ABI depends on torch version)
- [`../transformers/`](../transformers/) — another common consumer
