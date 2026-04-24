# `vllm/` — community vllm upstream

**Upstream repo**: `github.com/vllm-project/vllm` (community, not Huawei-owned)
**Edit scope**: **C-report only** — we don't have merge authority on community vllm, so we file issues upstream and ship downstream fixes in `vllm-ascend` instead.

## Experts in this folder

| Expert | What it does | Audience |
|---|---|---|
| [`port-expert/`](port-expert/) | Probe vllm 0.20/0.21 etc. Day-0 compatibility with Ascend NPU via vllm-ascend adapter. Surfaces drift; fixes land in `../vllm-ascend/port-expert/`. | vllm-ascend maintainers (receiving the probe results) |
| [`port-expert/_legacy-upgrade/`](port-expert/_legacy-upgrade/) | Shim-adapt vllm to a different version on a known-good NPU ecosystem. | EasyR1 / RL framework porters |

## Related folders

- [`../vllm-ascend/`](../vllm-ascend/) — where vllm Day-0 probes deliver their fixes
- [`../torch-npu/`](../torch-npu/) — torch / torch_npu Day-0 (deeper dep chain)
