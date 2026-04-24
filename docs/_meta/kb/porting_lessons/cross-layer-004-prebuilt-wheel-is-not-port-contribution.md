---
id: cross-layer-004
date: 2026-04-24
layer: cross-layer
title: Installing Ascend-prebuilt wheel is not a port contribution
trigger:
  - "pip install torch_npu==<ver>"
  - "pip install triton-ascend==<ver>"
  - "overlay FROM quay.io/ascend/... + pip install torch_npu"
symptom_in_wild:
  - "claim 'torch-npu 2.11.0rc1 works on NPU' based on pip install + import smoke"
  - "claim 'triton-ascend port done' via Ascend's released wheel"
root_cause: >
  Ascend-prebuilt wheels are the upstream team's work, not ours. Installing
  them proves nothing about our port contribution to that layer. If we're
  doing a port, the minimum evidence of our contribution is a commit on our
  personal fork of gitcode.com/Ascend/pytorch (or similar), touching real
  source, addressing a real gap the prebuilt wheel does not cover.
mistake_pattern: "confusing upstream's release with our contribution"
correction:
  - "When a layer is 'working' only because of Ascend's prebuilt wheel, the layer's port status is NOT STARTED or CONSUMER-ONLY, not DONE."
  - "Real layer contribution requires: (a) find a gap the prebuilt wheel doesn't cover (via L1 op bench failures, new upstream ver API drift, etc.), and (b) write C++/Python source in the Ascend fork to fill that gap."
  - "Demonstrable to customer = branch + diff on the Ascend fork, not 'we installed Ascend's wheel'."
evidence:
  - "2026-04-23 iter 20 image: torch_npu 2.11.0rc1 wheel from Ascend. Zero source edits on torch-npu side from our session."
  - "User 2026-04-24T02:47Z: '你在 torch npu 上基本上没做什么工作'"
---

# What port contribution looks like per layer

- **torch-npu**: fork `gitcode.com/Ascend/pytorch` → `<user>/pytorch` → branch
  `torch-<target>_auto_porting` → commits touching `torch_npu/` source.
- **triton-ascend**: fork `gitcode.com/Ascend/triton-ascend` → branch
  `triton-<target>_auto_porting` → commits touching kernel source.
- **transformers**: fork `github.com/huggingface/transformers` → branch
  `transformers-<target>_auto_porting` → commits touching
  `src/transformers/integrations/npu_*` (Ascend-owned area) or proposing PR
  upstream if in community area.
- **vllm-ascend**: fork `github.com/vllm-project/vllm-ascend` → branch
  `vllm-<target>_auto_porting` → commits touching vllm-ascend source.
