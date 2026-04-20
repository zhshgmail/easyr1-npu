"""V1.3 rollout smoke: load Qwen2-0.5B-Instruct via vllm_ascend and generate.

Expected outcome: a few hundred tokens of coherent text for each prompt.
If this works end-to-end, vllm_ascend covers the forward path for Qwen2
on this image.
"""

import os
import sys

# Ensure our triton-ascend fix is applied.
import torch  # triggers torch_npu auto-load
print("torch:", torch.__version__, "torch.npu.is_available():", torch.npu.is_available())

from vllm import LLM, SamplingParams

_user = os.environ.get("NPU_USER", os.environ.get("USER", "nobody"))
MODEL_PATH = os.environ.get("MODEL_PATH", f"/data/{_user}/models/Qwen2-0.5B-Instruct")
print("loading:", MODEL_PATH)

# vllm_ascend is registered as a platform plugin; `LLM(...)` should find it.
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,   # vllm calls it gpu_memory_* even on NPU
    max_model_len=1024,
    enforce_eager=True,            # skip compile; faster to start and fewer moving parts
)

sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64)
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "def fibonacci(n):",
]
outputs = llm.generate(prompts, sampling)
print("---")
for out in outputs:
    print("PROMPT:", repr(out.prompt))
    print("OUTPUT:", repr(out.outputs[0].text))
    print()

# Minimal sanity: every generation produced at least 1 token.
assert all(len(o.outputs[0].token_ids) > 0 for o in outputs), "some generation produced 0 tokens"
print("V1.3 ROLLOUT SMOKE PASSED")
