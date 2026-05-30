"""Sanity smoke for sglang-on-NPU.

Goal: verify sglang.Engine can load a small model on Ascend NPU and respond
to a one-shot generate. Establishes the sgl-on-NPU baseline before wiring
it into the miles RL driver.

Run inside verl-sglang container (NOT tlrescue), since this image ships
the matching torch 2.8 / sglang 0.5.10 / sgl_kernel_npu 2026.2.1 stack.
"""
import os
import sys
import time

# Same vllm-flavor fix: editable install gets shadowed by namespace
# resolution because '/' is on sys.path. Strip before any sgl import.
sys.path = [p for p in sys.path if p not in ("", "/")]

# Container has only /dev/davinci1 mounted; inside container that maps
# to logical index 0. Don't set ASCEND_RT_VISIBLE_DEVICES — it'd filter
# AWAY the single mounted chip.

MODEL = "/host-models/Qwen2-0.5B-Instruct"


def main():
    print(f"[smoke] importing sglang ...")
    import sglang as sgl
    print(f"[smoke] sglang OK")

    print(f"[smoke] importing sgl_kernel_npu ...")
    import sgl_kernel_npu
    print(f"[smoke] sgl_kernel_npu OK")

    print(f"[smoke] starting offline engine for {MODEL} ...")
    t0 = time.time()
    llm = sgl.Engine(
        model_path=MODEL,
        dtype="bfloat16",
        device="npu",
        mem_fraction_static=0.15,  # ~10 GiB on a 64 GiB chip
        tp_size=1,
        disable_cuda_graph=True,
    )
    print(f"[smoke] engine init in {time.time()-t0:.1f}s")

    prompts = ["What is the capital of France?", "Hello, my name is"]
    print(f"[smoke] generating {len(prompts)} prompts ...")
    t1 = time.time()
    outs = llm.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 16})
    print(f"[smoke] generate in {time.time()-t1:.1f}s")

    for i, out in enumerate(outs):
        text = out.get("text") if isinstance(out, dict) else getattr(out, "text", str(out))
        print(f"  [{i}] {prompts[i]!r} -> {text!r}")

    print("[smoke] PASS")


# sglang uses multiprocessing with spawn start method, which re-imports
# the main module in subprocesses. Without `if __name__ == '__main__'`
# guard, all the engine init code runs in the child too -> infinite
# recursion + 'attempt to start a new process before bootstrap' error.
if __name__ == "__main__":
    main()
