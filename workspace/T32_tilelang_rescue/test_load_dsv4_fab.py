"""Quick test: can sglang load the fabricated DSv4 ckpt?

If this passes, sglang's deepseek_v2.py DSA_NPU path accepts our random
weights and is ready for the 5-step driver. If it fails, the error tells
us what config field or weight key is missing.
"""
import sys
sys.path = [p for p in sys.path if p not in ("", "/")]

import os
os.environ.setdefault("SGLANG_USE_AG_AFTER_QLORA", "0")

MODEL = "/host-models/dsv4_1layer_fab"


def main():
    print(f"[test] importing sglang ...")
    import sglang as sgl
    print(f"[test] OK")

    print(f"[test] starting sgl.Engine for {MODEL} ...")
    import time
    t0 = time.time()
    llm = sgl.Engine(
        model_path=MODEL,
        dtype="bfloat16",
        device="npu",
        mem_fraction_static=0.10,
        max_total_tokens=1024,
        max_running_requests=1,
        max_prefill_tokens=256,
        chunked_prefill_size=256,
        enable_memory_saver=False,  # glm5 image lacks torch_memory_saver pkg
        disable_radix_cache=True,
        tp_size=1,
        disable_cuda_graph=True,
        trust_remote_code=False,
    )
    print(f"[test] engine init in {time.time()-t0:.1f}s")

    print(f"[test] one-shot generate ...")
    t1 = time.time()
    outs = llm.generate(["Hello"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})
    print(f"[test] generate in {time.time()-t1:.1f}s")
    for o in outs:
        text = o.get("text") if isinstance(o, dict) else getattr(o, "text", str(o))
        print(f"  -> {text!r}")

    print(f"[test] PASS")


# sglang Engine uses multiprocessing spawn; need __main__ guard
if __name__ == "__main__":
    main()
