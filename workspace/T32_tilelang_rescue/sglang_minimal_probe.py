"""Probe: how small can we make sglang's HBM footprint on NPU?

Goal: validate whether the sgl-on-NPU path (§5.2 (3)) is viable when we
only have a few GB of headroom alongside another user's long-running job.

Strategy: launch sglang.Engine with the most aggressive low-memory knobs:
  * mem_fraction_static=0.05 (target: ~3.2 GiB ceiling)
  * max_total_tokens=1024 (tiny KV pool)
  * max_running_requests=1 (no concurrency)
  * max_prefill_tokens=256, chunked_prefill_size=256
  * enable_memory_saver=True (defers weights load until generate)
  * disable_radix_cache=True (no shared-prefix KV)
  * disable_cuda_graph=True (already set; no graph capture buffer)

Then poll npu-smi periodically and report actual HBM used at:
  * after import sglang
  * after Engine() init
  * after one generate
  * after Engine.shutdown()

This is the empirical answer to 'can we run a sglang baseline at < 5 GiB?'
"""
import os
import sys
import time

sys.path = [p for p in sys.path if p not in ("", "/")]

MODEL = "/host-models/Qwen2-0.5B-Instruct"


def get_hbm_mb():
    """Read this container's NPU HBM usage in MB."""
    import subprocess
    try:
        # /usr/local/sbin/npu-smi info -t proc-mem -i 0 -c 0 for the single mounted device
        out = subprocess.run(
            ["/usr/local/sbin/npu-smi", "info", "-t", "proc-mem", "-i", "0", "-c", "0"],
            capture_output=True, text=True, timeout=10,
        )
        # parse "Process memory(MB):XXX" lines for our PID
        my_pid = os.getpid()
        for line in out.stdout.splitlines():
            if f"Process id:{my_pid}" in line:
                # ...Process memory(MB):XXX
                mb = int(line.split("Process memory(MB):")[-1].strip())
                return mb
    except Exception as e:
        return f"err:{e}"
    return None


def main():
    print(f"[probe] HBM at start: {get_hbm_mb()} MB")

    print(f"[probe] importing sglang ...")
    import sglang as sgl
    print(f"[probe] HBM after import: {get_hbm_mb()} MB")

    print(f"[probe] starting offline engine for {MODEL} with low-mem knobs ...")
    t0 = time.time()
    llm = sgl.Engine(
        model_path=MODEL,
        dtype="bfloat16",
        device="npu",
        mem_fraction_static=0.05,
        max_total_tokens=1024,
        max_running_requests=1,
        max_prefill_tokens=256,
        chunked_prefill_size=256,
        enable_memory_saver=True,
        disable_radix_cache=True,
        tp_size=1,
        disable_cuda_graph=True,
    )
    print(f"[probe] engine init in {time.time()-t0:.1f}s, HBM after init: {get_hbm_mb()} MB")

    prompts = ["Hi"]
    t1 = time.time()
    outs = llm.generate(prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 4})
    print(f"[probe] generate in {time.time()-t1:.1f}s, HBM after gen: {get_hbm_mb()} MB")

    for out in outs:
        text = out.get("text") if isinstance(out, dict) else getattr(out, "text", str(out))
        print(f"  -> {text!r}")

    print("[probe] PASS")


if __name__ == "__main__":
    main()
