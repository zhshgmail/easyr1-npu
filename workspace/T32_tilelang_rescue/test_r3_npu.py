"""Test: return_routed_experts on NPU with our 1-layer DSv4-Flash fab ckpt.

Note: our fab ckpt has first_k_dense_replace=1 and num_hidden_layers=1, so
the SINGLE layer is dense -- it WON'T route experts. So we won't see expert
IDs in the output meta_info. But we can:
  1. Verify the flag is accepted (no error)
  2. Verify the meta_info structure shape

To truly test R3 on NPU we'd need a fab ckpt with at least one MoE layer
(first_k_dense_replace=0 + n_routed_experts>0). For now, just verify the
flag flows end-to-end.
"""
import sys
sys.path = [p for p in sys.path if p not in ("", "/")]

import os
# don't set ASCEND_RT_VISIBLE_DEVICES on a 1-device container; would filter away the sole chip  # let container default mapping use davinci1


def main():
    print(f"[test] importing sglang ...")
    import sglang as sgl
    print(f"[test] OK")

    MODEL = "/host-models/dsv4_1layer_fab"

    print(f"[test] starting sgl.Engine ...")
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
        enable_memory_saver=False,
        disable_radix_cache=True,
        tp_size=1,
        disable_cuda_graph=True,
        trust_remote_code=False,
    )
    print(f"[test] engine init in {time.time()-t0:.1f}s")

    # First: generate WITHOUT R3 (baseline)
    print(f"[test] baseline generate (return_routed_experts=False) ...")
    t1 = time.time()
    outs = llm.generate(["Hi"], sampling_params={"temperature": 0.0, "max_new_tokens": 4})
    print(f"  baseline {time.time()-t1:.1f}s; out: {outs[0]}")
    baseline_keys = set(outs[0].keys()) if isinstance(outs[0], dict) else set()

    # Now: with R3
    print(f"[test] R3 generate (return_routed_experts=True) ...")
    t2 = time.time()
    outs_r3 = llm.generate(
        ["Hi"],
        sampling_params={"temperature": 0.0, "max_new_tokens": 4},
        return_routed_experts=True,
    )
    print(f"  R3 {time.time()-t2:.1f}s; out: {outs_r3[0]}")
    r3_keys = set(outs_r3[0].keys()) if isinstance(outs_r3[0], dict) else set()

    print(f"[test] baseline keys: {sorted(baseline_keys)}")
    print(f"[test] R3 keys: {sorted(r3_keys)}")
    print(f"[test] new in R3: {sorted(r3_keys - baseline_keys)}")

    # Look for routing info in meta_info
    if isinstance(outs_r3[0], dict):
        mi = outs_r3[0].get("meta_info", {})
        print(f"[test] meta_info keys: {list(mi.keys()) if isinstance(mi, dict) else type(mi)}")
        for k in mi if isinstance(mi, dict) else []:
            if "expert" in k.lower() or "rout" in k.lower() or "topk" in k.lower():
                v = mi[k]
                print(f"  meta_info[{k!r}] type: {type(v).__name__} len: {len(v) if hasattr(v, '__len__') else 'n/a'}")

    print(f"[test] PASS")


if __name__ == "__main__":
    main()
