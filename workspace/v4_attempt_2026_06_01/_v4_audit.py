import importlib, traceback
mods = [
    "sglang.jit_kernel.dsv4",
    "sglang.srt.layers.attention.dsa.utils",
    "sglang.srt.layers.attention.dsv4.compressor",
    "sglang.srt.layers.attention.dsv4.indexer",
    "sglang.srt.layers.mhc",
    "sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc",
    "sglang.srt.models.deepseek_v4",  # final entry point
]
for m in mods:
    try:
        mod = importlib.import_module(m)
        f = getattr(mod, "__file__", "?")
        print(f"OK   {m:65s} {f}")
    except Exception as e:
        print(f"FAIL {m:65s} {type(e).__name__}: {e}")
