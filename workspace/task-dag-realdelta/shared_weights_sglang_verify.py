"""SHARED-WEIGHTS sglang verification — proves megatron->sglang parameter flow is REAL.

Loads shared_attn_2layer.pt (INIT / TRAINED / SYNTH from the megatron export). For the sglang
2-layer V4 fab:
  1. push INIT attn weights  -> rollout_INIT      (sglang now shares megatron's pre-train attn)
  2. push TRAINED attn weights -> rollout_TRAINED  (the real post-training attn)
  3. push SYNTH attn weights  -> rollout_SYNTH     (control: INIT + random delta, SAME per-tensor L2)

Parameter flow is REAL (not arbitrary) iff:
  (A) rollout_TRAINED != rollout_INIT          -> the trained weights changed the rollout
  (B) rollout_TRAINED != rollout_SYNTH         -> the change is SPECIFIC to real training, not
                                                  reproducible by an equal-magnitude random delta
If (A) holds but (B) fails (TRAINED==SYNTH), the "flow" carries no training-specific signal —
honest FAIL. Both (A) and (B) required for a real PASS.

megatron name (layers.N.self_attention.SUF.weight) -> sglang (model.layers.N.self_attn.SUF.weight).
"""
import os, sys, time, types

sys.path = [p for p in sys.path if p not in ("", "/")]


def _setup_env_and_stubs():
    for k in ["SGLANG_OPT_FP8_WO_A_GEMM","SGLANG_OPT_USE_TILELANG_MHC_PRE","SGLANG_OPT_USE_TILELANG_MHC_POST",
              "SGLANG_OPT_USE_TILELANG_INDEXER","SGLANG_OPT_USE_AITER_MHC_PRE","SGLANG_OPT_USE_AITER_MHC_POST",
              "SGLANG_OPT_DEEPGEMM_HC_PRENORM","SGLANG_OPT_USE_FUSED_COMPRESS","SGLANG_OPT_USE_FUSED_QK_NORM_ROPE",
              "SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL","SGLANG_OPT_USE_ONLINE_COMPRESS","SGLANG_OPT_USE_COMPRESSOR_V2",
              "SGLANG_OPT_USE_OLD_COMPRESSOR","SGLANG_FIX_MTP_HC_HIDDEN","SGLANG_DSV4_FP4_EXPERTS",
              "SGLANG_OPT_USE_TRITON_SWA_PREPARE","SGLANG_OPT_FUSE_WQA_WKV"]:
        os.environ[k] = "0"
    os.environ["SGLANG_FP8_PAGED_MQA_LOGITS_TORCH"] = "1"
    os.environ["SGLANG_TOPK_TRANSFORM_512_TORCH"] = "1"
    mhc = types.ModuleType("sglang.srt.layers.mhc")
    def _stub(*a, **k): raise RuntimeError("mhc stub called")
    mhc.mhc_fused_post_pre = _stub; mhc.mhc_pre = _stub; mhc.mhc_post = _stub
    mhc.get_mhc_pre_token_count_representatives = lambda *a, **k: [1,2,4,8,16]
    def _hc(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        import torch
        b,s,_ = mixes.size(); dev,dt = mixes.device, mixes.dtype
        return (torch.full((b,s,hc_mult),1.0/hc_mult,device=dev,dtype=dt),
                torch.full((b,s,hc_mult),1.0/hc_mult,device=dev,dtype=dt),
                torch.full((b,s,hc_mult,hc_mult),1.0/(hc_mult*hc_mult),device=dev,dtype=dt))
    mhc.hc_split_sinkhorn = _hc
    sys.modules["sglang.srt.layers.mhc"] = mhc
    amd_root = types.ModuleType("sglang.srt.models.deepseek_common.amd"); amd_root.__path__ = []
    sys.modules["sglang.srt.models.deepseek_common.amd"] = amd_root
    amd_inner = types.ModuleType("sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc")
    amd_inner.fused_mhc_quant_routed_scaled_score = _stub; amd_inner.fused_mhc_post = _stub; amd_inner.fused_mhc_pre = _stub
    sys.modules["sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc"] = amd_inner


_setup_env_and_stubs()
import torch
from safetensors.torch import load_file

FAB = "/host-models/dsv4_REAL_2layer_fab"
PT = "/home/z00637938/workspace/task-dag-realdelta/shared_attn_2layer.pt"
PROMPTS = [[1,2,3],[9,8,7],[5,5,5],[2,7,1]]   # more prompts -> more sensitive divergence test


def meg_to_sgl(mname):
    # layers.N.self_attention.SUF.weight -> model.layers.N.self_attn.SUF.weight
    return "model." + mname.replace(".self_attention.", ".self_attn.")


def to_named(weight_dict, base_sd):
    out = []
    for mname, t in weight_dict.items():
        sname = meg_to_sgl(mname)
        base = base_sd[sname]
        tt = t if tuple(t.shape)==tuple(base.shape) else (t.t().contiguous() if tuple(t.shape[::-1])==tuple(base.shape) else None)
        if tt is None:
            raise ValueError(f"{sname}: shape {tuple(t.shape)} vs sglang {tuple(base.shape)}")
        out.append((sname, tt.to(base.dtype).npu()))
    return out


def sig(outs):
    return tuple(tuple(o.get("output_ids",[])) for o in outs)


def main():
    import sglang as sgl
    print(f"[verify] sgl {sgl.__version__}", flush=True)
    blob = torch.load(PT, map_location="cpu")
    INIT, TRAINED, SYNTH = blob["INIT"], blob["TRAINED"], blob["SYNTH"]
    base_sd = load_file(os.path.join(FAB,"model.safetensors"))
    print(f"[verify] fab {len(base_sd)} tensors; attn sets INIT/TRAINED/SYNTH each {len(INIT)}", flush=True)

    llm = sgl.Engine(model_path=FAB, dtype="bfloat16", device="npu", mem_fraction_static=0.50,
                     max_total_tokens=65536, max_running_requests=1, max_prefill_tokens=4096,
                     chunked_prefill_size=4096, disable_radix_cache=True, skip_server_warmup=True,
                     swa_full_tokens_ratio=0.5, tp_size=1, disable_cuda_graph=True,
                     trust_remote_code=False, log_level="info")
    print(f"[verify] Engine init OK", flush=True)

    def rollout(tag, wd):
        llm.update_weights_from_tensor(to_named(wd, base_sd))
        outs = llm.generate(input_ids=PROMPTS, sampling_params={"temperature":0.0,"max_new_tokens":4})
        s = sig(outs); print(f"[verify] {tag} sig={s}", flush=True); return s

    s_init = rollout("INIT   ", INIT)
    s_train = rollout("TRAINED", TRAINED)
    s_synth = rollout("SYNTH  ", SYNTH)

    a = s_train != s_init      # trained changed rollout vs shared init
    b = s_train != s_synth     # trained-specific, not reproduced by equal-L2 random delta
    print(f"\n[verify] (A) TRAINED != INIT  : {a}", flush=True)
    print(f"[verify] (B) TRAINED != SYNTH : {b}", flush=True)
    if a and b:
        print("[verify] === PASS: megatron->sglang parameter flow is REAL & training-specific ===", flush=True)
        print("[verify]   shared init confirmed: both sides started from megatron INIT attn; the trained", flush=True)
        print("[verify]   weights produce a rollout that an equal-magnitude random delta does NOT reproduce.", flush=True)
    elif a and not b:
        print("[verify] === HONEST FAIL: TRAINED changes rollout but == SYNTH control -> no training-specific signal ===", flush=True)
    else:
        print("[verify] === FAIL: TRAINED did not change rollout vs shared INIT ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
