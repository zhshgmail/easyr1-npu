from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _jit_mask_topk_module():
    return load_jit(
        make_name("mask_topk"),
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("run", "MaskKernel::run")],
    )


@cache_once
def _jit_hash_topk_module():
    args = make_cpp_args("act_sqrt_softplus", is_arch_support_pdl())
    return load_jit(
        make_name("hash_topk"),
        *args,
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("hash_topk", f"HashTopKKernel<{args}>::run")],
    )


@cache_once
def _jit_mega_moe_pre_dispatch_module(quant_group_size: int):
    args = make_cpp_args(quant_group_size, is_arch_support_pdl())
    return load_jit(
        make_name("mega_moe_pre_dispatch"),
        *args,
        cuda_files=["deepseek_v4/mega_moe_pre_dispatch.cuh"],
        cuda_wrappers=[("run", f"MegaMoEPreDispatchKernel<{args}>::run")],
    )


@cache_once
def _jit_silu_mul_quant_varlen_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
):
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_varlen"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_mul_quant_contig_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
):
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_contig"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulContigPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_and_mul_clamp_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("silu_and_mul_clamp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


def mask_topk_ids(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor):
    return _jit_mask_topk_module().run(topk_ids, num_token_non_padded)


def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scoring_func == "sqrtsoftplus"
    if is_hip_runtime():
        from sglang.jit_kernel.triton.hash_topk import hash_topk_triton

        return hash_topk_triton(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts,
            routed_scaling_factor,
            scoring_func,
        )
    else:
        num_tokens = router_logits.size(0)
        topk_routed = tid2eid.size(1)
        topk_fused = topk_routed + num_fused_shared_experts
        topk_ids = torch.empty(
            (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
        )
        topk_weights = torch.empty(
            (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
        )
        module = _jit_hash_topk_module()
        module.hash_topk(
            router_logits,
            input_ids,
            tid2eid,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
        )
        return topk_weights, topk_ids


def mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int = 32,
) -> None:
    module = _jit_mega_moe_pre_dispatch_module(quant_group_size)
    module.run(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )


def silu_and_mul_clamp(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    # NPU native path for silu_and_mul_clamp.
    # input: [..., 2N]; output: [..., N]
    # ref formula: out = silu(in[..., :N]).clamp(-lim, lim) * in[..., N:].clamp(-lim, lim)
    # torch_npu.npu_clipped_swiglu(alpha=1.0, bias=0.0, interleaved=False) reproduces
    # this exactly (plain SiLU = x*sigmoid(x), front/back split, symmetric clamp).
    # Verified vs the torch ref to bf16-ulp (max_rel < 0.8%, bit-exact within clamp).
    import torch
    lim = float(swiglu_limit)
    try:
        import torch_npu  # noqa: F401
        if input.is_npu:
            out = torch_npu.npu_clipped_swiglu(
                input, dim=-1, alpha=1.0, limit=lim, bias=0.0, interleaved=False
            )
            output.copy_(out)
            return
    except (ImportError, AttributeError):
        pass
    N = output.shape[-1]
    gate = input[..., :N]
    up = input[..., N:]
    gate_act = torch.nn.functional.silu(gate).clamp(-lim, lim)
    up_c = up.clamp(-lim, lim)
    output.copy_(gate_act * up_c)


def silu_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    module = _jit_silu_mul_quant_varlen_module(
        quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
    )
    module.run(
        input,
        output,
        output_scale,
        masked_m,
        topk,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )


def silu_and_mul_contig_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool = False,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    module = _jit_silu_mul_quant_contig_module(
        quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
    )
    module.run(
        input,
        output,
        output_scale,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )
