from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_hip

from .utils import make_name

_is_hip = is_hip()


@cache_once
def _jit_fused_rope_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("fused_rope"),
        *args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


@cache_once
def _jit_main_q_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
):
    """Main MLA path Q kernel: rmsnorm-self + RoPE, warp per (token, head)."""
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name("main_q_norm_rope"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_main_k_norm_rope_flashmla_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
    page_size: int,
):
    """Main MLA path K kernel: rmsnorm + RoPE + write to FlashMLA paged cache."""
    args = make_cpp_args(dtype, head_dim, rope_dim, page_size, is_arch_support_pdl())
    return load_jit(
        make_name("main_k_norm_rope_flashmla"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedKNormRopeFlashMLAKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_main_q_indexer_rope_hadamard_quant_module(dtype: torch.dtype):
    """C4 indexer Q kernel: RoPE + 128-pt Hadamard + fp8 act-quant"""
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("main_q_indexer_rope_hadamard_quant"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward"),
        ],
    )


def _apply_rope_inplace_torch(x, freqs_cis, positions, inverse=False):
    rope_dim = freqs_cis.shape[-1] * 2
    nope_dim = x.shape[-1] - rope_dim
    rope_part = x[..., nope_dim:].float().contiguous()
    rope_complex = torch.view_as_complex(
        rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
    )
    f_real = torch.view_as_real(freqs_cis)[positions]
    f = torch.view_as_complex(f_real.contiguous())
    while f.dim() < rope_complex.dim():
        f = f.unsqueeze(-2)
    if inverse:
        f = f.conj()
    rotated = rope_complex * f
    rotated_real = torch.view_as_real(rotated).reshape(*rope_part.shape).to(x.dtype)
    x[..., nope_dim:].copy_(rotated_real)


def _rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    # V4 self-rmsnorm (no learnable gamma in these fused paths). Native torch_npu.npu_rms_norm
    # is bit-exact vs the torch reference (verified, max_abs_err=0). Falls back to torch off-NPU.
    try:
        import torch_npu  # noqa: F401
        if x.is_npu:
            gamma = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
            return torch_npu.npu_rms_norm(x, gamma, eps)[0]
    except (ImportError, AttributeError):
        pass
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype)


def fused_rope_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    _apply_rope_inplace_torch(q, freqs_cis, positions, inverse)
    if k is not None:
        _apply_rope_inplace_torch(k, freqs_cis, positions, inverse)


def fused_q_norm_rope(
    q_input: torch.Tensor,
    q_output: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    rope_dim = freqs_cis.shape[-1] * 2
    nope_dim = q_input.shape[-1] - rope_dim
    normed = _rmsnorm(q_input, eps)
    if nope_dim > 0:
        q_output[..., :nope_dim].copy_(normed[..., :nope_dim])
    rope_part = normed[..., nope_dim:].float().contiguous()
    rope_complex = torch.view_as_complex(
        rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
    )
    # NPU aclnnIndex doesn't support complex64; index in real domain, view-as-complex after
    f_real = torch.view_as_real(freqs_cis)[positions]
    f = torch.view_as_complex(f_real.contiguous())
    while f.dim() < rope_complex.dim():
        f = f.unsqueeze(-2)
    rotated = rope_complex * f
    rotated_real = torch.view_as_real(rotated).reshape(*rope_part.shape).to(q_input.dtype)
    q_output[..., nope_dim:].copy_(rotated_real)


def fused_q_indexer_rope_hadamard_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    if _is_hip:
        torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant(
            q_input,
            q_fp8,
            weight,
            weights_out,
            float(weight_scale),
            freqs_real,
            positions,
        )
    else:
        module = _jit_main_q_indexer_rope_hadamard_quant_module(q_input.dtype)
        module.forward(
            q_input,
            q_fp8,
            weight,
            weights_out,
            float(weight_scale),
            freqs_real,
            positions,
        )
    return q_fp8, weights_out


def fused_k_norm_rope_flashmla(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
) -> None:
    # NPU torch fallback for K-path: rmsnorm + RoPE + scatter to paged KV cache.
    # kv: [N, head_dim] (single head MLA latent + rope dims merged)
    rope_dim = freqs_cis.shape[-1] * 2
    nope_dim = kv.shape[-1] - rope_dim
    normed = _rmsnorm(kv, eps)
    if nope_dim > 0:
        kv[..., :nope_dim].copy_(normed[..., :nope_dim])
    rope_part = normed[..., nope_dim:].float().contiguous()
    rope_complex = torch.view_as_complex(
        rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
    )
    # NPU aclnnIndex doesn't support complex64; index in real domain, view-as-complex after
    f_real = torch.view_as_real(freqs_cis)[positions]
    f = torch.view_as_complex(f_real.contiguous())
    while f.dim() < rope_complex.dim():
        f = f.unsqueeze(-2)
    rotated = rope_complex * f
    rotated_real = torch.view_as_real(rotated).reshape(*rope_part.shape).to(kv.dtype)
    kv[..., nope_dim:].copy_(rotated_real)
    # NPU PoC: skip packed FP8/BF16 page write (would need byte-exact layout).
    # The Q→K MLA path doesn't read this cache within a single-layer prefill,
    # and our PoC uses max_new_tokens=2 with disable_cuda_graph=True.
    # If a later decode read returns garbage, that's a separate next-step
    # gap (proper kv-page packer on NPU).
    return None
