import torch


def _fp8_e4m3_round(t: torch.Tensor) -> torch.Tensor:
    """Round an fp32 tensor to the fp8_e4m3 representable grid WITHOUT the fp8 dtype
    (torch_npu lacks the Float8_e4m3fn cast op). e4m3: 1-4-3, 3-mantissa-bit step per
    binade. Quantizes the VALUE (QAT only needs the quant error; the caller dequantizes)."""
    sign = torch.sign(t)
    a = t.abs().clamp(min=2.0 ** -9)  # smallest e4m3 normal ~ 2^-6; subnormals ~2^-9
    e = torch.floor(torch.log2(a))
    step = torch.pow(2.0, e - 3)       # 3 mantissa bits
    return sign * torch.round(a / step) * step


def _act_quant_npu(x: torch.Tensor, block_size: int):
    """NPU path: block-wise fp8_e4m3 act quant via torch (dispatches to CANN aclnn);
    matches the tilelang act_quant math (per-(row,block) amax -> scale=amax/448 -> clamp ->
    fp8-grid round). Returns fp8-simulated value in fp32 (no Float8_e4m3fn dtype on torch_npu)."""
    FP8_MAX = 448.0
    N = x.shape[-1]
    nblk = N // block_size
    xb = x.float().reshape(*x.shape[:-1], nblk, block_size)
    amax = torch.clamp(xb.abs().amax(dim=-1, keepdim=True), min=1e-4)
    scale = amax / FP8_MAX
    yq = _fp8_e4m3_round(torch.clamp(xb / scale, -FP8_MAX, FP8_MAX))  # fp8-grid value, fp32
    return yq.reshape(*x.shape), scale.squeeze(-1)


def fp8_simulate(x: torch.Tensor, block_size: int):
    if x.is_npu:
        # NPU: _act_quant_npu already returns the fp8-grid VALUE in fp32 + per-block scale.
        y, scale = _act_quant_npu(x.contiguous(), block_size)
        y = y.unflatten(-1, (-1, block_size)) * scale.unsqueeze(-1)
        return y.flatten(-2).to(x.dtype)
    from .kernel.act_quant import act_quant  # lazy (CUDA-only)
    y, scale = act_quant(x.contiguous(), block_size, "ue8m0")
    y = y.unflatten(-1, (-1, block_size)).float() * scale.unsqueeze(-1)
    return y.flatten(-2).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
