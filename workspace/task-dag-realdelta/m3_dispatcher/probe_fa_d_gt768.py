"""1-min probe: does npu_fusion_attention accept head_dim D>768 on A3? RUN vs 161002 reject."""
import torch, torch_npu
dev = "npu:0"
B, N, S = 1, 1, 128
scalefn = lambda D: 1.0 / (D ** 0.5)
for D in (768, 1024, 1280):
    try:
        q = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
        k = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
        v = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
        out = torch_npu.npu_fusion_attention(q, k, v, N, input_layout="BNSD", scale=scalefn(D))
        o = out[0] if isinstance(out, (tuple, list)) else out
        torch.npu.synchronize()
        print(f"D={D:5d}  RUN ok  out{tuple(o.shape)} finite={torch.isfinite(o.float()).all().item()}")
    except Exception as e:
        msg = str(e).replace("\n", " ")
        code = "161002" if "161002" in msg else ("" )
        print(f"D={D:5d}  REJECT  {code}  {type(e).__name__}: {msg[:160]}")
print("DONE")
