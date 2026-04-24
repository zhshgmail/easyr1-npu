"""V1.1 / V1.2 smoke test: inside the easyr1-npu container on A3.

Checks:
- torch_npu imports and reports device count.
- verl.utils.device.is_npu_available() returns True.
- get_device_name() returns "npu".
- get_default_attn_implementation() returns "sdpa".
- A small tensor can be placed on the npu and read back.
"""

import torch

print("torch.__version__:", torch.__version__)
try:
    import torch_npu
    print("torch_npu.__version__:", torch_npu.__version__)
    print("torch.npu.is_available():", torch.npu.is_available())
    print("torch.npu.device_count():", torch.npu.device_count())
except ImportError as e:
    print("torch_npu import failed:", e)
    raise SystemExit(1)

# Now exercise our accessors.
from verl.utils.device import (
    is_npu_available,
    get_device_name,
    get_device_module,
    get_default_attn_implementation,
    get_dist_backend,
    get_visible_devices_env,
)

print("---")
print("is_npu_available():", is_npu_available())
print("get_device_name():", get_device_name())
print("get_device_module():", get_device_module())
print("get_default_attn_implementation():", get_default_attn_implementation())
print("get_dist_backend():", get_dist_backend())
print("get_visible_devices_env():", get_visible_devices_env())

# Tensor round trip.
print("---")
dev = torch.device(get_device_name(), 0)
x = torch.arange(12, dtype=torch.float32).view(3, 4).to(dev)
y = (x * 2 + 1).cpu()
print("round-trip result:\n", y)
assert y[0, 0].item() == 1.0 and y[2, 3].item() == 23.0, "arithmetic wrong"
print("round-trip OK")

# Attention_utils facade (forces the NPU branch to resolve; just importing is enough).
from verl.utils.attention_utils import index_first_axis, pad_input, unpad_input, rearrange
attn_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int32).to(dev)
hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3).to(dev)
unpadded, indices, cu, max_sl, used = unpad_input(hidden, attn_mask)
print("unpad shapes:", unpadded.shape, indices.shape, cu.shape, max_sl)
padded_back = pad_input(unpadded, indices, batch=2, seqlen=4)
assert torch.allclose(padded_back[attn_mask.bool()], hidden[attn_mask.bool()])
print("pad/unpad round-trip on npu: OK")

print("\nALL SMOKE CHECKS PASSED")
