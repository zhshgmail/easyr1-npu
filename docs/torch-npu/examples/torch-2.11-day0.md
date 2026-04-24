# 在 Ascend NPU 上跑 PyTorch 2.11

## 场景

社区刚发布 PyTorch 2.11.0，但 Ascend 官方 NPU 镜像里只打包到 torch 2.9 +
torch_npu 2.9。PyPI 上有 `torch-npu==2.11.0rc1`（预发布版本），本文演示
如何在一台 A3（910C）机器上装起来，并验证基础的 NPU 张量运算能通。

适用读者：想在 Ascend NPU 上用 PyTorch 2.11 的开发者，手上有一台 A3
机器和 Ascend 官方 verl 基础镜像。

---

## 前置条件

- 一台 A3 NPU 机器，至少 1 张空闲的 NPU 卡
- 本地已有基础镜像：
  `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`
  （里面带 CANN 8.5.1 + Python 3.11.14 + torch 2.9 + torch_npu 2.9）
- 磁盘空间 ≥ 5 GB（overlay 大约增 3 GB）

检查 NPU 空闲：

```bash
npu-smi info -t proc-mem -i 0 | grep "Process id" | head -1
# 输出为空 = 卡空闲，可以继续
```

---

## 步骤 1：构建 overlay 镜像

在一个干净目录里建 `Dockerfile`：

```dockerfile
ARG BASE_IMAGE=quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5
FROM ${BASE_IMAGE}

# 升级 torch/torch_npu。--no-deps 保留基础镜像里的 numpy/safetensors 等
RUN pip install --no-deps \
        torch==2.11.0+cpu \
        torchvision==0.26.0+cpu \
        torchaudio==2.11.0+cpu \
        torch_npu==2.11.0rc1 \
        --extra-index-url https://download.pytorch.org/whl/cpu/

# 只用 pip metadata 确认版本。注意：build 阶段不要 `import torch`，
# 因为 torch 2.11 会自动 dlopen torch_npu -> libascend_hal.so，
# docker build 容器里没挂 NPU device，会直接挂。
RUN python3 -c "from importlib.metadata import version as v; \
    print('torch', v('torch')); print('torch_npu', v('torch_npu'))"
```

构建：

```bash
docker build -t mytorch211:latest .
```

---

## 步骤 2：冒烟测试（smoke test，即最小可运行验证）

跑一个容器，挂载 NPU 设备，执行下面的 Python 代码：

```bash
docker run --rm -it \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    mytorch211:latest \
    python3 -c '
import torch, torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("device count:", torch.npu.device_count())

# 一次最小 NPU matmul
a = torch.randn(4, 4).npu()
b = torch.randn(4, 4).npu()
c = a @ b
print("matmul OK, shape:", c.shape, "device:", c.device)
'
```

期望输出：

```
torch: 2.11.0+cpu
torch_npu: 2.11.0rc1
device count: 1
matmul OK, shape: torch.Size([4, 4]) device: npu:0
```

跑到这里就说明 torch 2.11 在 NPU 上基本可用了。

---

## 已知不能直接用的部分

**vllm-ascend 的 C++ 扩展会崩。**
基础镜像里预装的 `vllm_ascend_C.cpython-*.so` 是针对 torch 2.9 ABI 编译
的。升级到 torch 2.11 后，调用 `torch.ops._C_ascend.npu_add_rms_norm_bias`
之类的算子会 SIGSEGV。

如果只做训练/推理脚本里的纯 PyTorch 逻辑，不碰 vllm，可以忽略。

如果需要 vllm 推理，两个办法：

1. **绕开 native 路径**：设置 `VLLM_BATCH_INVARIANT=1`，让 vllm-ascend
   走 Python fallback，速度略慢但能跑。
2. **重编译 `.so`**：在 overlay 容器里跑：
   ```bash
   cd /path/to/vllm-ascend
   # 放宽 CMakeLists.txt 里的 torch 版本硬钉（=2.9 → 允许 2.11.x）
   python3 setup.py build_ext --inplace
   ```
   编出的 `vllm_ascend_C.cpython-*.so`（约 472 KB）覆盖镜像里的旧文件。

**其它已观察到的陷阱：**

- `import torch` 在 torch 2.11 会触发 `_import_device_backends()` 自动
  加载 torch_npu，进而依赖 `libascend_hal.so`。在没挂 NPU 的环境
  （例如 `docker build` 阶段、CI）里，不要 import torch——只读 pip
  metadata 确认版本即可。

---

## 参考实测记录

2026-04-23 在 A3 机器上用上述步骤构建，torch 2.11.0 + torch_npu
2.11.0rc1 的基础 smoke 6 项全通过（pip metadata / import torch /
import torch_npu / device count / NPU matmul / API 存在性）。
