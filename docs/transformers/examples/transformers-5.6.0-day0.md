# 在 Ascend NPU 上跑 transformers 5.6.0

## 场景

HuggingFace 刚发布 `transformers==5.6.0`（2026-04-22），但 Ascend 官方
verl 镜像里装的是 `transformers==5.3.0.dev0`。本文演示如何在基础镜像
上叠加一层安装 5.6.0，并用 EasyR1 做一次 rollout + 两步训练来验证
数值正确。

适用读者：想在 Ascend NPU（A3 / 910C）上用 transformers 5.6.0 的开发者。

---

## 前置条件

- 一台 A3 NPU 机器，至少 2 张空闲卡（训练验证需要）
- 基础镜像：
  `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`
- EasyR1 代码：`github.com/zhshgmail/EasyR1`，分支
  `ascend-port-transformers-upgrade`（这个分支已经打好 transformers
  5.x 的向后兼容补丁）
- 磁盘空间 ≥ 3 GB（overlay 大约增 1 GB）

检查 NPU 空闲：

```bash
for i in 0 1; do
    npu-smi info -t proc-mem -i $i | grep "Process id" | head -1
done
# 两行都为空 = 两张卡都空闲
```

---

## 步骤 1：构建 overlay 镜像

建 `Dockerfile`：

```dockerfile
ARG BASE_IMAGE=quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5
FROM ${BASE_IMAGE}

# --no-deps 避免 pip 把已经装好的 torch / torch_npu / numpy 等重拉一遍
RUN pip install transformers==5.6.0 --no-deps

# 仅通过 pip metadata 确认版本。build 阶段不要 import transformers，
# 它会触发 torch_npu 去 dlopen libascend_hal.so，docker build 容器里
# 没 NPU device 会挂。
RUN python3 -c "from importlib.metadata import version as v; \
    print('transformers', v('transformers'))"
```

构建：

```bash
docker build -t mytrans56:latest .
```

---

## 步骤 2：冒烟测试（smoke test，即最小可运行验证）

### 2a. 基础 import 检查

```bash
docker run --rm -it \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    mytrans56:latest \
    python3 -c '
import torch, torch_npu, transformers
print("transformers:", transformers.__version__)
from transformers import AutoTokenizer, AutoModelForCausalLM

# 最小 tokenizer + 推理，使用 Qwen2-0.5B
model_id = "/root/models/Qwen2-0.5B"   # 按你本地路径改
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).npu()
inputs = tok("Hello, my name is", return_tensors="pt").to("npu")
out = model.generate(**inputs, max_new_tokens=10)
print(tok.decode(out[0]))
'
```

期望输出：能打印出一句完整的续写，比如 `Hello, my name is Sarah and
I am a 20`。

### 2b. 端到端训练验证（可选，需要 2 卡）

用 EasyR1 跑一个两步 GRPO，比对 loss 数值：

```bash
# 消费者侧放松 transformers 上界（如果原来钉在 <5.0.0）
cd /path/to/EasyR1
git checkout ascend-port-transformers-upgrade

# 跑 2-step GRPO on Qwen2-0.5B + math12k
bash examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

期望：日志里 step-1 `entropy_loss` 落在 `[1.21, 1.34]` 区间
（2026-04-23 实测 `1.310156412422657`，和基础镜像原生
transformers 5.3.0.dev0 的数值基本一致）。

---

## 已知不能直接用的部分

**消费者 requirements.txt 里的 transformers 上界。**
EasyR1 / vllm / 你自己的项目可能把 transformers 钉在 `<5.0.0`。
这只是 pip metadata 层面的钉，runtime 本身能接受 5.6。放松即可：

```bash
sed -i 's|transformers>=4.54.0,<5.0.0|transformers>=4.54.0,<6.0.0|' \
    requirements.txt
```

**大版本跳跃（4.x → 5.x）可能改模块路径。**
典型的是 `transformers.integrations.npu_flash_attention` 接口签名
变化、`ALL_ATTENTION_FUNCTIONS` 里新加或删除的 attention key、
`modeling_utils` 的私有 hook。踩到了就在消费者 repo 里加一层适配
（多一个 kwarg 兼容 / 多一个 fallback handler）。

**torch.compile inductor 在 NPU 上会崩。**
和 transformers 版本无关，但升级后仍需要注意：训练脚本里要确保
`use_torch_compile=false`。

**build 阶段不要 import transformers。**
原因同上面 torch 的 doc——会间接触发 torch_npu dlopen 需要 NPU 设备，
docker build 环境里没挂设备会挂。只读 pip metadata 确认版本。

---

## 参考实测记录

2026-04-23 在 A3 上用上述步骤构建镜像 `easyr1-npu-trans56:...`
（27 GB），EasyR1 端到端跑通：

- 基础 smoke（device + pad/unpad 等）PASS
- rollout 推理 PASS（三条 prompt 续写正常）
- 2-step GRPO 训练 step-1 entropy_loss=1.310，step-2 entropy_loss=1.286
  （与基础镜像原生 5.3.0.dev0 的基线一致）
