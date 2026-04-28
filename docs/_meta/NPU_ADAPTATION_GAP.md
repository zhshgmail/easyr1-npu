# NPU 适配 gap — 下一波 day-0 候选清单

> verl 镜像（`verl-8.5.2-a3` / `easyr1-npu:integrated-20260427`）里**主要软件**按"已 NPU 适配 / 部分适配 / 完全未适配"三档盘点，作为下一波 day-0 / port skill 的目标库。
>
> 当前已被 4 条工具链 + 1 条 integrated overlay 覆盖的 5 项不在本文档；那部分见 [`README.md`](../../README.md) 的当前状态表与 [`UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md)。
>
> 最后更新：2026-04-28（含 on-A3 实测 probe）。

---

## 0. 镜像里有 ≠ "不需要适配"——必须实测分类

第一直觉："镜像里既然装了，多半通过 torch_npu 透明支持了"。**这是对的，但只对一半**——必须用 on-A3 probe 区分三种情况：

- **真透明 NPU 支持**：`.to("npu:0")` + 算子调用全程 NPU-resident，无 fallback warning
- **静默 CPU fallback**：返回 tensor `device=npu:0` 看似在 NPU，但 PyTorch 后端实际把这个 op fallback 到 CPU 跑，再把结果 copy 回 NPU。**性能完全是 CPU 级**，看 device 标签会被骗
- **直接报错 / 不可用**：import 即失败 / 调用即抛异常

判定方法（在 NPU 容器内）：

```bash
# 1) 直接调一次 op
python3 -c "import torch, torch_npu; <call op on .to('npu:0') tensor>"

# 2) 看 stderr 有没有这种 warning（决定性证据）
W428 ... VariableFallbackKernel.cpp:250] Warning: CAUTION:
  The operator 'X::Y' is not currently supported on the NPU backend
  and will fall back to run on the CPU.
```

**有 fallback warning = 静默 CPU fallback，是真 gap**。无 warning + 算子吃满 NPU AICore = 真透明支持。

本文档下面所有"NPU 状态"列都是基于 2026-04-28 在 `easyr1-npu:integrated-20260427` 上实测的结果。

---

## 1. 档 A — 已覆盖（参照基线，本文档不再讨论）

| 软件 | 入口 |
|---|---|
| torch + torch_npu | `/torch-npu-day0` |
| vllm + vllm-ascend | `/vllm-ascend-day0` |
| transformers（含 NPU FA 集成） | `/transformers-day0` |
| triton-ascend | `/triton-ascend-port` |
| EasyR1（consumer） | v1 / v2 PORT-GUIDE |

---

## 2. 档 A' — 镜像里**通过 torch_npu 透明支持**，无需单独 day-0

实测 PASS：调用 op 无 CPU-fallback warning，返回 `device=npu:0`，算子在 NPU 跑。这些**不是 gap**——继续保持现状即可，**不需要新建 day-0 skill**。

| 软件 | 镜像版本 | 实测结论 | 证据 |
|---|---|---|---|
| **HCCL 集合通信** | `hccl 0.1.0` | ✓ 真支持 | `torch.distributed.is_hccl_available()=True`；`is_backend_available("hccl")=True` |
| **accelerate** | 1.13.0 | ✓ 真支持 NPU device_type | `accelerate.utils.is_npu_available()=True`（上游 1.0+ 已集成 NPU） |
| **tensordict** | 0.10.0 | ✓ 真支持 | `TensorDict.to("npu:0")` + 后续 `*2` 算子全程 `npu:0` |
| **torchaudio** | 2.11.0+cpu | ✓ FFT 路径在 NPU 跑 | `torchaudio.functional.spectrogram(input.npu(), window.npu(), n_fft=512)` 返回 `device=npu:0`，无 fallback warning |

**结论**：用户判断正确——这 4 个**镜像里有就是因为 torch_npu 已透明支持**，不是垃圾包。**这些从档 B 移除**。

---

## 3. 档 B' — 镜像里有，但**特定 op 静默 CPU fallback**（真 gap）

实测有 `npu_cpu_fallback` warning，性能等同 CPU 跑：

| 软件 | NPU 状态 | 已确认 fallback 的 op | 风险 / 何时阻断 |
|---|---|---|---|
| **torchvision** 2.11/0.26 | ⚠ 部分 op fallback CPU | `torchvision::roi_align`、`torchvision::nms`（实测 warning 命中）；可能还有 `roi_pool`、`deform_conv2d`、`ps_roi_align` | VLM 里 detection head / 视觉 RL（YOLO / DETR 类）触发，性能 ↘↘↘；对 LLM-only 路线无影响 |
| **peft** 0.19.1（待重测） | ❓ Module 加载 OK，初始 forward 测试因 peft 0.19 API 变化未跑通 | （未确认） | LoRA SFT / RLHF / QLoRA-without-bnb 时若有 fallback op 会暴露 |
| **vllm Paged Attention / 高级 backend** | ❓ NPU kernel 已实现基础路径，`FLASH_ATTN_VLLM_V1` / `TRITON_ATTN_VLLM_V1` / chunked prefill 边角未充分实测 | 长 context >32k、speculative decoding、chunked prefill | 高级 inference 场景（V1.5+） |

### 档 B' 处理方法

1. **客户用例确认走到该 op** → 在容器里跑一次 probe 看是否有 `npu_cpu_fallback` warning
2. 有 warning + 性能不可接受 → 进入档 C 处理流程（写 NPU 实现）
3. 无 warning → 加进档 A'

不需要立刻建 day-0 skill；**触发后**按场景验证 + 加测进对应 day-0 skill。

---

## 4. 档 C — 完全没装但 RL / 训练常用（**预期 gap**）

GPU 上常用，NPU 镜像里要么装不上、要么故意不装。这是下一波**真**的 day-0 / port 目标库。

| 软件 | GPU 用途 | NPU 状态 | 替代 / 下一步 | 优先级 |
|---|---|---|---|---|
| **flash-attn FA-2** | 标准 FA | 装不上（无 NPU build） | 已替代：transformers `integrations.npu_flash_attention`（CANN FA op）✓ | 低（已替代） |
| **flash-attn FA-3** | 长 context 优化 FA | NPU 上**无原生实现** | 客户走长 context 触发；需 NPU FA-3 kernel | 中 |
| **liger-kernel** | Triton-based fused op（RMSNorm / SwiGLU / Cross-Entropy / Embedding / LayerNorm） | 装不上 | 写 CANN ATB 实现 / 用 triton-ascend kernel 重写。**当前没人做**——典型档 2 / 档 3 任务 | **高**（GPU 训练性能优势核心） |
| **deepspeed** | ZeRO / pipeline parallel / MoE 训练 | 上游有 NPU 适配项目（`ds-ascend`），**镜像没装** | 路径：image 里装 ds-ascend；先做兼容性 smoke；再写 day-0 skill 跟进 | **高**（ZeRO-3 是大模型训练刚需） |
| **megatron / mindspeed** | 大模型 model-parallel | mindspeed 是 Huawei megatron NPU 版本；**8.5.0 镜像有，8.5.2 已移除** | 决定题：客户走 vllm-ascend + FSDP 还是回 mindspeed？8.5.2 选了前者；MoE / 大模型 model-parallel 仍是后者优势 | 中（大模型场景） |
| **apex** | NVIDIA fused optimizer / FA | 装不上 | 已替代：torch_npu fused AdamW（功能等价）✓ | 低（已替代） |
| **xformers** | 推理 FA family | 不适用 | NPU 推理路径在 vllm-ascend 内 ✓ | 低（不需要） |
| **bitsandbytes** | 8-bit / 4-bit 量化训练 + QLoRA | **完全无 NPU 支持** | 路径：要么 port bitsandbytes-NPU，要么用 torch_npu native int8/int4 + 自写 LoRA | **高**（QLoRA 是行业刚需） |
| **torchao** | Inductor-based 量化 | 依赖 torch._inductor NPU codegen | torch_npu 有 inductor 但覆盖不全；专项任务 | 中（量化训练路线） |
| **TensorRT / torch_tensorrt / onnxruntime** | 推理加速 | NPU 用 ATB / vllm-ascend 替代生态 | 不是同生态，本仓不追 | 低（生态切换） |
| **sglang / TGI** | 替代 inference engine | 完全没装 | 客户问 sglang on NPU → 当前 N/A，是新 day-0 目标 | 低（vllm-ascend 已盖） |
| **trl / axolotl / unsloth** | 上层 RL / SFT 框架 | 完全没装 | EasyR1 / verl 已 fill 这块 | 低（已替代） |

### 档 C 处理方法

每个档 C 项要变成下一波 day-0 / port，**先做 3 步评估**再决定开 skill：

1. **Demand probe**：是否有客户用例 / EasyR1 master 已 import？没需求 → 不做
2. **Existence probe**：有没有现成 NPU 上游项目（如 ds-ascend、bitsandbytes-NPU PR、liger-NPU fork）？有 → 走 day-0 模式追上游；没有 → 进入 NPU 适配三档：
   - **档 1**（本仓直接做）：Python 层 shim / 写 wrapper
   - **档 2**（委托姐妹项目）：kernel 实现 → `a5_ops` / kernel 仓
   - **档 3**（只能提需求）：CANN runtime 层 → 提 issue 给 Ascend
3. **Cost probe**：估开发周期。Kernel 级（liger / bnb）周期月级；Python 层（deepspeed wrapper）周级

---

## 5. 推荐下一波优先级（按"客户/团队会真触发的概率 × 阻断程度"）

1. **deepspeed-NPU 集成验证**（高优）——客户切到 ZeRO-3 / pipeline parallel 时是 blocker；NPU 上游 ds-ascend 项目存在可以接
2. **liger-kernel NPU 实现**（高优，长期）——GPU 训练性能优势核心；当前完全 gap，需要 kernel 团队配合
3. **bitsandbytes-NPU / QLoRA 路径**（高优）——QLoRA 是中下游客户刚需；NPU 完全不支持是商业卡点
4. **mindspeed 复活决策**（中优）——是否 8.5.2 镜像把它移除是对的？MoE 大模型场景需要回它
5. **torchvision NPU op 补齐**（中优，VLM 路线触发）——`nms` / `roi_align` 已确认 CPU fallback；做 detection head 的 RL 是 blocker
6. **HCCL 多机压力测试**（中优，规模触发）——单节点 2-chip OK；4-chip / 8-chip / 多机扩展时压测必查
7. **peft NPU 路径完整 smoke**（中优）——LoRA SFT / RLHF 触发；目前测试因 peft 0.19 API 变化未跑通

---

## 6. 如何用这份清单

- 客户提需求或 EasyR1 master 引入新依赖 → 先看本表对照"在哪一档"
- 档 A' 命中 → 不做新 skill；继续保持
- 档 B' 命中 → 在既有 day-0 skill 里加一次 on-A3 probe 看是否真 fallback；若是再决定档 C 化
- 档 C 命中 → 按 §"档 C 处理方法" 3 步评估，再决定是否新建 day-0
- 评估结论 / 实际进展 → 回写本表的 NPU 状态列 + 优先级（带日期）

新发现的档 B' / 档 C 项 → 直接追加到对应表，**不要新开文档**。

## 见也

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 整体架构 + 4 上游覆盖位置
- [`UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md) — 当前 4 fork 分支（档 A）
- [`SKILLS-USAGE.md`](SKILLS-USAGE.md) — 当前 5 条 slash command
- [`knowledge/images/verl-8.5.2-a3.md`](../../knowledge/images/verl-8.5.2-a3.md) — 镜像详细 inventory
