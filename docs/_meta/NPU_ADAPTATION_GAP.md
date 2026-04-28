# NPU 适配 gap — 下一波 day-0 候选清单

> verl 镜像（`verl-8.5.2-a3` / `easyr1-npu:integrated-20260427`）里**主要软件**按"已 NPU 适配 / 部分适配 / 完全未适配"三档盘点，作为下一波 day-0 / port skill 的目标库。
>
> 当前已被 4 条工具链 + 1 条 integrated overlay 覆盖的 5 项不在本文档；那部分见 [`README.md`](../../README.md) 的当前状态表与 [`UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md)。
>
> 最后更新：2026-04-28。

---

## 档 A — 已覆盖（参照基线，本文档不再讨论）

| 软件 | 入口 |
|---|---|
| torch + torch_npu | `/torch-npu-day0` |
| vllm + vllm-ascend | `/vllm-ascend-day0` |
| transformers（含 NPU FA 集成） | `/transformers-day0` |
| triton-ascend | `/triton-ascend-port` |
| EasyR1（consumer） | v1 / v2 PORT-GUIDE |

---

## 档 B — 镜像里有但**没专门 NPU 适配**

依赖通用 torch / torch_npu 层，今天 EasyR1 跑得通；但场景一变就暴露 gap。

| 软件 | 镜像版本 | NPU 状态 | 风险 / 何时变成下一目标 | 优先级 |
|---|---|---|---|---|
| **HCCL 集合通信** | `hccl 0.1.0`（CANN 内嵌 wheel） | NPU 集合通信原生后端，PyTorch `init_process_group("hccl")` 可用，但与 GPU `nccl` 路径不完全等价（all_reduce、reduce_scatter、all_gather 已 OK；某些 collective 仍 fallback CPU） | 多卡 / 多机 RL 触发；目前 EasyR1 单节点 2-chip 没碰到，**4-chip+ 或多机时压力测试必查** | 中（EasyR1 V1.5+ 触发） |
| **accelerate** | 1.13.0 | 上游已支持 `device_type="npu"`；FSDP plugin 部分路径 NPU 不完整 | 若 EasyR1 切到 `accelerate.Accelerator()` 包装会暴露（目前 verl 走原生 FSDP，没碰到） | 低（不主动用） |
| **torchaudio / torchvision** | 2.11.0+cpu / 0.26.0+cpu | `+cpu` 版本——NPU 专用 op（audio FFT、视觉 decode）走 CPU fallback | VLM / 视频 RL 路径性能 ↘ 或失败 | 中（VLM/video 路线） |
| **tensordict** | 0.10.0 | PyTorch native；NPU dispatch OK；但 fused ops（`tensordict.nn.functional.*`）某些 path 仍走 CPU | RL trajectory 累积型操作量大时是性能瓶颈 | 中（性能优化项） |
| **peft (LoRA)** | 0.19.1 | LoRA 模块继承 `torch.nn.Module`，NPU 路径基本 OK；但 `peft.tuners.lora.layer.LoraLayer.update_layer` 中的部分 dtype 转换需测 | LoRA SFT / RLHF / QLoRA-without-bnb 时触发 | 中（客户 LoRA 场景） |
| **vllm Paged Attention / FA backend 选项** | 在 vllm-ascend 内 | NPU kernel 已实现，但 vllm 的 `attention_backend=FLASH_ATTN_VLLM_V1` / `TRITON_ATTN_VLLM_V1` 选项不全适配 | 长 context（>32k）、speculative decoding、chunked prefill 复杂场景 | 中（高级 inference 场景） |

### 档 B 处理方法

不需要立刻新建 day-0 skill；触发后按需处理：

1. 客户用例进到该软件范畴 → 跑一次 on-A3 smoke 看是否真的有 gap
2. 真有 gap → 在对应已存在的 day-0 skill 框架下加场景验证（如 HCCL 多机进 vllm-ascend day-0；torchaudio/vision 进 transformers / triton-ascend）
3. 不要先创新 skill；先看是否能借既有

---

## 档 C — 完全没装但 RL / 训练常用（**预期 gap**）

GPU 上常用的库，在 NPU 镜像里要么装不上、要么故意不装。这是下一波**真**的 day-0 / port 目标库。

| 软件 | GPU 用途 | NPU 状态 | 替代 / 下一步 | 优先级 |
|---|---|---|---|---|
| **flash-attn** | FA-2 / FA-3 kernel | 装不上（无 NPU build） | **已替代**（FA-2）：transformers `integrations.npu_flash_attention`（CANN FA op）。**FA-3** 客户场景 NPU 上**没有原生实现** | 中（FA-3 长 context 客户触发） |
| **liger-kernel** | Triton-based fused op（RMSNorm / SwiGLU / Cross-Entropy / Embedding / Layer-norm） | 装不上 | 替代：写 CANN ATB 实现 / 用 triton-ascend kernel 重写。**当前没人做**——典型档 2 / 档 3 任务 | **高**（GPU 训练性能优势核心） |
| **deepspeed** | ZeRO / pipeline parallel / MoE 训练 | 上游有 NPU 适配项目（`ds-ascend`），**镜像没装** | 路径：image 里装 ds-ascend；先做兼容性 smoke；再写 day-0 skill 跟进 | **高**（ZeRO-3 是大模型训练刚需） |
| **megatron / mindspeed** | 大模型 model-parallel | mindspeed 是 Huawei megatron NPU 版本；**8.5.0 镜像有，8.5.2 已移除** | 决定题：客户走 vllm-ascend + FSDP 还是回 mindspeed？8.5.2 选了前者；MoE / 大模型 model-parallel 仍是后者优势 | 中（大模型场景） |
| **apex** | NVIDIA fused optimizer / FA | 装不上 | 已替代：torch_npu fused AdamW（功能等价） | 低（已替代） |
| **xformers** | 推理 FA family | 不适用 | NPU 推理路径在 vllm-ascend 内 | 低（不需要） |
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

## 推荐下一波优先级（按"客户/团队会真触发的概率 × 阻断程度"）

1. **deepspeed-NPU 集成验证**（高优）——客户切到 ZeRO-3 / pipeline parallel 时是 blocker；NPU 上游 ds-ascend 项目存在可以接
2. **liger-kernel NPU 实现**（高优，长期）——GPU 训练性能优势核心；当前完全 gap，需要 kernel 团队配合
3. **bitsandbytes-NPU / QLoRA 路径**（高优）——QLoRA 是中下游客户刚需；NPU 完全不支持是商业卡点
4. **mindspeed 复活决策**（中优）——是否 8.5.2 镜像把它移除是对的？MoE 大模型场景需要回它
5. **HCCL 多机 / 多卡压力测试**（中优）——当前只 2-chip 单节点验证；4-chip / 8-chip / 多机扩展时必查
6. **accelerate FSDP NPU 路径回归**（中优）——若 EasyR1 / 客户切到 accelerate 包装会暴露
7. **VLM / video 触发的 torchaudio / torchvision NPU op 适配**（中优）——VLM RL 路线触发

---

## 如何用这份清单

- 客户提需求或 EasyR1 master 引入新依赖 → 先看本表对照"在哪一档"
- 档 B 命中 → 在既有 day-0 skill 里加一次 on-A3 smoke 验证
- 档 C 命中 → 按 §"档 C 处理方法" 3 步评估，再决定是否新建 day-0
- 评估结论 / 实际进展 → 回写本表的 NPU 状态列 + 优先级（带日期）

新发现的档 B / 档 C 项 → 直接追加到对应表，**不要新开文档**。

## 见也

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 整体架构 + 4 上游覆盖位置
- [`UPSTREAM_FORKS.md`](UPSTREAM_FORKS.md) — 当前 4 fork 分支（档 A）
- [`SKILLS-USAGE.md`](SKILLS-USAGE.md) — 当前 5 条 slash command
- [`knowledge/images/verl-8.5.2-a3.md`](../../knowledge/images/verl-8.5.2-a3.md) — 镜像详细 inventory
