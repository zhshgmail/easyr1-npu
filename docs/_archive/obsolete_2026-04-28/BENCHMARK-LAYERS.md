# 分层 Benchmark 标准

**用途**：给每一个上游 module 定义"**port 完成**"的客观判据。这样对客户能讲清楚"我们在这一层做了什么、验证到什么程度"。

**背景**（user 2026-04-24T04:04Z / 04:05Z）：过去两周的问题是"port = smoke 能跑"这种模糊标准。每一层得有**自己层级的 benchmark**，底层不通过不做上层，上层 benchmark 失败时能判断锅在哪一层。

---

## 总体原则

1. **每层只测自己职责的范围**。torch_npu 层不管模型、transformers 层不管 RL、vllm 层不管训练。
2. **下层 benchmark 全过才做上层**。否则上层失败原因混淆。
3. **Benchmark 要有 pass/fail 数字判据**，不是"看起来能跑"。
4. **对比基线明确**：NPU 结果 vs CPU fp32 reference / GPU reference / 旧版 NPU reference 之一。
5. **可复现**：固定 seed、固定 input、固定 model。任何人在新 image 上跑都能复现同样数字。

---

## Layer 1 — torch + torch_npu

**职责**：PyTorch op 在 NPU 上数值正确性 + 基础运行时能力（device 可见、HCCL 可通信、torch.compile 能 trace）。

### L1.1 Op-level numerical correctness

覆盖的 op subset（按 transformer 模型常用程度排序）：

| Op | 形状 | dtype | 验证方式 |
|---|---|---|---|
| `torch.matmul` | (B, N, K) × (K, M), B=1/4/16 | bf16, fp16, fp32 | NPU vs CPU fp32 allclose `rtol=1e-2, atol=1e-3` |
| `torch.nn.functional.linear` | 同上 | 同上 | 同上 |
| `torch.nn.functional.layer_norm` | (B, N, D) | bf16, fp16 | allclose `rtol=1e-2, atol=1e-3` |
| `torch.nn.functional.softmax` | (B, H, N, N) | bf16, fp16 | 同上 |
| `torch.nn.functional.scaled_dot_product_attention` | 标准 causal | bf16 | 同上 |
| `torch_npu.npu_rotary_embedding` | (B, N, H, D) | bf16 | NPU vs cpu reference of (cos, sin) manual 计算 |
| `torch_npu.npu_flash_attention` | 同 sdpa | bf16 | 同上 |
| `torch.nn.functional.cross_entropy` | logits (B, N, V) + labels (B, N) | fp32 | allclose `atol=1e-4` |
| 所有 backward（`autograd.grad`） | 上述 op 的 grad | bf16, fp16 | NPU grad vs CPU grad allclose `rtol=1e-2` |

**Pass 标准**：100% op 通过（任何 op 失败就不能说 torch_npu layer port done）。

### L1.2 Runtime / collective

- `torch.distributed.init_process_group(backend="hccl")` 2 chip init + broadcast + allreduce → 结果 allclose single-chip reference
- `torch.compile(fn, backend="inductor")` + execute → 输出 allclose eager
- `torch.npu.synchronize()` + memory query

**Pass 标准**：3/3 通过。

### L1.3 实现

- `docs/torch-npu/benchmark/` 下一份 `bench_ops.py` 和 `bench_runtime.py`，独立 python script
- 跑法：`docker run <npu-image> python -m docs.torch-npu.benchmark.bench_ops`
- 产出：JSON 格式每个 op 的 pass/fail + 数值 diff
- 客户可展示：一个 markdown 表格 "torch-npu 2.11.0rc1 on A3: 64/64 ops PASS" + 可复现 command

### L1.4 当前状态

**未做**。目前只能 `import torch, torch_npu; x = torch.randn(2,3).npu()` 这种 import smoke。远不够。

---

## Layer 2 — transformers (HF)

**职责**：`AutoModel.from_pretrained` + forward 的数值正确性，在 NPU 下模型加载 + 推理没有 drift vs CPU/GPU reference。

### L2.1 Model loading

覆盖 model list：
- Qwen2-0.5B-Instruct（EasyR1 v1 smoke model）
- Qwen2.5-7B-Instruct（常见生产 size）
- Llama3.1-8B-Instruct（跨家族）

**Pass 标准**：`AutoModel.from_pretrained(path, torch_dtype=bfloat16).to("npu")` 不 OOM 不 crash，weight 和 metadata 完整。

### L2.2 Forward logits correctness

- 固定 input (e.g. `[1, 2, 3, ..., 32]` 的 token ids)
- NPU forward logits → last-token top-5 比对 CPU fp32 reference
- allclose `rtol=0.05` (top-5 ranking 应一致，absolute value drift <5%)

### L2.3 Generation coherence

- `model.generate(greedy, temperature=0, max_new_tokens=32)` 
- prompt = "Hello, my name is"
- NPU output = CPU output 至少 28/32 token match（允许 bf16 尾部微小分叉）
- logprobs KL per step < 0.01

### L2.4 NPU-specific integration

- `src/transformers/integrations/npu_flash_attention.py` 里的 path 被触发（可通过 `attn_implementation="npu_flash_attention"` 切）
- NPU-FA forward 数值 vs eager SDPA allclose

### L2.5 实现

- `docs/transformers/benchmark/` 下 `bench_loading.py` + `bench_forward.py` + `bench_generate.py`
- 同样 docker run 跑，产出 JSON + markdown

### L2.6 当前状态

**未做**。目前 `transformers-5.6.0-day0.md` HowTo 里只写了"`pip install` + 跑个脚本 import 不 crash"。没有数值验证。

---

## Layer 3 — vllm-ascend (on top of torch_npu + transformers)

**职责**：vllm 推理 serving 在 NPU 上的正确性 + 吞吐。

### L3.1 Inference fidelity

- 固定 prompt set：用 **MMLU-subset (100 prompts)** 或 自选 100 个 diverse English prompts
- greedy decode, temperature=0, max_tokens=128
- 对比基线：同 prompt 在 **vllm GPU reference** 或 **vllm CPU reference** 的输出
- **Pass 标准**：
  - ≥99% prompts 整句 token 完全一致
  - 平均每 token top-5 logprob KL divergence < 0.05
  - 不存在 catastrophic divergence（某个 prompt 输出变纯噪声 token）

### L3.2 Throughput

- tokens/sec @ batch 1, 8, 32
- first-token latency @ batch 1
- TTFT / ITL（time to first token / inter-token latency）
- 合格：相比 GPU reference 在合理 ratio 内（e.g. A3 ≥ 30% of H100 throughput）

### L3.3 Continuous batching correctness

- 混合长度 prompt（50–512 tokens）+ 异步 queue
- 对每个 prompt 单独在 GPU reference 上跑，然后对比 vllm-ascend 是否每个 prompt 得到同样 output
- Pass：100% prompts 输出一致

### L3.4 实现

- `docs/vllm-ascend/benchmark/` 下 `bench_fidelity.py` / `bench_throughput.py`
- 数据集 checkin 在 `benchmark/data/` 里
- 产出客户可展示的 `RESULTS.md`

### L3.5 当前状态

**未做**。目前只有我手工的 8-token greedy smoke（`v13_token_diff.py`）对比 iter 20 vs Fix C，不是 prompt set，不是 throughput，不是客户能复用的东西。

---

## Layer 4 — RL 消费者 (EasyR1, 或其他 RL 框架)

**职责**：GRPO / PPO 算法在 NPU 上真收敛 + 训练指标和 GPU reference 一致。

### L4.1 短跑收敛指标

- Qwen2-0.5B + math12k, GRPO, max_steps=10
- **Pass 标准**：
  - Step 1 entropy_loss 落在 `[1.21, 1.34]`（v2 image baseline band）
  - Step 10 reward 较 step 1 有显著上升
  - 无 NaN / Inf

### L4.2 长跑验证

- 同 recipe, max_steps=200
- **Pass 标准**：最终 reward ≥ GPU reference baseline 的 90%

### L4.3 实现

- `docs/easyr1/benchmark/` 下 `bench_grpo_short.py` + `bench_grpo_long.py`

### L4.4 当前状态

**做了一部分**：V1.4 短跑跑通过（Fix C + 旧 entropy_loss=1.275）。长跑没做过。iter 20 上重跑 V1.4 entropy_loss=3.213（不在 band）—— 但已知根因在 Layer 3 以下没真做 benchmark，先不混淆这层。

---

## 总览表（当前所有层状态）

| Layer | 做了什么 | Benchmark PASS 比例 | 可给客户看 |
|---|---|---|---|
| L1 torch_npu 2.11.0rc1 | 装 Ascend 预编译 wheel、一个 import smoke | 0/64（没跑 op suite） | ❌ |
| L2 transformers 5.3 / 5.6 | `pip install --no-deps` + 模型加载 smoke | 未定义 | ❌ |
| L3 vllm-ascend 0.17 + 我的 iter 20 patches | 18 iter drift patches + ABI guard fix + `do_kv_cache_update` | 1 prompt / 8 tokens greedy bit-exact vs Fix C | 🟡（单点证据不是 suite） |
| L4 EasyR1 GRPO | V1.3 推理 PASS + V1.4 entropy 旧 PASS | 混了其他层的问题 | ❌ |

---

## 下一步（按层顺序）

1. **Layer 1 benchmark 实现**：写 `bench_ops.py`，跑 24 个核心 op，产出 JSON + markdown report。目标交付给客户一份 `L1_benchmark_report.md`，里面有数字。
2. **Layer 1 benchmark 跑出结果**：在 iter 20 image（torch 2.11.0rc1 + torch_npu 2.11.0rc1）上跑，看哪些 op 过哪些不过
3. 根据 L1 结果决定：
   - 如果 100% 过 → L1 层 **可以**对外声称 "torch_npu 2.11.0rc1 on A3 verified"
   - 如果有 op fail → 在 `zhengshencn_hwca/pytorch` fork 上 (如果能 fork，等 user 确认 gitcode fork 路径) 的 `torch-2.11.0_auto_porting` branch 修
4. 重复到 L2、L3、L4

**每层完成标准**：`L<N>_benchmark_report.md` 在对应 `docs/<upstream>/benchmark/` 下，客户打开能看到数字。

**当前卡点**：我建议立刻实施 L1 benchmark。实现 + 在 iter 20 image 跑一次大约 1 小时。
