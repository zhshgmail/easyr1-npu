# DeepSeek-V4 / GLM-5 真参数全栈承载分析(Ascend A3)

> 维护者笔记:这份文档跟着实际 patch 实测同步更新,**不写未验证的猜测**。每条结论后面带"测试方法 / 实测结果 / 源材料"。

## 目标

让 miles 在 Ascend A3 上跑通 **真 DSv4 / GLM-5 参数**(非减层、非缩 hidden、非凭空假设的 config)。当前 fork 还没到这一步;这份文档记录:

1. 真 DSv4 / GLM-5 的算子层 shape 是什么(从 HF config.json 直接读出来,不靠记忆)
2. 我们的 4 个 NPU tilelang 算子在真 shape 下能不能跑(独立测,不混在模型里)
3. 每个不能跑的算子,精确的失败原因 + 在编译器哪一层修
4. miles 周边组件(Megatron / TE / Apex / sglang)在 NPU 上需要什么 shim、shim 的代价、什么是 workaround、什么是真正的修复
5. Roadmap:把每条都落到一个具体的 patch/commit/PR

## 1. 真 DSv4 / GLM-5 算子层 shape

### 数据来源(全部从 HF `config.json` 直读)

* DeepSeek-V3.2-Exp: <https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json>
* DeepSeek-V4-Flash: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json>

GLM-5 (Z AI) 没公开 config.json,但官方文档说架构是 DSv3.2 同族,所以下面用 DSv3.2 + DSv4-Flash 的字段直接代表 GLM-5。

### 关键字段对照

| 字段 | 含义 | DSv3.2-Exp | DSv4-Flash |
|---|---|---|---|
| `hidden_size` | 模型 hidden 维度 | 7168 | 4096 |
| `num_attention_heads` | MLA Q-head 数 | 128 | 64 |
| `num_hidden_layers` | 总层数 | 61 | 43 |
| `q_lora_rank` | Q LoRA 投影 rank | 1536 | 1024 |
| `kv_lora_rank` | KV LoRA 投影 rank | 512 | (隐式 512;见下) |
| `qk_nope_head_dim` | QK 非 RoPE 部分每 head 维度 | 128 | (n/a, 见下) |
| `qk_rope_head_dim` | QK RoPE 部分每 head 维度 | 64 | 64 |
| `head_dim` | V head 维度 | (= `qk_nope_head_dim` 即 128) | **512** |
| `v_head_dim` | V head 维度 | 128 | (= `head_dim`) |
| `index_head_dim` | lighting indexer head 维度 | 128 | 128 |
| `index_n_heads` | lighting indexer head 数 | 64 | 64 |
| `index_topk` | sparse 选 topk | 2048 | 512 |
| `vocab_size` | tokenizer vocab | 129280 | 129280 |
| `torch_dtype` | base 权重 dtype | bfloat16 | bfloat16 |
| `quantization_config` | inference 量化 | fp8 dynamic | fp4+fp8 mixed (DSv4) |

### sparse_mla 算子层最关键的一个 shape:`dim_plus_tail_dim`

miles 在 `tilelang_sparse_mla_fwd.py:185` 写死:
```python
assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
```

`576` 怎么来:在 GLM-5 / DSv4 的 LoRA-absorb 模式下,**absorbed q 的 last dim** = `kv_lora_rank + qk_pos_emb_head_dim`。`512 + 64 = 576`。
也是 DSv4-Flash 直观看的 `head_dim + qk_rope_head_dim = 512 + 64 = 576`。

### 因此 NPU 算子层"真 shape"应该是

```
H_MLA            = 64          (DSv4-Flash; DSv3.2 是 128)
D_V              = 512         (sparse_mla output 每 head 维度)
D_TAIL           = 64
DQK = D_V + D_TAIL = 576       (sparse_mla Q/K 每 head 总维度;miles 写死)
kv_lora_rank     = 512

H_INDEXER        = 64          (lighting indexer head 数)
D_INDEXER        = 128         (lighting indexer head 维度)
index_topk       = 512 (DSv4-Flash) / 2048 (DSv3.2)
SKV typical      = 2048 ~ 16384  (训练 context 长度,index 选 topk 跨这么大窗口)
```

## 2. 4 个 NPU tilelang 算子在真 shape 下的实测结果

测试 driver: `miles_plugins/models/glm5/ops/_npu/_real_shape_smoke.py`(commit `<pending>` on `zhshgmail/miles@npu-tilelang-dispatch`)

测试入口:`docker exec tlrescue` 通过 PYTHONPATH 加载 `miles_plugins.models.glm5.ops._npu`(我们的 NPU dispatch 包),分别调用 `npu_indexer_fwd_interface / npu_indexer_bwd_interface / npu_sparse_mla_fwd_interface / npu_sparse_mla_bwd`,SEQ=1 单 query,SKV / topk 用真 DSv3.2 值。

| 算子 | 输入 shape | 编译 + 运行 | 备注 |
|---|---|---|---|
| `lighting_indexer_fwd` | q=[1, 64, 128] bf16; k=[2048, 128] bf16; w=[1, 64] fp32; topk=512 | ✅ PASS 2.5s | logits finite max≈60.9 |
| `lighting_indexer_bwd` | 同上 + grad_scores=[1, 512] fp32 | ✅ PASS 2.3s | dq / dw / dk 全 finite |
| `sparse_mla_fwd` | q=[1, 64, 576] fp16; kv=[2048, 1, 576] fp16; topk=512 | ❌ FAIL | bishengir "ub overflow, 2494464 > 1572864 bits" |
| `sparse_mla_bwd` | 同上 + dO=[1, 64, 512] fp16 | ❌ FAIL | 同上(走 fwd 先就死) |

### sparse_mla_fwd 真 shape 失败的精确量化(由我们 PR #80 的 `CheckUBBudget` 算出)

```
total requested:  403968 B
A3 UB capacity:   196608 B  (192 KB on dav-c220 / 910B)
soft budget:      157286 B  (= 80% of UB, 留给 sync flags / pipeline)

Per-allocation breakdown (largest first):
    131072 B  acc_o          [block_M=64, D_V=512]    fp32    ← 单一最大,占总数 1/3
     65536 B  KV_shared      [block_N=64, D_V=512]    fp16
     65536 B  O_cast         [block_M=64, D_V=512]    fp16
     65536 B  Q_shared       [block_M=64, D_V=512]    fp16
     16384 B  tmp            [block_M=64, block_N=64] fp32
     16384 B  scores         [block_M=64, block_N=64] fp32
     16384 B  scales         [block_M=64, block_N=64] fp32
      8192 B  scores_cast    [block_M=64, block_N=64] fp16
      8192 B  K_tail_shared  [block_N=64, D_TAIL=64]  fp16
      8192 B  Q_tail_shared  [block_M=64, D_TAIL=64]  fp16
      ... (10 个 < 256 B 的小 fragment 累计 ~2.5 KB)
```

**根因**:`block_M=64`(我们 kernel 写死 `block_M=heads`)× 三大 buffer (`acc_o`, `KV_shared`, `Q_shared`)各 ≥ 64 KB,直接撑爆 192 KB UB。

### lighting_indexer 真 shape 为什么不爆

`lighting_indexer_fwd` 内部最大 fragment 是 `scores [block_N=64, block_Q*heads=8*64=512] fp32` = 128 KB,但 indexer kernel 不留 `acc_o [64, 512]` 这种持续累加的大 buffer(只有滑窗 reduce 到 `logits [block_N, block_Q] = 8 KB`),所以总 UB 占用 ~160 KB,**刚好在 192 KB 之内**。

## 3. 编译器层修复路径

### 编译器栈分层

```
tilelang Python DSL
    │ T.alloc_fragment([block_M, D], dtype)
    ▼
TIR 中间层(tilelang/engine/phase.py)
    │ LowerAndLegalize → OptimizeForTarget → SplitHostDevice
    ▼
NPUIR codegen(src/target/codegen_npuir_dev.cc)
    │ AllocateNode → mlir::tensor::empty()
    ▼
MLIR(hivm dialect,在 AscendNPU-IR 子模块)
    │ bishengir-compile pipeline
    │   ├─ HIVMLoweringPipeline
    │   ├─ HIVMTilingPass        ← "ub overflow" 在这里报
    │   └─ HIVMBufferAllocation
    ▼
bisheng clang (闭源,只在 CANN 安装包里)
    │ AICore .o
    ▼
NPU runtime
```

### 我们能改的层 vs 不能改的层

| 层 | 可改吗 | 文件 / repo |
|---|---|---|
| tilelang Python DSL | ✅ 直接改 | `tilelang-mlir-ascend/tilelang/` |
| TIR pass(Python or C++)| ✅ 直接改 | `tilelang-mlir-ascend/tilelang/transform/`, `src/transform/` |
| NPUIR codegen | ✅ 直接改 | `tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc` |
| HIVM dialect / pipeline | ✅ 改(在 AscendNPU-IR 子模块,源码开放)| `3rdparty/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/` |
| bisheng clang (AICore 后端)| ❌ 闭源 | 只能向 Huawei 报 issue |

### 修复方案(按已实施 / 进行中 / 待办分类)

#### A. CheckUBBudget 早期诊断 pass ✅ 已实施(PR #80)

* 文件:`tilelang/transform/check_ub_budget.py`
* 位置:`tilelang/engine/phase.py` NPUIR pipeline 在 `LowerOpaqueBlock` 之后
* 行为:遍历每个 `AllocateNode`,sum bytes,与 `tilelang.utils.npu_arch.CHIP_SPECS[chip]["UB"]` 比较,超 80% 软预算就 raise RuntimeError 并给出 per-allocation breakdown + 建议 block_M
* 不改 IR,纯诊断;可通过 `tl.disable_ub_budget_check=True` opt-out
* 单测 5/5 PASS,3 个 known-good 小 shape kernel 无误报,真 DSv4 sparse_mla 触发清晰诊断
* PR: <https://github.com/tile-ai/tilelang-mlir-ascend/pull/80>

#### B. AutoTileFragmentBlockM rewrite pass 🚧 进行中(下一个 PR)

* 目标:让 sparse_mla_fwd / sparse_mla_bwd 在 H=64 真 DSv4 shape 下能编译
* 思路:不指望 bishengir 内部 auto-fission;tilelang 这层把 `block_M=64` 拆成 `block_M_inner=16` × 4 个 inner tile,outer-product 维度 grid block 数从 1 个变成 4 个,UB 用量减半再减半
* 不能简单缩 `block_M` 参数 — kernel 用户写的 `T.alloc_fragment([block_M, D], dtype)` 是 outer block_M;我们要在 TIR 层做 split
* 入口:在 `phase.py` 中 `CheckUBBudget` 失败前,先跑一遍 `AutoTileFragmentBlockM`(detects overflow, rewrites alloc + outer loop, re-checks)
* 状态:设计中

#### C. (可选)C++ port of CheckUBBudget

如果 upstream maintainer 想 in-tree,把 Python pass 翻成 C++ TIR pass。诊断 pass 小且无副作用,翻起来不难。等 PR #80 reviewer 反应再决定。

#### D. (可选)bishengir HIVM 层的 auto-fission

最深层修复:让 `HIVMTilingPass` 自己识别 oversized fragment 并自动 spill / split。这是 AscendNPU-IR 上游的 PR,需要熟悉 HIVM dialect,工作量 ~1 week。暂搁。

## 4. miles 周边组件适配分析

测试时我让 miles 的 `DSAMLASelfAttention` 真正在 Megatron-core 引擎里 init + forward。下面是所有遇到的 blocker + 修复方法 + 这个修复是"workaround"还是"真适配"。

### 4.1 Megatron-core 在 NPU 上的 CUDA 假设

| 站点 | 现状 | 我的做法 | 类型 |
|---|---|---|---|
| `torch.cuda.current_device()` 在 `ColumnParallelLinear._initialize_affine_weight_gpu` 等多处 | 假设 CUDA 已 init | monkey-patch `torch.cuda.current_device = torch.npu.current_device` | **workaround**,但是行业标准做法(MindSpeed 也这么干);上游 Megatron 接受 NPU PR 之前必须这么干 |
| `torch.cuda.get_rng_state` / `set_rng_state` / `manual_seed` 等 RNG 系列 | 同上,torch_npu 有对应的但不同 signature | monkey-patch 全套 | **workaround** |
| `torch.cuda.Stream` / `current_stream` / `default_stream` | 同上 | monkey-patch | **workaround** |
| `torch.cuda.random.get_rng_state` (submodule path) | Megatron 有时走这条路径 | 也要 monkey-patch 这条 | **workaround** |

> 这套 shim **是行业标准** — `Ascend/MindSpeed` 内部就是这么干的(他们把 shim 打包成 `mindspeed.patch_utils.aspm`)。我们这版是直接的 11 行 shim,跟 MindSpeed 等价但更轻。如果以后 install MindSpeed,这套 shim 自动被 MindSpeed 覆盖,无冲突。

### 4.2 TransformerEngine 缺失(Ascend 上没有 TE)

| 站点 | 现状 | 我的做法 | 类型 |
|---|---|---|---|
| `TELinear` / `TEColumnParallelLinear` 在 miles glm5.py | CUDA-only NVIDIA TransformerEngine | miles 自己已经支持把这些换成 `ColumnParallelLinear` / `RowParallelLinear` | **真适配** — miles 本身有 fallback path |
| `ColumnParallelLayerNormLinear`(LayerNorm 融合 Linear)| TE-only | 我写了 `_LayerNormColumnParallelLinear` shim,基于 `ColumnParallelLinear` + 学习版 RMSNorm scale + `.layer_norm_weight` 属性 | **介于 workaround 和真适配**:数学等价,但缺 TE 的 LN+Linear fused kernel,慢一些 |
| `parallel_mode="duplicated"` / `skip_weight_param_allocation` kwargs 在 TELinear | TE-specific | `IndexerColumnParallelLinear` shim 把这些 kwargs strip 掉 | **workaround**(应该向 miles 上游提 PR 让 glm5.py 在非 TE backend 不传这些 kwargs) |
| `FusedLayerNorm` 在 Apex | Apex CUDA-only | 用 `torch.nn.LayerNorm` 替代 | **workaround**,丢 fused kernel 性能 |

### 4.3 Apex(NVIDIA fused-ops library)缺失

| 站点 | 现状 | 我的做法 | 类型 |
|---|---|---|---|
| `apex.transformer.functional.fused_apply_rotary_pos_emb_thd` 在 glm5.py:484 | Apex CUDA-only | `sys.modules` 注入 stub,内部用 pure-torch impl(per-token 位置从 cu_seqlens 反推) | **workaround**;真适配是 torch_npu 提供 fused rotary,或者 MindSpeed 自带 NPU 版 |

### 4.4 Megatron MoE 路径上的 NameError(`te_general_gemm`)

| 站点 | 现状 | 我的做法 | 类型 |
|---|---|---|---|
| `megatron.core.transformer.moe.moe_utils.RouterGatingLinearFunction` 引用 `te_general_gemm` 但在 ImportError 路径下没定义 | Megatron 0.16 的 bug(本意是有 guard `if te_general_gemm is not None` 但 NameError 在 guard 之前就抛了)| 给 module 注入 `te_general_gemm = None` placeholder | **真修复**(Megatron 上游也需要这个 patch,可以提到 NVIDIA/Megatron-LM) |

### 4.5 sglang on NPU 是坏的

| 站点 | 现状 | 处理 | 类型 |
|---|---|---|---|
| sglang glm5-poc image (`lmsysorg/sglang:cann8.5.0-a3-glm5`)| `triton==3.6.0` + `triton-ascend==3.2.0` ABI mismatch,sglang scheduler 在 first triton-kernel JIT 时 crash | 完全不走 sglang,用 mock advantages 跑 training loop | upstream block;真适配等 `Ascend/triton-ascend` 出 `release/3.6.x` wheel 或者重新 pin image |

我提的 issue:<https://github.com/triton-lang/triton-ascend/issues/277> (cross-ref #234)

### 4.6 miles 自己的 SparseMLA contract

| 站点 | 现状 | 我的做法 | 类型 |
|---|---|---|---|
| `tilelang_sparse_mla_fwd.py:185` 写死 `assert dim_plus_tail_dim == 576` | miles 假设永远 GLM-5 / DSv4 shape | NPU dispatch 不走 miles 这层 assertion,直接路由到我们的 `npu_sparse_mla_fwd_interface`,把 d_v 从 `o.shape[-1]` 推出来 | **真适配**(我们的 NPU 算子比 GPU 版本更通用) |
| `tilelang_indexer_fwd.py` 里 `tilelang.PassConfigKey.TL_ENABLE_FAST_MATH` 在 module-load 时引用,NPU 上没这个属性 | miles GPU 文件假设 NVIDIA tilelang | NPU 路径上 lazy-load:把 `from .tilelang_*_{fwd,bwd} import ...` 移到 `*_interface` 函数体里,只在非 NPU 路径才 import | **真适配** — 应该向 miles 上游提 PR |

## 5. Roadmap 到"真参数全栈承载"

(每条 row 后面带 owner 和 status)

| # | 任务 | 描述 | Owner | Status |
|---|---|---|---|---|
| 1 | UB-budget 早期诊断 | 把 bishengir 的 opaque error 提前到 tilelang TIR pass 报清楚 | self | ✅ PR #80 |
| 2 | AutoTileFragmentBlockM | tilelang 层自动 split block_M,让 sparse_mla 在 H=64 编译 | self | 🚧 in progress |
| 3 | sparse_mla 真 shape e2e | 验证 #2 之后 sparse_mla_fwd / bwd 在真 shape 跑通 | self | 待 #2 完成 |
| 4 | miles upstream PR: 在 ops/{indexer,sparse_mla}.py lazy-load GPU tilelang | 移除 module-scope CUDA-only import | self | 待 #2 完成后一起提 |
| 5 | miles upstream PR: `glm5.py` 在非 TE backend 不传 `parallel_mode="duplicated"` | 移除 IndexerColumnParallelLinear shim 的必要性 | self | 低优,reviewer 反馈后再说 |
| 6 | Megatron-LM upstream PR: `te_general_gemm = None` placeholder | 修复 NameError-on-no-TE bug | upstream | 提小 issue 即可 |
| 7 | torch_npu fused rotary | 让 apex 那条路径在 NPU 上有 fused 版本 | Huawei | 等 |
| 8 | triton-ascend release/3.6.x wheel | unblock sglang | Huawei | 已开 issue #277 |
| 9 | sgl-kernel-npu / vllm-ascend inference path | rollout 侧 | 上游 | 跟踪 |
| 10 | end-to-end DSv4 真权重 1 iter training | 装权重 mock,跑 1 个完整 GRPO step | self | 等 #1-#3 完成 |

## 6. 分析方法论(给后来人 / 自己复盘)

写这份文档过程中我用的分析方法:

1. **永远从真 HF config.json 读 shape**,不要从记忆或者论文摘要里抓数字。具体 URL 已在 §1 列出。
2. **算子单独测,不混在模型里**:`_real_shape_smoke.py` 不走 miles glm5 layer,直接调 `_npu/` 子包里的算子,一个 fail 就能精确指出是 lighting_indexer 还是 sparse_mla。
3. **拿到失败时的精确错误后,先找触发栈**:bishengir 报 "ub overflow" 看上去是 runtime,但顺着 tilelang `jit_npu.py:1178 → engine/lower.py:276 → device_codegen.py:180` 可以追到我们能控制的 layer(TIR pass)。
4. **量化失败原因**(bytes 量化优于"看上去很大"):写 `_collect_ub_allocs` 把所有 fragment 列出来 + bytes,这样建议"`block_M <= 32`"才有依据。
5. **shim vs 真适配**严格区分:不要把所有 monkey-patch 都叫 "适配",凡是丢 fused kernel / 改 numerics / 长期不可持续的都列为 workaround,roadmap 里指出真适配路径。
6. **真适配是上游 PR**,不是 fork 私改:#1 / #2 / #4 / #5 都目标提到 `tile-ai/tilelang-mlir-ascend` 或 `radixark/miles`,#6 提到 `NVIDIA/Megatron-LM`,这样这些改不需要 fork 长期维护。

## 7. 进度更新区

(每完成一个 roadmap row,在这里追加一段 dated note)

* **2026-05-28 早**:CheckUBBudget pass 完成,PR #80 开。
* **2026-05-28 早**:确认 lighting_indexer fwd/bwd 真 shape 跑通;sparse_mla fwd/bwd 真 shape 失败原因 quantified 到 `acc_o [64, 512] fp32 = 131072 B` 单一最大。
* **2026-05-28 中**:**block_M_inner 头维度切分原型**(commit `4983264` on miles fork branch `npu-tilelang-dispatch`):
  - `sparse_mla_fwd_kernel.py` + `sparse_mla_bwd_kernel.py` 加 `block_M_inner` / `block_H_inner` 参数,grid 从 `batch*seq_len` 变成 `batch*seq_len*head_groups`,每个 NPU block 处理 `block_M_inner=16` 个 head 而不是全部 H=64
  - `sparse_mla.py` dispatcher 在 `H % 16 == 0 and H > 16` 时自动选 `block_M_inner=16`
  - **实测结果**:
    * `sparse_mla_fwd` 真 DSv4 shape (H=64) **编译成功**(CheckUBBudget 通过,bishengir 通过)
    * **但输出有 NaN** — 每个 head-group 只有约 1/8 行 finite,模式像 UB 跨 block 复用未清零。在 H=32 (head_groups=2) 没问题,只 H=64 (head_groups=4) 出错
    * `sparse_mla_bwd` 切分后**仍然 UB 溢出**(289 KB 超过 192 KB):bwd 状态比 fwd 多(`acc_dq` / `acc_dq_tail` / `acc_dkv` / `acc_dkv_tail` 同时活),需要进一步把 `block_size=32` 也缩小
    * `lighting_indexer_bwd` 真 DSv4 shape 也 UB 溢出(259 KB),之前的 PASS 是 cache 命中误判。需要同样的 head-split 处理
  - 已分析但未提 PR:还在调 NaN,需要排查"是 init bug 还是 NPU 同步问题"。Roadmap #2 状态从 in-progress 改成 partial。

### 未解决的失败模式(供下次接手)

1. **H=64 head-split NaN**:head_groups=2 (block_M_inner=32) 也超 UB 装不下;head_groups=4 (block_M_inner=16) 装下但 NaN。可能根因:
   - acc_o 在 vbrc 后跨 grid block 边界没被真正清零(NPU UB 内容在 block 间复用)
   - 或者 head-group 之间的 sync barrier 缺失(虽然各 block 写不同 H 区间,但 ATB 在 wrapper 可能批量执行)
   - 或者 bishengir 在 head_groups=4 时给 acc_o 分配了一个对齐错的位置
   - **调试建议**:加 `T.npuir_clear` 显式清零 acc_o;或者在 fwd 单 head 范围(`Output[..., h_start:h_start+block_M, :]`)外加 `T.npuir_barrier` 看 NaN 是否消失
2. **bwd UB 溢出 289 KB**:需要把 `block_size=32`(topk 切块大小)缩到 16,或者把 `acc_dkv [BS=32, D=512] fp32 = 64 KB` 在 bwd 拆 split_store(upstream miles 注释里提过 `split_store` 的设计)。这是 follow-up #3。
3. **lighting_indexer_bwd UB 溢出 259 KB**:同样需要 block_H_inner 切分(目前 kernel 写死 `pad_heads = max(heads, 16)`,改成支持外部传入小 inner 值)。Follow-up。
