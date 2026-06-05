# miles NPU 适配 rebase-到-最新-upstream 工作量评估（2026-06-05）

> 回答 owner 问题：「我们是否该用最新版 miles？upstream 是否也要更新？」
> 全程本地 git 分析，零 NPU / 零 A3 写入。

## 1. 版本现状

| 维度 | 事实 |
|---|---|
| 我们的工作分支 | `zhshgmail/miles npu-tilelang-ops`，HEAD `ff0161cc0`（2026-05-31） |
| fork 点（merge-base） | `f7960bb2`（与 origin/main 的共同祖先） |
| 上游 `radixark/miles` main | HEAD `74198b45`（2026-06-04，非常活跃） |
| 落后 | **115 commits** |
| 我们分支相对 main 的净改动 | 14 files / +1970 / -4 |

## 2. 我们的 NPU 贡献到底是什么（实查）

我们分支相对 main 多出的 NPU 内容**全在 `miles_plugins/models/glm5/ops/_npu/`**（**不是 deepseek_v4！**）：
- `_npu/_sparse_mla_fwd_kernel.py`、`_sparse_mla_bwd_kernel.py`、`_lighting_indexer_fwd_kernel.py`、`_lighting_indexer_bwd_kernel.py`、`indexer.py`、`sparse_mla.py`、`__init__.py`
- `tests/fast/test_npu_kernel_safety_guards.py`
- 这些用 **NPU idiom**（`T.Kernel(..., is_npu=True)` 单 block + decode-inside）。

## 3. 关键发现：DSv4 plugin 是 fork 点之后才进 main 的，且纯 GPU

- 整个 `deepseek_v4/` plugin（19 files / 2393 行）在 fork 点 `f7960bb` **之后**才落地 main → 我们分支里**根本没有 deepseek_v4 plugin**。
- main 的 deepseek_v4 tilelang kernel = **纯 CUDA tilelang**：`sparse_mqa_fwd` 用 `T.Kernel(seq_len*REPLICATE_H, batch, threads=256)`（GPU 2D-grid + threads=），grep `is_npu|alloc_ub|npuir|ascend` = **0**。
- 且**接口已演进**：main `sparse_mqa_fwd(q, kv, attn_sink, topk_idxs, sm_scale, block_I=64, num_stages, threads)` ≠ 我们 glm5 的 `sparse_mla_fwd(q, kv, indices)`（多了 attn_sink / topk_idxs / block_I / MQA 化）。

## 4. 结论：不是 "git rebase"，是 "用已验证的 NPU idiom 重新移植演进后的 DSv4 算子"

| 问题 | 答案 |
|---|---|
| 最新 miles 能否「免 NPU 适配」？ | ❌ 否。最新 miles DSv4 = 纯 GPU/CUDA，**零 NPU 路径**。换最新版只是拿到更成熟的 GPU 基线，NPU 适配工作照做。 |
| 我们的 glm5 `_npu/` 能直接 rebase 过去吗？ | 🟡 部分。**算法同族**（sparse MLA/MQA + lighting indexer），NPU-idiom 移植知识（is_npu 单 block decode 模式）**可迁移**；但 (1) 目标模块不同（deepseek_v4 vs glm5）(2) 接口演进（attn_sink/topk_idxs/block_I/MQA）(3) 新 kernel 是 GPU idiom，每个都要做 GPU→NPU idiom 重写。 |
| upstream 是否该更新？ | ✅ 是。upstream miles DSv4 = GPU-only，**NPU 路径正是缺口**，是该收的贡献。 |

## 5. 工作量估计（粗）

把 6 个演进后的 DSv4 算子用我们已验证的 NPU idiom 重新移植 + 对齐新接口：
- sparse_mqa_fwd / bwd、lighting_indexer fwd / bwd（4 个，重写量大，我们 glm5 版可作模板）+ qat（TileKernels 路，A3 fp8 硬件墙问题仍在，见 [[fp8_is_a3_hardware_limit_not_software]]）+ precision_aligned linear（bf16-fp32，A3 可直接用）。
- **非机械 rebase**；估计与当初做 glm5 `_npu/` 同量级（数天），因接口演进 + 需在 A3 实测（当前 A3 被他人 job 占，需排期）。

## 6. 建议

值得做，但分两步、按 owner 节奏：
1. （本地，已完成）= 本评估。
2. （需 owner go + A3 空闲）= 在最新 main 上建新分支，把 6 个 DSv4 算子按 NPU idiom 移植 + UT + A3 e2e，达 [[feedback_pr_quality_bar]] 后 PR。**先不动手，等 owner 拍。**

相关：`project_v4_ops_cann_native_mapping`（V4 算子另有 CANN-native 路，可能比 tilelang 移植更直接）、`tilelang_npu_idiom_vs_gpu_idiom`。

---

## 7. CANN-native 覆盖度核实（2026-06-05，A3 只读 `dir(torch_npu)`，零 NPU 计算）

把**最新 main DSv4 plugin 的核心算子** ↔ torch_npu native op 逐个映射（torch_npu 实测有这些 op，比 memory 记的还全）：

| 最新 DSv4 算子（main） | CANN-native (torch_npu) | fwd | bwd/grad |
|---|---|---|---|
| sparse MQA fwd/bwd (`sparse_mqa_fwd/bwd_interface`) | `npu_nsa_select_attention` | ✅ | ✅ `npu_nsa_select_attention_grad` |
| compress attention (compressor) | `npu_nsa_compress_attention` | ✅ | ✅ `npu_nsa_compress_grad` |
| lightning indexer fwd/bwd (`tl_indexer_*`) | `npu_lightning_indexer` | ✅ | ✅ `npu_lightning_indexer_grad` + `npu_sparse_lightning_indexer_grad_kl_loss` |
| rms_norm | `npu_rms_norm` | ✅ | ✅ `npu_rms_norm_backward` |
| MLA prolog | `npu_mla_prolog_v3` | ✅ | functional 变体 |
| precision_aligned linear (bf16→fp32) | 纯 torch（`torch.mm out_dtype`）| ✅ A3 直接可用 | ✅ |
| act_quant (fp8) | ❌ 无 native（且 A3 fp8 硬件墙） | — | A3 QAT-off 路不需要 |

**结论（决定性）**：CANN-native **覆盖最新 DSv4 全部核心 attention/indexer/norm 算子，fwd+bwd 都有**。唯一缺口是 fp8 `act_quant`——而它本就是 A3 硬件墙、A3 走 bf16/QAT-off 路根本不调它。
→ **Path A（CANN-native）对 A3 产品级运行层是清晰可行且更优**：覆盖全 + 绕开整个 tilelang re-port + #100/codegen，且这些 native op 之前在 A3 跑过。最新 DSv4 是 GPU-tilelang 不影响——A3 把它们指到 CANN-native 即可。
→ **Path B（tilelang 移植）只在 CANN 无 native 的算子才需要**——按上表，核心算子无此需要。

**修正建议**：不必做 115-commit 的 tilelang re-port（第 5 节估的数天工作量大半可省）。产品路 = miles DSv4 运行层接 CANN-native（dispatcher 把 6 个 tilelang 调用换成 torch_npu native）。
