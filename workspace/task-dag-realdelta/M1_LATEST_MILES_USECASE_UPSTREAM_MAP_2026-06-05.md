# M1 — 最新 miles DSv4 用例 ↔ cookbook ↔ CANN-native 归属映射（2026-06-05）

> Milestone M1（本地、只读）。基线：upstream `radixark/miles` main `74198b45`（2026-06-04）。
> 证据全部来自 `git show origin/main:...` + A3 只读 `dir(torch_npu)`（零 NPU 计算）。

## A. 最新 DSv4 plugin 实际 wiring 的算子（deepseek_v4.py / compressor.py / v4_indexer.py import 实查）

| DSv4 算子（最新 main） | 来源文件 | 模型里调用点 |
|---|---|---|
| `sparse_attn_tilelang`（sparse MQA attn） | `ops/kernel/tilelang_sparse_mla.py` | `deepseek_v4.py:35` |
| `V4Indexer` → `batched_indexer_fwd`（lightning indexer） | `ops/v4_indexer.py` → `ops/kernel/tilelang_indexer_fwd.py` | `deepseek_v4.py:38` |
| `DeepSeekV4Compressor`（compress attn） | `ops/compressor.py` | `deepseek_v4.py:27` |
| `linear_bf16_fp32`（bf16→fp32 精度对齐 linear） | `ops/kernel/precision_aligned_ops.py` | `compressor.py:8` |
| `fp8_simulate_qat`（fp8 QAT fake-quant） | `ops/qat.py` | `deepseek_v4.py:252`、`compressor.py`、`v4_indexer.py`（env-gated） |
| `apply_rotary_emb`（RoPE） | `ops/rope.py` | `deepseek_v4.py:37` |
| `RMSNorm`（纯 PyTorch FP32，刻意对齐 SGLang norm） | `ops/compressor.py:14` | compressor |

## B. CANN-native 归属（每算子：A3 走 native / 纯 torch / tilelang 退路）— torch_npu 已实测

| DSv4 算子 | CANN-native (torch_npu) | A3 归属 | bwd |
|---|---|---|---|
| sparse MQA attn | `npu_nsa_select_attention` | ✅ native | ✅ `npu_nsa_select_attention_grad` |
| compress attn | `npu_nsa_compress_attention` | ✅ native | ✅ `npu_nsa_compress_grad` |
| lightning indexer | `npu_lightning_indexer` | ✅ native | ✅ `npu_lightning_indexer_grad` / `npu_sparse_lightning_indexer_grad_kl_loss` |
| MLA prolog | `npu_mla_prolog_v3` | ✅ native | functional 变体 |
| RMSNorm | `npu_rms_norm`（或保留纯 torch FP32 对齐 SGLang） | ✅ native 可选 | ✅ `npu_rms_norm_backward` |
| linear_bf16_fp32 | 纯 torch（`torch.mm out_dtype=fp32`） | ✅ A3 直接可用 | ✅ |
| RoPE apply_rotary_emb | 标准 torch / `npu_*rope*` | ✅ | ✅ |
| fp8_simulate_qat | ❌ 无 native（fp8 = A3 硬件墙） | ⛔ QAT-off 路不调；QAT-on 需 A5 | — |

→ **核心 attention/indexer/norm 算子 100% 有 CANN-native fwd+bwd**。唯一不可达 = fp8 QAT（A3 硬件，QAT-off 不影响 bf16 训练）。**tilelang re-port 对核心算子无必要**（Path A 成立）。

## C. cookbook 重整决定（→ M2 执行）

| cookbook | 现状 | M2 动作 |
|---|---|---|
| `miles-001`（tilelang NPU port pattern） | 把 miles 算子 port 到 tilelang-ascend | **降级**为"仅 CANN 无 native 覆盖时的退路"；加 cross-link 到 B 表 |
| `miles-002`（V4 ops CANN-native-first） | 已是 native-first 方向 | **更新**：换最新 DSv4 op 集 + B 表的实测 torch_npu 映射（含 bwd op 名） |
| `miles-003`（V4 megatron layer on NPU） | 减层 layer fwd+bwd | **校对**对齐最新 plugin 层结构 + 接口演进（sparse_mqa attn_sink/topk_idxs） |
| `sglang-004/005`（V4 推理 hooks/SWA） | V4 推理侧 | **校对**是否仍适用最新 miles（推理侧变化小，预计仍适用） |
| `cross-layer-012/013`（RL 权重同步） | tensor-not-disk / same-weights | **校对**（机制级，预计仍适用） |
| 新增？ | — | 评估是否需「DSv4 CANN-native dispatcher」cookbook（M3 PoC 产出后定） |

## D. upstream 账本更新决定（→ M1 内更新 UPSTREAM_FORKS.md miles 行）

- miles fork `npu-tilelang-ops` 落后 main 115 commit；其 NPU 工作在 `glm5/_npu/`，非 DSv4。
- 决策：DSv4 NPU 运行层走 CANN-native（非 tilelang re-port）；upstream 贡献候选 = 「DSv4 plugin 的 NPU dispatcher 路」（M3 PoC 后评估，M5 定）。

## E. 验收

每行 claim 有证据指针（main 文件:行 / torch_npu op 名，均已实查）。待**独立 agent 对抗验证**（核 native op 真存在 + main 文件真有该算子 + import wiring 属实）。
