# 基于最新版 miles 重新整理用例 + upstream — Milestone 计划（2026-06-05）

> Owner 指令 2026-06-05：「基于最新版 miles 重新整理所有用例和 upstream。你来整理、计划。按步骤驱动。使用 loop 把任务端到端完成，注意 kb 更新和 report 更新，每个 milestone 都使用独立 agent 做验证。」

## 0. 现状基线（M0，已盘清，本地）

- **最新 miles** = upstream `radixark/miles` main HEAD `74198b45`（2026-06-04，活跃）。我们 fork `npu-tilelang-ops` 落后 **115 commit**，且其 NPU 工作在 `glm5/ops/_npu/`，**不含** DSv4 plugin（DSv4 是 fork 点后才进 main 的，纯 GPU）。
- **关键决定性发现（已验证，A3 只读）**：torch_npu **CANN-native 覆盖最新 DSv4 全部核心算子 fwd+bwd**（`npu_nsa_select_attention(+grad)` / `npu_nsa_compress_attention(+grad)` / `npu_lightning_indexer(+grad)` / `npu_rms_norm(+backward)` / `npu_mla_prolog_v3`）；唯一缺口 fp8 act_quant = A3 硬件墙（bf16/QAT-off 不需要）。→ **NPU 运行层走 CANN-native，不必逐个 tilelang re-port**。详见 `MILES_REBASE_ASSESSMENT_2026-06-05.md`。
- **现有用例资产**：`docs/_meta/kb/porting_lessons/` ~35 条 cookbook（index.md keyword 表）+ `UPSTREAM_FORKS.md`（5 upstream 账本）+ 两份 report（`docs/_meta/DSV4_NPU_PORTING_REPORT.md` + `output/miles-dsv4-flash-poc/docs/REPORT.md`）。

## 1. 受最新-miles-rebaseline 影响的资产（要重整的范围）

| 资产 | 为何要重整 |
|---|---|
| cookbook `miles-001`（tilelang NPU port pattern）| 被 CANN-native-first 取代为非主路；标注"仅 CANN 无覆盖时" |
| cookbook `miles-002`（V4 ops CANN-native-first）| 用最新 DSv4 op 集 + 验证过的 torch_npu native 映射表更新 |
| cookbook `miles-003`（megatron layer on NPU）| 对齐最新 DSv4 plugin 的层结构 / 接口演进 |
| `sglang-004/005`、`cross-layer-012/013` | 校对是否仍适用最新 miles（接口/权重同步可能变） |
| `UPSTREAM_FORKS.md` miles 行 | 反映 115-commit gap + CANN-native 决策 + 新 fork 分支 |
| `DSV4_NPU_PORTING_REPORT.md` | 加「最新 miles 基线 + CANN-native 运行层」章 |
| `output/miles-dsv4-flash-poc/docs/REPORT.md` | 同步基线变更 |

## 2. Milestones（每个 = 交付 + 验收 + 独立 agent 验证）

> 纪律：每 milestone 完成 → KB+report 即时更新 → **独立 sub-agent 对抗验证**（三态裁决：CONFIRMED/REFUTED/UNVERIFIABLE）→ 过了才进下一个。NPU 活按 A3 排期（当前 host 被他人 job 占，precheck 不抢）。

### 进度（2026-06-05）
- **M1 ✅ 完成**（独立 agent 验证零 REFUTED）：映射表 + UPSTREAM_FORKS miles 行。
- **M2 ✅ 完成**（独立 agent 验证：编辑全 CONFIRMED + 抓到漏的 miles-003 stale 已补修 + 4 cookbook still-apply 注已 sediment）：miles-001 降级 / miles-002 强化 / miles-003 修 stale walls / index 更新 / sglang-004,005 + cross-layer-012,013 加 re-confirm 注。
- **M4 部分 ✅ 完成**（独立 agent 验证 PASS，零 dropped content）：DSV4 report 加 §九（最新 miles 基线 + CANN-native 策略）。M4 剩余（PoC report 同步 + M3 e2e 数）待 M3。
- **M3 ⏳ 待 A3**：CANN-native dispatcher PoC + e2e。A3 当前被他人 job 占（AscendOpGen/liger/vllm），按排期。
- **M5 ⏳ 待 owner**：upstream 贡献整理（不自动 PR）。

---

- **M1 — 最新 miles 用例/upstream 差异基线（纯本地，无 NPU）**
  - 交付：一张「最新 main DSv4 用例清单 ↔ 我们现有 cookbook ↔ 需 新增/更新/作废」映射表 + `UPSTREAM_FORKS.md` miles 行更新 + 决定 CANN-native vs tilelang 每算子归属。
  - 验收：映射表每行有证据指针（main 文件:行 / torch_npu op 名）。
  - 独立 agent 验证：核「映射表声称的 native op 真存在 / main 文件真有那算子」。

- **M2 — cookbook 重整（纯本地，无 NPU）**
  - 交付：更新 miles-001/002/003 + 校 sglang-004/005、cross-layer-012/013；index.md keyword 表同步；新增「CANN-native-first for DSv4」cookbook（若需）。
  - 验收：每条 cookbook frontmatter trigger/symptom 准确；index 一致。
  - 独立 agent 验证：核每条 cookbook 的 claim vs 源码 / torch_npu。

- **M3 — CANN-native dispatcher PoC（需 A3，按排期）**
  - 交付：最新 main 上建分支，把 DSv4 的 6 个 tilelang 调用在 dispatcher 接 torch_npu native；A3 单算子 e2e（fwd+bwd 数值对）。
  - 验收：达 `feedback_pr_quality_bar`（验证+UT+e2e 报告）。
  - 独立 agent 验证：复核 e2e 数值 + UT 覆盖。
  - **M3-readiness 预研（2026-06-05，本地只读，A3 空闲前先记下）：**
    - **最新 main `sparse_mqa_fwd_interface` 签名（实查 `tilelang_sparse_mla_fwd.py:152`）**：in `q[B,S,H,D]bf16` / `kv[B,S_kv,D]bf16`（MQA 单 KV 头）/ **`attn_sink[H]fp32`** / `topk_idxs[B,S,topk]int32`（pad 到 block_I=64 倍数）/ `sm_scale`；out `[B,S,H,D]bf16` + `lse[B,S,H]fp32`。
    - **native `npu_nsa_select_attention` 签名（memory 验证值）**：`(q,k,v,topk_indices,scale,head_num,select_block_size,select_block_count,actual_seq_qlen,actual_seq_kvlen)`，TND layout，D_qk=192/D_v=128/sbs=64/sbc=16，返回 attn+softmax max/sum。
    - **⚠️ M3 头号风险（本预研新发现）：最新 main 多了 `attn_sink`（softmax sink，[H]fp32），而 `npu_nsa_select_attention` 的 native 签名里没有显式 attn_sink 入参**。→ M3 的 A3 run **第一件要验证的就是 attn_sink 怎么接**：(a) native op 是否有隐藏/新版 attn_sink 支持，(b) 否则需在 native softmax 后做 sink 调整（post-hoc lse 合并），(c) 这可能让"100% native 覆盖"对最新 main 多一个 attn_sink 适配层。torch_npu C-binding 签名不可 introspect（`*args/**kwargs` wrap）→ 只能 A3 实跑确认,不能静态推。
    - 含义：M3 不是纯"换函数名",attn_sink 是真适配点;dispatcher 要处理 attn_sink + MQA 单头 broadcast + topk pad + TND layout 转换。预研到此为止（再深要 A3）。

- **M4 — report 重整（纯本地）**
  - 交付：DSV4 report + PoC report 加「最新 miles 基线 + CANN-native 运行层」章，旧 tilelang-port 结论降级为"CANN 无覆盖时的退路"。
  - 验收：`rewrite_must_preserve_content`（diff 旧→新不丢事实）。
  - 独立 agent 验证：核 report 新章 claim vs M1-M3 证据。

- **M5 — upstream 贡献整理（需 owner 二次确认才提 PR）**
  - 交付：哪些是该提给 upstream 的（miles NPU 路 / sglang V4 hooks / 等），各自达 PR-bar 的状态评估。
  - 验收：每个 PR 候选标注「验证状态 / 缺口 / 是否够 bar」。
  - 独立 agent 验证：核 PR 候选的验证证据是否真达 bar。
  - **不自动提 PR**——按 `feedback_pr_quality_bar` + owner 节奏。

## 3. 驱动方式

- loop 自驱，milestone 顺序 M1→M5；每个 milestone 内：做→KB/report 更新→独立 agent 验证→报 Discord→进下一个。
- A3 NPU 活（M3）按 host 空闲排期；纯本地 milestone（M1/M2/M4）不受 A3 占用影响，优先推进。
- 每个 milestone 完成即 Discord 报（含独立 agent 裁决结果）。

相关：`MILES_REBASE_ASSESSMENT_2026-06-05.md`、memory `project_v4_ops_cann_native_mapping` / `feedback_pr_quality_bar` / `project_report_verified_2026_06_05` / `version_aware_reviews`。
