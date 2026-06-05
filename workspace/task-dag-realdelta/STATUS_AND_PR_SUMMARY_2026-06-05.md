# 状态 + PR 总结(2026-06-05)

> owner 指令:更新 report/readme + 归档产出物 + 总结 PR 和状态 + commit/push。本文件 = 状态 + PR 汇总。

## 1. 「基于最新 miles 重新基线」5-milestone 状态

| Milestone | 内容 | 状态 | 验证 |
|---|---|---|---|
| **M1** | 最新 main DSv4 用例 ↔ CANN-native 映射 + UPSTREAM_FORKS miles 行 | ✅ 完成 | 独立 agent 零 REFUTED |
| **M2** | cookbook 重整(miles-001 降级 FALLBACK / 002 强化 / 003 修 stale / index / 4 still-apply) | ✅ 完成 | 独立 agent(抓到 miles-003 漏修、已补) |
| **M3** | DSv4 CANN-native 算子 A3 真机验证 | ✅ **核心 PASS**(3 算子真机跑通);未达 PR-bar 产品化待定 | A3 实跑 + 落盘 RESULT 文档 |
| **M4** | report 重整(§九) | ✅ 完成 | 独立 agent PASS、零 dropped |
| **M5** | upstream 贡献整理 → PR | ⏳ 待 owner 二次确认 | — |

## 2. M3 真机验证结果(2026-06-05,A3,reduced+watchdog)

3 个核心 DSv4 算子 CANN-native 在 A3 跑通(HBM 全程基线、不扰他人):
- `npu_nsa_select_attention`(sparse-MLA,最贵):**fwd + bwd** 全 finite(dq/dk/dv)= 训练路通。
- `npu_nsa_compress_attention`:fwd finite。
- `npu_lightning_indexer`:fwd 正确(indices + 未掩码分数;`-inf` 是 mode-3 causal 掩码、非 bug)。

**唯一真工程缺口 = attn_sink 适配**(native 签名无 attn_sink,最新 main 要传)。其余(各 bwd 补全 / 数值对 / dispatcher / UT)= 产品化。
详:`RESULT_M3_nsa_select_attention_e2e_2026-06-05.md`。

## 3. PR / Issue 总结

### 3.1 本次「重新基线」产生的 upstream 候选(M5,待 owner 确认后提)
| 目标 | 内容 | 状态 |
|---|---|---|
| `radixark/miles` | DSv4 NPU CANN-native dispatcher(upstream DSv4 = GPU-only,缺 NPU 路) | **未达 PR-bar**(M3 产品化未完);先做完 attn_sink+UT+e2e 再提 |

### 3.2 既有 prepared-but-unopened PR(承前,见 CLAUDE.md 表,未受本次影响)
- `radixark/miles` `npu-tilelang-ops`(glm5 `_npu/` tilelang port)— **注意:本次 re-baseline 后,DSv4 运行层改走 CANN-native,此 tilelang 路降级为 FALLBACK**(cookbook miles-001);该分支仍是 glm5 算子的 tilelang NPU port,与 DSv4-CANN-native 是两条路。
- tilelang-mlir-ascend PR #80 / #96、issue #97/#99/#100,AscendNPU-IR #251/#253/PR #1199 等——均承前,本次未动。

### 3.3 a5_ops(工具链,gitcode)我的 blue/pr 分支
- `blue/pr/kb-nd2nz-srcdvalue-overflow` / `blue/pr/o5-sync-timeout-env` / `blue/pr/perf-capture-canonical-na` — 全 unmerged、main 确认 KEEP(2026-06-05 cleanup)。

## 4. 归档产出物清单(本次,均已 commit+push `zhshgmail/easyr1-npu` main)
- `REBASE_ON_LATEST_MILES_PLAN_2026-06-05.md`(计划 + 进度)
- `M1_LATEST_MILES_USECASE_UPSTREAM_MAP_2026-06-05.md`(M1 映射)
- `MILES_REBASE_ASSESSMENT_2026-06-05.md`(rebase 评估 + CANN-native 覆盖表)
- `RESULT_M3_nsa_select_attention_e2e_2026-06-05.md`(M3 真机结果)
- `M3_HANDOFF_FOR_INTRANET_AGENT.md`(M3 接手指南)
- `STATUS_AND_PR_SUMMARY_2026-06-05.md`(本文件)
- cookbook:miles-001/002/003 + sglang-004/005 + cross-layer-012/013 + index 更新
- `docs/_meta/DSV4_NPU_PORTING_REPORT.md` §九 + `README.md` + `UPSTREAM_FORKS.md` miles 行

## 5. 下一步(待 owner / 内网 agent)
- M3 产品化(attn_sink 适配 + bwd 补全 + 数值对 + dispatcher + UT + e2e)→ PR-bar → M5 提 `radixark/miles` PR(无签名,充分验证后)。
- 可由内网 A3 agent 接(handoff doc self-contained)或 owner 给 A3/A5 排期我续。
