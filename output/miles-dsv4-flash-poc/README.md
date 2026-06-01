# miles + DeepSeek-V4-Flash on Ascend A3 NPU — PoC

> **kind**: poc · **status**: active · **target**: Ascend 910C A3 · **owner**: claude-opus-4-7
>
> 让 `radixark/miles`(基于 Megatron-LM 0.16.0rc0 的 RL post-training 框架)在 Ascend A3 NPU 上跑通 **DeepSeek-V4-Flash** 模型的 RL 后训练。

## 一句话

4 个 tilelang DSAMLA 算子(`lighting_indexer_fwd/bwd`、`sparse_mla_fwd/bwd`)在 A3 NPU 上编译跑通,52M-param Megatron+MindSpeed+tilelang 训练栈真 shape forward+backward+Adam PASS,vllm-ascend 拉 Qwen2-0.5B 单进程 RL step PASS(loss=-0.06163,12/12 finite grads)。剩下数值正确性 gap 收口在 R-KA-16 这一个上游编译器 bug。

## 当前状态(2026-06-01)

| 字段 | 值 |
|---|---|
| 进度 | 7 个上游 deliverable(5 PR + 2 Issue)全部 ready / open;13 条新 KB cookbook 沉淀 |
| 阻塞项 | R-KA-16(Huawei 编译器组接手)— blocks 真 shape NS≥2 数值回测 |
| 下一步 | 等 R-KA-16 修;期间继续 polling reviewer 反馈 |

## 上游 PR / Issue

详细每条状态 + 反馈历史见 [`artifacts/upstream-prs.md`](artifacts/upstream-prs.md);PROJECT.json 是 machine-readable 版本。

| # | 上游 | 类型 | 状态 | URL |
|---|---|---|---|---|
| 1 | `tile-ai/tilelang-mlir-ascend` | PR — CheckUBBudget pass + UT | reviewer feedback addressed `6b5c8df`,CI green | https://github.com/tile-ai/tilelang-mlir-ascend/pull/80 |
| 2 | `Ascend/AscendNPU-IR` | Issue — R-KA-16 ExtendedCanonicalizer | open,Huawei triage-review label | https://gitcode.com/Ascend/AscendNPU-IR/issues/251 |
| 3 | `radixark/miles` | PR — `_npu/` 子包 + 4 DSAMLA 算子 | reviewer feedback addressed `ff0161cc0` | https://github.com/radixark/miles/pull/1246 |
| 4 | `Ascend/MindSpeed` | PR — apex fused-rope-thd shim | open, no reviewer activity | https://gitcode.com/Ascend/MindSpeed/merge_requests/3509 |
| 5 | `sgl-project/sgl-kernel-npu` | PR — RMSNorm.bias getattr fix | open | https://github.com/sgl-project/sgl-kernel-npu/pull/531 |
| 6 | `sgl-project/sglang` | Issue — FusedMoE reload regression | open | https://github.com/sgl-project/sglang/issues/26794 |
| 7 | `triton-lang/triton-ascend` | (closed not-planned) | reframed to KB cookbook | https://github.com/triton-lang/triton-ascend/issues/306 |

## 目录结构

```
output/miles-dsv4-flash-poc/
├── PROJECT.json          # 元数据(schema:../_project_schema/PROJECT.schema.json)
├── README.md             # 本文件
├── docs/
│   ├── REPORT.md         # 详细 PoC 报告(原 docs/_meta/MILES_DSV4_NPU_POC_REPORT.md)
│   ├── REPRODUCE.md      # 从 0 到 RL step PASS 的复现步骤
│   └── kb_index.md       # 本 PoC 沉淀的 13 条 KB cookbook 列表 + 链接
└── artifacts/
    ├── upstream-prs.md   # PR / Issue 反馈历史 + 状态详表
    └── README.md         # 大型 forensic artifact 仍在 workspace/T32_tilelang_rescue/(理由见此文件)
```

## 快速复现

短答:看 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

长答 (3 step overview):

1. **构建 miles-on-NPU 训练栈**:`pip install -e` miles + MindSpeed(core_r0.16.0)+ tilelang;装 4 个 NPU 算子(`radixark/miles#1246` 分支)+ apex shim(`MindSpeed#3509` 分支)+ tilelang `CheckUBBudget`(`tile-ai/tilelang-mlir-ascend#80` 分支)+ `triton-ascend` workaround。
2. **真 shape 单步 e2e**:`python3 _e2e_megatron_step.py MILES_E2E_SHAPE=real`,期望 52M-param forward+backward+Adam PASS,sparse_mla_fwd NS=1 PASS / NS=2/4/8 NaN(R-KA-16,等修)。
3. **完整 RL step**:`python3 _e2e_rl_step_mindspeed.py`,期望 vllm-ascend rollout PASS + GRPO advantage + DSAMLA actor train PASS,loss 非零、12/12 finite grads。

详细每条命令、期望输出、撞坑 fallback 见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

## 沉淀(给后人用)

- **13 条新 KB cookbook**(详见 [`docs/kb_index.md`](docs/kb_index.md)):每条带 frontmatter `trigger` / `symptom_in_wild` / `root_cause` / `correction` / `evidence`,可被 `/npu-adapt-assist` skill 自动检索
- **`/npu-adapt-assist` skill**(`src/skills/npu-adapt-assist/`):retrieval-only,按 error trace match KB cookbook 返回 fix recipe;preflight 三层 fail-loud-early gate;cold-drive 3/3 PASS
- **PR polling 教训**(memory `pr_polling_must_check_reviews_field.md`):`gh pr view comments[]` 只显示 issue-comments,reviewer 的 review body 在 `reviews[]`、line-level 在 `/repos/.../pulls/N/comments`,不查就会漏
- **CI 验证教训**(memory `check_ci_on_my_own_push_before_declaring_done.md`):push 完必须等 CI 出结果再宣告 done,source-level grep ≠ runtime verification
