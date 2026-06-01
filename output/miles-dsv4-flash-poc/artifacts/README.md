# `artifacts/` — 大型 forensic 文件与 reproducer 脚本

## 为什么 `scripts/` 和 `ir-dumps/` 没在这里

按 a5_ops 标准模式,每个 sub-project 自带 `artifacts/` 放 reproducer。本 PoC 的对应文件在 [`workspace/T32_tilelang_rescue/`](../../../workspace/T32_tilelang_rescue/) 而**没有**复制到这里,理由:

1. **CLAUDE.md 把它们标为 "compact-survival index" 不可删/不可移**:`workspace/T32_tilelang_rescue/rka16_ns4_passes.txt`(2.9 MB)、`rka16_ns4_pass_index.txt`(311 lines)、`rka16_ns4.npuir`、`dump_rka16_ir.py`、`repro_rka16.py` 等被 Ascend/AscendNPU-IR Issue #251 引用;复制会让两份 hash 漂移
2. **跨 sub-project 共享 forensic 价值**:如果未来出 R-KA-17 / R-KA-18,bisect 配方和 IR dump 还会被复用 — 它们属于「项目级 forensic asset」,不属于「单一 sub-project 的复现脚本」
3. **物理体积**:`rka16_ns4_passes.txt` 2.9 MB,git LFS 不开 + 不复制是合理选择

**索引(从 `output/miles-dsv4-flash-poc/` 找过去)**:

| Artifact | 实际路径 | 用途 |
|---|---|---|
| R-KA-16 最小重现 | [`workspace/T32_tilelang_rescue/repro_rka16.py`](../../../workspace/T32_tilelang_rescue/repro_rka16.py) | `python3 repro_rka16.py --ns {1,2,4,8}` |
| 311-pass dump driver | [`workspace/T32_tilelang_rescue/dump_rka16_ir.py`](../../../workspace/T32_tilelang_rescue/dump_rka16_ir.py) | monkey-patch tilelang 把 NS=4 npuir 保存 + bishengir-opt 跑 print-after-all |
| 311-pass full dump | [`workspace/T32_tilelang_rescue/rka16_ns4_passes.txt`](../../../workspace/T32_tilelang_rescue/rka16_ns4_passes.txt) | 2.9 MB,Issue #251 引用 |
| Pass index | [`workspace/T32_tilelang_rescue/rka16_ns4_pass_index.txt`](../../../workspace/T32_tilelang_rescue/rka16_ns4_pass_index.txt) | line-number → pass-name,grep iter_args 用 |
| RL step driver(本 PoC 主路径)| [`workspace/T32_tilelang_rescue/_e2e_rl_step_mindspeed.py`](../../../workspace/T32_tilelang_rescue/_e2e_rl_step_mindspeed.py) | vllm-ascend rollout + MindSpeed actor train 单进程 |
| 5-step weight sync driver | [`workspace/T32_tilelang_rescue/test_5step_weight_sync.py`](../../../workspace/T32_tilelang_rescue/test_5step_weight_sync.py) | dense fab,5 step rollout outputs distinct |
| Megatron → HF rename map | [`workspace/T32_tilelang_rescue/megatron_to_hf_rename.py`](../../../workspace/T32_tilelang_rescue/megatron_to_hf_rename.py) | 12-key map byte-clean |
| sglang-on-NPU smoke | [`workspace/T32_tilelang_rescue/sglang_npu_smoke.py`](../../../workspace/T32_tilelang_rescue/sglang_npu_smoke.py) | Qwen2-0.5B engine init + generate(sidecar 容器)|
| R-KA-16 issue 草稿 | [`workspace/T32_tilelang_rescue/UPSTREAM_ISSUE_RKA16.md`](../../../workspace/T32_tilelang_rescue/UPSTREAM_ISSUE_RKA16.md) | gitcode #251 提交前的 body 草稿 |
| DSv4 真 shape 推导 | [`workspace/T32_tilelang_rescue/DSV4_REAL_SHAPE_FULLSTACK_ANALYSIS.md`](../../../workspace/T32_tilelang_rescue/DSV4_REAL_SHAPE_FULLSTACK_ANALYSIS.md) | DSv4-Flash 真 dim 推导 + miles 组件分类 |
| 上游全景图 | [`workspace/T32_tilelang_rescue/UPSTREAM_PATCH_MAP.md`](../../../workspace/T32_tilelang_rescue/UPSTREAM_PATCH_MAP.md) | 「**THE 全景图**」:enumerate 每条 miles-DSv4-RL-on-NPU 全链路上游 |

## 未来 sub-project 怎么用 `artifacts/`

新建 sub-project 时,**reproducer 默认放本目录**(`artifacts/scripts/`、`artifacts/ir-dumps/` 等),只有当 artifact 跨多个 sub-project 共享 OR 体积特别大时,才用本项目这种「索引到 workspace/」的方式。Template 在 [`output/_project_template/`](../../_project_template/) 给的是默认形态(`artifacts/scripts/`)。
