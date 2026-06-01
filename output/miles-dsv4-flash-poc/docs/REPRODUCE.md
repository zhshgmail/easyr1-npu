# miles + DeepSeek-V4-Flash on A3 — 复现步骤

> 「从零到 RL step PASS」— 给客户 / 接手 agent 用。
> 时间预算:首次 cold-start ~6 小时(含 tilelang/AscendNPU-IR build);
> 已 cache 环境 hot-start RL step ~40 秒。

## 0. 前置环境

| 资源 | 版本 / 标识 | 来源 |
|---|---|---|
| 硬件 | Ascend A3 host(910C / dav-c220) | `ssh -p 443 root@115.190.166.102`(本项目专用机) |
| 容器 image | `easyr1-npu:integrated-20260427`(28.2 GB) | A3 host 上已 build |
| 主 workdir | `/home/z00637938/workspace/` | bind-mount 到容器 |
| Python | 3.11 + torch 2.11 + torch_npu 2.12.0rc1 | image 内置 |
| CANN | 8.5.0 | image 内置 |

> **A3 chip 经济**:host 是共享的,跑前用 `npu-smi info` 看哪些 chip free,只占需要的(单 chip 验证用 1 颗;V1.4 e2e 用 2 颗)。memory `a3_chip_economy.md` 是 hard rule。

## 1. 装 NPU 适配上游分支

5 个上游各 fork 一个 `ascend-port/<slug>` 分支,本项目用以下 HEAD:

```bash
# (1) tile-ai/tilelang-mlir-ascend PR #80 - CheckUBBudget
cd /home/z00637938/workspace/tilelang-mlir-ascend
git remote add fork https://github.com/zhshgmail/tilelang-mlir-ascend.git
git fetch fork && git checkout fork/npuir-check-ub-budget
# (build + install:见 tilelang README,需要 LLVM + ninja)

# (2) radixark/miles PR #1246 - 4 个 DSAMLA NPU 算子
cd /home/z00637938/workspace/miles
git remote add fork https://github.com/zhshgmail/miles.git
git fetch fork && git checkout fork/npu-tilelang-ops
pip install -e .

# (3) Ascend/MindSpeed MR #3509 - apex shim
cd /home/z00637938/workspace/MindSpeed
git checkout core_r0.16.0
git fetch fork && git cherry-pick <commit-from-MR-3509>
pip install -e . --no-deps

# (4) sgl-project/sgl-kernel-npu PR #531 - RMSNorm.bias getattr
# 注:本 PoC 只在 sglang 路径用到;baseline e2e 走 vllm-ascend 不需要
# 详见 docs/_meta/kb/porting_lessons/sglang-002-rmsnorm-bias-attribute-getattr.md
```

撞 `triton` vs `triton-ascend` 命名空间冲突的话,按 KB cookbook `triton-ascend-002` 的 recipe:

```bash
pip uninstall -y triton
pip install --force-reinstall --no-deps triton-ascend
python -c "import triton; assert 'ascend' in triton.backends.backends"
```

## 2. 单算子真 shape smoke

```bash
cd /home/z00637938/workspace/miles
python3 miles_plugins/models/glm5/ops/_npu/_real_shape_smoke.py
```

期望输出:

```
lighting_indexer_fwd  : PASS
lighting_indexer_bwd  : PASS  1.2s
sparse_mla_fwd  NS=1  : PASS
sparse_mla_fwd  NS=2  : NaN  ← R-KA-16, expected, blocked on Ascend/AscendNPU-IR#251
sparse_mla_bwd  NS=8  : PASS  6.6s  UB 140/192 KB
```

3/4 PASS,1/4 NS≥2 NaN 是已知 R-KA-16,**这是 PoC 当前状态**,不是 fail。

## 3. Megatron 单步 e2e(真 shape)

```bash
MILES_E2E_SHAPE=real python3 miles_plugins/models/glm5/ops/_npu/_e2e_megatron_step.py
```

期望:52,270,848 params DSAMLASelfAttention,forward + backward + Adam 跑通,out shape `[2048, 1, 512]`,indexer score `[2048, 512]`。

## 4. 多层多步(MindSpeed 接入后,验证无 NaN drift)

```bash
python3 miles_plugins/models/glm5/ops/_npu/_e2e_megatron_multilayer_mindspeed.py
```

期望:2-layer × 3-iter,**25/25 finite 跨 3 iter,无 NaN drift**。

## 5. 完整 RL step

```bash
# 注:driver 在 workspace/T32_tilelang_rescue/,因为它是 PoC 期间临时驱动
# production 路径见 docs/REPORT.md §3.5
python3 /home/z00637938/workspace/easyr1-npu/repo/workspace/T32_tilelang_rescue/_e2e_rl_step_mindspeed.py
```

期望(2026-05-29 实测):

```
=== RL step summary ===
  total time: 38.7s(rollout 18.6s + actor train 8.4s)
  prompts: 2  rollouts/prompt: 2
  actor out: (16, 1, 128)
  loss: -0.06163(mla=-0.06163, idx=0.00000)
  finite grads: 12/12
  result: PASS
```

## 6. sglang 路径 baseline smoke(可选,production 路径)

miles 默认 rollout 引擎是 sglang(`miles/ray/rollout.py:16` 用 `SGLangEngine`);本 PoC 主路径用 vllm-ascend 是因为 tlrescue 容器装的是 vllm。sglang-on-NPU 的 baseline 已独立 smoke PASS:

```bash
docker run --rm -it --device /dev/davinci1 quay.io/ascend/verl:verl-sglang-8.5.0-... \
    python3 /workspace/workspace/T32_tilelang_rescue/sglang_npu_smoke.py
```

期望:Qwen2-0.5B engine init 20.1s + generate 16.9s + 输出语义正确("Paris"、"Sarah... UK student")。3 个必撞坑(sys.path `/` shadow / `ASCEND_RT_VISIBLE_DEVICES=1` 滤掉单 chip / Engine multiprocessing spawn 要 `__main__` guard)见 memory `feedback_sglang_npu_smoke_recipe.md`。

## 7. 撞坑了怎么办

按 error trace grep 找 cookbook:

```bash
# 装好 /npu-adapt-assist skill 后:
/npu-adapt-assist "<paste error trace>"
```

skill 会从 27 条 cookbook(`docs/_meta/kb/porting_lessons/*.md`)按 frontmatter `trigger` + `symptom_in_wild` 匹配,返回 top-1 + `correction` 段。详见 [`docs/kb_index.md`](kb_index.md)。
