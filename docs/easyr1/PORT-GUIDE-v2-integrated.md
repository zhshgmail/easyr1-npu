# EasyR1 → A3 NPU — v2 (integrated overlay path)

> Ship-ready 集成路径：一个 `docker run`，一个 EasyR1 master 工作树，一个 image，RL loop 端到端验证。
>
> 与 [v1 path](PORT-GUIDE.md) 区别：v2 把 4 个 NPU 上游 ascend-port 分支（vllm-ascend / torch-npu / transformers / triton-ascend）作为 overlay 叠到 vllm 0.20.0-era base 上，让 EasyR1 master（去掉 `transformers<5.0.0` cap）跑在更新的上游 stack 上。

最后更新 2026-04-28（T25.5 第二次独立验证 PASS）。

## TL;DR

```bash
# 在 A3 host 上一条命令：
ssh -p 443 root@<a3-host> "
  NPU_USER=<workspace-owner> \
  bash /home/<workspace-owner>/workspace/easyr1-npu/repo/src/scripts/run-npu-container.sh \
    --chips 0,1 \
    --image easyr1-npu:integrated-20260427 \
    --live-source /home/<workspace-owner>/workspace/easyr1-npu/upstream/EasyR1 \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
"
```

预期：2 GRPO 步 + post-train val ~10 min PASS，checkpoint 落 `/tmp/<NPU_USER>/easyr1_smoke_ckpt/global_step_2/`。

## Stack（验证于 2026-04-28）

| 组件 | 版本 | 来源 |
|---|---|---|
| Image | `easyr1-npu:integrated-20260427`（28.2 GB，SHA `044ba0b76183`） | A3 host 本地 |
| Image base | `easyr1-npu-vllm0200:iter20-abi-both` | A3 本地 |
| OS / Python | Ubuntu 22.04 / Python 3.11.14 | base |
| CANN | 8.5.x bundled | base |
| torch | 2.11.0+cpu | base |
| torch_npu | 2.11.0rc1 | base + `torch_npu/compat/` overlay（`ascend-port/torch-2.12-rc3`） |
| transformers | 5.3.0.dev0 | base（v5.4 outcome A-with-note，见 `docs/transformers/PR_MATERIAL_v5.4_outcome_A.md`） |
| vllm | 0.20.0（real，非 stub） | base |
| vllm_ascend | 0.17.0rc2.dev109+g54879467c + 3 compat shim | base + `ascend-port/vllm-main` overlay |
| triton-ascend | 3.2.0 image vendor（6/6 NPU smoke PASS） | base |
| EasyR1 | master + cap 解除 | [`zhshgmail/EasyR1` `ascend-port-integrated-20260427`](https://github.com/zhshgmail/EasyR1/tree/ascend-port-integrated-20260427) |

## Overlay 内容

1. **vllm-ascend compat shim**（`ascend-port/vllm-main`）：
   - `vllm_ascend/compat/__init__.py`
   - `vllm_ascend/compat/shared_fused_moe.py` — F1 shim（vllm PR #35782）
   - `vllm_ascend/compat/default_moe_runner.py` — F1 shim（vllm PR #40560）
   - `vllm_ascend/compat/spec_decode_base_proposer.py` — F2-path-move shim，用 `find_spec` + lazy `__getattr__` 规避 vllm-ascend plugin-init 顺序
   - 3 处 call-site swap（`vllm_ascend/{_310p,ops,spec_decode}/`）
2. **torch_npu compat**（`ascend-port/torch-2.12-rc3`，在 torch 2.11 上是 no-op）：
   - `torch_npu/compat/__init__.py`
   - `torch_npu/compat/inductor_codecache.py` — F2-path-move shim，`Union` re-export
3. **EasyR1 master + cap 解除**（`ascend-port-integrated-20260427`）：
   - `requirements.txt`：`transformers>=4.54.0,<5.0.0` → `transformers>=4.54.0`
4. **transformers**：image 自带 5.3.0.dev0，无 overlay
5. **triton-ascend**：image 自带 vendor 3.2.0，无 overlay

## 验证

V1.4 GSM8K-style GRPO smoke 跑通两次（独立 agent，不同 chip 对）：

| 运行 | 日期 | chips | accuracy_reward | reward_score | val_response_length |
|---|---|---|---|---|---|
| 第一次（fresh baseline） | 2026-04-27 | 0,1 | 0.014 | 0.0126 | 184.3 |
| 第二次（cold-drive replay） | 2026-04-28 | 2,3 | 0.014 | 0.013 | 184.3 |

两次均：2 GRPO step + post-train val PASS，checkpoint 保存，无 exception。

`entropy_loss` / `grad_norm` 不出现在 `experiment_log.jsonl` —— 这是该 image stack 上 trainer 端 jsonl writer 的既存 quirk，**不是** overlay 引起的回归。

## 已知 gotcha

1. **SSH-as-root + `$USER`**：SSH 到 A3 作为 `root` 时 `$USER=root`，runner 默认 mount `/home/root` / `/data/root`（空目录）。**必须传 `NPU_USER=<workspace-owner>`**。详见 [NPU-OPS-013](../../knowledge/npu-patterns.md)。

2. **`--chips` 与 `ASCEND_RT_VISIBLE_DEVICES`**：`--chips 0,1` 是 host phy-id；runner 自己把对应 `/dev/davinciN` 透传，并自动设 `ASCEND_RT_VISIBLE_DEVICES=0,1,...,N-1`（容器内 index）。**不要手 docker run 直接传 host phy-id**——否则 Ray 报 `Total available GPUs 0`。详见 [NPU-OPS-012](../../knowledge/npu-patterns.md)。

3. **A3 host 的 `repo/` 必须是 git clone**，不能是早期 layout 的手抄拷贝（`repo/scripts/...` 路径）。详见 [NPU-OPS-014](../../knowledge/npu-patterns.md)。

## Skill-chain 来源

这个 integrated image 由 4 个 NPU 上游 port-expert skill 的产出叠加产生：

- `/torch-npu-day0` — outcome A 于 image torch 2.9
- `/transformers-day0` — outcome A-with-note 于 v5.4，on-A3 forward pass PASS（Qwen2-0.5B）
- `/vllm-ascend-day0` — 3 shim，3/3 import smoke PASS 于 vllm 0.20.0 base
- `/triton-ascend-port` — vendor 3.2.0 wheel 6/6 NPU smoke PASS

每条 skill 的 artifact 在 [`docs/_meta/UPSTREAM_FORKS.md`](../_meta/UPSTREAM_FORKS.md)。本文档是 integrated 层的 hand-off；每个上游 fork 分支根目录的 `PR_MATERIAL.md` 是各自上游维护者的 hand-off。

## Next-version forward plan

- **Base image 升级**：当新 `easyr1-npu-vllm0200:*` 或后继 image ship 更新 vllm / torch，重新跑 4 条 day-0 skill 对照新 base；F-family shim 大都向后兼容，能继续用。
- **EasyR1 master 跟踪**：当 EasyR1 master 改 `requirements.txt` 或 `verl/integrations/transformers*` 时，重跑 transformers day-0 byte-compare。
- **bishengir LLVM 22 release**：Huawei 一旦 ship LLVM-22 版 bishengir-compile，重新跑 `ascend-port/triton-v3.6.0` 源码端到端验证（当前 BLOCKED，详见该分支 `PR_MATERIAL.md`）。

## 见也

- [`PORT-GUIDE.md`](PORT-GUIDE.md) — v1 path
- [`docs/_meta/ARCHITECTURE.md`](../_meta/ARCHITECTURE.md) — 整体架构与流程
- [`docs/_meta/UPSTREAM_FORKS.md`](../_meta/UPSTREAM_FORKS.md) — fork ledger
- [`src/skills/_shared/integrated-overlay-build/SKILL.md`](../../src/skills/_shared/integrated-overlay-build/SKILL.md) — `/integrated-overlay-build` skill
