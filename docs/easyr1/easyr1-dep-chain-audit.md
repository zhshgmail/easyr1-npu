# EasyR1 master 依赖链审计 —— A/B/C/D/E 分级

**审计目标**：逐个认领 EasyR1 master（`dd71bbd`）声明的 20 个 runtime 依赖，判断其在 NPU 生态的状态。

**为什么做这个**：解答第一性原则问题 —— "EasyR1 master 在 A3 上真的能跑通吗？还是有我们没意识到的 gap？"。同时建立一个**自动化判断的基线**，下次新版本升级可以 diff 对照。

**结论（2026-04-20）**：EasyR1 master 在 v1 image（`verl-8.5.0-a3`）上 **D 类 blocker = 0**。V1.4 smoke 已验证数值 PASS（`entropy_loss` step1=0.991），说明生态覆盖完整。**场景 P1（不需要额外 NPU 开发）已经闭环**，只需用 `ascend-port` 分支跑起来。drill 在 v2 image 上 D 类也是 0。

---

## 分级方法

| 档 | 名称 | 描述 | 本项目责任 |
|---|---|---|---|
| **A** | NPU 原生支持 | Upstream 本身有 NPU 兼容代码（无需我们改任何事） | 直接用 |
| **B** | 有 NPU 移植版本 | 存在一个 NPU-specific 的 drop-in / fork（如 `vllm` → `vllm-ascend`） | 在 image 里用 NPU 版本 |
| **C** | CUDA-only 但能绕过 | EasyR1 代码里可以用 NPU 替代路径 / shim / 跳过 feature | 写 Python 层 shim 在 EasyR1 fork 里（档 1） |
| **D** | CUDA-only 阻塞 | 必须要这个包才能跑，但 NPU 生态没有 → **P2 场景** | 建任务：档 2 委托 kernel 项目、或档 3 提需求、或档 1 自己做 Python 移植 |
| **E** | 纯 Python / CPU | 跟 accelerator 无关 | 直接装 |

---

## EasyR1 master 20 个声明依赖

以 v1 image（`verl-8.5.0-a3`，head `81dc6f925b40`，CANN 8.5.0 + torch_npu 2.8.0）为目标 base。每行标 image 里是否已装。

| # | Package | EasyR1 pin | v1 image 版本 | 档 | 我们做了什么 |
|---|---|---|---|---|---|
| 1 | `accelerate` | unpinned | 1.13.0 | **E** | 跟 NPU 无关，装了就能用 |
| 2 | `codetiming` | unpinned | 1.4.0 | **E** | 纯 Python 计时工具 |
| 3 | `datasets` | unpinned | 4.8.4 | **E** | HuggingFace datasets，CPU/磁盘 |
| 4 | `flash-attn` | `>=2.4.3` | ❌ 不装 | **C** | **CUDA-only**。NPU 走 `transformers.integrations.npu_flash_attention`（transformers 4.57 自带）+ 纯 torch 重写 bert_padding helpers（`verl/utils/npu_flash_attn_utils.py`）。见 commit `da2487f` / `fbaa983`。在 EasyR1 fork 里处理 |
| 5 | `liger-kernel` | unpinned | ❌ 不装 | **C** | **CUDA-only Triton kernel**。EasyR1 可以跑时不启用，`requirements.txt` 拆分后挪进 `gpu` extras。见 commit `7ee0f0b` |
| 6 | `mathruler` | unpinned | 0.1.0 | **E** | EasyR1 自带 reward util，纯 Python |
| 7 | `numpy` | unpinned | 1.26.4 | **E** | NPU 生态整体 pin numpy<2 |
| 8 | `omegaconf` | unpinned | 2.3.0 | **E** | 配置管理 |
| 9 | `pandas` | unpinned | 3.0.2 | **E** | 数据处理 |
| 10 | `peft` | unpinned | 0.19.1 | **A** | transformers 配套；自身无 CUDA 硬编码，走 transformers / torch 路径 |
| 11 | `pillow` | unpinned | ❌ 不在 image base，但 transitive 通过 torchvision 装进去 | **E** | 纯 image 处理 |
| 12 | `pyarrow` | `>=15.0.0` | 23.0.1 | **E** | 列式存储，CPU |
| 13 | `pylatexenc` | unpinned | 2.10 | **E** | LaTeX 处理，纯 Python |
| 14 | `qwen-vl-utils` | unpinned | 0.0.14 | **E** | Qwen VL helper，纯 Python |
| 15 | `ray[default]` | unpinned | 2.55.0 | **A** + **C 补丁** | Ray 2.55 自身 framework 支持 NPU custom resource，但有 2 个小 bug 需要 shim：`ASCEND_RT_VISIBLE_DEVICES` 被清（NPU-BUG-002）+ 没有 `num_gpus`-sugar 对应（NPU-CP-003）。我们的 `ray-npu-shim` skill 包了 |
| 16 | `tensordict` | unpinned | 0.10.0 | **A** | torch 扩展，走 torch_npu |
| 17 | `torchdata` | unpinned | 0.11.0 | **A** | torch 扩展 |
| 18 | `transformers` | `>=4.54.0,<5.0.0` | 4.57.6 | **A** | transformers 4.57 已有 `integrations/npu_flash_attention.py`（upstream 内置 NPU FA 支持）。EasyR1 用这条路径 |
| 19 | `vllm` | `>=0.8.0` | 0.13.0+empty（壳）+ `vllm_ascend 0.13.1` | **B** | vllm 本身有一个空壳版本，真实运行走 `vllm-ascend`。存在 2 个 API rename 需要 shim（NPU-CP-002 / CP-004），在 EasyR1 fork 里已处理 |
| 20 | `wandb` | unpinned | 0.26.0 | **E** | logging 工具 |

---

## 分档统计

| 档 | 数量 | 占比 | 涉及包 |
|---|---|---|---|
| A（NPU 原生） | 5 | 25% | peft, ray[default]*, tensordict, torchdata, transformers |
| B（NPU 移植版本） | 1 | 5% | vllm → vllm-ascend |
| C（CUDA-only 能绕过） | 2 | 10% | flash-attn, liger-kernel |
| **D（阻塞，需 NPU 开发）** | **0** | **0%** | — |
| E（纯 Python/CPU） | 12 | 60% | accelerate, codetiming, datasets, mathruler, numpy, omegaconf, pandas, pillow, pyarrow, pylatexenc, qwen-vl-utils, wandb |

(`*` = ray 同时落在 A + C，因为 framework 支持但有 Python 层小 shim)

**关键结论**：
- **D = 0**。EasyR1 master 在 v1 image（CANN 8.5.0）上跑不需要任何新的 NPU 算子 / 新的 Python 包移植 / 新的 C++ 开发
- **整个 port 只动 EasyR1 代码**（确认了 HANDOVER 和 PORT-GUIDE 里的"只改 EasyR1 一个 repo"说法是对的）
- **`ascend-port` 分支 20 个 commit 全部是 Python 层 shim + 配置** —— 对应上表的 C 和 A+补丁

---

## 对 v2 drill image 的同样审计（transformers 5.3.0.dev0 + vllm_ascend 0.17）

Drill image `verl-8.5.2-a3` 上的依赖分级 **没有改变**：
- transformers 4.57 → 5.3.0.dev0：仍然是档 A（`npu_flash_attention.py` 集成仍在）+ 有 1 个 Python 层 shim（`no_init_weights` import 搬家）
- vllm_ascend 0.13 → 0.17：仍然是档 B + 有 1 个 Python 层 shim（`SamplingParams.eos_token_id` 只读 property）
- torch_npu 2.8 → 2.9：仍然是档 A
- 其他包：不变

**Drill 也 D = 0**。drill 报告验证了 2-step + 20-step 数值稳定。

---

## 未来升级的自动化判断流程

场景 P1（无需 NPU 额外开发）的自动化闭环条件：

1. 给定新 EasyR1 commit + 目标 base image
2. 对 `requirements.txt` 做 diff，列出新增 / 升级的包
3. 对每个新 / 升级包，逐个判断档位：
   - 在 NPU image 里吗？→ A or B
   - 不在但是纯 Python？→ E
   - 不在且 CUDA-only 但 EasyR1 可绕过？→ C
   - 不在且必须用 → **D，场景 P2，要建任务**
4. 只要判断结果没有 D，就走标准 `image-upgrade-drill` 流程
5. 有 D，停下来走 P2 流程（见 [`npu-adaptation-tasks.md`](npu-adaptation-tasks.md)）

**这个流程可以自动化**：做成 `scripts/dep-gap-detect.sh` 或扩展 `image-upgrade-drill` skill 的 Step 2 pre-flight。见 task #25。

---

## 未来新 EasyR1 版本的判断 checklist

拿到新 EasyR1 commit，按这个顺序查：

1. `git diff <prev>..<new> -- requirements.txt pyproject.toml setup.py`
2. 对每个新 / 升级的包：
   - 查 `knowledge/images/<target>.md` 的 pip-freeze 是否已含
   - 如果版本差异，查 upstream `upstream-refs.md` 是否有 NPU 兼容 tag
3. 若全部是 A/B/C/E → 场景 P1，走 `PORT-GUIDE` + `image-upgrade-drill` 就够了
4. 若出现 D → 场景 P2，看 `npu-adaptation-tasks.md`，建任务

---

## 参考来源

- EasyR1 master `requirements.txt`: `upstream/EasyR1/requirements.txt`（at dd71bbd）
- veRL 对照: [`knowledge/verl-master-deps.md`](../knowledge/verl-master-deps.md)
- v1 image 完整 pip freeze: [`knowledge/images/verl-8.5.0-a3.md`](../knowledge/images/verl-8.5.0-a3.md)
- v2 image: [`knowledge/images/verl-8.5.2-a3.md`](../knowledge/images/verl-8.5.2-a3.md)
- v1 port 的 20 个 commit 按 archetype 分类: [`PORT-GUIDE.md §4`](PORT-GUIDE.md)
- NPU 坑目录: [`knowledge/npu-patterns.md`](../knowledge/npu-patterns.md)

---

## 更新维护

这个 doc 要在下列时点更新：
- EasyR1 master 有 `requirements.txt` / `pyproject.toml` 变动
- 切换 target image
- 新建 `scripts/dep-gap-detect.sh` 自动化工具后，改成 "diff 结果归档点"
