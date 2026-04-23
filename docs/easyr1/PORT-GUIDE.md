# EasyR1 → Ascend 910C (A3) 移植手册 — v1 生产路径

**目标读者**：想**在 A3 NPU 上把 EasyR1 master 跑起来**的工程师。

**本手册范围**：只覆盖已验证的生产路径 —— **v1 = CANN 8.5.0 image + EasyR1 master + `ascend-port` 分支**。

**不覆盖**：transformers 升级 / CANN 新版本 / 新 image 切换 → 见 [`UPGRADE-DRILL-STATUS.md`](UPGRADE-DRILL-STATUS.md)
**不覆盖**：从 0 重做移植流程 → 见 [`SKILLS-GUIDE.md`](SKILLS-GUIDE.md)

---

## 1. TL;DR — 依赖基线 + 为什么不需要改 upstream

**完整依赖 baseline 全部来自单个 docker image**：

- Image: `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`（14.1 GB）
- 国内镜像: `quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`

里面已经装好的关键组件：

| 组件 | 版本 | 来源 |
|---|---|---|
| Base OS | Ubuntu 22.04 | image |
| CANN | 8.5.0 | image (Ascend 官方 base) |
| Python | 3.11.14 | image |
| torch | 2.8.0+cpu | image |
| **torch_npu** | **2.8.0** | image（NPU pytorch plugin） |
| **transformers** | **4.57.6** | image |
| vllm | 0.13.0+empty | image |
| **vllm_ascend** | **0.13.1.dev18+g2e5f72f92** | image |
| **triton_ascend** | **3.2.0** | image（但装残，我们 Dockerfile 里 force-reinstall） |
| ray | 2.55.0 | image |
| accelerate | 1.13.0 | image |
| tensordict | 0.10.0 | image |
| datasets | 4.8.4 | image |

完整 pip freeze：[`knowledge/images/verl-8.5.0-a3.md`](../knowledge/images/verl-8.5.0-a3.md)。

### 为什么不需要改 upstream 库？

EasyR1 的所有依赖被 base image 的 verl-A3 build 完整覆盖，**我们只需要动 EasyR1 自己**。具体讲：

1. **EasyR1 所有 RL / 模型训练依赖**（transformers、vllm_ascend、torch_npu、ray、tensordict 等）**image 全部装好**，版本和 `torch_npu 2.8` 严格对齐——Ascend 团队在 base image 维护层已经做好了这套兼容性工作。
2. **EasyR1 声明但不适用于 NPU 的依赖**（`flash-attn`、`liger-kernel`）**根本装不上也不需要**——它们是 CUDA-only 的 Triton kernel，NPU 上用 `transformers.integrations.npu_flash_attention`（**transformers 4.57 里已经自带**，不是我们新写的）替代。
3. **NPU 特定的 API 不完善的地方**（e.g. `flash_attn.bert_padding`）**我们把需要的 helper 用纯 torch 重写到 EasyR1 内部**（`verl/utils/npu_flash_attn_utils.py`）——这个文件是 EasyR1 的一部分，不是改 upstream 包。
4. **vllm 0.13 跟 0.12 的 API rename**（`vllm.lora.models`、`get_tensor_model_parallel_group`）**在 EasyR1 代码里用 `try/except` 适配**——同样是改 EasyR1 不是改 vllm。

**结论**：**只有 `zhshgmail/EasyR1:ascend-port` 一个 repo 被修改。torch_npu / vllm_ascend / transformers / triton_ascend 等 upstream 库都是从 image 直接消费，源码零 patch。**

所以，跑 EasyR1：
- **用户需要**：本仓（脚本）+ EasyR1 fork（代码）+ image
- **用户不需要**：clone torch-npu / vllm-ascend / transformers / triton-ascend 任何 upstream 源码

---

## 2. EasyR1 版本 baseline

| 项 | 值 | 备注 |
|---|---|---|
| Upstream | `hiyouga/EasyR1` | GitHub |
| Upstream baseline commit | `dd71bbd` | `Update issue templates (#10956)`，2026-04 月上旬 |
| 我们的 fork | `zhshgmail/EasyR1` (GitHub, **public**) | 20 个 NPU port commit |
| **发布分支（你要用这个）** | **`ascend-port`** | head `ecce71d` |

**为什么是 `ascend-port` 不是 `main`**：`main` 是 upstream `hiyouga/EasyR1:main` 的 mirror，便于未来 rebase；没有任何 NPU 改动。**直接 clone 默认分支拿不到 NPU 代码。**

`ascend-port` 分支上 20 个 commit 的**分类和动机**见 §4。

---

## 3. 前置条件

- Ascend 910C (A3) NPU host（16 个 chip，通常）
  - 我们验证过：x86_64 + openEuler 22.03 LTS + kernel 5.10.0-60.18 + glibc 2.34
  - NPU driver ≥ 25.5.0（`npu-smi info` 能看到 chip，HBM 64 GB/chip）
- Docker ≥ 24
- **V1.4 smoke：2 个空闲 chip**；V1.5+：**4 个空闲**
- **≥ 20 GB 空闲磁盘**（image 14 GB）
- 能访问 HuggingFace（国内用 `HF_ENDPOINT=https://hf-mirror.com`，runner 脚本默认已设）
- **你对目标 A3 host 有 ssh + docker 权限**

**共享 host 礼仪**：A3 通常是共享机器。跑之前一定 `npu-smi info` 看哪些 chip 空，不抢别人在用的 chip。`run-npu-container.sh` 自带 chip 占用检查。

---

## 4. 我们改了什么 —— 20 个 commit 按关注点分组

完整 log：`git log --oneline main..ascend-port`（在 clone 后的 `zhshgmail/EasyR1` 里）。

| Archetype | commit 数 | 用途 |
|---|---|---|
| 设备 dispatch（CUDA 调用 → accelerator-aware） | 3 | `verl/utils/device.py` + 35 处 `torch.cuda.*` / `"cuda"` 字符串替换 |
| Attention 后端 | 3 | NPU 默认 sdpa；`flash_attn.bert_padding` helper 纯 torch 重写；NPU config gate |
| Ray NPU 集成 | 3 | Ray 自定义 resource `"NPU"`；`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`；`VLLM_ASCEND_ENABLE_NZ=0` |
| 平台 shim | 4 | `Dockerfile.npu`；triton-ascend force-reinstall；vllm 0.13 的 2 处 API rename 适配 |
| Smoke + backward-compat | 6 | V1.4 / V1.5 / V2.1 / V2.2 smoke 脚本；`no_init_weights` / `SamplingParams` 只读 property 的 backward-compat（兼容 transformers 4/5、vllm 0.13/0.18） |
| 依赖声明 | 1 | 拆 `requirements.txt` 为 common/gpu/npu extras |

**关键 commit 列表（想细看具体改动时用）**：

| Commit | 改动 |
|---|---|
| `7ee0f0b` | 拆 `requirements.txt` 为 common / gpu / npu extras |
| `72b564a` | 新增 `verl/utils/device.py`（accelerator accessor layer） |
| `7187b51` | 批量替换 10 个文件 35 处 `torch.cuda.*` |
| `496d198` | 补漏 `flat_param_to("cuda",...)` 等 |
| `6701a50` | `attn_implementation` 可配置；NPU 默认 sdpa |
| `da2487f` | 纯 torch 重写 `flash_attn.bert_padding` helpers |
| `ffafa0d` | NPU config-level gate |
| `fb1a223` | Ray 注册 NPU 为自定义 resource |
| `59641d4` | `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` |
| `cc8e794` | `VLLM_ASCEND_ENABLE_NZ=0` |
| `cbfe645` | `Dockerfile.npu` |
| `cd16649` | triton-ascend force-reinstall（修 NPU-BUG-001） |
| `87faff1` | vllm 0.13 `vllm.lora.models` rename 适配 |
| `2d8ee2c` | vllm 0.13 `get_tensor_model_parallel_group` rename 适配 |
| `906215d` | V1.4 smoke |
| `72a7f22` | V1.5 smoke |
| `fbaa983` | V2.1 padding_free=True（用 transformers 原生 `npu_flash_attention`） |
| `75bad74` | V2.1 smoke 关 `use_torch_compile`（绕 NPU-BUG-003） |
| `6f8197f` | V2.2 （+ ulysses_size=2） |
| `1f716ea`, `ecce71d` | transformers 5 / vllm 0.18 的 backward-compat cherry-pick（从 drill 分支过来的，但**不影响 transformers 4.57 / vllm 0.13**，backward-compat 写法） |

每个 archetype 的详细讲解 → [`knowledge/npu-patterns.md`](../knowledge/npu-patterns.md)（24 条 stable ID 目录）

---

## 5. 从 0 在一台 A3 host 上跑通 V1.4 smoke

### 5.1 Step 1 — Pull image（5-15 分钟，看网速）

```bash
# 国内推荐（NJU mirror 直连）
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest

# 让后面的命令不用带长 tag：
docker tag quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest \
           quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
```

**卡住怎么办**：如果 docker daemon 配了 HTTP 代理（`systemctl cat docker | grep HTTP_PROXY`），把 `quay.nju.edu.cn` 加进 `NO_PROXY` 再 pull。详见 [`knowledge/npu-patterns.md`](../knowledge/npu-patterns.md) 的 `NPU-OPS-006`。

### 5.2 Step 2 — Clone 本仓 + EasyR1 fork

```bash
mkdir -p "$HOME/workspace" && cd "$HOME/workspace"

# 本仓（拿 runner 脚本 + 部署 skills 可选）
git clone https://github.com/zhshgmail/easyr1-npu.git

# EasyR1 fork 的 ascend-port 分支（⚠️ 一定要指定 -b ascend-port，否则拿到的是无 NPU 改动的 upstream mirror）
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git
```

两个仓都是 public，直接 clone。如果访问 github 慢，目前没有 CN mirror；后续可能建 gitcode 镜像。

### 5.3 Step 3 — Build 层叠 image（3-5 分钟）

```bash
cd "$HOME/workspace/EasyR1"
docker build -t easyr1-npu:ascend-port -f Dockerfile.npu .
```

`Dockerfile.npu` 做的事：
- `FROM quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`
- `pip install --force-reinstall --no-deps triton-ascend==3.2.0`（修 NPU-BUG-001 triton 装残）
- `pip install -r requirements-npu.txt`（EasyR1 的 npu extras，无 flash-attn / liger）

### 5.4 Step 4 — 下载模型（5-10 分钟）

```bash
mkdir -p "/data/$USER/models" "/data/$USER/hf-cache"

HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --local-dir "/data/$USER/models/Qwen2-0.5B-Instruct"
```

### 5.5 Step 5 — 查 chip 占用

```bash
npu-smi info
```

找 4 个 AICore=0%、HBM-Usage < 5 GB 的 chip。下面以 `0,1,2,3` 为例；你可能需要换别的。

### 5.6 Step 6 — 跑 V1.4 smoke（2-chip GRPO 2-step，约 8 分钟）

```bash
cd "$HOME/workspace/easyr1-npu"
mkdir -p "/tmp/$USER/easyr1-logs"

bash scripts/run-npu-container.sh \
  --chips 0,1 \
  --live-source "$HOME/workspace/EasyR1" \
  -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh 2>&1 \
  | tee "/tmp/$USER/easyr1-logs/V1.4.log"
```

**期望结果（在 v1 CANN 8.5.0 image 上跑 V1.4 的基准）**：
- 走完 2 step
- `entropy_loss` step 1 ≈ **0.991**，step 2 ≈ **1.263**
- `grad_norm` step 1 ≈ **1.43**
- checkpoint 写在 `/tmp/$USER/easyr1_smoke_ckpt/`

如果 `entropy_loss` step 1 漂移超过 **±5%**（< 0.94 或 > 1.04），说明依赖 drift 了，停下来用 `skills/npu-image-inspect/` 对比一下 image 是不是你预期的那个。

**注意**：上面的数值是**在 CANN 8.5.0 image（本手册推荐）上**的基准。如果你跑在别的 image（e.g. CANN 8.5.1 drill image），基准会**不同**（V1.4 在 8.5.2 drill image 上 step 1 是 ~1.434，是正常的，不是 regression）—— 见 [`knowledge/smoke-ladder-convention.md`](../knowledge/smoke-ladder-convention.md) 的分 image 基准表。

### 5.7 继续走 smoke 梯子（可选）

V1.4 是最小验证。要完整验证：继续 V1.5（4-chip）→ V2.1（padding_free）→ V2.2（+ ulysses）。每级脚本见 [`scripts/smoke/README.md`](../scripts/smoke/README.md) 里的索引表。

如果目的是**用 EasyR1 做实际 RL 训练**，V1.4 过了就可以自己改 `max_steps` / 换模型 / 换数据集。

---

## 6. 遇到问题

按顺序试：

1. **立刻 grep [`knowledge/npu-patterns.md`](../knowledge/npu-patterns.md)** —— 24 条 stable ID 命中率很高
2. 检查 `npu-smi info`：chip 是不是被别人抢了？
3. 检查容器内 pip freeze 是否匹配 [`knowledge/images/verl-8.5.0-a3.md`](../knowledge/images/verl-8.5.0-a3.md) —— image drift?
4. 看 [`porting-journal.md`](porting-journal.md) —— 我们踩过的坑都记了
5. 都没命中 → 新坑 → 按 `npu-patterns.md` schema 加一条

---

## 7. FAQ

### 7.1 能不能不用 docker？

**不推荐**。CANN 跟 torch_npu 的版本对齐很紧，base image 已经验证过；自己装容易 drift。真要裸装，按 [`knowledge/upstream-refs.md`](../knowledge/upstream-refs.md) 的 ref 表找对应版本，CANN 从 `gitcode.com/cann` 拉。

### 7.2 能不能在 GPU 上跑？

本手册只讲 NPU。GPU 路径直接用 upstream EasyR1 + `pip install -r requirements.txt[gpu]`，不需要我们的 fork。

### 7.3 更大的模型 / 更长上下文？

v1 只 smoke 过 Qwen2-0.5B + context ≤ 2k，单节点。更大 scale 是 [`DELIVERABLE.md`](DELIVERABLE.md) 的 known debt。

### 7.4 我想用 transformers 5 / CANN 8.5.1 / 新 image 怎么办？

→ [`UPGRADE-DRILL-STATUS.md`](UPGRADE-DRILL-STATUS.md)

短版本：drill 分支验证过兼容，但**不是 v1 发布路径**。生产用 v1。

### 7.5 CANN / torch_npu / 其他库的版本跟不上怎么办？

**别手动升级单个**，换一个我们已验证的 image（version 对齐紧密，单独升会 drift）。

如果新 EasyR1 或新模型**要求**某个版本还没被我们验证过的依赖：
1. 先用 `skills/image-upgrade-drill/` 走演练流程评估 —— 产出 drill report 判断是否能切
2. 如果该依赖 NPU 生态还没适配（没有 NPU 版本、或 NPU 版本有 bug、或 NPU 完全没跟进），**这是需要驱动的 NPU 适配任务**，不是"不在 scope" —— 建任务到 `docs/easyr1/npu-adaptation-tasks.md`（待建）。按三档责任划分：
   - **档 1**（本仓直接做）：Python 层 shim / fork、向 vllm-ascend / triton-ascend / torch_npu 的 Python 层 提 issue / PR
   - **档 2**（委托姐妹项目）：kernel 实现 → `ascend-fused-accuracy-probe` / `a5_ops` / A3 kernel 仓。本仓识别 + track
   - **档 3**（只能提需求）：CANN runtime C 层框架 bug —— 提 issue 给 Ascend 团队 + 做 workaround

详见 [`SKILLS-GUIDE.md §6`](SKILLS-GUIDE.md)。

---

## 8. 验证清单

跑通的标志：

- [ ] `docker pull` 成功
- [ ] `docker build -t easyr1-npu:ascend-port` 成功
- [ ] `npu-smi info` 能看到 chip
- [ ] V1.4 smoke 跑完 2 step，entropy_loss 在 baseline ±5% 内
- [ ] checkpoint 正确写入 `/tmp/$USER/easyr1_smoke_ckpt/`

到这里你就可以基于 V1.4 smoke 脚本改参数跑自己的训练任务了。

---

## 9. 相关文档

- **重做移植 / 用 skills 自动化**：[`SKILLS-GUIDE.md`](SKILLS-GUIDE.md)
- **我想用 transformers 5 / 新 image**：[`UPGRADE-DRILL-STATUS.md`](UPGRADE-DRILL-STATUS.md)
- **NPU 坑 stable ID 目录**：[`knowledge/npu-patterns.md`](../knowledge/npu-patterns.md)
- **当前项目状态**：[`HANDOVER.md`](HANDOVER.md)
- **正式 sign-off 报告**：[`DELIVERABLE.md`](DELIVERABLE.md)
