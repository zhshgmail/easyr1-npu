# EasyR1 → Ascend 910C (A3) 移植手册

**目标读者**：拿到本仓希望**在 A3 NPU 上把 EasyR1 跑起来**的工程师。

**本手册回答**：
1. 我们移植了 EasyR1 的哪个版本？
2. 依赖 baseline 是什么？（CANN / torch_npu / transformers / vllm / triton-ascend）
3. 我们在源码层做了哪些改动？为什么？
4. 依赖库本身有没有改？（答：只改了 EasyR1 一个 repo，其它库**都是 base image 自带**）
5. 如何在一台干净的 A3 host 上从 0 把 smoke 跑绿？
6. 每一步的验证脚本和预期结果是什么？

配套文档：
- `SKILLS-GUIDE.md` — 如果你想**重做移植**（新版本 EasyR1、或移植别的 RL 框架到 NPU），读这本
- `knowledge/npu-patterns.md` — 23 条稳定 ID 的 NPU 坑点目录
- `scripts/smoke/README.md` — smoke 梯子索引表
- `upstream-refs.md` — 每个 upstream 库在每个 image 下对应的 branch/tag

---

## 1. EasyR1 版本 baseline

| 项 | 值 | 备注 |
|---|---|---|
| Upstream | `hiyouga/EasyR1` | |
| 基线 commit | `dd71bbd` | `Update issue templates (#10956)`，2026-04 月上旬 |
| 我们的 fork | `zhshgmail/EasyR1` (GitHub, **private**) | |
| 发布分支 | `ascend-port` | head `ecce71d`；**这是给别人用的分支** |
| drill 分支 | `ascend-port-transformers-upgrade` | head `2fd9337`；升级演练用，**不要发布** |

**为什么 `ascend-port` 而不是 `main`**：`main` 是 upstream mirror（追 `hiyouga/EasyR1:main`），保持原样便于 rebase。`ascend-port` 是在 `dd71bbd` 基础上加了 20 个 NPU commit 的发布分支。

---

## 2. 依赖 baseline（两套目标 image）

我们验证过**两套 image**。推荐新用户用 **v1（8.5.0）**，稳定、经过完整 smoke 梯子；**v2（8.5.2）** 是升级演练结果，backward-compat 已经验证（见 §3.4）但生产发布以 v1 为准。

### v1 — 生产 image（推荐）

- **Image**: `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`（14.1 GB）
- **CN 镜像**: `quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`（同样内容，中国大陆直连）

| 组件 | 版本 |
|---|---|
| Base OS | Ubuntu 22.04 |
| CANN | 8.5.0 |
| Python | 3.11.14 |
| torch | 2.8.0+cpu |
| torch_npu | 2.8.0 |
| transformers | 4.57.6 |
| vllm | 0.13.0+empty |
| vllm_ascend | 0.13.1.dev18+g2e5f72f92 |
| triton_ascend | 3.2.0 |
| ray | 2.55.0 |
| accelerate | 1.13.0 |
| tensordict | 0.10.0 |
| datasets | 4.8.4 |

完整 pip freeze：`knowledge/images/verl-8.5.0-a3.md`。

### v2 — 升级演练 image（可选）

- **Image**: `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`（24.2 GB）
- 注意：tag 虽然是 `8.5.2` 但 CANN 是 8.5.1；`.2` 是 verl image 的 revision 号

| 组件 | 版本 |
|---|---|
| CANN | 8.5.1 |
| torch_npu | 2.9.0 |
| transformers | 5.3.0.dev0 |
| vllm_ascend | 0.17.0rc2.dev109+g54879467c |

**`ascend-port` 分支同时兼容 v1 和 v2 两套 image**——我们把 v2 drill 中发现的两个 API break 修复用 `try/except` / `hasattr` 写成了 backward-compat 版本后 cherry-pick 进 `ascend-port`。

---

## 3. 我们做了哪些改动？

**只有 EasyR1 一个 repo 被改。**其它所有依赖（torch_npu、vllm_ascend、triton_ascend、transformers、CANN）**没有源码 patch**，都是从 base image 直接消费。

在 EasyR1 `ascend-port` 分支上有 **20 个 commit**（`git log --oneline main..ascend-port`）。按关注点分组如下：

### 3.1 依赖声明层（1 commit）

| Commit | 改动 |
|---|---|
| `7ee0f0b` | 拆分 `requirements.txt` 成 common / gpu / npu 三份；把 flash-attn / liger-kernel / vllm 挪进 `[gpu]` extras；补声明 jinja2 / psutil / pyyaml；tensordict 上限收紧 |

### 3.2 设备访问层（3 commits）

| Commit | 改动 |
|---|---|
| `72b564a` | 新增 `verl/utils/device.py`：`is_npu_available()`、`get_device_name()`、`get_device_module()`、`get_dist_backend()`、`get_default_attn_implementation()`、`get_ray_resource_name()`、`get_visible_devices_env()` |
| `7187b51` | 扫描并替换 10 个文件里的 35 处 `torch.cuda.*` / `init_device_mesh("cuda",...)` / `"cuda"` 字符串 / `device_map="cuda"` |
| `496d198` | 补漏：`load_fsdp_submodule` 里的 `flat_param_to("cuda",...)`；把裸 `current_device()` int 包进 `torch.device(device_name, index)` |

### 3.3 Attention 后端（3 commits）

| Commit | 改动 |
|---|---|
| `6701a50` | `from_pretrained` / `from_config` 里 `attn_implementation` 可配置；NPU 默认 `sdpa`，CUDA 默认 `flash_attention_2` |
| `da2487f` | 把 `flash_attn.bert_padding` 的 helpers 用纯 torch 重写到 `verl/utils/npu_flash_attn_utils.py`；加 `verl/utils/attention_utils.py` 懒加载门面 |
| `ffafa0d` | NPU 配置级 gate 从 `apply_ulysses_patch` 挪到 `_build_model_optimizer`；加 `tests/test_device.py` |

**V2.1 加强**（`fbaa983`）：启用 `padding_free=True`，改用 `transformers.integrations.npu_flash_attention`（upstream 已有的 NPU FA 集成，不是我们新写的）。

### 3.4 Ray NPU 集成（3 commits）

| Commit | 改动 |
|---|---|
| `fb1a223` | 把 NPU 注册为 Ray 自定义 resource；`_check_resource_available` 读 `available_resources()[get_ray_resource_name()]`；placement bundles + actor options 用 `resources={"NPU": n}` 替代 `num_gpus` |
| `59641d4` | `runtime_env.env_vars` 里加 `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`（Ray 2.55+ 否则会清掉 `ASCEND_RT_VISIBLE_DEVICES`） |
| `cc8e794` | 设 `VLLM_ASCEND_ENABLE_NZ=0`（vllm-ascend FRACTAL_NZ 会在 RL param sync 时 drift） |

### 3.5 平台 shim（4 commits）

| Commit | 改动 |
|---|---|
| `cbfe645` | `Dockerfile.npu` 基于 verl-8.5.0-a3 base |
| `cd16649` | Dockerfile.npu 加 `pip install --force-reinstall --no-deps triton-ascend==3.2.0`（修 NPU-BUG-001：base image 的 triton-ascend 装残） |
| `87faff1` | `verl/utils/vllm_utils.py` 兼容 vllm 0.13 的 `vllm.lora.models` → `vllm.lora.lora_model` 改名 |
| `2d8ee2c` | `verl/workers/sharding_manager/fsdp_vllm.py` 兼容 vllm 0.13 的 `get_tensor_model_parallel_group` → `get_tp_group` 改名 |

### 3.6 Smoke 脚本 + backward-compat（6 commits）

| Commit | 改动 |
|---|---|
| `906215d` | V1.4 2-chip GRPO smoke |
| `72a7f22` | V1.5 4-chip GRPO smoke |
| `fbaa983` | V2.1 padding_free=True 工作流（NPU-CP-007） |
| `75bad74` | V2.1 smoke 关掉 `use_torch_compile`（NPU-BUG-003 triton-ascend inductor crash） |
| `9e971f0` | V1.6 → V2.1 重命名（统一 milestone-vs-level 命名） |
| `6f8197f` | V2.2（4-chip + ulysses_size=2 + padding_free=True） |
| `1f716ea` | **v2 drill cherry-pick**：transformers 5.x 的 `no_init_weights` import try/except（transformers <5 和 ≥5 都可用） |
| `ecce71d` | **v2 drill cherry-pick**：vllm 0.18 `SamplingParams.eos_token_id` 变成只读 `@property`，contextmanager 里 `hasattr` + `descriptor.fset is None` 跳过（vllm <0.18 和 ≥0.18 都可用） |

### 3.7 改动类型总结

20 个 commit 可以归成 5 个 archetype（下一次移植会按相似比例复现）：

| Archetype | 本次 commit 数 | 例子 |
|---|---|---|
| 设备 dispatch（把硬编码 CUDA 换成 accelerator-aware helper） | 3 | NPU-CP-001 系列 |
| 版本 compat shim（上游 API 改名） | 2 | NPU-CP-002、NPU-CP-004 |
| Cross-cutting runtime registration（init 时告诉框架 NPU 存在） | 3 | NPU-CP-003 (Ray)、NPU-BUG-002 (visibility env)、NPU-ENV-002 (NZ knob) |
| Vendoring / imports（把 CUDA-only 包变成可选） | 2 | 重写 padding helpers、transformers FA shim |
| Smoke / 运维（Dockerfile、脚本、chip 检查、bind mount） | 10 | — |

---

## 4. A3 上从 0 把 EasyR1 跑起来

### 4.1 前置条件

- A3 host（Ascend 910C / 910B3，简称 "910_93"）
  - 我们验证过的环境：x86_64 + openEuler 22.03 LTS + kernel 5.10.0-60.18 + glibc 2.34
  - NPU driver ≥ 25.5.0（`npu-smi info` 能看到 ≥16 个 chip，HBM 64 GB/chip）
- Docker ≥ 24
- **至少 2 个空闲 chip**（V1.4 只用 2 chip）；跑 V1.5 以上需要 4 个
- **20+ GB 空闲 HBM**（v1 image 14 GB；留出 pull + run 空间。如果 `/var/lib/docker` 快满了先 `docker images | sort -k 7` 清冗余）
- Host 能访问 HuggingFace（国内用 `HF_ENDPOINT=https://hf-mirror.com` 走镜像；`run-npu-container.sh` 已默认设）

### 4.2 Step 1 — 拉 image

```bash
# 国外 / 有直连
docker pull quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest

# 国内 / GFW 后（推荐）
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
docker tag quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest \
           quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
```

**卡住怎么办**：如果 docker daemon 配了 HTTP 代理（`systemctl cat docker | grep HTTP_PROXY`），把 `quay.nju.edu.cn` 加进 `NO_PROXY` 再 pull。详见 `npu-patterns.md::NPU-OPS-006`。

### 4.3 Step 2 — Clone 我们的 EasyR1 fork

```bash
# 建议放进 $HOME/workspace/ 以跟容器 bind-mount 约定一致
mkdir -p "$HOME/workspace" && cd "$HOME/workspace"
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git
```

（如果 `zhshgmail/EasyR1` 是 private 且你没 access，找项目 owner 加 collaborator，或换成 patch set 的形式分发。）

### 4.4 Step 3 — Build 层叠 image

```bash
cd "$HOME/workspace/EasyR1"
docker build -t easyr1-npu:ascend-port -f Dockerfile.npu .
```

`Dockerfile.npu` 做的事：
- `FROM quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`
- `pip install --force-reinstall --no-deps triton-ascend==3.2.0`（修 NPU-BUG-001）
- `pip install -r requirements-npu.txt`（EasyR1 自己的 npu extras，没有 flash-attn / liger）

### 4.5 Step 4 — 下载模型

```bash
# 确保 /data/$USER 存在且 writable
mkdir -p "/data/$USER/models" "/data/$USER/hf-cache"

# 走 hf-mirror（国内）
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --local-dir "/data/$USER/models/Qwen2-0.5B-Instruct"
```

### 4.6 Step 5 — Clone 本仓（为了拿 `run-npu-container.sh`）

```bash
cd "$HOME/workspace"
git clone https://gitcode.com/zhengshencn_hwca/easyr1-npu.git
```

### 4.7 Step 6 — 跑 smoke 梯子

**先看 chip 占用** —— A3 是共享机器，`run-npu-container.sh` 自带 `npu-smi` 检查，但你也可以手动先看：

```bash
npu-smi info
```

找 4 个 AICore=0%、HBM<5 GB 的 chip。下面以 chip 0,1,2,3 为例。

**V1.4（2-chip GRPO 2-step）**：

```bash
cd "$HOME/workspace/easyr1-npu"
bash scripts/run-npu-container.sh \
  --chips 0,1 \
  --live-source "$HOME/workspace/EasyR1" \
  -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh 2>&1 \
  | tee "/tmp/$USER/V1.4.log"
```

**期望**：走完 2 step，entropy_loss step1 ≈ 0.991、step2 ≈ 1.263，grad_norm ≈ 1.43，checkpoint 写在 `/tmp/$USER/easyr1_smoke_ckpt/`。

**V1.5（4-chip GRPO 2-step）**：

```bash
bash scripts/run-npu-container.sh \
  --chips 0,1,2,3 \
  --live-source "$HOME/workspace/EasyR1" \
  -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke_4chip.sh 2>&1 \
  | tee "/tmp/$USER/V1.5.log"
```

**期望**：比 V1.4 快 ~1.7×；跨 chip HCCL 通；ranks 0-3 全部走到 step 2 并 save。

**V2.1（padding_free=True）** 和 **V2.2（+ ulysses_size=2）**：见 `scripts/smoke/README.md` 和 EasyR1 fork 的对应 example 脚本。

全部脚本索引：`scripts/smoke/README.md`。

### 4.8 挂了怎么办

按顺序试：

1. **立刻 `grep` `knowledge/npu-patterns.md`** —— 23 条稳定 ID 里大概率已经有
2. 检查 `npu-smi info` 看 chip 状态 —— 是不是被别人抢了
3. 看 `knowledge/images/verl-8.5.0-a3.md` vs `pip freeze`（在容器内跑）—— 是不是 image drift
4. 看 `docs/porting-journal.md` —— 过程中遇到的坑都记了
5. 如果这些都没命中，那就是**新坑** —— 按 `npu-patterns.md` 的 schema 加一条，commit 到本仓

---

## 5. 常见问题

### 5.1 能不能不用 docker？

**不推荐**。CANN 依赖一堆 host-level 安装（driver + toolkit + nnal/atb），版本得和 torch_npu 严格对齐。base image 把这些都装好了，自己装容易 drift。真要裸装，按 `upstream-refs.md` 的 ref 表找对应版本，CANN 从 `gitcode.com/cann` 拉。

### 5.2 能不能在 H100 / GPU 上跑？

本手册只覆盖 NPU port。GPU 路径用 upstream 的 `pip install -r requirements.txt[gpu]` 就行，不需要我们的 fork。

### 5.3 想要更大的模型 / 更长上下文？

目前只 smoke 过 Qwen2-0.5B、context 不超过 2k。没跑过多节点。这些是 `DELIVERABLE.md` 里标的 known debt。

### 5.4 v2 image（8.5.2）怎么用？

把 §4.2 的 pull、§4.4 的 build 换成 v2 image（tag `verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`）和对应 Dockerfile（drill 分支上有 `Dockerfile.npu-852`）。**`ascend-port` 代码已经 backward-compat**，smoke 脚本不用改。但请注意 v2 还没跑过完整 V1.1→V2.2 梯子，只跑了 2-step + 20-step drill；有 corner case 欢迎报。

### 5.5 CANN 版本跟 torch_npu 对不上怎么办？

**别手动升级其中一个。**换一个我们已验证的 image（v1 或 v2）。如果必须升级，用 `skills/image-upgrade-drill/` 走完整演练流程，见 `SKILLS-GUIDE.md`。

---

## 6. 验证清单（signoff）

移植跑通的标志：

- [ ] V1.1 smoke 过（device accessor 正确）
- [ ] V1.3 smoke 过（vllm_ascend rollout 生成合理文本）
- [ ] V1.4 smoke 过（GRPO 2-step，entropy_loss 匹配基线）
- [ ] V1.5 smoke 过（4-chip HCCL 正常）
- [ ] V2.1 smoke 过（padding_free=True）
- [ ] V2.2 smoke 过（+ ulysses_size=2）

如果你的目的是**用 EasyR1 做 RL 训练**，V1.4 + V2.2 跑绿就可以了，后面自己加 max_steps / 换模型 / 换数据集。如果你的目的是**验证移植质量**，整个梯子都要走一遍。

---

## 7. 相关文档

- **重做移植**（新 EasyR1 / 新 image / 新 RL 框架）：`SKILLS-GUIDE.md`
- **每个坑的 stable ID 详解**：`knowledge/npu-patterns.md`
- **当前项目状态 + 未结工作**：`HANDOVER.md`
- **正式 sign-off 报告**：`DELIVERABLE.md`
- **移植日志**（按时间）：`porting-journal.md`
- **v2 升级演练报告**：`transformers-upgrade-drill.md`
