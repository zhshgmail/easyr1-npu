# EasyR1 → Ascend 910C (A3) 移植手册 — v1 生产路径

**目标读者**：想在 A3 NPU 上把 EasyR1 master 跑起来的工程师。

**本手册范围**：v1 = `verl-8.5.0-a3` image + EasyR1 fork 的 `ascend-port` 分支。这是已验证最稳定的路径。

**v2 集成路径（最新）**：[`PORT-GUIDE-v2-integrated.md`](PORT-GUIDE-v2-integrated.md)。

---

## 1. 依赖基线

完整依赖来自单个 docker image：

- 镜像：`quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`（14.1 GB）
- 国内镜像：`quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`

| 组件 | 版本 |
|---|---|
| Base OS | Ubuntu 22.04 |
| CANN | 8.5.0 |
| Python | 3.11.14 |
| torch | 2.8.0+cpu |
| torch_npu | 2.8.0 |
| transformers | 4.57.6 |
| vllm | 0.13.0+empty |
| vllm_ascend | 0.13.1.dev18 |
| triton_ascend | 3.2.0（image 装残；Dockerfile.npu 里 force-reinstall） |
| ray | 2.55.0 |
| accelerate | 1.13.0 |
| tensordict | 0.10.0 |

完整 pip freeze：[`knowledge/images/verl-8.5.0-a3.md`](../../knowledge/images/verl-8.5.0-a3.md)。

### 1.1 为什么 v1 不需要改 upstream

EasyR1 的所有依赖被 base image 覆盖，**只需动 EasyR1 自己**：

1. RL / 模型训练依赖（transformers / vllm_ascend / torch_npu / ray / tensordict）image 全装好。
2. CUDA-only 依赖（flash-attn / liger-kernel）NPU 不装，用 transformers 4.57 自带的 `transformers.integrations.npu_flash_attention`。
3. NPU API 缺口（如 `flash_attn.bert_padding`）用纯 torch 重写到 EasyR1 内部（`verl/utils/npu_flash_attn_utils.py`）。
4. vllm 0.13 vs 0.12 的 API rename 在 EasyR1 里 try/except 适配。

**结论**：v1 仅 `zhshgmail/EasyR1:ascend-port` 一个 repo 被修改。

---

## 2. EasyR1 版本

| 项 | 值 |
|---|---|
| Upstream | `hiyouga/EasyR1` |
| Upstream baseline | `dd71bbd` |
| 本仓 fork | [`zhshgmail/EasyR1`](https://github.com/zhshgmail/EasyR1)（public） |
| 发布分支 | **`ascend-port`**（head `ecce71d`，20 个 NPU port commit） |

**为什么是 `ascend-port` 不是 `main`**：`main` 是 upstream mirror，无 NPU 改动。**直接 clone 默认分支拿不到 NPU 代码**。

---

## 3. 前置条件

- A3 host（16 chip 通常）；x86_64 + openEuler 22.03 LTS + kernel 5.10；NPU driver ≥ 25.5.0
- Docker ≥ 24
- V1.4 smoke：2 个空闲 chip；V1.5+：4 个空闲
- ≥ 20 GB 空闲磁盘
- HuggingFace 可达（国内：`HF_ENDPOINT=https://hf-mirror.com`，runner 默认已设）
- ssh + docker 权限

A3 通常是共享机。跑前必 `npu-smi info` 看哪些 chip 空。`run-npu-container.sh` 自带占用检查。

---

## 4. 改动概览（20 个 commit）

| Archetype | 数 | 用途 |
|---|---|---|
| 设备 dispatch（CUDA → accelerator-aware） | 3 | `verl/utils/device.py` + 35 处 `torch.cuda.*` 替换 |
| Attention 后端 | 3 | NPU 默认 sdpa；`flash_attn.bert_padding` 纯 torch 重写；config gate |
| Ray NPU 集成 | 3 | Ray `"NPU"` 自定义 resource；`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`；`VLLM_ASCEND_ENABLE_NZ=0` |
| 平台 shim | 4 | `Dockerfile.npu`；triton-ascend force-reinstall；vllm 0.13 的 2 处 rename 适配 |
| Smoke + backward-compat | 6 | V1.4/V1.5/V2.1/V2.2 smoke；transformers 4/5 + vllm 0.13/0.18 兼容 |
| 依赖声明 | 1 | `requirements.txt` 拆 common/gpu/npu extras |

完整 commit log：clone 后 `git log --oneline main..ascend-port`。

---

## 5. 在 A3 上跑通 V1.4 smoke

### 5.1 Pull image

```bash
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
docker tag quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest \
           quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
```

如果 docker daemon 配了 HTTP proxy 卡住：把 `quay.nju.edu.cn` 加进 `NO_PROXY` 后重试。详见 [NPU-OPS-006](../../knowledge/npu-patterns.md)。

### 5.2 Clone 本仓 + EasyR1 fork

```bash
mkdir -p "$HOME/workspace" && cd "$HOME/workspace"
git clone https://github.com/zhshgmail/easyr1-npu.git
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git
```

### 5.3 Build 层叠 image

```bash
cd "$HOME/workspace/EasyR1"
docker build -t easyr1-npu:ascend-port -f Dockerfile.npu .
```

`Dockerfile.npu`：
- `FROM` 基础 image
- `pip install --force-reinstall --no-deps triton-ascend==3.2.0`（修 NPU-BUG-001）
- `pip install -r requirements-npu.txt`

### 5.4 下载模型

```bash
mkdir -p "/data/$USER/models" "/data/$USER/hf-cache"
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --local-dir "/data/$USER/models/Qwen2-0.5B-Instruct"
```

### 5.5 查 chip 占用

```bash
npu-smi info
```

找 AICore=0% 的 chip。

### 5.6 跑 V1.4 smoke（约 8 分钟）

```bash
cd "$HOME/workspace/easyr1-npu"
mkdir -p "/tmp/$USER/easyr1-logs"

bash src/scripts/run-npu-container.sh \
  --chips 0,1 \
  --image easyr1-npu:ascend-port \
  --live-source "$HOME/workspace/EasyR1" \
  -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh 2>&1 \
  | tee "/tmp/$USER/easyr1-logs/V1.4.log"
```

**期望（在 v1 CANN 8.5.0 image 上）**：
- 走完 2 step
- `entropy_loss` step 1 ≈ **0.991**，step 2 ≈ **1.263**
- `grad_norm` step 1 ≈ **1.43**
- checkpoint 写在 `/tmp/$USER/easyr1_smoke_ckpt/`

`entropy_loss` step 1 漂移超 ±5% → 依赖 drift，对照 [`knowledge/upstream-refs.md`](../../knowledge/upstream-refs.md) 检查 image。

不同 image 上基准不同——见 [`knowledge/smoke-ladder-convention.md`](../../knowledge/smoke-ladder-convention.md) 的分 image 基准表。

---

## 6. 遇到问题

按顺序：

1. grep [`knowledge/npu-patterns.md`](../../knowledge/npu-patterns.md)（29 条 stable ID 命中率高）
2. `npu-smi info`：chip 被抢？
3. 容器内 pip freeze 与 [`knowledge/images/verl-8.5.0-a3.md`](../../knowledge/images/verl-8.5.0-a3.md) 对比
4. 都没命中 → 新坑 → 按 `npu-patterns.md` schema 加一条 + 提 issue

---

## 7. FAQ

### 7.1 能不能不用 docker？

不推荐。CANN ↔ torch_npu 版本对齐紧；base image 已验证。要裸装：[`knowledge/upstream-refs.md`](../../knowledge/upstream-refs.md) 找对应版本，CANN 从 `gitcode.com/cann`。

### 7.2 能不能在 GPU 上跑？

本手册只讲 NPU。GPU 路径用 upstream EasyR1 + `requirements.txt[gpu]`。

### 7.3 我想用 newer transformers / vllm / CANN？

走 v2 integrated 路径：[`PORT-GUIDE-v2-integrated.md`](PORT-GUIDE-v2-integrated.md)。已用 vllm 0.20 + torch 2.11 + 4 个 NPU 上游 ascend-port overlay 验证 V1.4 GRPO PASS（2026-04-27 + 2026-04-28 两次独立运行）。

### 7.4 CANN / torch_npu / 其他库版本跟不上？

不要单独手升某一个；换一个已验证的 image。新版本要适配 → 跑对应 day-0 / port skill：[`docs/_meta/SKILLS-USAGE.md`](../_meta/SKILLS-USAGE.md)。

---

## 8. 验证清单

- [ ] `docker pull` 成功
- [ ] `docker build` 成功
- [ ] `npu-smi info` 见 chip
- [ ] V1.4 smoke 走完 2 step，entropy_loss baseline ±5% 内
- [ ] checkpoint 写入 `/tmp/$USER/easyr1_smoke_ckpt/`

通过即可基于此脚本改 `max_steps` / 换模型 / 换数据集做实际训练。

---

## 见也

- [`PORT-GUIDE-v2-integrated.md`](PORT-GUIDE-v2-integrated.md) — v2 集成路径
- [`docs/_meta/ARCHITECTURE.md`](../_meta/ARCHITECTURE.md) — 整体架构与流程
- [`docs/_meta/SKILLS-USAGE.md`](../_meta/SKILLS-USAGE.md) — 升级工具链
- [`docs/_meta/UPSTREAM_FORKS.md`](../_meta/UPSTREAM_FORKS.md) — fork ledger
- [`knowledge/npu-patterns.md`](../../knowledge/npu-patterns.md) — NPU 坑 29 条 stable ID
