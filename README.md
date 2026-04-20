# easyr1-npu

**把 EasyR1（`hiyouga/EasyR1`）移植到 Ascend 910C (A3) NPU，并沉淀一套可复用的"GPU RL 框架→NPU"移植 skills。**

本仓交付两件事：
1. **EasyR1 在 A3 上能跑** —— 代码改动在 [`zhshgmail/EasyR1`](https://github.com/zhshgmail/EasyR1) 的 **`ascend-port`** 分支（20 个 commit）
2. **一套可复用的移植 skills** —— 下一次 EasyR1 版本升级 / CANN 升级 / 别的 RL 框架（OpenRLHF / TRL）移植都能套用

---

## 选你要走的路径

本仓服务**四类用户**，按你的目标选：

| # | 你想要 | 路径 | 读哪本 |
|---|---|---|---|
| 1 | **只想在 A3 上把 EasyR1 跑起来** | 用 v1 已验证发布路径（`ascend-port` 分支 + CANN 8.5.0 image） | [`docs/PORT-GUIDE.md`](docs/PORT-GUIDE.md) |
| 2 | **复现 / 自动化 EasyR1 移植** | 从 0 跑移植流程（用 7 个 skill + `fetch-upstream.sh`） | [`docs/SKILLS-GUIDE.md`](docs/SKILLS-GUIDE.md) |
| 3 | **用带新依赖（transformers 5 / CANN 8.5.1）的 EasyR1** | 看 drill 演练当前状态 + 自行验证 | [`docs/UPGRADE-DRILL-STATUS.md`](docs/UPGRADE-DRILL-STATUS.md) |
| 4 | **复现"把 EasyR1 + 新依赖一起移植到 NPU"的自动流程** | 走 `image-upgrade-drill` skill 7 步流程 | [`docs/SKILLS-GUIDE.md`](docs/SKILLS-GUIDE.md) §8 + [`docs/UPGRADE-DRILL-STATUS.md`](docs/UPGRADE-DRILL-STATUS.md) |

---

## 路径 1：只想把 EasyR1 跑起来（最快 ~30 分钟）

**前置**：A3 host（≥ 2 空闲 chip）+ Docker + HuggingFace 访问（国内走镜像）。

**你不需要**：clone 任何 upstream 库（`torch-npu` / `transformers` 等）。image 里全有。

**仓库**：本仓（`zhshgmail/easyr1-npu`）和 EasyR1 fork（`zhshgmail/EasyR1`）都是 **public**，直接 clone 即可。

```bash
# 1. Pull 镜像（国内走 NJU mirror；国外直接 quay.io/ascend）
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
docker tag quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest \
           quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest

# 2. Clone 本仓（拿 runner 脚本）+ EasyR1 fork 的 ascend-port 分支
mkdir -p "$HOME/workspace" && cd "$HOME/workspace"
git clone https://github.com/zhshgmail/easyr1-npu.git
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git

# 3. Build 层叠 image（~3 分钟）
cd EasyR1 && docker build -t easyr1-npu:ascend-port -f Dockerfile.npu .

# 4. 下载模型
mkdir -p "/data/$USER/models"
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --local-dir "/data/$USER/models/Qwen2-0.5B-Instruct"

# 5. 跑 V1.4 smoke（~8 分钟；期望 entropy_loss step1 ≈ 0.991）
cd "$HOME/workspace/easyr1-npu"
bash scripts/run-npu-container.sh --chips 0,1 \
    --live-source "$HOME/workspace/EasyR1" \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

完整细节 + 期望数值 + 遇到问题怎么办 → [`docs/PORT-GUIDE.md`](docs/PORT-GUIDE.md)

---

## 路径 2：复现 / 自动化 EasyR1 移植

适合：新版本 EasyR1 发布后你想自动化重跑这套移植流程；或要移植另一个 Ray-based RL 框架（OpenRLHF / TRL）到 NPU。

**额外需要**：upstream 库的本地 clone（供 skill grep 参考代码）。

```bash
# 1. Clone 本仓
git clone https://github.com/zhshgmail/easyr1-npu.git
cd easyr1-npu

# 2. 部署 skills 到 Claude Code（在本地开发机上，不是 A3）
bash scripts/install-skills.sh     # symlink skills/* 到 ~/.claude/skills/

# 3. 拉 upstream 参考代码（默认只拉 essential，加 --include-optional 拉全部）
bash scripts/fetch-upstream.sh --include-optional

# 4. 按 SKILLS-GUIDE.md 的 9 步 workflow 走
```

完整 workflow（什么 skill 在什么时候调用、输入输出是什么、决策点在哪）→ [`docs/SKILLS-GUIDE.md`](docs/SKILLS-GUIDE.md)

---

## 路径 3：用新依赖的 EasyR1（transformers 5 / 新 image）

**当前状态**：drill 分支验证过**在新 image（8.5.2）上跑通 2-step + 20-step smoke**，2 个 backward-compat fix 已 cherry-pick 到 `ascend-port` 分支。

**但**：没跑完整 V1.1 → V2.2 smoke 梯子，没做长训练收敛验证。**不推荐生产切过去**。

详细状态 + caveat + 要不要切的决策建议 → [`docs/UPGRADE-DRILL-STATUS.md`](docs/UPGRADE-DRILL-STATUS.md)

---

## 路径 4：复现"EasyR1 + 新依赖自动移植"的流程

适合：NPU 软件栈升级（CANN 9.x、torch_npu 2.10、transformers 6 之类）发布后，你要评估 "我们的 EasyR1 port 能不能跟上新 image"。

**明确的 scope**（路径 4 能做的）：
- 验证新 image 里的一整套新依赖（torch_npu、vllm_ascend、triton_ascend、transformers 等）跟我们的 EasyR1 `ascend-port` 分支是否兼容
- 修复 EasyR1 自己源码里的版本 compat 问题（e.g. transformers 5 改了 import 路径 → EasyR1 里加 try/except）
- 换 base image + rebuild 层叠 image

**不在 scope**（路径 4 不能做的）：
- ❌ **改 NPU 上游库的源码**（torch-npu C++ 层、vllm-ascend / triton-ascend 的 Python/C++）。这些库本项目**只消费 base image 的版本**，不 patch。要改，是 CANN / Ascend 团队或相关 upstream 项目的工作，见 [`SKILLS-GUIDE.md §6 "什么时候这些 skill 不够用"`](docs/SKILLS-GUIDE.md)
- ❌ **同时协调多个 NPU 上游库分支 commit 到 NPU 生态**。本项目的 `upstream-branch-hygiene` skill 只管 EasyR1 fork 的分支
- ❌ **改 CANN 本身**

用 [`image-upgrade-drill`](skills/image-upgrade-drill/SKILL.md) skill 的 7 步演练流程。产物：
- 一份带数字的 drill report（预测 vs 实际成本、LOC 变动、bug probe 结果）
- backward-compat commit 序列，可 cherry-pick 进 `ascend-port`
- 新 stable ID 条目加到 `knowledge/npu-patterns.md`
- PASS / BLOCKED 决策依据

**注意**：这个 skill 的"自动化复现能力"**还没完全证明** —— 2026-04-19 首次用 isolated agent 走完 7 步时卡住了（见 [`docs/UPGRADE-DRILL-STATUS.md`](docs/UPGRADE-DRILL-STATUS.md) §3）。skill 本身内容可用（人 + agent 协作跑过），但"一条命令交给 agent 就出结果"的 end-to-end 自动化还在迭代。

完整 skill 说明 → [`skills/image-upgrade-drill/SKILL.md`](skills/image-upgrade-drill/SKILL.md)
v2 drill 的首次实证报告 → [`docs/transformers-upgrade-drill.md`](docs/transformers-upgrade-drill.md)

---

## `zhshgmail/EasyR1` fork 的分支结构

Fork 上有 4 个分支，每个用途不同。**用户只切 `ascend-port`**，其它都是开发用。

| 分支 | 对应什么 | 用户该用吗？ |
|---|---|---|
| `main` | upstream `hiyouga/EasyR1:main` mirror；无 NPU 改动 | ❌ clone 默认分支会拿不到 NPU 代码 |
| **`ascend-port`** | `main` + 20 个 NPU port commit；已 backward-compat 兼容 v1/v2 两套 image | ✅ **发布分支，就用这个** |
| `ascend-port-transformers-upgrade` | `ascend-port` + 9 个 `[drill]` commit；实验性 Dockerfile + BUG probe 脚本 | ❌ drill 演练分支 |
| `ascend-port-transformers-upgrade-reproduce` | 从 `ascend-port` 派生的 skill 复现验证分支 | ❌ 复现实验分支 |

**关键**：
- `main` ≠ 发布分支。**一定用 `git clone -b ascend-port`**
- drill 分支跟 `ascend-port` **完全隔离**（drill 只加了新文件 `Dockerfile.npu-852`，没改 `Dockerfile.npu`）
- drill 的 backward-compat fix 通过 cherry-pick 进 `ascend-port`，**用 `try/except` + `hasattr` 写法同时兼容 transformers 4/5 和 vllm 0.13/0.18**

---

## 仓库布局

```
easyr1-npu/                           ← 本仓（github.com/zhshgmail/easyr1-npu）
├── README.md                         ← 你在这里
├── CLAUDE.md                         ← 项目指令（给 Claude Code 用）
├── docs/
│   ├── PORT-GUIDE.md                 ← 路径 1：怎么跑起来
│   ├── SKILLS-GUIDE.md               ← 路径 2 & 4：怎么用 skill 自动化移植
│   ├── UPGRADE-DRILL-STATUS.md       ← 路径 3：升级演练当前状态
│   ├── HANDOVER.md                   ← 当前状态 + 未结工作
│   ├── DELIVERABLE.md                ← 正式 sign-off 文档
│   ├── transformers-upgrade-drill.md ← v2 升级演练完整报告（drill 实证）
│   ├── porting-journal.md            ← 时间线日志
│   ├── design.md                     ← 原始需求 + 任务拆解
│   └── ...                           ← 其它分析 / codex review 存档
├── skills/                           ← 7 个可复用 skill
│   ├── npu-image-inspect/
│   ├── npu-code-path-sweep/
│   ├── npu-container-runner/
│   ├── upstream-branch-hygiene/
│   ├── ray-npu-shim/
│   ├── image-upgrade-drill/
│   └── codex-review/
├── scripts/
│   ├── install-skills.sh             ← 部署 skills 到 ~/.claude/skills/
│   ├── fetch-upstream.sh             ← 拉 upstream 库（路径 2 用）
│   ├── run-npu-container.sh          ← NPU 容器 launcher（路径 1 + 2 都用）
│   ├── inspect-ascend-image.sh       ← image 体检（skill 内部）
│   ├── code-path-sweep.sh            ← CUDA-only 扫描（skill 内部）
│   ├── smoke_v11_device.py           ← V1.1/V1.2 smoke
│   ├── smoke_v13_rollout.py          ← V1.3 smoke
│   └── smoke/README.md               ← 完整 smoke 梯子索引
└── knowledge/
    ├── npu-patterns.md               ← **23 条 NPU 坑 stable ID 目录**
    ├── smoke-ladder-convention.md
    ├── upstream-refs.md              ← image→upstream ref 对应表
    └── images/                       ← 每个目标 image 的 pip freeze + inventory
```

`upstream/` 不是本仓的一部分。走路径 2 的人用 `scripts/fetch-upstream.sh` 现拉现用，拉到 `../upstream/`（跟 `easyr1-npu/` 同级，不 track）。

---

## 前置条件

**路径 1（只跑 EasyR1）**：
- Ascend 910C (A3) NPU host（≥ 2 空闲 chip）
- NPU driver ≥ 25.5.0
- Docker ≥ 24
- ≥ 20 GB 空闲磁盘
- 能访问 HuggingFace（国内用 hf-mirror）

**路径 2 / 4（做或复现自动化移植）**：
- 路径 1 全部 +
- Claude Code CLI ≥ 2.0 安装好
- 开发机能访问 GitHub / GitCode（fetch-upstream.sh 用）
- `gh` CLI（`gh auth login` 登过）
- `codex` CLI（用于 `codex-review` skill）—— 可选

---

## 维护

- **坑目录**：`knowledge/npu-patterns.md` 按 `NPU-CP / NPU-BUG / NPU-ENV / NPU-OPS + NNN` schema 加新条目
- **新目标 image**：跑 `npu-image-inspect` skill，输出到 `knowledge/images/`，更新 `upstream-refs.md`
- **移植日记**：有状态变化写 `porting-journal.md`
- **交接**：维护 `HANDOVER.md` —— 下一个 session 接手的人只读这一篇就能上手

commit message **不要**加 Claude 相关文字（见 `CLAUDE.md`）。项目文档默认中文；代码注释、commit message、SKILL.md frontmatter 保持英文。

---

## 相关项目

**姐妹项目**：[`ascend-fused-accuracy-probe`](https://gitcode.com/zhengshencn_hwca/ascend-fused-accuracy-probe)（private）—— 检测融合算子精度对齐。两个项目都跑在同一个 A3 host 上，互不干扰但共享一些硬件 / 软件事实知识。
