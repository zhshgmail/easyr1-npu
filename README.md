# easyr1-npu

**把 EasyR1（`hiyouga/EasyR1`）移植到 Ascend 910C (A3) NPU，并沉淀一套可复用的"GPU RL 框架→NPU"移植 skills。**

本仓交付两件事：
1. **EasyR1 在 A3 上能跑** —— 代码改动在 [`zhshgmail/EasyR1` fork](https://github.com/zhshgmail/EasyR1) 的 **`ascend-port`** 分支
2. **一套可复用的移植 skills** —— 让别人能从 0 开始重做（或做 EasyR1 的下一个大版本升级、或移植 OpenRLHF/TRL 到 NPU）

---

## 从这里开始 — Step by Step

### 我想在 A3 上把 EasyR1 跑起来

→ 读 [`docs/PORT-GUIDE.md`](docs/PORT-GUIDE.md)

该手册说明：EasyR1 baseline commit、完整依赖版本（CANN / torch_npu / transformers / vllm_ascend / triton_ascend）、我们改了哪些代码以及为什么、如何在一台干净的 A3 host 上从 0 把 V1.4→V2.2 smoke 跑绿，每步的预期数值是什么。

**5 步摘要**：

```bash
# 1. Pull 镜像（国内走 NJU mirror）
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest

# 2. Clone EasyR1 fork 的 ascend-port 分支
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git \
    "$HOME/workspace/EasyR1"

# 3. Build 层叠 image
cd "$HOME/workspace/EasyR1" && docker build -t easyr1-npu:ascend-port -f Dockerfile.npu .

# 4. Clone 本仓拿 runner 脚本
git clone https://gitcode.com/zhengshencn_hwca/easyr1-npu.git "$HOME/workspace/easyr1-npu"

# 5. 跑 V1.4 smoke（2-chip GRPO 2-step）
cd "$HOME/workspace/easyr1-npu"
bash scripts/run-npu-container.sh --chips 0,1 \
    --live-source "$HOME/workspace/EasyR1" \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

完整细节 → [`docs/PORT-GUIDE.md`](docs/PORT-GUIDE.md)

### 我想从 0 重做一次移植（新 EasyR1 版本、新 image、或另一个 RL 框架）

→ 读 [`docs/SKILLS-GUIDE.md`](docs/SKILLS-GUIDE.md)

该手册说明：7 个 skill 各自做什么、输入输出是什么、从 0 开始按什么顺序调用、每一步的决策点。

**部署 skills 给 Claude Code 用**：

```bash
cd easyr1-npu
bash scripts/install-skills.sh       # symlink 到 ~/.claude/skills/
bash scripts/install-skills.sh --undeploy   # 卸载
```

---

## `zhshgmail/EasyR1` fork 的分支结构

`zhshgmail/EasyR1` fork 上有 **4 个分支**，每个用途不同。**给用户发布只推 `ascend-port`，其它分支用户不需要切。**

| 分支 | 对应什么 | 用途 | 用户该不该用 |
|---|---|---|---|
| `main` | upstream `hiyouga/EasyR1:main` mirror | 保持跟 upstream 同步，便于未来 rebase | ❌ 不要 |
| **`ascend-port`** | `main` + 20 个 NPU port commit | **v1 生产发布分支**。已 backward-compat 兼容 8.5.0 / 8.5.2 两套 image | ✅ **这个** |
| `ascend-port-transformers-upgrade` | `ascend-port` + 9 个 `[drill]` commit | v2 transformers 4→5 / vllm 0.13→0.18 升级演练分支。**带实验性 `Dockerfile.npu-852`、BUG-003 probe 脚本、20-step max_steps 等 drill-only artifact** | ❌ 不要 |
| `ascend-port-transformers-upgrade-reproduce` | 从 `ascend-port` 派生的复现验证分支 | 验证升级演练能否被 skill 从 0 复现；子 agent 跑一半卡住，留着给下次分段复现用 | ❌ 不要 |

**关键点**：
- **`main` 不是我们的 NPU 发布分支**。它是 upstream mirror。如果你 clone 默认分支会拿到**没有任何 NPU 改动的原版 EasyR1**
- `ascend-port` 和 drill 分支**完全隔离**——drill 分支上加了个新文件 `Dockerfile.npu-852`，**没改** `Dockerfile.npu`。drill 上的 v2 break 修复做成了 backward-compat 版本后 cherry-pick 进了 `ascend-port`，所以 `ascend-port` 既能跑 8.5.0 image 也能跑 8.5.2 image，**用户不需要切 drill 分支**
- 开发的时候 drill 分支带 `[drill]` commit 前缀是**契约**：cherry-pick 到 `ascend-port` 时跳过带前缀的，只挑 backward-compat fix

**用 `ascend-port` 的命令**：

```bash
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git
# 或
git clone https://github.com/zhshgmail/EasyR1.git && cd EasyR1 && git checkout ascend-port
```

---

## 仓库布局

```
easyr1-npu/                           ← 本仓（gitcode.com/zhengshencn_hwca/easyr1-npu）
├── README.md                         ← 你在这里
├── CLAUDE.md                         ← 项目指令（给 Claude Code 用）
├── docs/
│   ├── PORT-GUIDE.md                 ← "我要跑起来" 手册
│   ├── SKILLS-GUIDE.md               ← "我要重做移植" 手册
│   ├── HANDOVER.md                   ← 当前状态 + 未结工作
│   ├── DELIVERABLE.md                ← 正式 sign-off 文档
│   ├── transformers-upgrade-drill.md ← v2 升级演练完整报告
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
│   ├── run-npu-container.sh          ← NPU 容器 launcher（用户会直接用）
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

**开发本地还有一个 `upstream/` 目录**（跟 `easyr1-npu/` 同级，不在本仓的 git tracking 里）：

```
~/workspace/
├── easyr1-npu/     ← 本仓
└── upstream/       ← 每个子目录是各自 upstream 的独立 git clone
    ├── EasyR1/         ← zhshgmail/EasyR1 fork
    ├── verl/           ← GPU 参考
    ├── torch-npu/
    ├── vllm-ascend/
    ├── triton-ascend/
    └── transformers/
```

别人拿到本仓需要**按 `PORT-GUIDE.md` / `SKILLS-GUIDE.md` 的指引自行 clone 需要的 upstream**。

---

## 前置条件

**运行 EasyR1**：
- Ascend 910C (A3) NPU host（≥ 2 个空闲 chip；V1.5+ 需要 4 个）
- NPU driver ≥ 25.5.0
- Docker ≥ 24
- ≥ 20 GB 空闲磁盘给 docker

**重做移植 / 用 skills**：
- 上面全部 +
- Claude Code CLI ≥ 2.0 安装好
- 本地能访问 GitHub / GitCode（skill 会 fetch 上游代码）
- `gh` CLI（`gh auth login` 登过）
- `codex` CLI（用于 `codex-review` skill）—— 可选

---

## 贡献 / 维护

- **坑目录**：`knowledge/npu-patterns.md` 按 `NPU-CP / NPU-BUG / NPU-ENV / NPU-OPS + NNN` schema 加新条目
- **新目标 image**：跑 `npu-image-inspect` skill，输出到 `knowledge/images/`，加到 `upstream-refs.md`
- **移植日记**：每次有状态变化写到 `porting-journal.md`
- **交接**：维护 `HANDOVER.md` —— 接手下一个 session 的人只读这一篇就能上手

commit message **不要**加 Claude 相关文字（见 `CLAUDE.md`）。项目文档默认中文；代码注释、commit message、SKILL.md frontmatter 保持英文。

---

## 相关项目

**姐妹项目**：[`ascend-fused-accuracy-probe`](https://gitcode.com/zhengshencn_hwca/ascend-fused-accuracy-probe)（private）—— 检测融合算子精度对齐。两个项目都跑在同一个 A3 host 上，互不干扰但共享一些硬件/软件事实知识。
