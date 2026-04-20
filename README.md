# easyr1-npu

**把 EasyR1（`hiyouga/EasyR1`）移植到 Ascend 910C (A3) NPU，并沉淀一套可复用的"GPU RL 框架→NPU"移植 skills。**

本仓交付两件事：
1. **EasyR1 在 A3 上能跑** —— 代码改动在 [`zhshgmail/EasyR1`](https://github.com/zhshgmail/EasyR1) 的 **`ascend-port`** 分支（20 个 commit）
2. **一套可复用的移植 skills** —— 下一次 EasyR1 版本升级 / CANN 升级 / 别的 RL 框架（OpenRLHF / TRL）移植都能套用

> **📣 2026-04-20 Scope 说明（勘误）**：之前版本的 README / SKILLS-GUIDE 把"改 NPU 上游库"和"改 CANN / torch_npu C++ 层"都打成 ❌ 不在 scope 是**错误的表述**。
>
> 正确的区分（按"谁来做"分三档）：
> - ✅ **本仓直接做**：EasyR1 源码改动、Python 层 shim / fork、向 vllm-ascend / triton-ascend / torch_npu 上游提 issue 或 PR、识别 NPU 适配 gap 并 track
> - 🤝 **委托给姐妹项目 / 独立仓**：CANN 算子实现、kernel 数值精度验证 —— 用 `ascend-fused-accuracy-probe`（A3 kernel 验证）和 `a5_ops`（A5 kernel 生成，A3 有类似的独立仓）这些专门的项目做。本仓**识别 gap + 建接口 + track 适配**，不把 kernel commit 放本仓
> - 📣 **提需求给 Ascend 团队**：CANN runtime 框架级的 bug（ACL C 层本身的 crash、HCCL C 层协议问题）—— 这类我们做 workaround + 提 issue，修是 Ascend 团队的事
>
> **关键原则**：如果新 EasyR1 版本依赖一个 NPU 还没覆盖的包 / 算子，"不在 scope" 不是可接受答案 —— 项目目标就死了。**必须建任务推动适配**，哪怕具体 commit 落到别的仓。见路径 4 + `docs/npu-adaptation-tasks.md`（待建）。

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

适合：NPU 软件栈升级（CANN 9.x、torch_npu 2.10、transformers 6 之类）发布后，你要评估 "我们的 EasyR1 port 能不能跟上新 image"，**以及**识别出哪些新依赖是 NPU 生态还没适配的，驱动相应的 NPU 适配任务。

**第一性原则重申**：本项目的目标是"让 EasyR1 master 在 A3 上跑"。如果新 EasyR1 版本或新 image 引入一个 NPU 还没适配的依赖，**我们不能说"不在 scope" 就结束** —— 那等于放弃项目目标。正确的做法是**识别这个 gap，建 NPU 适配任务，推动它完成**（可能是我们自己做，可能是协调 Ascend / upstream 团队做）。见 [`docs/npu-adaptation-tasks.md`](docs/npu-adaptation-tasks.md)（**待建**，见本文档 §下一步）。

**本路径明确在 scope**：
- 验证新 image 的一整套新依赖（torch_npu、vllm_ascend、triton_ascend、transformers 等）跟我们的 EasyR1 `ascend-port` 分支是否兼容，做数值 smoke 验证
- 修复 EasyR1 自己源码里的版本 compat 问题（e.g. transformers 5 改了 import 路径 → EasyR1 里加 try/except）
- 换 base image + rebuild 层叠 image
- **识别 NPU 适配 gap**：新 EasyR1 依赖某个 CUDA-only 包、或某个包的 NPU 移植还没跟上 → 建任务 track。见 [`docs/npu-adaptation-tasks.md`](docs/npu-adaptation-tasks.md)（待建）
- **协调 NPU 适配工作落地**：比如需要给 vllm-ascend / triton-ascend 提 issue 或 PR、或跟 Ascend 团队提适配需求、或自己写 shim/fork

**本路径的工作委托给姐妹项目**（本仓识别 + 建接口 + track，具体实现在别的仓）：
- 🤝 **新 CANN 算子实现 / kernel 精度验证** → 委托给 `ascend-fused-accuracy-probe`（A3 kernel 验证）、`a5_ops`（A5 kernel 生成，A3 有类似的独立项目）这类专门 kernel 项目。本仓识别 "EasyR1 在 A3 上需要某个 fused attention / fused softmax 但没有现成 NPU 实现"，建 NPU 适配任务，协调姐妹项目完成

**真正上报给 Ascend 团队**（只有这一档是我们做不了 workaround 就只能等的）：
- 📣 **CANN runtime 的 C 层框架 bug**（ACL runtime 本身 crash、HCCL C 协议层问题）—— 我们能做的是提 issue + 做 workaround + 等修复

**Python 层的 shim、上游库的 issue、vllm-ascend 的移植协调、triton-ascend 的 wheel 整理 —— 都直接在本仓驱动**（哪怕具体 commit 是别的仓 merge 的）。

用 [`image-upgrade-drill`](skills/image-upgrade-drill/SKILL.md) skill 的 7 步演练流程。产物：
- 一份带数字的 drill report（预测 vs 实际成本、LOC 变动、bug probe 结果）
- backward-compat commit 序列，可 cherry-pick 进 `ascend-port`
- **NPU 适配 gap 清单**：哪些新依赖还没 NPU 适配，各自需要的工作量 / 谁去做 / 当前状态
- 新 stable ID 条目加到 `knowledge/npu-patterns.md`
- PASS / BLOCKED 决策依据（区分"EasyR1 代码 compat 完成" vs "NPU 适配 gap 完成"两件事）

**当前状态警告**：
- 2026-04-19 的 transformers 升级 drill 在 **drill 分支 + 8.5.2 image 上 PASS**（2-step 数值匹配 + 20-step 稳定），但 drill 的 fix cherry-pick 到 `ascend-port` 后**还没在 8.5.0 image 上跑过回归测**（HANDOVER §6.2 标的 P1）。所以 README / PORT-GUIDE 里说 "ascend-port 兼容 v1/v2 两套 image" 是**理论 backward-compat 写法，不是实测结论**
- "skill 自动化端到端复现" **没有证明**。2026-04-19 首次用 isolated agent 走完 7 步时卡住了（见 [`docs/UPGRADE-DRILL-STATUS.md`](docs/UPGRADE-DRILL-STATUS.md) §3）；2026-04-20 的 dry-run 验证发现当时的 SKILL.md 直接写了答案，agent 本质是"看着答案做"，**不是独立发现**（见 [`docs/skill-dry-run-2026-04-20.md`](docs/skill-dry-run-2026-04-20.md)）。已更新 SKILL.md 隐藏答案，但新版本还没真实场景测过
- **NPU 适配 gap 清单未建立** —— 下一步要做的事。目前 v1 (8.5.0) 上 EasyR1 master 能跑，说明在 v1 基线下 D 类 blocker 为 0；drill 在 v2 (8.5.2) 上能跑说明 v2 也没 D 类 blocker。但对未来 EasyR1 或更新 image 做了系统性分类 gap

完整 skill 说明 → [`skills/image-upgrade-drill/SKILL.md`](skills/image-upgrade-drill/SKILL.md)
v2 drill 的首次实证报告 → [`docs/transformers-upgrade-drill.md`](docs/transformers-upgrade-drill.md)
Skill dry-run 验证 → [`docs/skill-dry-run-2026-04-20.md`](docs/skill-dry-run-2026-04-20.md)

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
