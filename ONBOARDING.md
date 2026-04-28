# EasyR1 on Ascend 910C (A3) — onboarding

One-page quickstart. Two validated paths today:

| Path | Stack | Use case |
|---|---|---|
| **v1** (production) | verl-8.5.0-a3 image + EasyR1 ascend-port | "我只想今天就把 EasyR1 跑起来——最稳定的组合。" |
| **v2** (integrated) | vllm 0.20 + torch 2.11 + 4 个 NPU 上游 ascend-port overlay + EasyR1 master | "我要最新的 stack，4 个 NPU 上游漂移已经合掉。验证：T22.7 (2026-04-27) + T25.5 (2026-04-28) 两次独立 V1.4 GRPO PASS。" |

## Path v1 — fastest, most-tested

```bash
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git
git clone https://github.com/zhshgmail/easyr1-npu.git
NPU_USER=<your-workspace-owner> \
  bash easyr1-npu/src/scripts/run-npu-container.sh \
    --chips 0,1 \
    --image quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest \
    --live-source ./EasyR1 \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

完整 guide：[`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md)。

## Path v2 — integrated overlay (newest)

```bash
NPU_USER=<your-workspace-owner> \
  bash easyr1-npu/src/scripts/run-npu-container.sh \
    --chips 0,1 \
    --image easyr1-npu:integrated-20260427 \
    --live-source ./EasyR1 \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

`easyr1-npu:integrated-20260427`（28.2 GB，SHA `044ba0b76183`）由 A3 host 上从 `easyr1-npu-vllm0200:iter20-abi-both` base + 4 个 ascend-port overlay 层 build 出。要重建，参考 [`docs/easyr1/PORT-GUIDE-v2-integrated.md`](docs/easyr1/PORT-GUIDE-v2-integrated.md) 与 [`src/skills/_shared/integrated-overlay-build/SKILL.md`](src/skills/_shared/integrated-overlay-build/SKILL.md)。

完整 guide：[`docs/easyr1/PORT-GUIDE-v2-integrated.md`](docs/easyr1/PORT-GUIDE-v2-integrated.md)。

## Hard prerequisites (both paths)

- A3 host（Ascend 910C，x86 + NPU 卡，openEuler / Ubuntu）
- CANN ≥ 8.5.0 driver 安装于 `/usr/local/Ascend/`
- Docker，支持 `--privileged` + `--ipc=host`
- `/data/<your-workspace-owner>` ≥ 50 GB（HuggingFace 缓存 + checkpoint）
- **A3 host 上的 `easyr1-npu` 必须是 `git clone` 的工作树**（不能是早期 layout 的手抄拷贝）。本文档命令默认 `repo/src/scripts/...` 路径，旧 layout 在 `repo/scripts/...` 会报 No such file or directory。详见 [`knowledge/npu-patterns.md` NPU-OPS-014](knowledge/npu-patterns.md)。

## Common gotchas (both paths)

1. **SSH-as-root**：`ssh -p 443 root@...` 会让 `$USER=root`，runner 默认 mount `/home/root`（空目录）。**必须传 `NPU_USER=<workspace-owner>`**，否则模型/数据/checkpoint 路径都对不上。详见 NPU-OPS-013。
2. **`--chips` 含义**：`--chips 0,1` 是 host phy-id（`npu-smi info` 看到的那个号）。runner 自己把对应 `/dev/davinciN` 透传，并在容器里自动设 `ASCEND_RT_VISIBLE_DEVICES=0,1,...,N-1`（容器内 index）。**不要手 docker run**——容器内 index 与 host phy-id 是两个域，混淆即 Ray 报 `Total available GPUs 0`。详见 NPU-OPS-012。
3. **NPU 容器 bind set**：`run-npu-container.sh` 已经把 `/etc/ascend_install.info`、`/etc/ascend_driver.conf`、`/etc/ascend_filelist.info`、`/usr/local/dcmi/`、`/usr/local/bin/npu-smi` 等都 bind 进去；手 docker run 极易漏一个就 dcmi -8020。详见 NPU-OPS-009 / NPU-OPS-011。

## 入口索引

- 当前进度 + 入口表：[`README.md`](README.md)
- 整体架构与流程（含 mermaid 图）：[`docs/_meta/ARCHITECTURE.md`](docs/_meta/ARCHITECTURE.md)
- 上游 fork 分支权威表：[`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md)
- v1 详解：[`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md)
- v2 详解：[`docs/easyr1/PORT-GUIDE-v2-integrated.md`](docs/easyr1/PORT-GUIDE-v2-integrated.md)
- Skill 使用（给上游维护者）：[`docs/_meta/SKILLS-USAGE.md`](docs/_meta/SKILLS-USAGE.md)
- KB：[`knowledge/npu-patterns.md`](knowledge/npu-patterns.md)、[`docs/_meta/kb/porting_lessons/`](docs/_meta/kb/porting_lessons/)

## Issues / iteration

- Open issue：github.com/zhshgmail/easyr1-npu/issues
- Discord：项目 channel（internal）
