# easyr1-expert

**定位**：一个能自动完成 "EasyR1 → Ascend NPU 移植 + A3 上 V1.1..V2.2 smoke 验证" 的 skill + agent + KB 组合。

**作为产品**：
- **输入**：EasyR1 commit（默认 master 最新）+ target image（默认 v1 `verl-8.5.0-a3`）+ A3 host ssh access
- **输出**：
  - 一个新的 `ascend-port-<timestamp>` git 分支（含 port code commits）
  - 一个 docker image tag（`easyr1-npu:<tag>`）
  - Smoke ladder 报告：V1.1 / V1.3 / V1.4 / V1.5 / V2.1 / V2.2 各自 PASS/FAIL + 数值
  - `PROGRESS.md` + `RESULTS.md` 工作日志

**Scope 前提（Stage 0）**：当前 EasyR1 master 在 v1 image 上 D=0（见 `docs/easyr1-dep-chain-audit.md`）。本 expert 处理的就是这个 D=0 场景 —— 所有 port 工作都是 EasyR1 自己源码改动，不依赖新的 NPU 上游适配。

**不在 scope**（Stage 0）：
- ❌ EasyR1 某个新 commit 引入 D≥1 依赖（新的 CUDA-only 包 NPU 还没移植）—— 那时需要拆 `*-expert` 出来（见 `docs/design/SKILLS_ARCH_TARGET.md` 附录 A）
- ❌ 非 EasyR1 框架（OpenRLHF / TRL）—— 另一个消费者 expert

---

## 使用（计划中，Stage 0 S5/S6 完成后可用）

**冷启动**：
```bash
cd easyr1-npu
bash scripts/install-skills.sh  # 或 install-experts.sh 安装到 ~/.claude/skills/
```

**调用**：
```
/easyr1-port reproduce v1
# → easyr1-port-worker agent 被 spawn，Phase A/B/C/D 全程自动
# → 输出 port 分支 + image + smoke 数值
```

---

## 内部结构

```
easyr1-expert/
├── README.md                  # 本文
├── SKILL.md                   # /easyr1-port skill 入口
├── agent.md                   # easyr1-port-worker agent 定义
├── state_machine.yaml         # 内部工作流（P0..P7 + G1-G3 invariants）
├── references/                # KB（本 expert 自包含）
│   ├── ALWAYS_LOADED_RULES.md       # Phase A 必读
│   ├── KB_INDEX.md                  # Keywords/Aliases 索引
│   ├── CODE_PATH_PATTERNS.md        # NPU-CP-001..007 可执行模式
│   ├── ERROR_CORRECTIONS.md         # Traceback → fix
│   ├── PLATFORM_BUGS.md             # NPU-BUG-001..004
│   ├── SMOKE_BASELINE.md            # V1.1-V2.2 per-image 期望数值
│   └── patterns/
│       ├── device_dispatch.md       # NPU-CP-001 如何 apply
│       ├── ray_integration.md       # Ray NPU shim
│       ├── attention_backend.md     # flash_attn → transformers.integrations.npu_flash_attention
│       ├── vllm_compat.md           # vllm 0.13 API rename shims
│       ├── transformers_compat.md   # try/except import for 4.x / 5.x
│       └── dockerfile.md            # Dockerfile 模板 + common pitfalls
├── scripts/
│   ├── static_check.py              # py_compile + dry-import
│   ├── deploy_to_a3.sh              # tar → scp → docker cp
│   ├── smoke_validate.sh            # 跑 smoke + grep entropy_loss + assert band
│   └── code_path_sweep.sh           # (从 easyr1-npu/scripts/code-path-sweep.sh 搬)
├── tested_combinations.yaml   # 已验 tuple 存档
└── hooks/
    └── check_port_worker.sh   # Stop hook: static_check + PROGRESS 签名 + log evidence
```

---

## 验证标准

见 `docs/design/SKILLS_ARCH_TARGET.md` §3 Acceptance T0.1 / T0.2 / T0.3。

PASS 标志：冷启动 agent 自己跑通 V1.4 smoke 数值在 baseline 带。

---

## 历史 & 演进

- 2026-04-22 创建（V3.0 Stage 0 S2）
- 见 `docs/design/SKILLS_ARCH_TARGET.md` 历史版本 V1.0 → V2.0 → V3.0 的教训

**预期演进**：
- Stage 0 证明通过后（round 3 PASS），expert 结构稳定
- 真出现 D≥1 场景时，按 KB 所在 domain 把 `patterns/` 下面的对应文件拆出去成一个独立 expert（e.g. `transformers-expert`）
- 独立出去的 expert 跟本 expert 通过附录 A 描述的 contract 交互
- 本 expert 变成**消费者型** expert（保留，但不再 own 那些 dep 的 knowledge）
