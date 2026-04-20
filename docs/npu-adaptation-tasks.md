# NPU 适配任务清单

**这是本项目 track 所有"NPU 生态还没覆盖、需要驱动适配"工作的单一来源**。

**归属**：见 [`DOCS-CONVENTION.md`](DOCS-CONVENTION.md) —— 任何识别出的 NPU gap 必须登记在这里，track 到完成。

**责任档划分**（见 [`SKILLS-GUIDE.md §6`](SKILLS-GUIDE.md)）：
- **档 1** = 本仓直接做（EasyR1 源码 / Python shim / 上游 Python 层 PR）
- **档 2** = 委托姐妹项目（kernel 实现 → `ascend-fused-accuracy-probe` / `a5_ops` / A3 kernel 项目；torch_npu C++ ATen op → Ascend PyTorch 团队）
- **档 3** = 提需求给 Ascend 团队（CANN runtime C 层框架 bug，我们只能提 issue + workaround + 等修）

---

## 当前 active tasks

### Tier-3 已知项（上报 Ascend 团队 + workaround）

| ID | Task | 状态 | 当前 workaround | 下一步 |
|---|---|---|---|---|
| T3-001 | **NPU-BUG-003** — triton-ascend inductor log_probs 数值损坏（CANN 8.5.0 crash；CANN 8.5.1 silent corrupt + delayed crash） | **OPEN** | `use_torch_compile=false`（已在 V2.1 / V2.2 smoke 固定） | 等 triton-ascend 3.3+ 或 torch_npu 2.10 → 重跑 `bug003_probe`。见 [`transformers-upgrade-drill.md §follow-ups #3`](transformers-upgrade-drill.md) |
| T3-002 | **NPU-BUG-004** — upstream triton 3.6.0 + triton-ascend 3.2.0 共存时 `backends/amd/compiler.py` import `Language` 失败 | **WORKED-AROUND** | Dockerfile 里删 `backends/amd/` + `backends/nvidia/`（drill commit `a18d1f8`） | 监控 triton-ascend 是否升级对齐 upstream triton；若升级则可移除 workaround |
| T3-003 | **NPU-OPS-009** — A3 host 所有容器都无法枚举 NPU（`drvRetCode=87`、`device_count: 0`），但 host 外 `npu-smi info` 正常。sister container 也挂。2026-04-20 阻塞了 V1.4 回归测 | **BLOCKING** | 无——host-level 问题，容器侧修不了 | 请 host admin 重启 ascend driver services / shared memory，或 reboot。恢复后：立即跑 `T1-001`（已准备好） |

### Tier-1 已知项（本仓待跑）

| ID | Task | 状态 | 下一步 |
|---|---|---|---|
| T1-001 | `ascend-port` 两个 cherry-pick（`1f716ea` + `ecce71d`）在 8.5.0 生产 image 回归测 | **BLOCKED on T3-003**（2026-04-20 尝试启动，A3 host NPU 驱动状态坏，所有容器都枚举不到 NPU，非 port regression；准备就绪，host 恢复立即跑） | 等 T3-003 解开，跑 V1.4 smoke on 8.5.0，数值对比 0.991 基线。见 [`HANDOVER.md §6.2`](HANDOVER.md) |
| T1-002 | `ascend-port` 在 8.5.2 drill image 跑 V1.4 smoke（把 "兼容两套 image" 从理论推到实测） | **BLOCKED on T3-003** | 等 T3-003 解开，基准对比 drill 1.434 |
| T1-003 | 自动化 dep-gap 检测脚本（场景 P1 自动判断） | **PLANNED** | 扩展 `image-upgrade-drill` Step 2，或独立 `scripts/dep-gap-detect.sh`。输入 = 新 EasyR1 requirements + 新 target image pip-freeze，输出 = A/B/C/D/E 分级表 |
| T1-004 | Skill `image-upgrade-drill` 在"未知 break"场景下的 discovery 能力（skill 有效性验证） | **OPEN** | 2026-04-20 的 dry-run 用的 skill 漏了答案；已修（commit `66c5ce9`）。等下一次真升级（新 CANN / 新 transformers）做 clean test。见 [`skill-dry-run-2026-04-20.md`](skill-dry-run-2026-04-20.md) |

### Tier-2 已知项（委托姐妹项目）

*当前无 active tier-2 任务。EasyR1 master 的依赖审计（[`easyr1-dep-chain-audit.md`](easyr1-dep-chain-audit.md)）显示 D 类 blocker = 0，不需要新 kernel 实现。*

**如果未来识别出**：比如某个新 EasyR1 版本要求用一个 NPU 上不存在的 fused op，登记格式：

```
| T2-XXX | 描述（EasyR1 需要 X op，NPU 无实现） | OPEN | 委托给 [`ascend-fused-accuracy-probe`](https://gitcode.com/zhengshencn_hwca/ascend-fused-accuracy-probe) / `a5_ops` / A3 kernel 仓。姐妹项目做 kernel，本仓 track 接回 EasyR1 |
```

---

## 已完成 / closed tasks

（保留记录，不删除；未来反查用）

| ID | Task | 完成日 | 结果 |
|---|---|---|---|
| C-001 | transformers 5 / vllm 0.18 backward-compat 适配 | 2026-04-19 | Drill 分支 PASS 2-step + 20-step。backward-compat fix cherry-pick 到 `ascend-port`（`1f716ea` + `ecce71d`）。见 [`transformers-upgrade-drill.md`](transformers-upgrade-drill.md) |
| C-002 | EasyR1 master 依赖分级审计（D 类 blocker 识别） | 2026-04-20 | **D = 0**，P1 场景结构性闭环。见 [`easyr1-dep-chain-audit.md`](easyr1-dep-chain-audit.md) |
| C-003 | NPU-OPS-006/007/008（docker proxy / pip.conf / huaweicloud mirror）infra 坑识别 | 2026-04-19 | 已加到 [`npu-patterns.md`](../knowledge/npu-patterns.md) 作为 stable ID + drill Step 2 pre-flight |
| C-004 | NPU-CP-007（transformers 原生 NPU flash-attn 集成） | 2026-04-18 | V2.1 启用 padding_free=True 用 `transformers.integrations.npu_flash_attention`，numerical-equivalent V1.4 |
| C-005 | Ray NPU resource 注册 + `ASCEND_RT_VISIBLE_DEVICES` shim | 2026-04-18 | 通过 `ray-npu-shim` skill 沉淀 |

---

## 任务登记格式

新任务加到 "当前 active tasks" 对应档的表格。每条包含：

- **ID**：`T<tier>-<NNN>` 三位数递增
- **Task**：一句话描述问题或工作
- **状态**：`PLANNED` / `OPEN` / `IN-PROGRESS` / `BLOCKED (原因)` / `WORKED-AROUND`（有 workaround 但不是永久修复）
- **当前 workaround / workaround 有效性**（如适用）
- **下一步**：具体可执行动作，引用 doc 或 commit

完成后挪到 "已完成 / closed tasks"，保留记录。**不要直接删**。

---

## 维护时点

按 [`DOCS-CONVENTION.md §2`](DOCS-CONVENTION.md) 触发表：
- 识别新 NPU 适配 gap → 在这里加 task
- 完成一个 task → 挪到 closed
- 状态变化（workaround 失效 / 上游修了 / 优先级变了） → 就地更新

同步 HANDOVER §6（未结工作）。
