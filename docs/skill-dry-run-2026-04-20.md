# Skill dry-run 验证 — 2026-04-20

**目的**：验证 `image-upgrade-drill` skill 是否**可被 agent 独立执行**到产出正确代码改动 —— 回答 persona 2 的问题"用 skill 重跑移植能不能闭环"。

**方式**：dry-run（不上 A3），让一个隔离的 Explore agent 按 skill 说明执行 drill Step 1-4，产出代码 diff，和已知 ground truth（`1f716ea` + `ecce71d` 两个 cherry-pick）对比。

**结论**：**部分闭环成功**（agent 产出语义等价 fix），但有重要 caveat（skill 当时直接披露答案；agent 的 fix 2 有 Python 优先级 bug，需 A3 smoke 才能确认）。skill md 已针对此 finding 调整（见 §5）。

---

## 1. 实验设计

- **Agent 类型**：Explore，isolated context
- **授权读的文件**：`SKILLS-GUIDE.md`、`PORT-GUIDE.md`、`skills/*/SKILL.md`、`knowledge/images/verl-8.5.*-a3.md`、`knowledge/upstream-refs.md`、`knowledge/npu-patterns.md`（排除 drill 和 BUG-004 相关条目）、EasyR1 源码（限到 commit `6f8197f`，即 pre-drill state）、upstream 参考源码（transformers / vllm-ascend / 等）
- **禁止读的文件**：`transformers-upgrade-drill.md`、`HANDOVER.md` §6 之后、drill 分支 git log、drill 相关 commit SHA、NPU-BUG-004
- **任务**：从 6f8197f 出发，模拟跑 V2.2 smoke 在 v2 image 会撞的 API break，给出 backward-compat 修复

## 2. 结果

Agent 报告在 `/tmp/drill-replay-agent-report.md`。核心产出：

### Fix 1 — `no_init_weights` 迁移

| 维度 | Agent | Ground truth (`1f716ea`) |
|---|---|---|
| 目标文件 | `verl/workers/fsdp_workers.py:40` | 同 ✅ |
| Import 顺序 | v5 优先 / v4 fallback | v4 优先 / v5 fallback |
| 机制 | `try/except ImportError` | 同 ✅ |
| Backward compat | ✅ | ✅ |
| 语义等价 | ✅（顺序不影响行为） | — |

**判定**：通过。顺序差异不影响最终行为。

### Fix 2 — `SamplingParams.eos_token_id` 只读

| 维度 | Agent | Ground truth (`ecce71d`) |
|---|---|---|
| 目标文件 | `verl/workers/rollout/vllm_rollout_spmd.py` | 同 ✅ |
| 探测机制 | `inspect.getmembers + isdatadescriptor`，预先收集所有只读属性 | 对每个 key 用 `isinstance(cls_attr, property) and cls_attr.fset is None` |
| LOC | +9 | +9 |
| Backward compat | ✅ | ✅ |
| **正确性 bug** | **有**（见下） | 无 |

Agent 的实现有个 Python 优先级 bug：

```python
if isdatadescriptor(obj) and not hasattr(obj, 'fset') or (hasattr(obj, 'fset') and obj.fset is None):
```

`and` 优先于 `or`，这实际是 `(A and not B) or (B and C)`。若 `obj` 不是 data descriptor（A=False）但恰好 `fset=None`（B=True, C=True），会错误地加进 `read_only_attrs`。需要括号 `(A and (not B)) or (...)` → `A and ((not B) or (B and C))`。

这类 bug 只有 smoke / unit test 能抓。**dry-run 验证不足以最终定板**。

### 基础设施识别

- Agent 独立识别了 NPU-OPS-006/007/008（docker proxy / image 无 pip.conf / huaweicloud mirror 空）
- Agent 正确识别了 NPU-BUG-004（triton 3.6 / 3.2 共存冲突）作为 Dockerfile-level 修复

## 3. 数据点

| 指标 | Value |
|---|---|
| Agent 工作时间 | ~25 分钟 |
| 真实 drill 耗时（人 + agent 协作） | ~45 分钟 |
| Agent 产出 fix 数 | 2（正确数量） |
| 正确定位文件 | 2/2 ✅ |
| 正确 backward-compat 机制 | 2/2 ✅ |
| 可直接 apply 不需改 | 1/2（fix 2 有 bug） |

## 4. 主要发现 — skill 暴露了答案

读 `image-upgrade-drill/SKILL.md` 时 agent 发现 Step 4 "Concrete examples" 章节直接写了两个 break 的名字 + 修复模式（`no_init_weights` 搬家、`SamplingParams.eos_token_id` 只读 property）。

这意味着这次 dry-run 实际验证的是：
- ✅ skill 工作流能被 agent 正确**执行**（框架、步骤、输出格式都 OK）
- ✅ 即便 agent 得到答案，它能**独立 grep 出文件位置**
- ❓ skill 能否帮 agent **发现未知的 break** —— **未验证**，这次没有发现问题，agent 是看着答案做的

## 5. skill 已调整

已 commit 到 main：`image-upgrade-drill/SKILL.md` Step 4 删除了具体 break 的披露，改成讲**发现方法**：
- 读 traceback
- 按 upstream-refs.md 切对应 ref 的 upstream 源码
- 4 种常见 fix pattern（import moved / attr removed / property 只读 / API renamed）
- 指明 `transformers-upgrade-drill.md` 是答案 key，不要在 dry-run 时看

下一次做类似 dry-run（e.g. 新 CANN 9 出来时）可以测 skill 是否真的能驱动"discover unknown break"。

## 6. 对 persona 2 的结论

**部分闭环**：
- ✅ Skill + KB 可以驱动 agent 到**产出正确的 fix**（已知场景）
- 🟡 Agent 的 fix 需要 **A3 smoke** 做最终 validation（Python 优先级 bug 这类问题只有运行能抓）
- 🟡 "Discover unknown break" 还没验证 —— 要等下一个真升级场景才能测

**要完全闭环**：
- 下次真有新 EasyR1 或新 CANN 升级时，按新调整后的 skill 跑一次 dry-run，看是否能在**没有答案披露**的情况下定位到 break
- 跑一次 smoke on A3 来 validate agent 产出的 fix（不只是这次 dry-run 的两个，是任何将来 agent 产出的 fix）

## 7. 附件

- Agent 完整报告：`/tmp/drill-replay-agent-report.md`（仓外临时文件）
- Ground truth 对比：`upstream/EasyR1/ git show 1f716ea` 和 `git show ecce71d`
- 实验 prompt + 约束：见本 session 的 Explore agent 调用记录
