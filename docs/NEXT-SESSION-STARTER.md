# Next-session starter — 冷启动最小 checklist

如果你是接替这个 session 的下一位 Claude Code agent / contractor 来
继续项目，**先读这 7 条就能顶上**。约 10 分钟能看完。

## 1. 项目一句话定位

把 **EasyR1 (master, 2026-04)** 移植到 **Ascend 910C (A3) NPU**，并沉淀一套**可复用的"GPU-RL 框架移植到 NPU"** skills 库。详见 [`CLAUDE.md`](../CLAUDE.md)。

## 2. 今天（2026-04-23）最重要的新东西

**3-layer Day-0 chain 第一次打通并 codified**：
- `torch-day0-expert` — community PyTorch × NPU
- `vllm-ascend-day0-expert` — vllm-ascend 在新 torch 下的 ABI drift
- + 原有 `vllm-day0-expert`, `transformers-day0-expert`
- + 共享 pattern: `_shared/references/patterns/domains/day0-deploy-artifacts.md`

Skills 的地图入口：[`../src/experts/README.md`](../src/experts/README.md)（决策树 + 分 stage 列表）。

## 3. 2026-04-23 真跑通的链路

Manual 全过：torch 2.11.0+cpu + torch_npu 2.11.0rc1 + CANN 8.5.1 overlay → vllm-ascend Fix B+ patch (2 file-level edits in `upstream/vllm-ascend/`，trace-branch `ascend-day0-torch211-20260423` 在 `zhshgmail/vllm-ascend` mirror fork，authoritative deliverable 是 `workspace/vllm-ascend-day0-deploy-20260423-0655/PR_MATERIAL.md` 给 vllm-ascend maintainer) → V1.3 Qwen2-0.5B rollout **PASS**：
- `"Hello, my name is"` → `" Sarah and I am a 20"`
- `"2 + 2 equals"` → `" 4. 2 + 2"`

两个 overlay image 在 A3 保留：
- `easyr1-npu-torch211:torch-day0-manual-20260423-0537`
- `easyr1-npu-torch211-vllmascend-fixb:ascend-day0-torch211-20260423`

## 4. 但 **skill cold-drive 没做**

Session 内 manual 推进不算 0-interaction skill validation。
真正的 Phase 4 cold-drive 是你的下一步：

```
# 新 session，fresh state
/torch-day0 --target-torch-version 2.11.0 \
            --target-torch-npu-version 2.11.0rc1 \
            --base-image easyr1-npu-852:trans-upg-e2e-20260422-2200

# 等它跑完，outcome A，overlay image tag 产出
/vllm-ascend-day0 --target-delta torch-2.11 \
                  --base-image <上一步的 image tag>
# 期望 outcome C-patch + V1.3 PASS + PR material 文件
```

**预期等价**：你得到的结果应该和今天 manual 跑通的结果一致（同 overlay 版本组合 / 同 V1.3 text 输出 / 同 patch 内容）。如果有差异，说明 state_machine / KB 有 gap，按 memory `end_to_end_vs_described.md` 修 KB 再 retry。

## 5. 必读的 5 份文档

1. **[`HANDOVER.md`](HANDOVER.md)** — 本项目所有 transit state、未结工作、A3 状态、git 分支、chip 约定。**§6.6 今晚工作，§7.2 skill 系统，§11 下一步**是最新的。
2. **[`../CLAUDE.md`](../CLAUDE.md)** — 项目 instructions / working preferences / 下一个 agent 要注意的 user 偏好
3. **[`../src/experts/README.md`](../src/experts/README.md)** — 决策树 + 每个 expert 的用途
4. **[`../src/experts/_shared/references/ALWAYS_LOADED_UNIVERSAL.md`](../src/experts/_shared/references/ALWAYS_LOADED_UNIVERSAL.md)** — 所有 expert worker 无条件读的 OL 规则（OL-01 ~ OL-12）
5. **[`design/SKILLS_ARCH_TARGET.md`](design/SKILLS_ARCH_TARGET.md)** — skills 系统架构目标 + Day-0 reframing 段

## 6. 几条 session 级 memory（`~/.claude/projects/.../memory/`）

短清单（详见 MEMORY.md 索引）：

- `end_to_end_vs_described.md` — **CRITICAL**：不要把 codified / described 当 validated。真 validated 要 cold-drive。
- `no_trivial_pauses.md` — autonomous 模式不 pause 在 trivial 判断。
- `day0_upstream_deploy_chain.md` — manual port PASS 后插 Phase 2.5 deploy artifacts，下一层 expert 从已部署 state 起步。
- `day0_patch_scope.md` — C-patch 只改华为开源仓。
- `day0_follow_the_root.md` — 如果表层 gap 追到更深 dep，换 session 的上游到根 dep。
- `skill_production_order.md` — analysis → manual → codify → cold-drive。

## 7. 你**不**要做的事（ anti-patterns）

- **不要**立即新建 expert skeleton 做新 target — 先手动 port 通，再 codify（见 `skill_production_order.md`）
- **不要**在 `docker build` 里 `import torch`（OL-11；PyTorch 2.11 auto-load NPU driver 会炸）
- **不要**在 `enable_custom_op()` 之后才设 `VLLM_BATCH_INVARIANT`（OL-12；来不及）
- **不要**把 Stage 0 task 的 PASS claim 写成 manual-only（OL-02）
- **不要**未 precheck NPU 占用就 docker run（OL-05）
- **不要**为了 pause 而 pause — user 2026-04-23 明说了 "autonomous 模式不停"（`no_trivial_pauses.md`）

## 8. 你要做的事（immediate next steps 候选）

**HANDOVER §11** 是完整候选列表。高 priority：

1. **Cold-drive validate 两个新 Day-0 skill**（torch-day0 + vllm-ascend-day0）—— task #73
2. **Rebuild `vllm_ascend_C.so` against torch 2.11 headers** —— task #77 (Fix C tech debt)
3. **试跑 `/vllm-day0` against vllm 0.20.0**（真 Day-0 target，不是 0.19.1）
4. 根据你拿到的 user 指令决定方向

---

**Chip 状态现在**（2026-04-23 dusk session 结束时）：16/16 NPU 都闲置。A3 host: `ssh -p 443 root@115.190.166.102`。磁盘 93% 用量还好。

有疑问先看 HANDOVER.md，再看 CLAUDE.md，再搜 memory。**不要从头重建决策**。
