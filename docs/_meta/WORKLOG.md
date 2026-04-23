# WORKLOG — 当前 session 汇报入口

**用途**：user 2026-04-23T22:26Z 指令——"这么多工作，你分成一个工作序列从第一个任务开始做，所有任务都完成才算完成。你记录工作的文档先进行维持重构，把这些任务背景和状态放进去，用这个文件来汇报状态"。

本文件**必须**：
- 每完成一个 sub-task 立刻更新状态
- 每发现新问题立刻追加一行到未来任务
- auto-compact 之后下一 session 的第一入口就是读本文件
- user 想看状态就读这一份

## 任务背景

三件事交织：
1. **V1.4 debug on vllm 0.20**：iter 18 V1.3 bit-exact PASS 但 V1.4 entropy_loss=3.213 偏离 baseline 1.275 × 2.5，需要找到 rollout log-prob vs actor log-prob 的 KL 爆炸点
2. **结构重组（user 三连问）**：
   - 取消 "day0" vs "upgrade" 的假边界，每个上游合并为单一 expert
   - 删除 / 合并遗留 `skills/` 老目录到 `src/`
   - `docs/` 按 per-upstream 子目录重组（对齐 `src/experts/`）
3. **workspace/ 也按 per-upstream 组织**（目前是扁平 session-tag 目录）

## 任务序列（全做完才算完成）

### Phase 1 — 文档重构（本文件所在位置就是新结构的验证）

- [x] T1.1 创建 `docs/_meta/` 目录
- [x] T1.2 创建 `docs/_meta/WORKLOG.md`（就是本文件）
- [ ] T1.3 把项目级文档移到 `docs/_meta/`：DOCS-CONVENTION.md、GLOSSARY.md、HANDOVER.md、MODULE-PORT-STATUS.md、NEXT-SESSION-STARTER.md、design.md、e2e-validation-spec.md
- [ ] T1.4 创建 `docs/{vllm-ascend,vllm,torch-npu,transformers,easyr1}/` 目录，各自一份 README.md
- [ ] T1.5 分类剩余 `docs/` 顶层文件到对应上游：
  - `PORT-GUIDE.md`, `PORT-SUMMARY.md`, `porting-journal.md`, `DELIVERABLE.md`, `easyr1-dep-chain-audit.md`, `npu-adaptation-tasks.md`, `npu-gap-plan.md`, `code-path-sweep-EasyR1.md` → `docs/easyr1/`
  - `transformers-upgrade-drill.md`, `UPGRADE-DRILL-STATUS.md` → `docs/transformers/`（drill 是 transformers）
  - `examples/torch-2.11-day0.md` → `docs/torch-npu/examples/`
  - `skill-dry-run-2026-04-20.md`, `skills-design.md`, `SKILLS-GUIDE.md` → `docs/_meta/`
  - `codex-review-skills-audit.md`, `codex-signoff*.md`, `handoff-2026-04-19.md`, `P2-WORKFLOW.md`, `workflow/` → `docs/_archive/`（完成的历史 checkpoint）
- [ ] T1.6 rewrite 所有交叉引用路径（sed 一遍）
- [ ] T1.7 README 首屏指向 `docs/_meta/WORKLOG.md` 作为汇报入口

### Phase 2 — `src/` 全面对齐 per-upstream（结合 user 2026-04-23T22:29 重申）

**User 确定的目标结构**：
```
src/
├── skills/                       ← 原 src/experts/ 改名，因为 "expert" 是实现细节
│   ├── _shared/
│   ├── vllm-ascend/
│   │   └── port-expert/          ← 合并了 day0 + upgrade
│   ├── vllm/
│   │   └── port-expert/
│   ├── torch-npu/
│   │   └── port-expert/
│   ├── transformers/
│   │   └── port-expert/
│   ├── easyr1/
│   │   └── port-expert/          ← 原 easyr1-expert
│   └── dep-analysis/
│       └── expert/               ← 原 dep-analysis-expert
└── scripts/                      ← 原顶层 scripts/ 搬到 src/ 下
    └── ...（+ 每个上游独有的 scripts 搬到对应 skills/<upstream>/ 下）
```

同时处理遗留 `skills/` 顶层（老格式）：
- `skills/upstream-branch-hygiene/` — 通用 git 流程规则 → `src/skills/_shared/upstream-branch-hygiene/`
- `skills/codex-review/` + `skills/codex-signoff` — 评审流程 → `src/skills/_shared/codex-review/`
- `skills/dep-gap-detect/` — 通用依赖分析 → `src/skills/dep-analysis/` 合并
- `skills/image-upgrade-drill/` — 作为 consumer 的 EasyR1 port → `src/skills/easyr1/port-expert/` 的 drill 模式
- `skills/npu-code-path-sweep/`, `skills/npu-image-inspect/` — 通用 NPU 工具 → `src/skills/_shared/`
- `skills/npu-container-runner/` — 通用运行器 → `src/scripts/`
- `skills/ray-npu-shim/` — ray 相关通用 → `src/skills/_shared/`（consumer 用）

具体任务：
- [ ] T2.1 `src/experts/` → `src/skills/`（rename）
- [ ] T2.2 `scripts/` (top) → `src/scripts/`
- [ ] T2.3 在每个 `src/skills/<upstream>/` 下合并 day0-expert + upgrade-expert → `port-expert/`
- [ ] T2.4 把顶层 `skills/` 的 8 个老 skill 按上面清单迁入 `src/skills/_shared/` 或 `src/scripts/`
- [ ] T2.5 删除空的顶层 `skills/` 和 `scripts/`
- [ ] T2.6 更新每个 port-expert 的 SKILL.md + agent.md + state_machine.yaml 描述为"端到端 port"，不再分 day0/upgrade
- [ ] T2.7 更新 `src/skills/README.md` + `src/skills/<upstream>/README.md`
- [ ] T2.8 rewrite 所有路径引用（docs/ + src/ + knowledge/）

### Phase 3 — workspace/ 重组

- [ ] T3.1 把 workspace/ 下 21 个扁平 session 目录按上游分类：
  - `torch-day0-*`, `torch-npu-upgrade-*` → `workspace/torch-npu/`
  - `transformers-day0-*`, `transformers-upgrade-*` → `workspace/transformers/`
  - `vllm-day0-*`, `vllm-upgrade-*` → `workspace/vllm/`
  - `vllm-ascend-day0-*` → `workspace/vllm-ascend/`
  - `easyr1-port-*`, `dep-analysis-*`, `npu-port-*` → `workspace/easyr1/`
- [ ] T3.2 更新 MODULE-PORT-STATUS.md 路径
- [ ] T3.3 A3 上 `/tmp/z00637938/` 对应调整（可选，远端不一定必须重组）

### Phase 4 — GLOSSARY 更新，去掉 day0/upgrade 二分

- [ ] T4.1 GLOSSARY.md 把 "Day-0 skill outcome matrix" 改为 "port outcome matrix"，不再区分 day0/upgrade
- [ ] T4.2 删除 Fix-level 阶梯中隐含的 day0/upgrade 假设
- [ ] T4.3 "Fix A/B/B+/C" 部分继续保留（torch 2.11 历史速记），但注记为 legacy

### Phase 5 — V1.4 debug 回归（结构重组完成后继续）

- [ ] T6.1 iter 19：把 `(k_cache, v_cache)` tuple 替换为**直接从 `raw_kv_tensor` view 出的连续 5D tensor**（不经 stack，避免拷贝到独立 buffer）
- [ ] T6.2 rebuild overlay + V1.3 token diff + V1.4 smoke
- [ ] T6.3 如果 V1.4 entropy_loss 落入 band → 完成
- [ ] T6.4 如果仍偏离 → 追查 rollout vs actor log-prob 对齐（instrument rollout 输出 logprobs，和 actor 同 token 的 logprobs 做 diff）

## 当前状态（2026-04-23T22:27Z 本文件创建时）

- ✅ V1.3 bit-exact PASS on `easyr1-npu-vllm0200:vllm-day0-vllm0200-20260423-1623-iter18`
- ❌ V1.4 FAIL entropy_loss=3.213 on same image（攻关中，Phase 6）
- 🟡 `src/experts/` 已按 upstream 分类（Phase 0 完成），docs/ 待重组（Phase 1）
- 🟡 day0/upgrade 二分待合并（Phase 2）
- 🟡 `skills/` 老目录待处理（Phase 3）

## 已落的 commit（本 session）

| Commit | Summary |
|---|---|
| `001308c` | restructure: complete per-upstream folder layout for src/experts/ |
| `cf79e59` | restructure: move vllm-ascend-day0-expert under vllm-ascend/ |
| `42b8266` | docs: add MODULE-PORT-STATUS.md |
| `3479368` | docs(readme): link active session PROGRESS.md |
| `b9e6184` | docs(upstream-refs): reframe vllm-ascend entry |
| `cc5ec81` | docs: add GLOSSARY.md |
| `ef9ef46` | day0 skills: rewrite KB from upstream-maintainer perspective |
| `2f43d86` | docs(handover): add §6.8 vllm 0.20 Day-0 V1.3 PASS |
| `9d2bb81` | docs(vllm-day0): mark vllm 0.20 drift resolved at iter 15 |

## 已落的 trace-branch commit（zhshgmail/vllm-ascend）

| Commit | Summary |
|---|---|
| `0ab16321` | [BugFix] vllm 0.20: patched bind_kv_cache must assign single tensor |
| `ca384b09` | [BugFix] vllm 0.20: register kv_cache as stacked tensor |
| `3665da07` | [BugFix] vllm 0.20: override forward_includes_kv_cache_update=False + add do_kv_cache_update |
| `910b2f3d` | [Feature] block_table: add clear_row (vllm 0.20 new API) |
| `da27ea9e` | [BugFix] attention_v1 forward_impl: pre-populate self.key_cache |
| `b6b5462c` | [BugFix] model_runner_v1: port 11 .np/.cpu/.gpu/.copy_to_gpu sites |
| ... 前面还有 iter 1-11 的 commits | |

## V1.4 debug 路径 backlog（在 Phase 6 里展开）

- 当前推测：rollout（vllm）和 actor（FSDP transformer）对同一 token 的 log-prob 计算路径不同 → KL divergence 爆炸
- 验证方法：instrument vllm rollout 输出 `logprobs`，比较 actor forward 在同样 tokens 上的 output logits → logprobs
- 如果 rollout logprobs 差异 >>1%，那问题在 vllm 这边（可能是 iter 17 的 `torch.stack` 制造新 buffer 但 vllm 的 attention logit 读/写不一致）
- 如果 logprobs 相似，问题可能在别处（比如 reward 计算 / KL 系数 / entropy bonus 公式）
