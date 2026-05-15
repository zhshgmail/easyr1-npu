# 测试网 + 安全网 + 反馈环 — 设计文档

> **Status**: P0f 进行中（2026-05-15 起草），等 P0g codex-review 后进入实现。
>
> **Scope**: 借鉴 a5_ops 三件套（test net / safety net / feedback loop），落地 5 个子组件（P0h..P0l）到 easyr1-npu。**这是 design**，不是实现——读者：codex reviewer + next session implementer。
>
> **Source of inspiration**: a5_ops（[gitcode.com/zhengshencn_hwca/a5_ops](https://gitcode.com/zhengshencn_hwca/a5_ops)），具体借鉴：
> - `scripts/run_sanity_suite.sh` + `src/scripts/orchestrator/tests/` （50+ pytest files）
> - `src/scripts/scan_delegation_cheating.py`（mechanical safety scanner）
> - `src/scripts/orchestrator/finalize_pipeline.py`（GateID 多层独立 gate，1971 行）
> - `docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md`（典范 postmortem）
> - `REGRESSION_METHODOLOGY.md`（regression detection 工具栈）

---

## 0. 背景与决策

### 0.1 为什么需要这套（why now）

T22-T28 6 个 session 暴露了 4 类**结构性缺陷**：

| 事故 | 当时为什么没抓到 | 应该被什么抓到 |
|---|---|---|
| T22 walk-through PASS 包装成 end-to-end PASS（被 T25 推翻 3/4） | 没机制要求 outcome 声明带 log path + numeric evidence | **scanner**：扫 outcome claim 是否带 OL-09 provenance |
| T22.4 row 3 shim 3 假 PASS（base image 选错） | 没人独立 verify "shim 3 真在 vllm 0.20.0 import"——agent 自报 PASS | **GateID.OUTCOME_PROVENANCE**：终态前独立 grep + import 验证 |
| T25.5 `ASCEND_RT_VISIBLE_DEVICES` host-vs-container 索引域 bug 漏 10 个 commit | helper script 默认 `--chips 0,1`（happy path），没人压测非零 chip | **test net**：sanity suite 自动跑 helper-script 单元测 + 不同 chip 值 |
| T26 docs 整理后才发现 24 个文档已过时 | 没自动 stale-doc 检测；entry-point 链接没自动校验 | **scanner**：扫 docs/ 内 cross-ref 链接 + ROADMAP §6 staleness |

**共同根因**：当前所有 verify 依赖 LLM 自报；安全网完全为 0（除了人工 review）。

### 0.2 设计原则（5 条，借鉴 a5_ops postmortem）

1. **每层 gate 独立 verify 不同 invariant**——defense in depth；不互信。
2. **gate 优先做 mechanical check**（grep / file-exists / JSON schema），不做 LLM judgment。
3. **status=PASS 是 self-claim**，所有 PASS 必须 cite 一个 mechanical-verifiable artifact（log path / commit SHA / numeric value）。
4. **Safety net 不能 couple 到具体文件名**——name-coupling 是 SAFETY_NET_NAME_COUPLING_2026_05_14 这类 architectural fail 的根。
5. **任何 new feature 必须带 unit test**，commit 前必过 sanity suite。

### 0.3 不在 scope（明确划界）

- 不复制 a5_ops 1971 行 `finalize_pipeline.py` 的复杂度——我们 day-0 是 forward-compat shim work，artifact 数量 ≪ kernel work
- 不引入 GateID 的全部 15+ 个 a5_ops 用的 gate——day-0 场景下最多 5-6 个 mechanical-verifiable invariant
- 不做 `workflow_critic.py` 级别的 PreToolUse hook 拦截——那是 P0e（T29.5），独立工作
- 不做 hardware-level test（a5_ops `tests/hardware/ldg_test`）——我们没有 kernel work

---

## 1. 整体架构

```
┌────────────────────────────────────────────────────────────────────┐
│                       三层闭环架构                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  测试网      │    │  安全网      │    │  反馈环              │  │
│  │  (test net)  │    │  (safety net)│    │  (feedback loop)     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                       │              │
│         │                   │                       │              │
│  ┌──────▼──────────┐ ┌──────▼──────────────┐ ┌──────▼───────────┐  │
│  │ run_sanity_     │ │ scan_outcome_       │ │ snapshots/       │  │
│  │ suite.sh        │ │ claims.py           │ │ <date>_baseline  │  │
│  │ (P0h)           │ │ (P0i)               │ │ .yaml (P0k)      │  │
│  │ tests/          │ │                     │ │                  │  │
│  │ - test_ol_      │ │ finalize_day0_      │ │ postmortem/      │  │
│  │   catalog.py    │ │ check.py + GateID   │ │ POSTMORTEM_      │  │
│  │ - test_roadmap_ │ │ (P0j)               │ │ TEMPLATE.md (P0l)│  │
│  │   debt_fmt.py   │ │                     │ │                  │  │
│  │ - test_skill_   │ │                     │ │                  │  │
│  │   xref.py       │ │                     │ │                  │  │
│  └─────────────────┘ └─────────────────────┘ └──────────────────┘  │
│                                                                    │
│  事前：sanity suite   |  事中：scanner + gate  |  事后：snapshot+   │
│  pre-commit 拦截      |  outcome 声明前拦截    |       postmortem  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

      Pull-flow（实际触发顺序）：

      session work → outcome claim 准备
                            │
                            ▼
                   scan_outcome_claims.py  ← 安全网层 1：扫 provenance
                            │
                            ▼ (PASS)
                   finalize_day0_check.py  ← 安全网层 2：GateID 独立 verify
                            │
                            ▼ (PASS)
                       git commit
                            │
                            ▼
                   pre-commit hook         ← 安全网层 3：drift check
                       (existing)
                            │
                            ▼
                      git push origin
                            │
                            ▼
                   sanity suite (CI)       ← 测试网：unit tests
                            │
                            ▼ (next session)
                   snapshot diff           ← 反馈环：regression detection
                            │
                            ▼ (if regression)
                   postmortem 文件         ← 反馈环：教训沉淀
```

---

## 2. P0h — Sanity Suite Skeleton（测试网）

### 2.1 目录结构

```
repo/
├── scripts/
│   └── run_sanity_suite.sh         # 入口；pre-commit 前必跑；< 2s
├── tests/
│   ├── conftest.py                 # 共享 fixture（repo root, OL catalog 加载）
│   ├── test_ol_catalog.py          # OL-01..OL-27 在 OPERATIONAL_KNOWLEDGE.md 都有 grep keyword 行
│   ├── test_roadmap_format.py      # ROADMAP §6 DEBT-N 必须有 Trigger 字段；§2 P0xx 必须有 status
│   ├── test_skill_xref.py          # 所有 SKILL.md 引用的 OL-XX 都在 OL catalog 里存在
│   ├── test_npu_patterns_id.py     # knowledge/npu-patterns.md 的 NPU-XXX-NNN id 连续 + unique
│   └── test_upstream_forks_ledger.py  # docs/_meta/UPSTREAM_FORKS.md 每行有 branch + status + PR_MATERIAL link
└── src/skills/
    └── ...（已有）
```

### 2.2 入口脚本 `scripts/run_sanity_suite.sh`

```bash
#!/bin/bash
# Sanity test suite for easyr1-npu (P0h, T30.1).
#
# Runs pytest tests under tests/ — all 100% mechanical (no NPU, no LLM).
# Designed to run in < 2s before every commit.
#
# Usage:
#   bash scripts/run_sanity_suite.sh           # output to stdout
#   bash scripts/run_sanity_suite.sh -v        # verbose test names

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1

echo "== Sanity test suite (easyr1-npu) =="
echo "Repo: $(pwd)"
echo "HEAD: $(git rev-parse --short HEAD) ($(git log -1 --format=%s | head -c 60))"
echo ""

RC=0
python3 -m pytest tests/ $@ --tb=short 2>&1 || RC=$?

echo ""
if [[ $RC -eq 0 ]]; then echo "== SANITY SUITE: PASS =="
else echo "== SANITY SUITE: FAIL (exit $RC) =="
fi
exit $RC
```

### 2.3 首批测试（minimal viable）

每个测试都是 `< 100 ms` 的纯文件 IO + grep + parse，无任何 LLM / NPU / docker 依赖。

**test_ol_catalog.py**（核心，规模 ~30 行）：

```python
"""Assert OL-01..OL-27 catalog file is consistent with detail files."""
import re
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent

def test_ol_catalog_has_all_universal_ols():
    """Every OL-NN in ALWAYS_LOADED_UNIVERSAL.md must have a row in OPERATIONAL_KNOWLEDGE.md."""
    universal = (REPO / "src/skills/_shared/references/ALWAYS_LOADED_UNIVERSAL.md").read_text()
    catalog = (REPO / "src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md").read_text()

    universal_ols = set(re.findall(r"OL-(\d+[a-z]?)\b", universal))
    catalog_ols = set(re.findall(r"\| \*\*OL-(\d+[a-z]?)\*\*", catalog))

    missing = universal_ols - catalog_ols
    assert not missing, f"OL in ALWAYS_LOADED but not in catalog: {missing}"

def test_ol_catalog_no_duplicate_ids():
    catalog = (REPO / "src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md").read_text()
    ols = re.findall(r"\| \*\*OL-(\d+[a-z]?)\*\*", catalog)
    assert len(ols) == len(set(ols)), f"duplicate OL ids: {[x for x in ols if ols.count(x) > 1]}"
```

**test_roadmap_format.py**：

```python
def test_debt_entries_have_trigger():
    """Every DEBT-N row in ROADMAP §6 must have a non-empty Trigger column."""
    roadmap = (REPO / "docs/_meta/ROADMAP.md").read_text()
    # Find §6 block + parse table rows
    sec6 = roadmap.split("## §6")[1].split("## §7")[0]
    for row in re.findall(r"\| \*\*DEBT-\d+\*\* \|.*", sec6):
        cols = [c.strip() for c in row.split("|") if c.strip()]
        # Cols: DEBT-N | Why | What fix | Trigger | Effort
        assert len(cols) >= 5, f"DEBT row missing columns: {row[:80]}"
        assert cols[3] and cols[3] not in ("以后", "someday", "TBD"), \
            f"DEBT row Trigger empty / placeholder: {row[:80]}"

def test_p0_entries_have_status():
    """Every P0xx row in ROADMAP §2 must have Status column non-empty."""
    roadmap = (REPO / "docs/_meta/ROADMAP.md").read_text()
    sec2 = roadmap.split("## §2")[1].split("## §3")[0]
    for row in re.findall(r"\| \*\*P0[a-z]+\*\* \|.*", sec2):
        cols = [c.strip() for c in row.split("|") if c.strip()]
        assert len(cols) >= 4 and cols[-1], f"P0 row missing Status: {row[:80]}"
```

**test_skill_xref.py**：

```python
def test_skill_md_ol_references_exist():
    """Every OL-NN cited in any SKILL.md must exist in OL catalog."""
    catalog = (REPO / "src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md").read_text()
    catalog_ols = set(re.findall(r"\*\*OL-(\d+[a-z]?)\*\*", catalog))

    for skill_md in REPO.glob("src/skills/**/SKILL.md"):
        for ol_ref in re.findall(r"OL-(\d+[a-z]?)\b", skill_md.read_text()):
            assert ol_ref in catalog_ols, \
                f"{skill_md.relative_to(REPO)} cites OL-{ol_ref} not in catalog"
```

**test_npu_patterns_id.py**：scan `knowledge/npu-patterns.md`，确保 `NPU-CP-NNN`、`NPU-BUG-NNN`、`NPU-OPS-NNN` 编号连续 + 无重复 + 单调递增。

**test_upstream_forks_ledger.py**：每行 fork ledger 必须有 fork URL + branch + status + PR_MATERIAL link，且 link 路径 file-exists。

### 2.4 PASS 标准

- `bash scripts/run_sanity_suite.sh` 全绿（5 个 test file 全 PASS）
- 时间 < 2s（每个测试是纯文件 IO + 正则 + parse，不调外部服务）
- 集成进 README "Doing tasks" / CLAUDE.md 工作约定：**commit 前必跑**

---

## 3. P0i — Mechanical Scanner: Outcome Claim Provenance（安全网层 1）

### 3.1 目标

agent 准备 emit outcome A / A-with-note / B / C-patch / C-report 时，**先**让 mechanical scanner 验证 claim 是否带 OL-09 provenance。

借鉴 a5_ops `scan_delegation_cheating.py` 但 inverse：不是扫 forbidden patterns，是扫 required patterns。

### 3.2 路径

`src/scripts/safety/scan_outcome_claims.py`

```python
#!/usr/bin/env python3
"""scan_outcome_claims.py — verify outcome claims have OL-09 provenance.

Outcome A / B / C-patch / C-report claims must be backed by:
  - log_path: /tmp/<owner>/easyr1-logs/...  (smoke log)
  - commit_ref: <SHA>  (fork branch commit)
  - numeric_evidence: at least one quantitative value (entropy_loss / reward_score / shape / sha256)

Inputs: PR_MATERIAL.md files (per fork branch) and KB_INDEX.md "case registry" tables.

Exit codes:
  0 PASS — all claims have provenance
  1 FAIL — at least one claim lacks evidence (orchestrator must reject)
  2 usage error
"""
```

### 3.3 检测模式

Forbidden patterns（hedge words 不带证据）：

| Pattern | Failure reason |
|---|---|
| `outcome A.*应该\|probably\|likely\|should` | hedge without numeric proof |
| `PASS\|works\|imports clean` 但 line 周围 5 行内无 `[a-f0-9]{7,}` (commit sha) | no commit ref |
| `PASS\|works` 但 line 周围 10 行内无 `/tmp/.*\.log\|/data/.*\.json` | no log path |
| `entropy_loss\|reward_score\|val_response_length` 没数值 | numeric claim without number |

### 3.4 调用契约

Worker 在 emit outcome 前调：

```bash
python3 src/scripts/safety/scan_outcome_claims.py \
    --pr-material upstream/vllm-ascend/PR_MATERIAL.md
```

非零退出 = orchestrator 拒绝 outcome 写入；worker 必须补 provenance 才能继续。

### 3.5 反 name-coupling 设计

a5_ops postmortem 教训 #1：safety net 不能 couple 到具体文件名。

→ 我们的 scanner 接受 `--pr-material <any-path>` 参数（不 hard-code `PR_MATERIAL.md`）；同时支持 `--kb-case <path-to-KB_INDEX.md>` 扫 case registry 段。新 skill / 新 mode 出来不需改 scanner，只要 caller 把对的 path 传进来。

---

## 4. P0j — GateID + finalize_day0_check.py（安全网层 2）

### 4.1 目标

Day-0 skill 声明 done 前必过多层独立 gate；每个 gate verify **一个**且**不同** invariant。

借鉴 a5_ops `finalize_pipeline.GateID` 但**只保留与 day-0 适用的 6 个**。

### 4.2 路径

`src/scripts/safety/finalize_day0_check.py` + `src/scripts/safety/gate_ids.py`

### 4.3 GateID 列表（精简版，6 个，每个独立 invariant）

```python
class GateID(str, Enum):
    OUTCOME_PROVENANCE = "outcome_provenance"      # P0i scanner PASS 才放
    PR_MATERIAL_EXISTS = "pr_material_exists"      # fork branch 根目录有 PR_MATERIAL.md
    KB_CASE_REGISTRY = "kb_case_registry"          # KB_INDEX.md case registry 段加了本次新 case
    FORK_BRANCH_PUSHED = "fork_branch_pushed"      # gh api / gc api 验证 branch 真已 push
    SMOKE_LOG_PRESENT = "smoke_log_present"        # 至少有一个 A3 smoke log path 存在且 mtime < 24h
    OL_REGRESSION_FREE = "ol_regression_free"      # 没引入新 DEBT-N 与现有 OL 冲突
```

### 4.4 关键设计：mechanical 独立验证

每个 gate 是独立函数，**只**做 mechanical check：

```python
def gate_pr_material_exists(workspace: Path, claim: dict) -> Optional[Rejection]:
    """Verify PR_MATERIAL.md exists in workspace_root and has minimum sections."""
    pr_path = workspace / "PR_MATERIAL.md"
    if not pr_path.exists():
        return Rejection(GateID.PR_MATERIAL_EXISTS, "PR_MATERIAL.md missing",
                         expected=str(pr_path), actual="<not found>")
    content = pr_path.read_text()
    for required in ["## Outcome", "## Reproducer", "## Validation"]:
        if required not in content:
            return Rejection(GateID.PR_MATERIAL_EXISTS,
                             f"PR_MATERIAL.md missing section: {required}",
                             expected=required, actual=content[:200])
    return None
```

**关键**：gate 不读 verification.json 的 `status` 字段；它**只**看 artifact 是否存在 + 结构是否对（避免 a5_ops postmortem #2 的"trust status=PASS"陷阱）。

### 4.5 调用契约

```bash
python3 src/scripts/safety/finalize_day0_check.py \
    --workspace upstream/vllm-ascend \
    --claim '{"outcome": "C-patch", "fork_branch": "ascend-port/vllm-main", ...}'
```

非零退出 = 拒 done；输出列出每个 failed gate + fix 建议。

---

## 5. P0k — Regression Snapshot YAML（反馈环）

### 5.1 目标

每条 fork branch 当前 commit SHA + outcome 写入 baseline YAML；下次 cold-drive 时 diff 验证是否回归。

借鉴 a5_ops `output/<project>/src/kernels/<op>/` 归档 + cold-rerun + 对比 snapshot 模式。

### 5.2 路径

`docs/_meta/snapshots/<YYYY-MM-DD>_baseline.yaml`

### 5.3 Schema

```yaml
date: 2026-05-15
generated_by: T30.4
baselines:
  - skill: /vllm-ascend-day0
    fork: github.com/zhshgmail/vllm-ascend
    branch: ascend-port/vllm-main
    commit_sha: <head SHA>
    outcome: C-patch
    artifacts:
      - vllm_ascend/compat/shared_fused_moe.py
      - vllm_ascend/compat/default_moe_runner.py
      - vllm_ascend/compat/spec_decode_base_proposer.py
    validation_level: L4  # on-A3 import smoke 3/3 PASS
    smoke_log: /tmp/z00637938/easyr1-logs/T22a_vllm_ascend_smoke.log

  - skill: /torch-npu-day0
    ...

  - skill: /sglang-npu-day0
    fork: n/a (no fork — upstream actively maintained)
    image_tag: quay.nju.edu.cn/ascend/sglang:main-cann8.5.0-a3
    sgl_kernel_npu_version: 2026.3.1
    sglang_version: 0.5.10.post2.dev742+g47b8eadbc
    outcome: C-report  # T28 driver init regression
    upstream_issues:
      - https://github.com/sgl-project/sglang/issues/13648
```

### 5.4 Diff 验证脚本

`src/scripts/safety/diff_snapshot.py`：

```bash
python3 src/scripts/safety/diff_snapshot.py \
    --baseline docs/_meta/snapshots/2026-05-15_baseline.yaml \
    --current  # 现场 query fork branches + image versions
```

输出：每条 baseline vs current 比对——commit SHA 前进了？outcome 退化了？artifact 缺失了？

### 5.5 触发时机

- **创建 baseline**：每次 integrated overlay image build 完成时（手动跑或 hook 自动）
- **diff 验证**：每次 cold-drive replay 启动前（next session 入口加一行 "Read 最新 baseline + run diff_snapshot.py"）

---

## 6. P0l — Postmortem 模板 + T25.5 范例（反馈环）

### 6.1 目标

把 T22-T28 真实事故 codify 成 postmortem 文档，作 future regression detection 的"对照样本"。

借鉴 a5_ops `docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md` 结构。

### 6.2 路径

```
docs/_meta/postmortem/
├── POSTMORTEM_TEMPLATE.md
└── 2026-04-28_T25_5_ascend_rt_visible_devices_index_domain_bug.md
```

### 6.3 Postmortem 模板结构（借鉴 a5_ops）

```markdown
# Postmortem: <one-line title>

**Date**: YYYY-MM-DD
**Severity**: HIGH / MEDIUM / LOW
**Cost**: <hours wasted, $ if relevant>
**Reporter**: <user catch | self-discovery during cold-drive>
**Status**: <fix landed commit | followup pending>

## Summary
2-3 段说事故 + 损失.

## What happened (timeline)
按时间顺序 5-10 行.

## Root cause (architectural, not "agent forgot")
明确区分 immediate cause vs root cause.

## How it slipped past every layer
表格：每层应该 catch 但没 catch 的原因.

## Fix (landed <commit>)
- KB rule（如新 OL-N）
- Gate（如新 GateID）
- Test（如新 regression test）
- Doc（如新 CLAUDE.md 规则）

## What we couldn't fix yet
followup 清单.

## Lessons (永久记录)
按 numbered list 写，禁止"以后注意"措辞.
```

### 6.4 范例：T25.5 ASCEND_RT_VISIBLE_DEVICES 索引域 bug

写一篇真 postmortem 覆盖：
- 损失：10 个 commit 都没抓到；如果客户用 `--chips 2,3` 直接 0 GPUs
- root cause：helper script 默认 `--chips 0,1`，host phy-id 和 in-container index 在 0,1 同号下巧合 work；没 mechanical test 压非零 chip
- fix：NPU-OPS-014 / OL-13 + `IN_CONTAINER_CSV` 自动派生（commit `a6f3fca`）
- followup：DEBT-1（A3 stale repo 检测）

---

## 7. 依赖关系 + 实现顺序

```
P0f (本设计) → P0g (codex-review) → P0h → P0i → P0j → P0k → P0l
                                     │
                                     └─ 不强依赖：P0l 可以与 P0i 并行
```

实现严格按 ROADMAP 顺序，**每个子任务完成后跑 sanity suite + 写 commit**，commit message 引用 ROADMAP P0xx + 删 ROADMAP 对应行。

---

## 8. 与 a5_ops 的关键差异

| Dimension | a5_ops | easyr1-npu |
|---|---|---|
| 主体工作 | AscendC kernel 生成（per-op artifact 多） | day-0 forward-compat shim（per-fork artifact 少） |
| Verify 复杂度 | 1971 行 finalize_pipeline.py（精度 + perf + 多 case + bit-exact）| 200-300 行 finalize_day0_check.py（6 个 mechanical gate）|
| Test count | 50+ pytest files | 起步 5 个，按需扩 |
| Hardware test | 有（ldg_test / load_width_test）| 无（不涉及 kernel）|
| GateID 数 | 15+ | 6 |
| Regression snapshot | per-op kernel 文件 binary 对比 | per-fork commit SHA + outcome metadata YAML |
| Postmortem 频率 | 每个 significant fail | 用本设计的 4 个起步范例（T22/T22.4/T25.5/T26）|

**核心借鉴**：架构原则（5 条 §0.2）100% 适用；具体实现按我们规模缩减。

---

## 9. 风险 + open questions

### 9.1 已识别风险

| 风险 | 缓解 |
|---|---|
| Scanner 误报率太高 → agent 反复绕开 | 起步只扫 PR_MATERIAL.md / KB_INDEX.md 这 2 个文件类型；其他文档不扫，先验证 false-positive rate |
| Gate 阻塞合法工作 | 每个 gate 都支持 `--allow-exception <reason>` flag；exception 自动登记到 ROADMAP §6 作 DEBT-N |
| Sanity suite 时间膨胀 | 每个 test < 200 ms 上限；超出必须重构；目标全套 < 2s |
| Snapshot diff 假阳性（上游 fork 自然前进）| baseline YAML 含 `auto_advance_branches: [...]` 白名单，列在内的 commit SHA 前进不报警 |

### 9.2 Codex-review 要点

请 codex 重点 review：

1. **设计闭合性**：5 子任务（P0h-P0l）是否真覆盖 §0.1 的 4 类事故？还有第 5 类我们没想到的？
2. **Over-engineering**：6 个 GateID 是否过多？哪几个其实可以合并？
3. **依赖正确性**：P0h→P0i→P0j 严格串行有必要吗？还是 P0i/P0j 可以并行？
4. **a5_ops 还有什么没借鉴**：a5_ops 的 [aog-self-critic skill](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/src/skills/aog-self-critic/SKILL.md) 有 C1-C41 共 41 项 self-check，我们要不要也加？
5. **Name-coupling 防护**：本设计有没有重蹈 SAFETY_NET_NAME_COUPLING 覆辙？特别是 P0i scanner / P0j gate 的 path 参数是否 mode-invariant？

---

## 10. 见也

- [`docs/_meta/ROADMAP.md` §2 P0f-P0l](../ROADMAP.md)
- [`src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md`](../../src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md)（决策点 cite 映射）
- [`src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md`](../../src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md)（OL-01..27）
- a5_ops 参考：
  - [scripts/run_sanity_suite.sh](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/scripts/run_sanity_suite.sh)
  - [src/scripts/scan_delegation_cheating.py](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/src/scripts/scan_delegation_cheating.py)
  - [src/scripts/orchestrator/finalize_pipeline.py](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/src/scripts/orchestrator/finalize_pipeline.py)
  - [docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md)
