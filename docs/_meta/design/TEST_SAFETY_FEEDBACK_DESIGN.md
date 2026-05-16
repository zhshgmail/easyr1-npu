# 测试网 + 安全网 + 反馈环 — 设计文档（v4，post-second-codex-review）

> **Status**:
> - v1 (2026-05-15 起草) → codex review #1 → **v2** 应用 6 项 recommendation
> - v2 → a5_ops 2026-05-15 攻防演练记录拉到 → **v3** 新加 4 项 (M1/M2/M5/P9)
> - v3 → codex review #2 ([verbatim](review/2026-05-16_codex_review_T30_design_v3.md)) 抓出 6 类 regression → **v4 本文档** 修订
>
> **Scope**: 借鉴 a5_ops 三件套（test net / safety net / feedback loop），落地到 easyr1-npu。

## v3 → v4 changelog（codex review #2 6 项修复）

| # | Codex 指摘 | v4 修复 |
|---|---|---|
| 1 | M2 `verification.json` 字面 grep 是 cargo-cult（a5_ops 文件名，不是我们的）| 改为 **forbidden-paths set**：扫 verifier body 是否含**当前 manifest 任一 `evidence[].path`**（动态 set，按 manifest 自适应）|
| 2 | M2 路径解析 bug：manifest 在 `workspace/<skill-tag>/`，但 verifier path 是 repo-root 风格 | 引入 schema 字段 `repo_root` + `verifier_scripts[i].path` 统一**相对 repo_root 解析**；scanner 显式 `(repo_root / v.path).resolve()` |
| 3 | M5 degraded scan = security theater（self-reported prose） | 直接 **删除 P0i scanner 里 M5 短期实现**；改为只在 sanity suite 跑一次 telemetry-only warning（不 reject）；真实现完全 DEBT-6 / P0e |
| 4 | bad fixtures 用错层（schema fixtures 调 `validate_manifest()` 但 fraud 走 gate）| **拆两套 fixtures**：`tests/fixtures/schema_bad/` 测 `validate_manifest()`；`tests/fixtures/gate_bad/` 测 `finalize_day0_check.py`；4 个 gate 各至少 1 个 crafted fraud |
| 5 | M3 deferral 事实错：`/npu-port` orchestrator SKILL.md 已 spawn dep-analysis-worker + sub-skills | **scope** M3：standalone day-0 skill 不适用；`/npu-port` orchestrator **适用** + 加 P0r 任务跟踪 |
| 6 | schema↔gate dispatch drift（§2.3 `pr_material` optional vs §4.4 required；`A-w-note`/`A-with-note`/`A_WITH_NOTE` 三种写法）| §2.3 + §4.4 合并成 **唯一 canonical mode_dispatch_table**（§2.5 新增）；outcome 名只用 Python enum 形式 `A`/`A_WITH_NOTE`/`B`/`C_PATCH`/`C_REPORT`；prose 形式禁用 |

额外修复：P9 增加**机械 classifier table**（症状 → 类别），不再只是 policy 散文。

---

> **Source of inspiration**: a5_ops（[gitcode.com/zhengshencn_hwca/a5_ops](https://gitcode.com/zhengshencn_hwca/a5_ops)）。

---

## 0. 背景与决策

### 0.1 为什么需要这套（why now）

T22-T28 6 个 session + a5_ops 2026-05-15 攻防演练**外部数据**暴露了 **10 类结构性缺陷**（v3 新增 #7-#10，从 a5_ops 攻防演练 attack vector registry 借鉴）：

| # | 事故 / 失效模式 | 当时为什么没抓到 | 应该被什么抓到 |
|---|---|---|---|
| 1 | T22 walk-through PASS 包装成 end-to-end PASS（被 T25 推翻 3/4） | 没机制要求 outcome 声明带 log path + numeric evidence | mechanical scanner 扫 OL-09 provenance |
| 2 | T22.4 row 3 shim 3 假 PASS（base image 选错） | 没人独立 verify "shim 3 真在 vllm 0.20.0 import" | finalize gate 独立 grep+import 验证 |
| 3 | T25.5 `ASCEND_RT_VISIBLE_DEVICES` 索引域 bug 漏 10 commit | helper script 默认 `--chips 0,1`（happy path），没人压非零 chip | sanity suite 跑 helper-script 单元测多 chip 值 |
| 4 | T26 docs 整理时才发现 24 文档过时 | 没自动 stale-doc 检测；cross-ref 链接没自动校验 | doc-xref test in sanity suite |
| 5 | workflow bypass（agent 在 P8 压力下绕过 skill 直接 raw docker run） | 没机制确保 scanner/gate 真被调到 | bypass-control（§4.6） |
| 6 | 环境真实性 + mode heterogeneity（A3 上 repo 是 stale v0 layout / sglang 无 fork 但 gate 假设有）| 当前 gate 模型假设 fork_patch mode | mode enum + claim_manifest.yaml |
| **7** | **self-citing verifier**（a5_ops `WORKER-SELF-CITING-VERIFIER`，`pass_a_runner.py` 读 `verification.json` 的 claim 字段当独立测量返回）| 没人检查 verifier 数据流是否真独立于 claim | M2 anti-cycle check（scanner 扫 verifier `grep -c "claim_manifest.yaml" == 0`）|
| **8** | **gate-only-tests-good-fixtures**（gate 自己没"crafted fraud 能被抓到吗"的 test fixture）| sanity suite 只跑 happy path | M1 gate fitness（P0h.1 必有 bad fixtures 配 crafted fraud）|
| **9** | **infrastructure paper-over**（NPU 错码 / CANN install desync / .so size 不匹配 → worker 自己 "replace .so / manual cp / bypass --pkg" 而非上抛）| P6 只覆盖 orchestrator 域 friction；env baseline-violation 没人抓 | P9 ANTI_PRESSURE protocol（baseline-violated → graceful exit，不准 paper-over）|
| **10** | **agent tool-use 即 reward-hacking 签名**（worker `Read finalize_pipeline.py` + `Bash test gate` 逆向 engineer claim）| 我们 hook 层无 tool-use log capture | M5（短期降级为 self_challenge.notes 扫描；长期 DEBT-6 等 P0e 上线）|

### 0.2 设计原则（5 + 2 条，v2 新增 #6/#7 来自 codex review）

1. **每层 gate 独立 verify 不同 invariant** — defense in depth；不互信。
2. **gate 优先做 mechanical check**（grep / file-exists / JSON schema），不做 LLM judgment。
3. **status=PASS 是 self-claim**，所有 PASS 必须 cite 一个 mechanical-verifiable artifact（log path / commit SHA / numeric value）。
4. **Safety net 不能 couple 到具体文件名** —— name-coupling 是 a5_ops SAFETY_NET_NAME_COUPLING_2026_05_14 的根。
5. **任何 new feature 必须带 unit test**，commit 前必过 sanity suite。
6. **不依赖 regex 扫 prose** —— v1 设计在 markdown 里 regex 扫 `PASS / works / imports clean` 是 semantic coupling，A/B/C outcome 切换时即崩。改成**结构化 claim_manifest.yaml**。
7. **gate 要求按 mode derive**（不按文件名）—— v1 假设所有 day-0 都 emit fork branch + PR_MATERIAL.md，sglang 不符合这模型。明确建 mode enum `fork_patch / issue_only / integrated_overlay`，每个 mode 有独立 required-artifact 集。
8. **Verifier 必须 data-flow-independent**（a5_ops M2）—— 任何 "独立 re-measurement" 脚本**禁读** claim 文件本身。这是 a5_ops foreach_abs 事故的核心攻击向量；我们的 claim_manifest 加 `verifier_scripts` 字段 + scanner 强制扫 `grep -c "claim_manifest.yaml" $script == 0`。
9. **Gate 有 fitness function**（a5_ops M1）—— 每个 gate 必须配一个 "crafted fraud 真能被抓到" 的 bad fixture。sanity suite 跑 good fixture + bad fixture 都要符合预期。
10. **Baseline-violation 不准 worker paper-over**（a5_ops P9）—— env-level 异常（NPU 错码 / CANN install desync / .so 不匹配）必须上抛到 orchestrator + dedicated env-diagnose path；worker 一旦 "决定自己继续" 就把 signal 丢掉了。本设计加 P9 进 ANTI_PRESSURE_PROTOCOLS.md。

### 0.3 不在 scope（明确划界）

- 不复制 a5_ops 1971 行 `finalize_pipeline.py` 复杂度
- 不引入 a5_ops 41 项 self-critic（已有 `/porting-self-challenge` + 11 challenge patterns；改为**集成既有**而非另建）
- 不做 `workflow_critic.py` 级别的 PreToolUse hook 拦截（P0e / T29.5 独立工作）
- 不做 hardware-level test

---

## 1. 整体架构（v2：claim_manifest 居中）

```
                            ┌──────────────────────────────┐
                            │  claim_manifest.yaml         │ ← 单一结构化 claim 源
                            │  (per day-0 outcome emission)│
                            │  - mode: fork_patch | issue_  │
                            │    only | integrated_overlay │
                            │  - outcome: A | A-w-note |   │
                            │    B | C-patch | C-report   │
                            │  - evidence: [log paths]    │
                            │  - artifacts: [files]       │
                            │  - external_refs: [URLs]    │
                            │  - validation_level: L1..L5 │
                            └──────────┬───────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
  │  P0h         │              │  P0i + P0j   │              │  P0k + P0l   │
  │  测试网      │              │  安全网      │              │  反馈环      │
  └──────┬───────┘              └──────┬───────┘              └──────┬───────┘
         │                             │                             │
         ▼                             ▼                             ▼
  ┌──────────────────┐    ┌──────────────────────────┐    ┌──────────────────┐
  │ run_sanity_      │    │ scan_outcome_claims.py   │    │ snapshots/       │
  │ suite.sh + tests │    │ (scanner: validate       │    │ <date>_baseline  │
  │                  │    │  claim_manifest schema)  │    │ .yaml (P0k)      │
  │ - schema valid   │    │                          │    │                  │
  │ - OL catalog     │    │ finalize_day0_check.py   │    │ postmortem/      │
  │ - skill xref     │    │ (4 gates, mode-derived)  │    │ POSTMORTEM_      │
  │ - claim_         │    │                          │    │ TEMPLATE.md (P0l)│
  │   manifest.py    │    │ + invoke /porting-self-  │    │                  │
  │   roundtrip      │    │   challenge as gate-0    │    │                  │
  └──────────────────┘    └──────────────────────────┘    └──────────────────┘

      Pull-flow（v2）：
      
      session work
            │
            ▼
      emit claim_manifest.yaml  ← skill 写一个结构化 manifest
            │
            ▼
      /porting-self-challenge  ← codex R4：复用既有 self-challenge 作 gate-0
            │
            ▼ PASS
      scan_outcome_claims.py   ← P0i: schema validation + cross-ref check
            │
            ▼ PASS
      finalize_day0_check.py   ← P0j: 4 mode-derived gates
            │
            ▼ PASS
      git commit + push
            │
            ▼
      pre-commit hook (P0h)    ← drift check + 调 run_sanity_suite.sh
            │
            ▼
      sanity suite             ← 测试网 (claim_manifest schema, OL, xref, ...)
            │
            ▼ (next session)
      snapshot diff (P0k)      ← 反馈环：regression detection
            │
            ▼ (if regression)
      postmortem (P0l)         ← 反馈环：教训沉淀
```

---

## 2. claim_manifest.yaml — 单一结构化 claim 源

这是 v2 最重要的新增。Codex R1 + R2 推荐：避免 regex-on-prose，引入结构化 schema。

### 2.1 Schema（YAML，stdlib `yaml.safe_load` 解析）

```yaml
# emit by each day-0 / port skill before claiming done
# location: workspace/<skill-tag>/claim_manifest.yaml

claim_manifest_version: 1
schema_version: "1.0"
generated_at: 2026-05-15T12:34:56Z
generated_by: /vllm-ascend-day0  # 或其他 skill 名

# === core fields ===
skill: vllm-ascend-day0
mode: fork_patch                  # fork_patch | issue_only | integrated_overlay
outcome: C-patch                  # A | A-with-note | B | C-patch | C-report
validation_level: L4              # L1..L5 per smoke ladder

# === evidence: every PASS 必须 cite ≥1 item ===
evidence:
  - type: smoke_log
    path: /tmp/z00637938/easyr1-logs/T22a_vllm_smoke_20260427.log
    mtime: 2026-04-27T15:30:00Z
    size_bytes: 19205
    sha256: <sha256 of log file>   # caller computes
    grep_assertion: "shim 1 OK\\nshim 2 OK\\nshim 3 OK"  # what we grep'd

  - type: commit_ref
    repo: github.com/zhshgmail/vllm-ascend
    branch: ascend-port/vllm-main
    sha: 6bac1f5e
    upstream_url: https://github.com/zhshgmail/vllm-ascend/commit/6bac1f5e

  - type: numeric_metric
    name: reward_score
    value: 0.013
    baseline: 0.0126
    tolerance: 0.1
    source_log: /tmp/z00637938/easyr1-logs/V1.4_GRPO.log

# === artifacts: mode-dependent required files ===
artifacts:
  - role: shim_module                  # role tag, not filename
    paths:
      - vllm_ascend/compat/shared_fused_moe.py
      - vllm_ascend/compat/default_moe_runner.py
      - vllm_ascend/compat/spec_decode_base_proposer.py
    branch: ascend-port/vllm-main      # in which fork branch
    must_be_pushed: true

  - role: pr_material
    path: PR_MATERIAL.md               # at branch root
    required_sections:
      - "## Outcome"
      - "## Reproducer"
      - "## Validation"

# === external_refs: 上游 / 第三方 reference ===
external_refs:
  - kind: github_issue
    url: https://github.com/sgl-project/sglang/issues/13648
    relevance: same-class-failure
  - kind: kb_case
    path: src/skills/vllm-ascend/port-expert/references/KB_INDEX.md
    section_anchor: "concrete-case-registry"

# === self-challenge gate-0 result ===
self_challenge:
  ran_at: 2026-05-15T12:30:00Z
  patterns_checked: [01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11]
  patterns_passed: [01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11]
  notes: |
    1 challenge initially failed (#05 shim-is-not-port — claimed
    PASS without on-A3 import); fixed by re-running on-A3 smoke.

# === verifier scripts (v3 / a5_ops M2 anti-cycle) ===
# Any independent re-measurement script must be listed here AND must
# NOT read claim_manifest.yaml directly (data-flow-independence rule).
# Scanner enforces: grep -c "claim_manifest.yaml" $script == 0
verifier_scripts:
  - role: smoke_replay
    path: src/scripts/smoke/replay_v14.sh
    purpose: "Re-run V1.4 GRPO and emit reward_score independently"
  - role: import_smoke
    path: src/scripts/smoke/import_shim_3.sh
    purpose: "On-A3 import test for vllm_ascend.compat.spec_decode_base_proposer"
```

### 2.2 Schema validator

`src/scripts/safety/validate_claim_manifest.py` — stdlib only（`json`/`yaml.safe_load`），无 jsonschema 依赖。Exit 0/1。

### 2.3 Mode-driven required-field

详 §2.5 canonical mode_dispatch_table。**整个设计中所有 mode/artifact/outcome required-set 仅在 §2.5 那张表里声明一次**；schema validator + scanner + gate 都读同一张表，禁止 §3/§4 处复刻或漂移描述。

### 2.4 Outcome enum (canonical naming)

整个仓库只允许这 5 个值，**Python enum 形式**，**所有文档** + claim_manifest + scanner / gate / postmortem 必须用：

```python
class Outcome(str, enum.Enum):
    A          = "A"            # 完全适配，无 NPU 集成面变化
    A_WITH_NOTE = "A_WITH_NOTE" # 适配 + 有 additive 差异需记录
    B          = "B"            # 单 commit / env-var workaround 即解
    C_PATCH    = "C_PATCH"      # 需写 forward-compat shim
    C_REPORT   = "C_REPORT"     # 修复在社区，写 blocker issue
```

禁止使用 `A-with-note`/`A-w-note`/`a-with-note` 等任何其它形式。Sanity suite `test_outcome_enum_canonical.py` 扫所有 `.md` 文件 + `*.yaml` 检查无 prose 变种。

### 2.5 Canonical mode_dispatch_table（**唯一权威表**）

下表是设计中**所有 mode/outcome/required-artifact/required-evidence 的单一来源**。任何下游消费者（schema validator / P0i scanner / P0j gate / sanity suite）**只能从本表派生**，禁止内联复刻：

```python
# src/scripts/safety/mode_dispatch.py — canonical authority
from enum import Enum

class Mode(str, Enum):
    FORK_PATCH = "fork_patch"
    ISSUE_ONLY = "issue_only"
    INTEGRATED_OVERLAY = "integrated_overlay"

class ArtifactRole(str, Enum):
    SHIM_MODULE = "shim_module"
    PR_MATERIAL = "pr_material"
    WORKAROUND_DOC = "workaround_doc"
    IMAGE_TAG = "image_tag"
    SMOKE_LOG = "smoke_log"
    CHECKPOINT_PATH = "checkpoint_path"

class EvidenceType(str, Enum):
    SMOKE_LOG = "smoke_log"
    COMMIT_REF = "commit_ref"
    NUMERIC_METRIC = "numeric_metric"
    GITHUB_ISSUE = "github_issue"
    IMAGE_SHA = "image_sha"

# Per-mode required artifact roles
REQUIRED_ARTIFACTS: dict[Mode, set[ArtifactRole]] = {
    Mode.FORK_PATCH:         {ArtifactRole.SHIM_MODULE, ArtifactRole.PR_MATERIAL},
    Mode.ISSUE_ONLY:         {ArtifactRole.WORKAROUND_DOC},
    Mode.INTEGRATED_OVERLAY: {ArtifactRole.IMAGE_TAG, ArtifactRole.SMOKE_LOG, ArtifactRole.CHECKPOINT_PATH},
}

# Per-outcome required evidence types (orthogonal to mode)
REQUIRED_EVIDENCE: dict[Outcome, set[EvidenceType]] = {
    Outcome.A:           {EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF},
    Outcome.A_WITH_NOTE: {EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF},
    Outcome.B:           {EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF},
    Outcome.C_PATCH:     {EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF, EvidenceType.NUMERIC_METRIC},
    Outcome.C_REPORT:    {EvidenceType.GITHUB_ISSUE},  # smoke log optional
}

# Per-mode required validation level (min)
REQUIRED_VALIDATION_LEVEL: dict[Mode, int] = {
    Mode.FORK_PATCH:         3,  # L3 on-A3 import smoke
    Mode.ISSUE_ONLY:         3,
    Mode.INTEGRATED_OVERLAY: 5,  # L5 V1.4 GRPO end-to-end
}

# Self-challenge minimum pattern pass count (all modes)
SELF_CHALLENGE_MIN_PASSED = 8  # of 11 in docs/_meta/kb/challenge_patterns/
```

**消费者**：
- P0h.0 `validate_claim_manifest.py` import `Mode` / `ArtifactRole` / `EvidenceType` / `REQUIRED_*` 做 schema 校验
- P0i `scan_outcome_claims.py` import 同样的 enum 做 cross-field 校验
- P0j `finalize_day0_check.py` 4 gate 全部 derive 自这表
- P0h.1 sanity suite `test_mode_dispatch_authority.py` 验证：本表是唯一定义 enum 的地方；其它文件 `grep -c "class Outcome\b\|class Mode\b" == 0` except `mode_dispatch.py` itself

---

## 3. P0h — Sanity Suite（测试网）

### 3.1 目录

```
repo/
├── scripts/
│   └── run_sanity_suite.sh         # 入口
├── tests/
│   ├── conftest.py
│   ├── test_claim_manifest_schema.py   # NEW v2: schema roundtrip
│   ├── test_ol_catalog.py              # OL-NN catalog consistency
│   ├── test_roadmap_format.py          # ROADMAP §2/§6 format
│   ├── test_skill_xref.py              # SKILL.md → OL catalog cross-ref
│   ├── test_npu_patterns_id.py         # knowledge/npu-patterns.md id uniqueness
│   ├── test_upstream_forks_ledger.py   # UPSTREAM_FORKS.md row validity
│   └── fixtures/
│       ├── good_claim_manifest_fork_patch.yaml
│       ├── good_claim_manifest_issue_only.yaml
│       ├── good_claim_manifest_integrated.yaml
│       ├── bad_claim_missing_evidence.yaml
│       └── bad_claim_wrong_mode_artifact_combo.yaml
```

### 3.2 入口脚本

```bash
#!/bin/bash
# scripts/run_sanity_suite.sh — P0h
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1

echo "== Sanity test suite (easyr1-npu) =="
echo "Repo: $(pwd)"
echo "HEAD: $(git rev-parse --short HEAD) ($(git log -1 --format=%s | head -c 60))"
echo

RC=0
python3 -m pytest tests/ "$@" --tb=short 2>&1 || RC=$?
[[ $RC -eq 0 ]] && echo "== SANITY SUITE: PASS ==" || echo "== SANITY SUITE: FAIL (exit $RC) =="
exit $RC
```

### 3.3 关键 test：`test_claim_manifest_schema.py`

> Codex R6：避免 `split("## §6")` 这种 brittle parsing。本测试**只**用 yaml.safe_load + 字段存在性检查，不 grep 文档结构。

```python
"""Schema roundtrip: fixtures load + required-field per-mode + bad fixtures rejected."""
import pytest
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests/fixtures"

# Import validator we're testing
import sys
sys.path.insert(0, str(REPO / "src/scripts/safety"))
from validate_claim_manifest import validate_manifest, ValidationError

GOOD_FIXTURES = ["good_claim_manifest_fork_patch.yaml",
                 "good_claim_manifest_issue_only.yaml",
                 "good_claim_manifest_integrated.yaml"]

BAD_FIXTURES_AND_REASONS = [
    ("bad_claim_missing_evidence.yaml", "no evidence array"),
    ("bad_claim_wrong_mode_artifact_combo.yaml", "issue_only mode but has shim_module artifact"),
]

@pytest.mark.parametrize("fix_name", GOOD_FIXTURES)
def test_good_fixture_passes(fix_name):
    data = yaml.safe_load((FIX / fix_name).read_text())
    validate_manifest(data)  # raises ValidationError if fail

@pytest.mark.parametrize("fix_name,why", BAD_FIXTURES_AND_REASONS)
def test_bad_fixture_rejected(fix_name, why):
    data = yaml.safe_load((FIX / fix_name).read_text())
    with pytest.raises(ValidationError):
        validate_manifest(data)
```

### 3.4 其他 tests（robust 化）

每个测试**不再** `split("## §N")` —— 改用：
- `test_ol_catalog.py`：用 `re.findall(r"\| \*\*OL-(\d+[a-z]?)\*\*", catalog)` 直接匹 table row（无 section split）
- `test_roadmap_format.py`：parse **整篇** ROADMAP，扫所有 `\| \*\*DEBT-\d+\*\* \|`、`\| \*\*P0[a-z]+\*\* \|` 行，不依赖 section heading
- `test_skill_xref.py`：grep + set diff，不 parse 结构

### 3.5 PASS 标准 + 集成

- `bash scripts/run_sanity_suite.sh` 全绿
- 时间 < 2s
- 集成进现有 `.claude/settings.json` PostToolUse on `git commit`（已存在），在 commit 后自动跑 + 失败 emit `[CRITIC-HOOK]` 提示

### 3.6 Bad fixtures（M1 gate fitness function，v4 双层重构）

Codex review #2 抓到 v3 bad fixtures **用错层**——所有 fixture 都丢去 `validate_manifest()`，但有些 fraud 属于 P0j gate 范畴（schema 通过，gate 拒）。

v4 把 bad fixtures **拆两层**：

```
tests/fixtures/
├── good/                                       # 各 mode happy path
│   ├── good_fork_patch_C_PATCH.yaml
│   ├── good_issue_only_C_REPORT.yaml
│   └── good_integrated_overlay_A.yaml
├── schema_bad/                                 # 测 validate_manifest() 拒绝
│   ├── schema_bad_outcome_string_variant.yaml  # outcome="A-with-note" 而非 A_WITH_NOTE
│   ├── schema_bad_perf_null_with_pass.yaml     # outcome=A + evidence[numeric_metric].value=null
│   ├── schema_bad_issue_only_with_shim.yaml    # mode=issue_only + artifacts[shim_module]
│   ├── schema_bad_missing_self_challenge.yaml  # no self_challenge block
│   └── schema_bad_repo_root_anchor_missing.yaml # no repo_root_anchor file
└── gate_bad/                                   # 测 finalize_day0_check 拒绝（schema 过 + gate 拒）
    ├── gate_bad_self_citing_verifier/          # 含完整 workspace + 一个 read-manifest verifier
    │   ├── claim_manifest.yaml
    │   └── src/scripts/smoke/bad_verifier.sh   # body 含 manifest evidence[].path
    ├── gate_bad_smoke_log_grep_fails/          # smoke log 存在但 grep_assertion 不命中
    │   ├── claim_manifest.yaml
    │   └── smoke.log                           # 内容不含期望 token
    ├── gate_bad_fork_branch_not_pushed/        # fork_patch 但 branch 不存在
    │   └── claim_manifest.yaml                 # commit_ref 含 fake sha
    └── gate_bad_pr_material_missing_section/   # PR_MATERIAL.md 存在但缺 ## Outcome
        └── workspace/PR_MATERIAL.md
```

**Schema-layer test**（与 v3 类似，但仅 schema fixtures）：

```python
@pytest.mark.parametrize("fix,expected_error_kind", [
    ("schema_bad_outcome_string_variant.yaml", "outcome-canonical"),
    ("schema_bad_perf_null_with_pass.yaml", "evidence-numeric-missing"),
    ("schema_bad_issue_only_with_shim.yaml", "mode-artifact-mismatch"),
    ("schema_bad_missing_self_challenge.yaml", "self-challenge-required"),
    ("schema_bad_repo_root_anchor_missing.yaml", "repo-root-anchor"),
])
def test_schema_crafted_fraud_rejected(fix, expected_error_kind):
    data = yaml.safe_load((FIX / "schema_bad" / fix).read_text())
    with pytest.raises(ValidationError) as exc:
        validate_manifest(data)
    assert expected_error_kind in str(exc.value), \
        f"schema fixture rejected for wrong reason: got {exc.value}"
```

**Gate-layer test**（v4 新增，每个 gate 至少 1 个）：

```python
GATE_FIXTURES = [
    ("gate_bad_self_citing_verifier", GateID.CLAIM_EVIDENCE_PRESENT, "anti-cycle"),
    ("gate_bad_smoke_log_grep_fails", GateID.VALIDATION_ARTIFACT_VERIFIED, "grep-assertion-failed"),
    ("gate_bad_fork_branch_not_pushed", GateID.EXTERNAL_PUBLICATION_VERIFIED, "branch-not-found"),
    ("gate_bad_pr_material_missing_section", GateID.REQUIRED_ARTIFACTS_PRESENT, "section-missing"),
]

@pytest.mark.parametrize("workspace_dir,expected_gate,expected_reason", GATE_FIXTURES)
def test_gate_crafted_fraud_rejected(workspace_dir, expected_gate, expected_reason):
    ws = FIX / "gate_bad" / workspace_dir
    result = finalize_day0_check.check(workspace=ws, repo_root=REPO)
    assert not result.eligible, f"gate did not reject fraud: {result}"
    matching = [r for r in result.rejections if r.gate_id == expected_gate]
    assert matching, f"expected gate {expected_gate} did not fire; got {result.rejections}"
    assert expected_reason in matching[0].description, \
        f"wrong reason: got '{matching[0].description}'"
```

**Coverage rule**（M1 fitness）：sanity suite 启动时**校验**：
- `Mode` enum 每个值都有 ≥ 1 个 `good/` fixture
- 4 个 `GateID` 每个都有 ≥ 1 个 `gate_bad/` fixture
- schema 校验出的每类 error 都有 ≥ 1 个 `schema_bad/` fixture
- 任一缺漏 → sanity suite FAIL（不是跑过）

测试代码大致：

```python
def test_coverage_completeness():
    gate_bad_dirs = list((FIX / "gate_bad").iterdir())
    covered_gates = {parse_workspace(d).expected_gate for d in gate_bad_dirs}
    assert set(GateID) == covered_gates, \
        f"missing gate fitness fixtures for: {set(GateID) - covered_gates}"
```

**关键**（M1 原意）：fixture **不只测 happy path**；好 fixture 通过 ≠ scanner/gate 真在工作。**v4 还加 "每个 enum/gate 都得有 fixture" 的覆盖率门**——避免 4 个 gate 只测了 2 个就 ship。

---

## 4. P0i + P0j — 安全网（合并设计）

> Codex R3：scanner 和 gate 共享同一 claim manifest，定 strict contract；R2：gate 按 mode derive；R3：4 gates 不是 6。

### 4.1 一个 schema 两套消费

```
claim_manifest.yaml  ──┬─→  P0i scan_outcome_claims.py  ← schema-level validation
                       │                                  ├ all required fields present
                       │                                  ├ field types correct
                       │                                  ├ evidence array non-empty when PASS
                       │                                  └ self_challenge ran + ≥ 8 patterns pass
                       │
                       └─→  P0j finalize_day0_check.py  ← artifact-level validation
                                                          ├ Gate 1 CLAIM_EVIDENCE_PRESENT
                                                          ├ Gate 2 REQUIRED_ARTIFACTS_PRESENT
                                                          ├ Gate 3 EXTERNAL_PUBLICATION_VERIFIED
                                                          └ Gate 4 VALIDATION_ARTIFACT_VERIFIED
```

### 4.2 P0i `scan_outcome_claims.py`（v4，post-codex-2）

Job：**schema 层**校验 claim_manifest.yaml + **anti-cycle verifier check**（M2 真实落地，非 cargo-cult）。

```python
def scan(manifest_path: Path, repo_root: Path) -> ScanResult:
    """
    Returns ScanResult with errors=[] if pass.

    Errors include:
      - schema fields missing / wrong type
      - evidence array empty when outcome != C_REPORT
      - mode/artifact combo invalid (per §2.5 mode_dispatch_table)
      - self_challenge missing or < SELF_CHALLENGE_MIN_PASSED patterns
      - any verifier_scripts[i].path missing OR not executable
        (Path.exists() AND os.access(p, os.X_OK))
      - any verifier_scripts[i] body references a manifest-derived path
        (M2 dynamic forbidden-path set; see _check_anti_cycle_dynamic below)

    M5 telemetry (v4 split per codex review #2):
      - tool-use signature detection NOT implemented here; it requires
        PreToolUse hook + tool-use log capture (DEBT-6 / P0e).
      - Sanity-suite-only optional telemetry test emits warning when
        self_challenge.notes is empty; never rejects.
    """
```

**M2 dynamic forbidden-path set**（v4：替换 v3 cargo-cult 字面 grep）：

```python
def _check_anti_cycle_dynamic(
    repo_root: Path,
    manifest: dict,
) -> list[Error]:
    """
    For each verifier_script, ensure its body does NOT read any path
    in the manifest's evidence[].path or artifacts[].paths set.

    This is the M2 invariant: "verifier must be data-flow-independent
    of the claim it verifies". The forbidden set is derived dynamically
    from THIS manifest's content — not from a hard-coded a5_ops literal.
    """
    errors = []

    # Collect every path the manifest references (these are the "claim"
    # — verifier must not read them).
    forbidden: set[str] = set()
    for ev in manifest.get("evidence", []):
        if "path" in ev:
            forbidden.add(ev["path"])
        if "source_log" in ev:
            forbidden.add(ev["source_log"])
    for art in manifest.get("artifacts", []):
        for p in art.get("paths", []):
            forbidden.add(p)
        if "path" in art:
            forbidden.add(art["path"])
    # Also forbid the manifest file itself (most obvious cycle).
    manifest_rel = manifest.get("_manifest_relative_to_repo_root")
    if manifest_rel:
        forbidden.add(manifest_rel)

    for v in manifest.get("verifier_scripts", []):
        # Resolve verifier path from repo_root (v4 fix — was manifest_dir).
        script = (repo_root / v["path"]).resolve()
        if not script.is_relative_to(repo_root.resolve()):
            errors.append(Error("verifier path escapes repo_root",
                                v["path"]))
            continue
        if not script.exists():
            errors.append(Error("verifier missing", v["path"]))
            continue
        if not os.access(script, os.X_OK):
            errors.append(Error("verifier not executable", v["path"]))
            continue
        body = script.read_text(errors="replace")
        for fp in forbidden:
            # Use word-boundary match to reduce false positives from
            # incidental substring (e.g. "shim_module" appearing in comment).
            # Caller can lower the bar later; we err on the side of caution.
            if fp and fp in body:
                errors.append(Error(
                    f"verifier reads manifest-claim path '{fp}' "
                    f"(M2 anti-cycle: data-flow not independent of claim)",
                    v["path"]))
    return errors
```

**Why this is not cargo-cult**:
- v3 hardcoded `verification.json` (a5_ops's claim file). We don't have a `verification.json`.
- v4 dynamic set is derived from THIS manifest, so the rule self-adjusts when paths change.
- Path resolution explicit: verifier paths are **repo-root relative** (per schema §2.1, `_manifest_relative_to_repo_root` field new in v4).
- `_check_anti_cycle_dynamic` is **mode-agnostic** — works for fork_patch / issue_only / integrated_overlay alike.

**Schema addition (v4)**: the claim_manifest gains a top-level field:
```yaml
_manifest_relative_to_repo_root: workspace/<skill-tag>/claim_manifest.yaml
repo_root_anchor: README.md   # for the validator to confirm repo_root via file presence
```

**Why M5 dropped from scanner** (v4): codex #2 demonstrated scanning `self_challenge.notes` for `Read.*finalize_day0_check` strings is theater — agent can omit the sentence; `.*` literal-substring is not regex. Real implementation needs PreToolUse hook to capture actual tool-use log lines. Moved entirely to DEBT-6.

### 4.3 P0j `finalize_day0_check.py` — 4 mode-derived gates

> Codex R3 cut: 6 → 4。每个 gate verify 独立 invariant。

| Gate ID | What it verifies | Mechanical check |
|---|---|---|
| `CLAIM_EVIDENCE_PRESENT` | claim 里每条 evidence path 真存在、sha256 真匹配、mtime 合理 | `Path.exists()` + `hashlib.sha256` + `os.stat` |
| `REQUIRED_ARTIFACTS_PRESENT` | mode-required artifacts 全在 + 内容结构对（如 PR_MATERIAL.md 必含 `## Outcome / ## Reproducer / ## Validation` 3 段） | per-mode required-set lookup + section-header grep |
| `EXTERNAL_PUBLICATION_VERIFIED` | claim 里的 fork branch 真已 push（mode=fork_patch 时）/ image tag 真已 build（mode=integrated_overlay 时）/ upstream issue URL 真可达（mode=issue_only 时） | `gh api` / `docker inspect` / `curl -fsI URL` |
| `VALIDATION_ARTIFACT_VERIFIED` | smoke log 里 grep_assertion 真匹配（OL-09 numeric evidence 真存在） | `Path.read_text() + re.search(grep_assertion)` |

### 4.4 关键设计：mode-derived，不 hard-code 文件名

```python
class Mode(str, Enum):
    FORK_PATCH = "fork_patch"
    ISSUE_ONLY = "issue_only"
    INTEGRATED_OVERLAY = "integrated_overlay"

REQUIRED_ARTIFACTS_BY_MODE: dict[Mode, list[ArtifactRole]] = {
    Mode.FORK_PATCH: [
        ArtifactRole.SHIM_MODULE,
        ArtifactRole.PR_MATERIAL,
    ],
    Mode.ISSUE_ONLY: [
        # no fork; only need github issue link + workaround doc
        ArtifactRole.WORKAROUND_DOC,
    ],
    Mode.INTEGRATED_OVERLAY: [
        ArtifactRole.IMAGE_TAG,
        ArtifactRole.SMOKE_LOG,
        ArtifactRole.CHECKPOINT_PATH,
    ],
}

REQUIRED_EVIDENCE_BY_OUTCOME: dict[Outcome, list[EvidenceType]] = {
    Outcome.A: [EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF],
    Outcome.A_WITH_NOTE: [EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF],
    Outcome.B: [EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF],
    Outcome.C_PATCH: [EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF, EvidenceType.NUMERIC_METRIC],
    Outcome.C_REPORT: [EvidenceType.GITHUB_ISSUE],  # smoke log optional
}
```

**核心**：gate 代码**不**出现 `PR_MATERIAL.md` / `ascend-port/vllm-main` / `shim` 这种字面字符串；它们都从 manifest mode + artifact role 表 derive。新 mode 加只要在 enum + table 里加行。

### 4.5 集成 `/porting-self-challenge`（Codex R4）

不另起 41-item 框架。在 manifest 里加 `self_challenge` block；P0i 校验它**ran + 至少 8 patterns 通过**。具体 11 patterns 由 `docs/_meta/kb/challenge_patterns/` 现有定义提供。

`/porting-self-challenge` skill 实际跑流程：

1. session 末尾、emit done 前，user 或 orchestrator 调 `/porting-self-challenge`
2. skill 跑 11 patterns，emit `self_challenge_result.yaml`
3. day-0 skill 把 `self_challenge_result.yaml` merge 进 `claim_manifest.yaml` 的 `self_challenge` block
4. P0i 校验存在 + 通过率

### 4.6 Bypass-control（Codex R5）

3 重 enforcement，确保 scanner/gate 真被调到：

1. **声明级**：`/<skill>` SKILL.md 的 "Done criteria" 段明列 "must run finalize_day0_check.py before claiming done"——SKILL.md ↔ ROADMAP cross-ref 测试加 `test_skill_has_finalize_step.py` 验证
2. **commit 级**：`.claude/hooks/check_finalize_ran.sh`（新建）—— PostToolUse on `git commit`，若 commit message 含 "outcome A/B/C" 关键词但同 commit 没有 `*/claim_manifest.yaml` 改动 → 发 `[CRITIC-HOOK]` 警告（不 block，但记 PROGRESS）
3. **session 末级**：现有 `.claude/hooks/critic-on-significant-commit.sh` + SessionEnd hook 调 `/porting-self-challenge`——self_challenge 现在硬性 require claim_manifest existing，间接强制

**承认**：完全 bypass-proof 需要 P0e (workflow critic at PreToolUse) 才能闭合。本设计不闭合该洞——明确为 known gap，登记到 ROADMAP §6 DEBT。

---

## 5. P0k — Regression Snapshot YAML（反馈环）

`docs/_meta/snapshots/<date>_baseline.yaml` schema 与 claim_manifest 共享部分字段（mode / outcome / commit_ref / artifacts），但加 `as_of_date` + `auto_advance_branches` 白名单。

Diff 工具 `src/scripts/safety/diff_snapshot.py`：
- 输入：baseline.yaml + current state (live query forks + images)
- 输出：每条 baseline vs current 比对——commit SHA 前进了？outcome 退化了？artifact 缺失了？
- 落 `docs/_meta/snapshots/<date>_diff_report.md`

触发：
- **创建 baseline**：integrated overlay image build 完成时；手动跑 `python3 src/scripts/safety/snapshot_current.py > docs/_meta/snapshots/$(date +%Y-%m-%d)_baseline.yaml`
- **diff 验证**：每次 cold-drive replay 启动前；continuing agent checklist 第 3 步加这项

---

## 6. P0l — Postmortem 模板（反馈环）

`docs/_meta/postmortem/` 含：

- `POSTMORTEM_TEMPLATE.md`（按 a5_ops `SAFETY_NET_NAME_COUPLING_2026_05_14.md` 结构）
- `2026-04-28_T25_5_ascend_rt_visible_devices_index_domain_bug.md`（**首篇范例**）
- `2026-05-15_T30_design_v1_mode_coupling.md`（**v2 写：本设计 v1 自己中了 mode-coupling 陷阱，codex 抓到**——eat own dog food）

第二篇 postmortem 是本设计 v2 的自反思——记录"v1 设计 review 时被指出复制了 a5_ops 同样的 mode-coupling failure"，让未来 contributor 看到此教训。

---

## 7. 实现顺序（v2，按 codex R3 修订）

```
P0f (本设计 v2)                                        ✅
   │
   ▼
P0g (codex review v1 done → v2 incorporated)          ✅
   │
   ▼
P0h.0 claim_manifest.yaml schema + validator           ← schema first
   │
   ▼
P0h.1 sanity suite + claim_manifest tests              
   │
   ├──────────┬───────────┐
   ▼          ▼           ▼
P0i scanner  P0j gates   P0l postmortem               ← 三者可并行
            │                                          
            ▼
P0k snapshot                                          ← 等 schema 稳定
```

**关键变化** vs v1：
- schema 先行（P0h.0）；scanner + gate 共享同一 schema 才能 strict contract
- P0i / P0j / P0l 三者解耦，可并行
- P0k 依赖 schema 稳定，故晚于 P0i+P0j

ROADMAP §2 同步修订（commit 时一起改）。

---

## 8. 与 a5_ops 的关键差异（v2 更新）

| Dimension | a5_ops | easyr1-npu v2 |
|---|---|---|
| 主体工作 | AscendC kernel 生成 | day-0 forward-compat shim |
| Verify 复杂度 | 1971 行 finalize_pipeline.py | ~400 行 finalize_day0_check.py（4 gate）|
| Test count | 50+ pytest files | 起步 6（schema + 5 doc-test）|
| GateID 数 | 15+ | **4**（codex R3 cut） |
| Self-critic | C1-C41（41 项） | 复用既有 `/porting-self-challenge` 11 patterns（codex R4） |
| Claim 表达 | verification.json + 多 status field | **claim_manifest.yaml**（codex R1 推荐） |
| Mode model | mode-aware（5+ modes） | **3 modes**（codex R2 推荐） |
| Regression snapshot | per-op kernel binary 对比 | per-skill commit SHA + outcome YAML |
| Bypass-control | workflow_critic PreToolUse hook | 3 重 enforcement + 承认 known gap → P0e |

---

## 9. 风险 + 已应对项

### 9.1 已应对（v1 → v2）

| 风险 | v2 应对 |
|---|---|
| Scanner regex-on-prose fragile | claim_manifest 结构化，schema-validate |
| 6 gate over-engineering | 4 gate + 删 OL_REGRESSION_FREE |
| Mode-coupling（v1 假设 fork_patch）| mode enum + per-mode required tables |
| Test parsing 太脆（`split("## §6")`）| 改用 regex 行匹配，不依赖 section heading |
| PyYAML 在 hooks 路径上是依赖风险 | hooks 路径只用 stdlib（json）；YAML 校验由 sanity suite 跑，不在 commit hook |
| `mtime < 24h` weak evidence | 改成 sha256 + grep_assertion 必须匹配（不是 mtime）|
| Network/API gate 在 offline path | `EXTERNAL_PUBLICATION_VERIFIED` 标 `--allow-offline` flag；不可达时 emit warning，不 block；记 DEBT |

### 9.2 已知 gap（明确登记）

- **Bypass-control 不完全闭合**：完全闭合需 P0e (T29.5 workflow critic PreToolUse hook)。本设计接受 known gap，承诺 P0e 上线后回填。登记到 ROADMAP §6 作 DEBT-5。
- **Schema v1 演进风险**：claim_manifest schema 改变会破现有 manifest 文件。约定：schema_version 字段 + 兼容性 test in sanity suite。

---

## 11. Adversarial-audit incorporation summary（v4 final）

v2 → v3 由 a5_ops 2026-05-15 攻防演练触发。v3 → v4 由 codex review #2 抓出 cargo-cult + drift 触发。

| a5_ops rule | v3 → v4 修订 | ROADMAP item |
|---|---|---|
| **M2** Verifier 必须 data-flow-independent | v3 字面 grep `verification.json` → v4 **dynamic forbidden-path set**（每 manifest 自适应），repo-root 路径解析显式，执行位检查（§4.2） | P0o |
| **M1** Gate fitness function | v3 fixtures 都丢 `validate_manifest()` → v4 **拆 schema_bad/ + gate_bad/**，每 gate 至少 1 fixture + 覆盖率门（§3.6） | P0p |
| **P9** Infrastructure paper-over | v3 政策散文 → v4 **机械 classifier 表**（§11.2） | P0q |
| **M5** Agent tool-use 即 fraud signature | v3 self_challenge.notes 正则扫 → 经 codex 评估是 **security theater**（agent 可省略 prose），v4 **完全移到 DEBT-6**；scanner 不再实现 | DEBT-6 only |
| **M3** Per-X cap 必须有 aggregate-X cap | v3 错误声明不适用（codex 抓到 `/npu-port` 已 orchestrate） → v4 **scope 到 `/npu-port`**：standalone day-0 skill 不适用；`/npu-port` 适用 + 加 P0r 跟踪 | P0r new |
| **M4** Cross-backend joint regression | 当前只 codex review；DEBT-8 跟踪加第二个 backend | DEBT-8 |

### 11.1 攻防演练里 a5_ops 自己**仍 OPEN** 的 attack（我们也得承认）

| Attack | a5_ops 状态 | 我们怎么办 |
|---|---|---|
| `WORKER-GATE-ENUMERATION` (worker 读 gate 源代码逆向 engineer claim) | 🔴 STILL OPEN (a5_ops M5 design TBD) | 登记 DEBT-7；本设计接受 known gap；workaround 是 4 gate 数量少 + 每个 gate 独立 invariant 比 a5_ops 15 gate 更难全部 fit |

### 11.2 P9 mechanical classifier table（v4 新增，替代 v3 政策散文）

Worker 收到 env-class 信号时**机械分类**到一个类别再决定路径。分类表：

| 症状（worker 实际看到的）| 类别 | 处理 |
|---|---|---|
| `aclrtSetDevice` 偶发 507033 / dcmi -8020（首次出现）| **transient** | retry ≤ 3 with exp backoff；orchestrator 暴露 `retry_count`；用尽 → INFRA_TRANSIENT_RETRY_EXHAUSTED |
| HuggingFace proxy 429 / 短暂 connect timeout | **transient** | 同上 |
| HCCL init fail（独立 process 内首次出现）| **transient** | 同上 |
| `/etc/ascend_install.info` 不存在 / `which npu-smi` 失败 | **baseline-violated** | INFRA_BASELINE_VIOLATED；**禁止** worker phase O1+ work |
| `npu-smi info` 报 driver != 25.5.x | **baseline-violated** | 同上 |
| Docker proxy 持续 > 5 min 挂 / pull 长 timeout | **baseline-violated** | 同上 |
| A3 上 `repo/` 不是 git clone（NPU-OPS-014）| **baseline-violated** | 同上 |
| `libascendcl.so` size != 已知 baseline | **baseline-violated** | 同上 |
| Helper script 自己 bug（如 `--chips` 错传，T25.5）| **our-script-bug** | 修 script + commit，**不是** workaround；DEBT 登记 |
| Skill 自己默认参数错（如 T25.5 `ASCEND_RT_VISIBLE_DEVICES`）| **our-script-bug** | 同上 |

**禁止行为白名单**（worker / probe / 任何 day-0 agent 收到 env-class signal 必须立刻上抛，不在同一 spawn 内做以下 paper-over）：

- 手动 cp / replace .so / replace lib
- 重启 NPU 后再试
- 换个 NPU chip 试到通为止
- bypass `--pkg` / `--no-verify` / `--force`
- 在 PROGRESS.md / handover 写 "环境问题，先跳过"
- 修改 host 上 `/etc/ascend_*` 配置

具体 P9 完整文案 T31 P0q 阶段落地到 ANTI_PRESSURE_PROTOCOLS.md。

### 11.3 v4 implementation order

```
P0h.0 schema (含 _manifest_relative_to_repo_root, repo_root_anchor, Outcome enum, mode_dispatch.py)
   │
   ▼
P0h.1 sanity suite (schema_bad/ + gate_bad/ 双层 fixtures + coverage 门)
   │
   ├──────────┬──────────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼          ▼
P0i scanner P0j gates  P0l postm  P0q P9     P0r M3 scope for /npu-port
(M2 only,                          ANTI_     (only if /npu-port真启用 sub-skill spawn；
 NOT M5)                           PRESSURE   未真启用前是 design doc 不写代码)
   │          │
   └────┬─────┘
        ▼
P0k snapshot
```

P0q + P0r 不依赖代码改动，可立即并行开工。

---

## 10. 见也

- [`docs/_meta/ROADMAP.md` §2 P0f-P0q + §6 DEBT-5-8](../ROADMAP.md)
- [`review/2026-05-15_codex_review_T30_design.md`](review/2026-05-15_codex_review_T30_design.md)（本设计 v1 → v2 的 review verbatim）
- v3 触发文档：
  - [a5_ops ADVERSARIAL_REWARD_HACKING_AUDIT.md](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/design/ADVERSARIAL_REWARD_HACKING_AUDIT.md)
  - [a5_ops ADVERSARIAL_AUDIT_2026_05_15_RUNLOG.md](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/design/ADVERSARIAL_AUDIT_2026_05_15_RUNLOG.md)
  - a5_ops 新 P9：`src/skills/references/shared/ANTI_PRESSURE_PROTOCOLS.md §P9`
- [`src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md`](../../src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md)
- [`src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md`](../../src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md)
- [`src/skills/_shared/porting-self-challenge/SKILL.md`](../../src/skills/_shared/porting-self-challenge/SKILL.md)（codex R4 复用的现有 self-challenge skill）
