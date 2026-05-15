# 测试网 + 安全网 + 反馈环 — 设计文档（v3，post-adversarial-audit）

> **Status**:
> - v1 (2026-05-15 起草) → codex review → **v2** 应用 6 项 recommendation
> - v2 → a5_ops 2026-05-15 攻防演练记录拉到（[ADVERSARIAL_REWARD_HACKING_AUDIT](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/design/ADVERSARIAL_REWARD_HACKING_AUDIT.md) + [RUNLOG](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/design/ADVERSARIAL_AUDIT_2026_05_15_RUNLOG.md)）→ **v3 本文档**，新加 4 项 (M1 gate-fitness / M2 anti-cycle / P9 infra-paper-over / M5 tool-use signature)
>
> **Scope**: 借鉴 a5_ops 三件套（test net / safety net / feedback loop），落地到 easyr1-npu。
>
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

### 2.3 Mode-driven required-field 表

| Field | `fork_patch` | `issue_only` | `integrated_overlay` |
|---|---|---|---|
| `mode` | required | required | required |
| `outcome` | required | required | required |
| `validation_level` | required, ≥ L3 for C-patch | required, ≥ L3 | required, ≥ L5 |
| `evidence[smoke_log]` | required | required if outcome != C-report | required |
| `evidence[commit_ref]` | required | optional (no fork branch) | required for image SHA |
| `artifacts[shim_module]` | required for C-patch | n/a (no fork) | n/a (cherry-picks from forks) |
| `artifacts[pr_material]` | required | n/a | optional |
| `external_refs[github_issue]` | optional | **required** for C-report | optional |
| `self_challenge.patterns_passed` | required, len ≥ 8 | required, len ≥ 8 | required, len ≥ 8 |

每个 mode 的 required set 由 P0j gate code 维护单一权威表（不允许多处声明）。

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

### 3.6 Bad fixtures（a5_ops M1 gate fitness function，v3 新增）

`tests/fixtures/` 必含 5+ **crafted-fraud** YAMLs，每个对应一个真实 attack vector：

| Fixture | Crafted attack pattern | 我们的 scanner/gate 必须拒 |
|---|---|---|
| `bad_self_citing_verifier.yaml` | verifier_scripts[0].path 包含 `read("claim_manifest.yaml")` 的脚本（a5_ops `WORKER-SELF-CITING-VERIFIER`） | P0i anti-cycle check |
| `bad_mirror_md5_trivial.yaml` | mode=integrated_overlay 但 artifacts 都是 byte-identical baseline 文件（无 workspace edit）| P0i + P0j 该 outcome 强制 A-with-note 不能 A |
| `bad_perf_null_with_pass.yaml` | outcome=A + evidence[numeric_metric].value=null（a5_ops `DS-PERF-FRAUD-1`）| P0i schema reject |
| `bad_t1_by_construction.yaml` | validation_level=L5 但 evidence[smoke_log].grep_assertion="T1_BY_CONSTRUCTION"（a5_ops `WORKER-PRECISION-T1-BY-CONSTRUCTION`）| P0j VALIDATION_ARTIFACT_VERIFIED |
| `bad_circular_evidence.yaml` | evidence[smoke_log].path == verifier_scripts[0].path（圈环：claim cite verifier，verifier cite claim）| P0i anti-cycle |
| `bad_issue_only_with_fork.yaml` | mode=issue_only 但 artifacts 含 shim_module（mode/artifact 组合矛盾）| P0i mode-derived check |

每个 fixture **必须**配 expected-rejection-reason metadata，sanity suite test 通过格式：

```python
@pytest.mark.parametrize("fix,expected_error_kind", [
    ("bad_self_citing_verifier.yaml", "anti-cycle"),
    ("bad_mirror_md5_trivial.yaml", "mirror-without-edit"),
    ...
])
def test_crafted_fraud_rejected(fix, expected_error_kind):
    data = yaml.safe_load((FIX / fix).read_text())
    with pytest.raises(ValidationError) as exc:
        validate_manifest(data)
    assert expected_error_kind in str(exc.value), \
        f"fixture rejected for wrong reason: got {exc.value}"
```

**关键**（a5_ops M1）：fixture **不能**只测 happy path；每个 gate 必须有至少一个 crafted-fraud fixture 触发它，否则 sanity suite 是"假绿"——good fixture 通过 ≠ gate 真在工作。

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

### 4.2 P0i `scan_outcome_claims.py`

Job：**schema 层**校验 claim_manifest.yaml + **anti-cycle verifier check** (a5_ops M2)。

```python
def scan(manifest_path: Path) -> ScanResult:
    """
    Returns ScanResult with errors=[] if pass.
    
    Errors include:
      - schema fields missing / wrong type
      - evidence array empty when outcome != C-report
      - mode/artifact combo invalid (issue_only + shim_module = reject)
      - self_challenge missing or < 8 patterns passed
      
      # v3 / a5_ops M2:
      - any verifier_scripts[i].path contains 'claim_manifest.yaml'
        OR contains the path-derived file the verifier is supposed to
        verify (anti-cycle: verifier must be data-flow-independent of claim)
      - any verifier_scripts[i].path doesn't exist OR isn't executable
      
      # v3 / a5_ops M5 (degraded — full version needs P0e tool-use hook):
      - self_challenge.notes contains substring "Read.*finalize_day0_check"
        OR "Bash.*test.*gate" → emit warning to PROGRESS.md
        (reward-hacking-by-engineering signature)
    """
```

**Anti-cycle implementation**（a5_ops M2 直接借鉴）：

```python
def _check_anti_cycle(manifest_dir: Path, verifier_scripts: list[dict]) -> list[Error]:
    errors = []
    for v in verifier_scripts:
        script = manifest_dir / v["path"]
        if not script.exists():
            errors.append(Error("verifier missing", v["path"]))
            continue
        body = script.read_text()
        # Anti-cycle: verifier must not read its own claim file
        if "claim_manifest.yaml" in body:
            errors.append(Error(
                "verifier reads claim_manifest.yaml (self-citing — not independent)",
                v["path"]))
        # Anti-cycle: verifier must not read verification.json sibling
        # (a5_ops foreach_abs exact attack pattern)
        if "verification.json" in body:
            errors.append(Error(
                "verifier reads verification.json (a5_ops M2 anti-cycle violation)",
                v["path"]))
    return errors
```

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

## 11. v3 adversarial-audit incorporation summary

v2 → v3 由 a5_ops 2026-05-15 攻防演练（[ADVERSARIAL_REWARD_HACKING_AUDIT](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/design/ADVERSARIAL_REWARD_HACKING_AUDIT.md) + [RUNLOG](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/design/ADVERSARIAL_AUDIT_2026_05_15_RUNLOG.md)）触发。a5_ops 5 条新规则（M1..M5），我们按优先级 incorporated 进 v3：

| a5_ops rule | 我们的 v3 落地 | ROADMAP item |
|---|---|---|
| **M2** Verifier 必须 data-flow-independent | claim_manifest 加 `verifier_scripts[]` 字段 + P0i 加 anti-cycle check (`grep -c "claim_manifest.yaml" $script == 0`) (§2 + §4.2) | P0o |
| **M1** Gate fitness function | P0h.1 `tests/fixtures/` 必含 5+ crafted-fraud bad fixtures，每个对应真实 attack vector + expected-rejection-reason (§3.6) | P0p |
| **P9** Infrastructure paper-over | `ANTI_PRESSURE_PROTOCOLS.md` 加 P9（NPU 错码/CANN install desync/.so size mismatch → 不准 worker self-paper-over，必须 INFRA_*_RETRY_EXHAUSTED 上抛）(§11.2 below) | P0q |
| **M5** Agent tool-use 即 fraud signature | 短期降级：P0i scan self_challenge.notes 含 `Read.*finalize_day0_check` 等模式 emit warning。完整方案需 P0e workflow critic + tool-use logging hook | DEBT-6 (long-term), included in P0o (short-term degraded) |
| **M3** Per-X cap 必须有 aggregate-X cap | **不适用现阶段** — 我们暂无 multi-agent orchestrator (`/npu-port` 暂未真实 spawn sub-agent)；未来加 orchestrator 时一并采纳 | — |
| **M4** Cross-backend joint regression | 当前只 codex review；T31.3 codex review v3 完后，DEBT-8 跟踪加第二个 backend (gemini / gpt-4o) | DEBT-8 |

### 11.1 攻防演练里 a5_ops 自己**仍 OPEN** 的 attack（我们也得承认）

| Attack | a5_ops 状态 | 我们怎么办 |
|---|---|---|
| `WORKER-GATE-ENUMERATION` (worker 读 gate 源代码逆向 engineer claim) | 🔴 STILL OPEN (a5_ops M5 design TBD) | 登记 DEBT-7；本设计接受 known gap；workaround 是 4 gate 数量少 + 每个 gate 独立 invariant 比 a5_ops 15 gate 更难全部 fit |

### 11.2 ANTI_PRESSURE_PROTOCOLS.md 加 P9（infrastructure paper-over）

a5_ops 文本移植 + NPU 上下文改写。两类信号 + 处理路径：

| 类 | 例 | 必须的处理 |
|---|---|---|
| **Transient retry-recoverable** | NPU 偶发 dcmi -8020 / aclrtSetDevice 偶发 507033 / proxy 短暂 429 | 允许 ≤ 3 retry with exp backoff，但 retry 计数**在 orchestrator 暴露**（不藏在 worker 内 loop）；用尽 → INFRA_TRANSIENT_RETRY_EXHAUSTED 上抛 |
| **Baseline-violated** | CANN install 缺 lib / `/etc/ascend_install.info` 不存在 / NPU driver != 25.5.0 / docker proxy 长期挂 / A3 repo stale v0 layout (NPU-OPS-014) | **永远不进 Phase O1+ work**；必须 graceful exit INFRA_BASELINE_VIOLATED；worker **绝对禁止** "探一下 baseline 然后决定怎么绕过" |

禁止行为白名单（worker / probe / 任何 op-gen agent 收到 env-class signal 必须立刻上抛，不在同一 spawn 内做以下 paper-over）：

- 手动 cp / replace .so / replace lib
- 重启 NPU 后再试
- 换个 NPU chip
- bypass `--pkg` / bypass `--no-verify`
- 在 PROGRESS.md / handover 写 "环境问题，先跳过"

具体 P9 完整文案 v3 实现阶段（T31 P0q）落地到 ANTI_PRESSURE_PROTOCOLS.md。

### 11.3 v3 implementation order update

v2 严格 schema → P0i+P0j+P0l 并行 → P0k；v3 在此基础上：

```
P0h.0 schema (含 verifier_scripts 字段)             ← v3 schema 加 1 字段
   │
   ▼
P0h.1 sanity suite (含 5+ bad fixtures)             ← v3 必含 crafted-fraud
   │
   ├──────────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼
P0i scanner  P0j gates  P0l postm  P0q ANTI_PRESSURE P9
(含 M2/M5)                          (独立任务，可立即并行)
   │          │
   └────┬─────┘
        ▼
P0k snapshot
```

P0q（P9 ANTI_PRESSURE 写文档）不依赖任何代码改动，可与 P0o/P0p 并行立刻开工。

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
