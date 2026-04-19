---
name: codex-review
description: Get a second-opinion review of code, docs, designs, or plans from the locally-installed `codex` CLI (OpenAI's gpt-5.4 agent). Use when the user asks to "review with codex", "get a second opinion", "have codex check this", or wants an independent pass on work before declaring it done. The skill is not specific to code — works equally well on design docs, dependency analyses, migration plans, architecture proposals.
---

# codex-review

Run a non-interactive review with the locally-installed `codex` CLI and use its findings as a second opinion. Codex runs a different model (gpt-5.4) from Claude, so it catches things the author missed — factual errors, logical gaps, internal inconsistencies, unstated assumptions.

## When to use

- After writing or editing a non-trivial doc / design / plan / code change, before declaring it done.
- When the user asks for a "second opinion," "review," "sanity check," or "independent pass."
- When the author (you) suspects you might have blind spots — e.g. you wrote it, the user hasn't seen it yet, and you want an adversarial reader.

## When NOT to use

- Trivial changes (typo fix, a single-line refactor). Codex review takes 1–5 minutes and tokens; not worth it for small edits.
- Tasks where the user just wants you to execute, not get a review.
- When `codex` is not installed — `which codex` returns nothing. Tell the user rather than skipping silently.

## Prerequisites

- `codex` CLI installed (`which codex` should find it; version ≥ 0.121 tested).
- `codex` authenticated (it will prompt if not; tell the user to run `codex login` if so).

## How to invoke

Use `codex exec` in non-interactive mode. Two flags matter:

- `--skip-git-repo-check` — always include. Without it, codex errors out when run outside a git-trusted directory.
- `--dangerously-bypass-approvals-and-sandbox` — **include by default for doc/code reviews.** On some hosts the default bubblewrap sandbox fails to create network namespaces (`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`), which silently prevents codex from reading local files. When it can't read files, it falls back to web search + inference and produces plausible-sounding but ungrounded findings. The bypass flag is safe here because review prompts are read-only by nature — codex has no write side effects in a review workflow.
  - Counter-example: if the user asks codex to *make changes* (not just review), re-evaluate the sandbox choice — `-s workspace-write` may be appropriate.

Baseline command:

```bash
cat /tmp/review-prompt.txt | codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox
```

## Prompt template

Write the prompt to a file first (Bash HEREDOC is fine). The template:

```
You are reviewing [N documents / a code change / a design] for [project context in one sentence].

Read these files carefully:
1. /absolute/path/to/file1 — [one-line description of what's in it]
2. /absolute/path/to/file2 — [...]

For additional context you may also read:
- /absolute/path/to/background/README.md
- /absolute/path/to/related/file.py

Review focus:
- [Bullet — a specific thing to check]
- [Bullet — another specific thing to check]
- [...]

Output format:
- STRENGTHS: bullets of what's right.
- ISSUES: bullets, each with severity (CRITICAL / MAJOR / MINOR) and a specific file:section reference.
- SUGGESTIONS: bullets, each actionable.
- VERDICT: one paragraph — ready to proceed, or needs another pass?

Keep under [N] words. Be direct; don't hedge. If something is fine, say so briefly and move on.
```

Key prompt-writing rules:
- **Always use absolute paths.** Codex's CWD is the directory you invoked it from; relative paths are unreliable.
- **Enumerate the target files explicitly.** Don't say "review the docs dir" — say which files in which order.
- **Ask for severity labels.** CRITICAL / MAJOR / MINOR forces codex to prioritize; without labels everything looks equally important.
- **Ask for a VERDICT paragraph.** Forces codex to commit to a position rather than listing findings without a stance.
- **Cap word count.** Codex will pad if unconstrained. 500–800 words is usually enough.

## Verifying the review

Critical: **trust but verify.** Before acting on codex's concrete claims:

1. **Did codex actually read the files?** Check the output for `exec` blocks showing file reads. If you see `error=exec_command failed` or the review reads suspiciously generic, codex likely fell back to web search + inference. Re-run with `--dangerously-bypass-approvals-and-sandbox`.
2. **Spot-check specific claims.** If codex says "file X line Y does Z," grep/read to confirm. Claude's observed failure mode with codex: it hallucinates file paths and line numbers when it couldn't read the source. Don't edit blindly based on "codex said so."
3. **Weight severity appropriately.** CRITICAL findings warrant correction; MAJOR warrant discussion with the user; MINOR warrant a judgment call.
4. **Don't over-index on any one reviewer.** Codex has its own blind spots (sycophancy, hedge-heavy writing when the prompt permits it, occasional over-confidence on library APIs).

## Reference-first for port reviews

Before asking codex to review a proposal that says "write a new X" for a port, **first grep the reference port and upstream library for X**. If either already has X, the review should be on "should we adopt their pattern?" not "is our design for X good?"

Mechanism: a proposal to write a substantial new adapter / shim / backend for a port is suspicious if it doesn't cite:
- what the reference port (e.g. veRL for any RL framework port) does for the same concern,
- whether the upstream library (e.g. transformers, vllm, ray) already ships the integration.

Codex can fact-check these claims, but only if they're in the prompt. If the prompt says "we need to build X from scratch" without referencing those two places, that's usually a red flag — ask codex to check both before designing.

Concrete incident (2026-04-18): v2 proposal estimated 2 days to write an `npu_fusion_attention` adapter. User asked "how does verl do it?" before authorizing. veRL does it with a 4-line import swap from `transformers.integrations.npu_flash_attention` (which ships `npu_flash_attn_varlen_func` already). Total work dropped from ~2 days to ~1 hour. See `repo/knowledge/npu-patterns.md::NPU-OPS-005`.

## Version-aware reviews

**Critical for multi-version projects (like any port against specific SDK releases):** code behavior differs between branches/tags/master. A review of master when production runs a release branch produces false positives and false negatives.

Before asking codex to review code that targets a specific SDK version:

1. **Identify the matching ref** — release branch, version tag, or vendored commit — for the dependency in the target environment (docker image, deployed runtime, etc.). Ship-ready production versions usually ship from `releases/vX.Y` or `vX.Y.Z-SDK-version` style branches, NOT master.
2. **Check out that ref in the local clone** before invoking codex, or include the exact ref in the prompt: "inspect `upstream/torch-npu/` at commit `<SHA>` / branch `origin/v2.8.0-7.3.0` — not master."
3. **Probe the compat table** — most NPU projects (torch-npu, vllm-ascend) ship a README table mapping SDK version to branch. Use that table; don't guess.
4. **Record the ref used** — in review output and in any follow-up notes, so later readers know what version the finding was made against.

Example failure mode we hit: codex reviewed device-API calls against `torch-npu` master and cited a known quirk in `torch.npu.get_device_name(device)`. The behavior on the actual target branch (`v2.8.0-7.3.0`) may differ. Not fatal, but worth verifying on the right ref before acting.

## Post-review workflow

After the review:

1. Capture the full raw output somewhere retrievable (e.g. `/tmp/codex-review-<topic>.txt`). Useful if the user asks "what did codex say exactly."
2. Triage findings into (a) factual errors you'll fix now, (b) material omissions you'll add, (c) scope/design calls the user should make.
3. Verify (a) and (b) against actual source before editing. Then apply edits.
4. Mirror the milestone summary to the user (with Discord if that's the channel). Include: what codex flagged, what you verified, what you fixed, what remains open.
5. If codex caught something you should have caught yourself, consider whether it's worth a `feedback` memory entry to avoid repeating the pattern.

## Generalizes beyond code

Codex works as a reviewer for any reasoning-heavy artifact:

- Code changes / PRs.
- Design docs, architecture proposals, migration plans.
- Dependency analyses, porting checklists.
- Technical specifications.
- Commit-message drafts (for high-stakes commits where message quality matters).
- Decision memos.

Not a fit for:
- Pure formatting / style review.
- Creative writing.
- Review of the codex tool itself (obviously).

## Examples

### Doc review

Prompt file lists the doc paths + "review focus" bullets + output format. Run:
```bash
cat /tmp/doc-review.txt | codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox
```

### Code review

Same shape; add "check for bugs, correctness issues, security issues, missing error handling" to the focus. Consider passing the relevant files via `-i FILE` if codex is struggling to locate them, though file paths in the prompt usually suffice.

### Plan review

Frame as: "review this migration plan for completeness and risk. Are there steps missing? Irreversible actions without rollback? Misordered steps? Hidden assumptions?"

## Known gotchas

- **Silent sandbox failure**: as above — always pass `--dangerously-bypass-approvals-and-sandbox` for local-doc review unless you have reason not to.
- **Timeout**: codex review can take 3–10 min on long docs. Pass a Bash `timeout` of at least 10 min (`600000ms`).
- **First run might auth-prompt**: if `codex login` wasn't done, the first invocation opens a browser. Warn the user if you suspect auth isn't set up.
- **Output size**: long reviews can exceed Bash output caps. Tee to a file: `codex exec ... 2>&1 | tee /tmp/review.txt | tail -200`.
- **Context scope**: codex reads only the files you name. It won't proactively search your project. If the review needs broader context, enumerate the files explicitly.
