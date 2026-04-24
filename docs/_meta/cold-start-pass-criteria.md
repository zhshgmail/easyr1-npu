# Cold-start 0-interaction PASS criteria (2026-04-25)

**Purpose**: before running a cold-start skill test, define the PASS
criterion. Without this, post-hoc evaluation drifts into "well, it
mostly worked" — a form of self-deception. Each criterion must be
mechanically checkable against the file system + the agent's final
output.

**What "0-interaction" means here**:
1. I hand the agent ONE initial prompt containing the assignment +
   the repo root path + the fork branch name convention.
2. I do NOT respond to the agent during its run. Not even to say
   "yes continue" or "try again". The agent either succeeds on its
   own or reports specific blockage and exits.
3. The agent may call sub-tools (grep, bash, git, python) — that's
   fine. "0-interaction" is between me and the agent, not between
   the agent and its tools.
4. "Pre-existing state" is allowed — the agent can see today's
   fork-branch commits and the KB case registry. It may decide to
   redo the port on a throwaway branch. What counts is whether IT
   can reach the same (or equivalent) end state.

**What counts as PASS per component**:

## vllm-ascend cold-start

**Assignment given to agent**:
"You are a vllm-ascend maintainer. Port vllm-ascend to vllm main
(post-v0.20.0 state). Follow `src/skills/vllm-ascend/port-expert/SKILL.md`.
Work on a fresh branch `vllm-main_cold_<timestamp>` on the personal
fork. Do not touch `vllm-main_auto_porting`. Report at the end."

**PASS criteria (all must hold)**:

1. **Scanner ran**: agent invoked `scripts/sweep.sh` or its 4 component
   scripts AND wrote a JSON result under `/tmp/`. Evidence: file exists.

2. **Drift classified**: agent identified BOTH `SharedFusedMoE` and
   `DefaultMoERunner` as F1 drifts. Evidence: grep agent's transcript
   for both symbols' names.

3. **Compat shim written**: on the fresh branch, files
   `vllm_ascend/compat/shared_fused_moe.py` and
   `vllm_ascend/compat/default_moe_runner.py` exist AND
   `python3 -m py_compile` both succeed. Evidence: `git show` on fresh
   branch.

4. **Call sites swapped**: at least `ops/fused_moe/fused_moe.py` has
   both imports changed from `vllm.*` to `vllm_ascend.compat.*`.
   Evidence: `git diff` on fresh branch.

5. **Drift-port-validate ran**: agent invoked the validation skill.
   Evidence: validation script output or log file.

6. **Push attempted**: agent ran `git push personal <branch>` OR
   reported a specific authentication/network reason it couldn't.
   Evidence: git reflog or agent transcript.

**FAIL triggers**:
- Agent asked me for clarification mid-run.
- Agent invented a drift that doesn't exist in the current vllm main.
- Agent skipped scanner and guessed from memory.
- Agent modified the existing `vllm-main_auto_porting` branch.
- Final output omits explicit PASS/FAIL self-report.

## torch-npu cold-start

**Assignment**:
"You are a torch_npu maintainer. Port torch_npu to torch v2.12.0-rc3
from current baseline v2.11.0. Follow
`src/skills/torch-npu/port-expert/SKILL.md`. Work on a fresh branch
`torch-2.12-rc3_cold_<timestamp>` on the personal fork. Do not touch
`torch-2.12_auto_porting`. Report at the end."

**PASS criteria**:

1. **Scanner ran**: `scripts/sweep.sh` or its 4 component scripts
   produced output. Evidence: file exists under /tmp/ or OUT_DIR.

2. **Real drift identified**: agent reported `Union` re-export drop
   from `torch._inductor.codecache`. Evidence: grep transcript.

3. **Compat written**: on fresh branch, file
   `torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/codecache.py` has
   the `try/except` for `Union`. py_compile passes. Evidence:
   `git show`, `python3 -m py_compile`.

4. **F2-path-move shims**: either created all 5 compat modules
   (sympy_functions / inductor_ir / inductor_codegen_common /
   inductor_codegen_simd / dynamo_utils) OR correctly concluded they
   are defensive-only (old paths still re-export). Either verdict is
   PASS; the agent must pick one explicitly based on verification.
   Evidence: transcript has explicit reasoning.

5. **F7/F8 verdict**: agent ran `check_f7_f8.py` AND either wrote
   override subclass code OR documented "inherits safely, no fix
   needed" with grep evidence. Evidence: transcript.

6. **Push attempted**: same as vllm-ascend.

**FAIL triggers**: same structure as vllm-ascend.

## transformers cold-start

**Assignment**:
"You are a transformers NPU-integration maintainer. Probe transformers
main HEAD for NPU drift. Follow
`src/skills/transformers/port-expert/SKILL.md`. Report outcome A / B / C
with evidence. Do not apply any patch or push anything if outcome is A."

**PASS criteria** (different shape because transformers today is
outcome A — no drift to patch):

1. **Stage 0 decision tree applied**: agent read the 4 static checks
   in `SKILL.md` Stage 0 AND executed at least 3 of them
   (byte-match `npu_flash_attention.py`, import chain unchanged,
   upstream consumption path unchanged). Evidence: transcript.

2. **Outcome classified**: agent reported outcome A (or B / C with
   specific evidence of what changed). Evidence: transcript contains
   the single letter `A` / `B` / `C` as the verdict.

3. **Correct classification**: if outcome is A, independent check
   (I run `git show origin/main:src/transformers/integrations/npu_flash_attention.py`
   and compare to the KB baseline signature) MUST agree with A.
   If outcome is B / C, the diff the agent points to MUST actually
   exist.

4. **No false action**: agent did NOT apply patches / push / open PRs
   on an outcome-A result. The skill explicitly says "do not apply
   any patch if outcome is A".

**FAIL triggers**:
- Agent applied patches / pushed anything on outcome A.
- Agent's classification disagrees with independent verification.
- Agent asked for clarification mid-run.
- Agent fabricated a drift that doesn't exist.

## Meta-rule for all three

If the agent exits with its own self-reported FAIL AND the reason is
"skill/KB had a gap I couldn't resolve autonomously" — that is a
cold-start PASS for the skill-as-tested-today (the skill correctly
failed safe) BUT a FAIL for skill completeness. Both get logged;
iteration P5 fixes the skill, then P2/P3/P4 re-runs.

If the agent exits with a FAIL reason that amounts to "I needed more
info", the skill needs a gap-fix before it's considered complete.
