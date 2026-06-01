# RESULTS — V4 real-delta RL-loop bridge (task-dag-planner cold-drive)

> DAG `task_dag.json` executed to completion 2026-06-01. Goal: replace the V4 RL loop's
> synth-delta with the REAL trained-weight delta from the megatron 1-layer training iteration
> (reduced-layer basis). Also the first cold-drive of the new `/task-dag-planner` skill.

## Outcome: ALL NODES COMPLETE (with honest caveats)

| Node | Status | Evidence | Verify met? |
|---|---|---|---|
| n1_export (needs-NPU, tlrescue) | ✅ done | `trained_attn_delta.pt` (816MB); all 5 attn tensors nonzero finite delta (L2 0.09–0.29) | YES — 5/5 nonzero (DAG asked ≥1) |
| n2_remap (local) | ✅ done | `megatron_to_sglang_attn_remap.py` unit-tested vs n1 keys | YES — name-map proven; layout deferred to n3 shape-assert (honest) |
| n3_bridge (needs-NPU, sgl_probe) | ✅ done | `v4_rl_loop_REAL_delta_RESULT.txt` | YES — see below |

## What was proven (real, on-disk evidence)

**The V4 RL loop closes end-to-end with the REAL trained delta**, not synthetic noise:
- `update_weights_from_tensor` returned `(True, 'Success')` for the real remapped delta at every step (attention-only path, dodges #26794 — KB cross-layer-012).
- The **applied delta L2 exactly equals the real trained-delta L2** at step1 (wq_a=0.1318, wkv=0.2908, wo_a=0.1926, …) and scales linearly across steps. This is the honesty-gate proof that the weights pushed are the actual AdamW-trained delta from the megatron stack — NOT a synth perturbation.
- The cross-stack bridge works: delta produced in the **megatron/mindspeed** stack (tlrescue) → exported to a plain `.pt` on the shared mount → loaded + remapped (`self_attention`→`self_attn`) + shape-asserted → pushed into the **sglang.Engine** stack (sgl_probe). Two different runtimes, one file boundary.
- Re-rollout output changed: `distinct_vs_step0 = 1/5` (the step5 rollout flipped from step0).

## Honest boundary (do NOT overclaim)

- **Weak divergence (1/5, not 5/5).** The rollout only changed at step5 (the 5× amplified real delta); steps 1–4 produced identical output to step0. Reason: the real delta comes from a **toy-MSE-loss 3-step AdamW** training iteration whose attention grad-norms are tiny (~1e-6, see n1 log). So the trained delta's magnitude is small — it took 5× amplification before bf16 greedy-decode flipped a token. The earlier synth-delta run got 5/5 only because random noise at scale 0.5 perturbs far more aggressively than this weak real signal.
- **This proves the REAL-delta PLUMBING end-to-end, not a real RL reward signal.** "rollout → real-trained-weight sync → re-rollout, output demonstrably changes" is honestly closed. "the model learned something useful from RL" is NOT claimed — the training step used a placeholder loss, not a reward.
- **Still reduced-layer (1-layer fab).** NOT full 43-layer. NOT multi-chip.
- The next step to a *real* RL signal: replace the toy MSE loss in the training iteration with an actual miles RL objective (reward-weighted), so the attention delta carries real learning. The plumbing this DAG proved does not change — only the loss that produces the delta.

## Cold-drive of /task-dag-planner (skill validation)

The skill's Phase A–D ran as designed on a real task:
- Phase A/B/C produced `task_dag.json` (3-node serial chain) + CC tasks 322→323→324 with blockedBy edges.
- Phase D executed in topological order; each node's claimed output was verified on disk before marking complete (n1's nonzero-delta gate, n2's unit test, n3's applied-L2==real-L2 gate).
- The honesty gate caught + reported the weak-divergence reality instead of declaring a clean 5/5 PASS. This is exactly the `end_to_end_vs_described` discipline the skill encodes — working as intended.

## Artifacts

- `task_dag.json` — the DAG
- `n1_export_real_delta.py` + `trained_attn_delta.pt` (on A3) — real delta export
- `megatron_to_sglang_attn_remap.py` — verified remap
- `n3_rl_loop_real_delta.py` + `v4_rl_loop_REAL_delta_RESULT.txt` — the bridge run + result
