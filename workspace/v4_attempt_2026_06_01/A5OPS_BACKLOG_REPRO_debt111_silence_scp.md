# a5_ops backlog repro — DEBT-111 + silence-timeout-on-slow-scp

For main → ROADMAP §6 + dev routing. Observed live during `/ascendc-op-gen hc_split_sinkhorn`
run#2 (2026-06-01) on the easyr1 setup: **blue's separate host** drives op-gen, build/run
target = remote container **`a5ops-a3`** (davinci2) on the A3 host (115.190.166.102), reached
over SSH alias `easyr1-a3` (port 443). `.ascendc_env` TARGET=a3.

This topology (driver host ≠ build host, deploy over a slow long-haul SSH link) is what
surfaces both gaps; co-located team setups (A5 host == build host) likely never hit them.

---

## ① DEBT-111 — build harness not present in the remote build container

**Symptom**: first build attempt failed `deploy script EXIT: 2`, empty stderr marker.

**Root cause (worker diagnosis, log lines 144-165)**:
- deploy script does `BUILD_ROOT="${BENCHMARK_ROOT}"` → `/root/AscendOpGenAgent`, expects `utils/build_ascendc.py` there.
- On `a5ops-a3`, `/root/AscendOpGenAgent/` contained ONLY `current_task/` — **no `utils/`**, no `build_ascendc.py`, `verification_ascendc.py`, `performance.py`.
- On the **driver host**, `vendor/AscendOpGenAgent` submodule was **uninitialized** → `build_ascendc.py` absent there too; only `src/scripts/patches/build_ascendc.py` (the DEBT-20 overlay patch) existed.

**Worker self-recovery (legitimate, worked)**:
1. `git submodule update --init vendor/AscendOpGenAgent` → `vendor/AscendOpGenAgent/utils/build_ascendc.py` now exists.
2. ran `setup_a3_utils.sh` (the documented harness-provisioning script) → provisioned 8 files to remote `a5ops-a3` (`build_ascendc.py`, `verification_ascendc.py`, `performance.py`, ...).
3. overlaid the **patched** `build_ascendc.py` (393 lines, DEBT-20 per-source COMPILE_DEFINITIONS) over the vendor version on the remote.
4. re-ran build → SUCCESS.

**Gap**: a fresh remote build container has NO build harness; nothing in the orchestrator
preflight provisions it. The worker had to discover `setup_a3_utils.sh` + init the submodule
+ overlay the patch by itself. **Fix-for-next-customer**: orchestrator preflight (or
`/aog-preflight`) should detect missing `utils/build_ascendc.py` on the target container and
auto-run `setup_a3_utils.sh` + submodule-init + DEBT-20 overlay BEFORE the first build, so the
worker never sees `deploy EXIT 2`. (Bonus: `ref_preflight` could include a harness-presence
check in its RUNNABLE verdict.)

---

## ② silence-timeout-on-slow-scp — 19MB edge_dataset bundled into deploy tarball

**Symptom**: `silence-timeout #1/2 on aog-kernel-worker: stdout silent for 600s ... SIGTERMed
subprocess. Respawning fresh.` (log line 170, 2026-06-01 03:26:19) — worker killed mid-build.

**Root cause (worker diagnosis, log lines 215-242)**:
- the deploy tarball was **~24MB**, dominated by `edge_dataset.pt` (19MB) + `edge_inputs.pt`.
- upload over the long-haul SSH link to a5ops-a3 ran at **~1.8MB/min** → ~13+ min for 24MB.
- the worker's deploy/poll step produced **no stdout** during the SCP → tripped the 600s
  silence watchdog → SIGTERM + respawn (consuming 1 of 2 retries before hard abort).
- the SCP was NOT actually hung — `ls -la` on the remote showed the tarball growing
  (6.5MB → 8.3MB → ...); it was just slow.

**Worker mitigation (worked, but fragile)**:
- after respawn, worker appended progress notes to PROGRESS.md as anti-silence heartbeat.
- then **re-deployed a tiny kernel-only tarball** (kernel/*.h/.cpp + .py, no .pt) which uploads
  in seconds; edge_dataset re-uploaded separately only when Pass B needed it.

**Gap**: (a) deploy bundles the large edge_dataset.pt into the same tarball as the kernel,
making the *build* deploy needlessly huge; (b) a slow-but-progressing SCP looks identical to a
hang to the 600s stdout-silence watchdog. **Fix-for-next-customer**:
1. **Split deploy**: kernel sources (build-critical, tiny) deploy first; edge_dataset.pt
   deploys lazily, only for Pass B, in parallel with the build. (The worker did this by hand
   on the respawn — make it the default deploy path.)
2. **SCP progress → watchdog liveness**: the deploy step should emit periodic stdout (e.g.
   `scp` without `-q`, or a byte-count heartbeat) so a slow transfer resets the silence timer
   instead of being mistaken for a hang. Or: exempt the deploy/upload phase from the
   stdout-silence watchdog (use a transfer-stall detector on tarball-size-delta instead).

---

## ③ O5 count-basis mismatch — pass_a case-set disagrees between worker and O5 (opgen-style op)

**Symptom**: after build+verify PASS, finalize was REFUSED twice by post-verify gates:
```
04:04:55 O5 MISMATCH (2 discrepancies):
  worker claimed pass_a = {tier1_pass:6, total:6}
  O5 re-measured  pass_a = {tier1_pass:28, total:28}   (both status=PASS — NOT a precision regression)
```

**Root cause (worker-3 diagnosis)**: this op has **no dedicated `hc_split_sinkhorn.json` benchmark
file** (it's opgen-style, driven by `edge_dataset.pt`). So the two sides counted different case-sets:
- worker recorded pass_a from `model.py::get_input_groups()` → **6 typical shapes** (N=[1,7,256,512,1009,1024]).
- O5 re-measured pass_a using the **28-case edge_dataset** → 28.
Both PASS, but the count *basis* is systematically different → O5 flags MISMATCH and refuses finalize.
(main: same family as **P0cc dual-count schema / flat_quant** worker-vs-O5 disagreement.)

**Resolution**: worker-3 reconciled pass_a to the authoritative edge_dataset 28-case set → pass_a 28/28
== O5's 28 → O5 VERIFIED → finalize proceeded.

**Fix-for-next-customer**: for opgen-style ops (no benchmark.json), either (a) worker reports pass_a on
the SAME case-set O5 uses (edge_dataset), or (b) O5 re-measures on the worker's get_input_groups set, or
(c) split into two explicit count fields (Pass-A typical-shapes vs edge-cases) so the gate compares
like-for-like instead of conflating them.

## ④ P0ee perf-methodology gate — symmetric-measurement declaration required (and it caught an inflated number)

**Symptom**: finalize REFUSED a 3rd time:
```
04:07:27 P0ee METHODOLOGY_DECLARATION: performance.ratio=17.52 > 1.0× claim WITHOUT
         performance.method declaration. Default-deny: speedup claim MUST declare symmetric measurement.
```

**Root cause + value**: worker's verification.json had `ratio_mean=51.09` + a `harness` string, but NOT the
explicit `method_symmetric=true / same_wrapper` declaration the gate requires. When worker-4 added an
*honest symmetric* re-measurement, the number dropped **51× → 5.34×** — i.e. the original 51× was an
asymmetric/inflated measurement. **The P0ee gate's default-deny on undeclared-methodology speedup claims
directly prevented an inflated perf number from finalizing.** (Also surfaced: on V220 Ascend910_9382 the
`torch_npu.profiler` device-duration path fails — `Failed to get acl to npu flow events` — and the harness
falls back to `time.perf_counter()`; worker-4 correctly declared the perf_counter symmetric path and did
NOT claim the profiler signal.)

**Fix-for-next-customer**: the kw/worker brief should make the perf-methodology declaration a
*first-class required field* the worker fills on first authoring (not discovered only at the P0ee rollback),
and the perf harness should emit the `method_symmetric` flag automatically when both sides go through the
same wrapper — so the honest number is recorded the first time instead of after a rollback.

## Evidence pointers
- orchestrator log: `/tmp/orch_20260601_024716_104980.log` (on blue's host).
  - DEBT-111: lines 144-165 · silence-timeout: line 170 · slow-scp: lines 215-242
  - O5 count mismatch: lines 526-537 (iter#1 finalize) · O5 VERIFIED: 04:04:55 (iter#2)
  - P0ee perf-method rollback: line 601 (04:07:27) · honest re-measure done: worker-4 04:15:06
- final HONEST result (4 worker spawns: author→build→count-reconcile→perf-method):
  precision PASS pass_a 28/28 + pass_b 28/28 T1_STRICT, determinism 2/2, **perf 5.34× mean symmetric**.
- the two honesty gates (O5 count + P0ee perf-method) each caught a premature/inflated claim — high value.
