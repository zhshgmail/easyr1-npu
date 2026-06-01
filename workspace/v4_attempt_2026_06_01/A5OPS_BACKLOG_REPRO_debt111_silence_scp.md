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

## Evidence pointers
- orchestrator log: `/tmp/orch_20260601_024716_104980.log` (on blue's host) — DEBT-111 lines
  144-165, silence-timeout line 170, slow-scp lines 215-242.
- final result (the recovery worked): precision PASS 6/6 T1 + 28/28 edge, perf 51.1× mean.
