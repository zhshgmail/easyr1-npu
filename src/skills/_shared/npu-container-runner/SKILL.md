---
name: npu-container-runner
description: Launch a docker container on an Ascend NPU host with the correct device passthrough, HCCL host IPC/net, user-scoped bind mounts, HF / vllm-ascend / Ray environment defaults, and a chip-occupancy precheck. Use whenever you need to run something inside an NPU container — smoke tests, training, inference, debug shells. Encodes the operational rules from `memory/a3_server.md` (shared host, three standard paths, never clobber others).
---

# npu-container-runner

## When to use

Any of:
- Running a smoke test, training step, or inference inside an NPU container on a shared host.
- Starting a long-lived dev container attached to a bind-mounted source tree.
- Running a one-shot diagnostic (e.g. `python3 -c "import torch_npu; ..."`).

Don't use for:
- Pulling or building docker images (use docker directly; no NPU devices needed).
- Pure host-side work (git pull, inspect log files, read knowledge docs).

## The rules this encodes

1. **Never grab NPUs someone else is using.** Before binding chips, check `npu-smi info -t proc-mem -i <npu_id>`. If another process owns the HBM, fail loud — don't silently share.
2. **Three user-scoped bind mounts.** `/home/<user>`, `/data/<user>`, `/tmp/<user>` are always mounted. Anything produced inside the container that matters must land in one of these three paths — `--rm` destroys everything else.
3. **Device passthrough minimum set.** `/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`, plus one `/dev/davinci<chip>` per requested chip.
4. **Host libs for Ascend tooling.** `/usr/local/Ascend/driver`, `/usr/local/bin/npu-smi`, `/usr/bin/msnpureport`, `/etc/hccn.conf` bind-mount into the container at the same paths.
5. **Host IPC + host net.** HCCL needs both. `--ipc=host --network=host`. Shared-memory size bumped (`--shm-size=64g`).
6. **Env vars the NPU stack needs** (set as defaults, overridable):
   - `ASCEND_RT_VISIBLE_DEVICES=<chips>` — the NPU equivalent of CUDA_VISIBLE_DEVICES.
   - `HF_ENDPOINT=https://hf-mirror.com` — host is in China; hf.co is blocked.
   - `HF_HOME=/data/<user>/hf-cache` — cache survives container death.
   - `VLLM_ASCEND_ENABLE_NZ=0` — required for RL (vllm-ascend FRACTAL_NZ drifts params during sync; see `NPU-ENV-002`).
7. **Live source bind-mount** over the image's `/opt/<project>` path — so `git pull` on the host takes effect without a docker rebuild. Guards against `NPU-OPS-001`.

## The script

`scripts/run-npu-container.sh` is the implementation. Invoked as:

```bash
bash scripts/run-npu-container.sh \
    [--chips 0,1] \
    [--image easyr1-npu:ascend-port] \
    [--live-source /home/$USER/workspace/easyr1-npu/upstream/EasyR1] \
    -- <cmd...>
```

Defaults:
- `--chips 0,1` (one A3 card = two chips).
- `--image easyr1-npu:ascend-port`.
- `--live-source` = env var `LIVE_EASYR1` or the canonical project path.

The script:
1. Prechecks chip occupancy via `npu-smi info -t proc-mem -i <chip>`; if any chip has a non-self process holding HBM, aborts with a clear message pointing at `npu-smi info` output.
2. Builds the device-flag list from `--chips`.
3. `docker run --rm` with the mounts and env vars listed above.
4. Passes `"$@"` verbatim as the container command.

## Checklist before invoking

- [ ] I have checked `npu-smi info -t proc-mem -i <chip>` for each chip I plan to use.
- [ ] The chips are either idle or occupied only by processes I own (e.g. my own prior runs — use `docker ps` to verify).
- [ ] I have pulled the latest source on the host (`git pull personal ascend-port` or equivalent) so the live bind mount reflects the intended code.
- [ ] If I'm running a long job, I've redirected output to `/tmp/<user>/<logs-dir>/<timestamp>.log` so it survives ssh disconnects.
- [ ] If the job uses significant HBM, I've picked chips outside what another user appears to be on (defaults to 0,1; chips 8-15 were seen at 52 GB/chip on 2026-04-17 — occasionally occupied).

## Gotchas

- **Container WORKDIR overrides host cwd.** `cd $HOME/project` on the host doesn't reach inside the container. If your inner command uses relative paths, prepend `cd /opt/<project> && ` to the command string inside the container.
- **`__pycache__` can shadow source changes**. After a live-bind-mount source swap, stale `.pyc` may win. Either clear `__pycache__` before the run, or set `-e PYTHONDONTWRITEBYTECODE=1` in the runner. `NPU-OPS-002`.
- **Docker hub is slow from this host.** Prefer `quay.io/ascend/*` (direct) or `docker.m.daocloud.io` mirror for pulls. Don't run `docker pull` from inside the container runner — build/pull images separately.
- **`--ipc=host`** can leak shared memory to other users on a shared host. Acceptable here because Ascend HCCL requires it and we clean up via `--rm`, but be aware.

## Related knowledge / skills

- `knowledge/npu-patterns.md` — the 7+ NPU-specific findings this runner encodes defaults for.
- `knowledge/a3_server.md` (in agent memory) — host-specific paths and shared-host etiquette.
- `skills/upstream-branch-hygiene/SKILL.md` — the "edit locally, push, A3 pull" flow that this runner assumes.
