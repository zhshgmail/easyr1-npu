---
id: cross-layer-011
date: 2026-04-22
layer: cross-layer
title: Container can't see NPU but host can — dmesg uda_occupy_dev_by_ns first, not driver reboot
trigger:
  - "container's torch_npu reports no device but host npu-smi info shows the chip"
  - "After a previous training run crashed, new containers can't access NPU"
  - "Zombie Ray raylet processes on host"
  - "Tempting to docker restart or driver reboot"
symptom_in_wild:
  - "Inside container: torch_npu.npu.is_available() False; npu-smi info empty"
  - "Outside container: npu-smi info works, chip is healthy"
  - "After OS reboot the problem disappears (but reboot is wrong fix)"
  - "Multiple containers on same chip — only one works at a time"
root_cause: >
  Ascend driver enforces a namespace lock via `uda_occupy_dev_by_ns`. Only
  one Linux process namespace at a time can hold the lock. When a previous
  training run (often Ray-based) crashed without cleaning up its raylet
  zombie, the zombie still holds the namespace lock. New containers (in a
  different ns) can't acquire the device.

  Symptom looks like a driver bug — host works, container doesn't. The
  actual cause is a userspace zombie process.
mistake_pattern: "userspace cleanup not done after process crash; resource lock held by zombie"
correction:
  - "First diagnostic: `dmesg | grep uda_occupy_dev_by_ns` inside or outside container. If you see 'already occupied by ns X' or similar, this is the bug."
  - "Find the zombie: `ps -ef | grep -i 'ray\\|raylet\\|training'` on host"
  - "Kill it (only your own processes! never another user's): `kill -9 <pid>`"
  - "Restart the container or just retry; lock should release within a few seconds"
  - "Do NOT reboot the driver. The lock will clear cleanly once the zombie dies."
  - "Long-term fix: every training driver MUST have a finally-block that explicitly releases NPU resources before exit"
evidence:
  - "Reproduced 2026-04-22 on A3 host after a Ray-based training run crashed"
  - "Memory: a3_uda_ns_conflict.md"
  - "NPU-OPS-006: documented operational lesson — 'first check uda_occupy_dev_by_ns, not driver bug'"
  - "After killing zombie raylet: container immediately sees NPU"
---

# cross-layer-011 — uda namespace lock zombie diagnosis

## Why this matters

The wrong fix here is "reboot the NPU driver" or "reboot the host". Both work but both nuke any concurrent users' work — and on a shared dev host they're explicitly forbidden. Knowing the zombie-process root cause means the fix is 30 seconds of `kill` instead of 10 minutes of nuke-and-pray.

## Diagnostic order (memorize)

1. `dmesg | grep uda_occupy_dev_by_ns` — names the ns holding the lock
2. `ps -ef | grep -iE 'ray|raylet|train|miles'` — find candidate zombies (filter to your own pids)
3. `kill -9 <pid>` — only your own pids; never `pkill -f` because someone else's process might match
4. Wait 5-10s; retry container

## Shared-host etiquette

A3 is a shared host. Never:
- `kill -9` another user's process
- `docker container prune` (kills others' stopped containers)
- `npu-smi` reset or driver-level commands
- OS-level operations (`reboot`, `systemctl restart` of ascend services)

Only kill your own zombie. If you can't find your own zombie and the lock is held by another user, ask them on Discord rather than acting.

## Prevention

Every training driver should have:
```python
import atexit
def _cleanup():
    try:
        import torch_npu
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
    except Exception:
        pass
atexit.register(_cleanup)
```

This is best-effort — a hard crash will still leak the lock. But normal exits will clean up.
