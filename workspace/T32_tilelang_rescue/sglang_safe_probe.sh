#!/bin/bash
# Safe-mode sglang minimal probe with watchdog.
#
# Concurrent setup:
#   * Our probe runs in sgl_probe container holding /dev/davinci1
#   * Watchdog (this script) polls npu-smi every 2s
#   * Watchdog kills our probe if:
#       (a) other-user HBM on our chip rises above safe ceiling, OR
#       (b) total HBM on our chip exceeds 95% (60 GB / 64 GB)
#
# Definition of "other-user HBM": total HBM used - our process's HBM
# Safe ceiling: 50 GB on our chip (other user starts at ~37 GB; if they
# climb to 50 GB we vacate to leave 14 GB headroom for their growth)
#
# Exit reasons emitted to stderr:
#   "OTHER_USER_GREW" -- other proc HBM exceeded ceiling
#   "TOTAL_HIGH"      -- combined HBM > 60 GB
#   "PROBE_DONE"      -- probe finished naturally
#   "PROBE_TIMEOUT"   -- probe ran longer than safe budget (300s)
set -u

PROBE_LOG=/home/z00637938/workspace/sgl_minimal_probe.log
WATCHDOG_LOG=/home/z00637938/workspace/sgl_watchdog.log
SAFE_OTHER_HBM_MB=50000   # other user HBM ceiling = 50 GB
TOTAL_HBM_CEILING_MB=60000 # chip-total HBM ceiling = 60 GB

> "$PROBE_LOG"
> "$WATCHDOG_LOG"

# launch probe in background
docker exec -d sgl_probe bash -c "cd /tmp && python /home/z00637938/workspace/sglang_minimal_probe.py > $PROBE_LOG 2>&1"
echo "[wd] probe launched at $(date -u +%FT%TZ)" | tee -a "$WATCHDOG_LOG"

# physical chip-id for /dev/davinci1: this corresponds to npu-smi NPU=0 Chip=1
# (see npu-smi info table). proc-mem -i 0 -c 1 is what we sample.
NPU_I=0
NPU_C=1

# Find our container's main python proc PID (host-side)
sleep 4
OUR_HOST_PID=$(docker top sgl_probe -o pid,comm 2>&1 | grep -E "python" | head -1 | awk '{print $1}')
echo "[wd] our host pid: $OUR_HOST_PID" | tee -a "$WATCHDOG_LOG"

START=$(date +%s)
MAX_SECONDS=300

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    # Probe naturally finished?
    if grep -q "\[probe\] PASS" "$PROBE_LOG" 2>/dev/null; then
        echo "[wd] PROBE_DONE at +${ELAPSED}s" | tee -a "$WATCHDOG_LOG"
        break
    fi
    if grep -qE "Traceback|RuntimeError|FAIL" "$PROBE_LOG" 2>/dev/null; then
        echo "[wd] PROBE_ERROR at +${ELAPSED}s (probe self-failed)" | tee -a "$WATCHDOG_LOG"
        break
    fi

    # Timeout
    if (( ELAPSED > MAX_SECONDS )); then
        echo "[wd] PROBE_TIMEOUT after ${MAX_SECONDS}s -- killing" | tee -a "$WATCHDOG_LOG"
        docker exec sgl_probe bash -c "pkill -9 -f sglang_minimal_probe" 2>&1
        break
    fi

    # Sample HBM
    PROCMEM_LINE=$(/usr/local/sbin/npu-smi info -t proc-mem -i $NPU_I -c $NPU_C 2>&1)
    # Sum all process memory (MB) from output, then identify others vs ours
    TOTAL_HBM=$(echo "$PROCMEM_LINE" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END {print s+0}')

    # Our HBM (our container python proc)
    if [ -n "$OUR_HOST_PID" ]; then
        OUR_HBM=$(echo "$PROCMEM_LINE" | grep "Process id:$OUR_HOST_PID" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | head -1)
        OUR_HBM=${OUR_HBM:-0}
    else
        OUR_HBM=0
    fi
    OTHER_HBM=$((TOTAL_HBM - OUR_HBM))

    echo "[wd] +${ELAPSED}s total=${TOTAL_HBM}MB our=${OUR_HBM}MB other=${OTHER_HBM}MB" | tee -a "$WATCHDOG_LOG"

    if (( OTHER_HBM > SAFE_OTHER_HBM_MB )); then
        echo "[wd] OTHER_USER_GREW (${OTHER_HBM} > ${SAFE_OTHER_HBM_MB}) -- killing our probe" | tee -a "$WATCHDOG_LOG"
        docker exec sgl_probe bash -c "pkill -9 -f sglang_minimal_probe" 2>&1
        break
    fi
    if (( TOTAL_HBM > TOTAL_HBM_CEILING_MB )); then
        echo "[wd] TOTAL_HIGH (${TOTAL_HBM} > ${TOTAL_HBM_CEILING_MB}) -- killing our probe" | tee -a "$WATCHDOG_LOG"
        docker exec sgl_probe bash -c "pkill -9 -f sglang_minimal_probe" 2>&1
        break
    fi

    sleep 2
done

echo "[wd] watchdog exit" | tee -a "$WATCHDOG_LOG"
