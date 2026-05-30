#!/bin/bash
# Watchdog around dsv4_1layer_hbm_probe.py running inside tlrescue.
# Same safety contract as sglang_safe_probe.sh:
#   * Polls npu-smi every 2s
#   * Kills our process if other-user HBM > 50 GB OR total > 60 GB OR timeout
#   * Never touches other containers, never touches other host pids
#
# Probe runs in tlrescue (has patched MindSpeed + miles + tilelang); chip 1.
set -u

PROBE_LOG=/home/z00637938/workspace/dsv4_probe.log
WATCHDOG_LOG=/home/z00637938/workspace/dsv4_watchdog.log
SAFE_OTHER_HBM_MB=50000
TOTAL_HBM_CEILING_MB=60000
MAX_SECONDS=300

> "$PROBE_LOG"
> "$WATCHDOG_LOG"

# Launch probe inside tlrescue
docker exec -d tlrescue bash -c "
  cd /tmp
  export TILELANG_ASCEND_MODE=Developer
  export ASCEND_RT_VISIBLE_DEVICES=14
  export RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29520 LOCAL_RANK=0
  export PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/Megatron-LM-miles:/home/z00637938/workspace/tilelang-mlir-ascend:/home/z00637938/workspace/MindSpeed-clone:\$PYTHONPATH
  python /home/z00637938/workspace/dsv4_1layer_hbm_probe.py > $PROBE_LOG 2>&1
"
echo "[wd] dsv4 probe launched at $(date -u +%FT%TZ)" | tee -a "$WATCHDOG_LOG"

# tlrescue holds /dev/davinci14 which is NPU=3 Chip=0 (PCI 0000:85:00.0 = chip index 6 in npu-smi)
# Wait, let me re-read the original tlrescue config: --device /dev/davinci14
# Mapping davinci14 to physical chip: davinciN where N is the logical (renumbered) device.
# We need to find which physical chip tlrescue is on. Look at the python pids on each chip.
sleep 4
OUR_HOST_PID=$(docker top tlrescue -o pid,comm 2>&1 | grep -E "python" | head -1 | awk '{print $1}')
echo "[wd] our host pid: $OUR_HOST_PID" | tee -a "$WATCHDOG_LOG"

# Discover which physical chip our PID is on by sampling all 8 proc-mem outputs
CHIP_FOUND=""
for c in 0 1 2 3 4 5 6 7; do
    i=$((c/2)); j=$((c%2))
    PROCMEM=$(/usr/local/sbin/npu-smi info -t proc-mem -i $i -c $j 2>&1)
    if echo "$PROCMEM" | grep -q "Process id:$OUR_HOST_PID"; then
        CHIP_FOUND="$c"
        NPU_I=$i
        NPU_C=$j
        break
    fi
done

if [ -z "$CHIP_FOUND" ]; then
    echo "[wd] could not find our process on any chip yet; will retry mid-loop" | tee -a "$WATCHDOG_LOG"
    NPU_I=0; NPU_C=0  # default; will re-detect inside loop
fi
echo "[wd] our chip: $CHIP_FOUND (NPU=$NPU_I Chip=$NPU_C)" | tee -a "$WATCHDOG_LOG"

START=$(date +%s)

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if grep -q "\[probe\] PASS" "$PROBE_LOG" 2>/dev/null; then
        echo "[wd] PROBE_DONE at +${ELAPSED}s" | tee -a "$WATCHDOG_LOG"
        break
    fi
    if grep -qE "^Traceback|RuntimeError|FAIL" "$PROBE_LOG" 2>/dev/null; then
        echo "[wd] PROBE_ERROR at +${ELAPSED}s (probe self-failed)" | tee -a "$WATCHDOG_LOG"
        break
    fi
    if (( ELAPSED > MAX_SECONDS )); then
        echo "[wd] PROBE_TIMEOUT after ${MAX_SECONDS}s -- killing" | tee -a "$WATCHDOG_LOG"
        docker exec tlrescue bash -c "pkill -9 -f dsv4_1layer_hbm_probe" 2>&1
        break
    fi

    # Re-detect chip if still unknown
    if [ -z "$CHIP_FOUND" ]; then
        for c in 0 1 2 3 4 5 6 7; do
            i=$((c/2)); j=$((c%2))
            PROCMEM=$(/usr/local/sbin/npu-smi info -t proc-mem -i $i -c $j 2>&1)
            if echo "$PROCMEM" | grep -q "Process id:$OUR_HOST_PID"; then
                CHIP_FOUND="$c"; NPU_I=$i; NPU_C=$j
                echo "[wd] our chip detected: $CHIP_FOUND" | tee -a "$WATCHDOG_LOG"
                break
            fi
        done
    fi

    PROCMEM_LINE=$(/usr/local/sbin/npu-smi info -t proc-mem -i $NPU_I -c $NPU_C 2>&1)
    TOTAL_HBM=$(echo "$PROCMEM_LINE" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END {print s+0}')
    OUR_HBM=$(echo "$PROCMEM_LINE" | grep "Process id:$OUR_HOST_PID" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | head -1)
    OUR_HBM=${OUR_HBM:-0}
    OTHER_HBM=$((TOTAL_HBM - OUR_HBM))

    echo "[wd] +${ELAPSED}s chip=$CHIP_FOUND total=${TOTAL_HBM}MB our=${OUR_HBM}MB other=${OTHER_HBM}MB" | tee -a "$WATCHDOG_LOG"

    if (( OTHER_HBM > SAFE_OTHER_HBM_MB )); then
        echo "[wd] OTHER_USER_GREW (${OTHER_HBM} > ${SAFE_OTHER_HBM_MB}) -- killing" | tee -a "$WATCHDOG_LOG"
        docker exec tlrescue bash -c "pkill -9 -f dsv4_1layer_hbm_probe" 2>&1
        break
    fi
    if (( TOTAL_HBM > TOTAL_HBM_CEILING_MB )); then
        echo "[wd] TOTAL_HIGH (${TOTAL_HBM} > ${TOTAL_HBM_CEILING_MB}) -- killing" | tee -a "$WATCHDOG_LOG"
        docker exec tlrescue bash -c "pkill -9 -f dsv4_1layer_hbm_probe" 2>&1
        break
    fi

    sleep 2
done

echo "[wd] watchdog exit" | tee -a "$WATCHDOG_LOG"
