#!/bin/bash
# Run the test_load smoke under same watchdog discipline as previous probes.
set -u
LOG=/home/z00637938/workspace/test_load_smoke.log
WDLOG=/home/z00637938/workspace/test_load_wd.log
> "$LOG"; > "$WDLOG"

docker exec -d sgl_probe bash -c "cd /tmp && python /home/z00637938/workspace/test_load_dsv4_fab.py > $LOG 2>&1"
echo "[wd] test launched" | tee -a "$WDLOG"
sleep 4
PID=$(docker top sgl_probe -o pid,comm 2>&1 | grep -E "python" | head -1 | awk '{print $1}')
echo "[wd] sgl pid: $PID" | tee -a "$WDLOG"

CHIP=""
for c in 0 1 2 3 4 5 6 7; do
    i=$((c/2)); j=$((c%2))
    if /usr/local/sbin/npu-smi info -t proc-mem -i $i -c $j 2>&1 | grep -q "Process id:$PID"; then
        CHIP="$c"; NI=$i; NJ=$j; break
    fi
done
echo "[wd] chip: $CHIP" | tee -a "$WDLOG"

START=$(date +%s)
while true; do
    EL=$(( $(date +%s) - START ))
    if grep -q "\[test\] PASS" "$LOG" 2>/dev/null; then echo "[wd] PASS at +${EL}s" | tee -a "$WDLOG"; break; fi
    if grep -qE "Traceback|RuntimeError|AttributeError|ValueError|FAIL" "$LOG" 2>/dev/null; then echo "[wd] FAIL at +${EL}s" | tee -a "$WDLOG"; break; fi
    if (( EL > 300 )); then echo "[wd] TIMEOUT" | tee -a "$WDLOG"; docker exec sgl_probe bash -c "pkill -9 -f test_load_dsv4_fab || true"; break; fi
    if [ -z "$CHIP" ]; then
        for c in 0 1 2 3 4 5 6 7; do
            i=$((c/2)); j=$((c%2))
            if /usr/local/sbin/npu-smi info -t proc-mem -i $i -c $j 2>&1 | grep -q "Process id:$PID"; then
                CHIP="$c"; NI=$i; NJ=$j; break
            fi
        done
    fi
    if [ -n "$CHIP" ]; then
        PM=$(/usr/local/sbin/npu-smi info -t proc-mem -i $NI -c $NJ 2>&1)
        TOT=$(echo "$PM" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END {print s+0}')
        OUR=$(echo "$PM" | grep "Process id:$PID" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | head -1); OUR=${OUR:-0}
        OTH=$((TOT - OUR))
        echo "[wd] +${EL}s chip=$CHIP tot=${TOT}MB our=${OUR}MB other=${OTH}MB" | tee -a "$WDLOG"
        if (( OTH > 50000 )); then echo "[wd] OTHER_USER" | tee -a "$WDLOG"; docker exec sgl_probe bash -c "pkill -9 -f test_load_dsv4_fab || true"; break; fi
        if (( TOT > 60000 )); then echo "[wd] TOT_HIGH" | tee -a "$WDLOG"; docker exec sgl_probe bash -c "pkill -9 -f test_load_dsv4_fab || true"; break; fi
    fi
    sleep 3
done
echo "[wd] exit" | tee -a "$WDLOG"
