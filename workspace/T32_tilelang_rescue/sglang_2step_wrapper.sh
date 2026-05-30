#!/bin/bash
# Wrapper: start sglang HTTP server in sgl_probe, run 2-step rollout smoke,
# tear down server, monitor HBM throughout.
set -u

SERVER_LOG=/home/z00637938/workspace/sgl_2step_server.log
SMOKE_LOG=/home/z00637938/workspace/sgl_2step_smoke.log
WATCHDOG_LOG=/home/z00637938/workspace/sgl_2step_watchdog.log
SAFE_OTHER_HBM_MB=50000
TOTAL_HBM_CEILING_MB=60000
MAX_SECONDS=600

> "$SERVER_LOG"
> "$SMOKE_LOG"
> "$WATCHDOG_LOG"

# 1. Start sglang HTTP server inside sgl_probe (background)
echo "[wrap] starting sglang HTTP server in sgl_probe ..." | tee -a "$WATCHDOG_LOG"
docker exec -d sgl_probe bash -c "
  cd /tmp
  python -c \"
import sys
sys.path = [p for p in sys.path if p not in ('', '/')]
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
sa = ServerArgs(
    model_path='/host-models/Qwen2-0.5B-Instruct',
    host='127.0.0.1',
    port=30000,
    dtype='bfloat16',
    device='npu',
    mem_fraction_static=0.05,
    max_total_tokens=1024,
    max_running_requests=1,
    max_prefill_tokens=256,
    chunked_prefill_size=256,
    enable_memory_saver=True,
    disable_radix_cache=True,
    tp_size=1,
    disable_cuda_graph=True,
)
launch_server(sa)
\" > $SERVER_LOG 2>&1
"

# 2. Wait for server to be healthy from sgl_probe perspective
echo "[wrap] waiting for sglang health ..." | tee -a "$WATCHDOG_LOG"
SERVER_READY=0
for i in $(seq 1 90); do
    if docker exec sgl_probe curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health_generate 2>/dev/null | grep -q 200; then
        SERVER_READY=1
        echo "[wrap] server healthy at +${i}s" | tee -a "$WATCHDOG_LOG"
        break
    fi
    sleep 2
done

if [ "$SERVER_READY" != "1" ]; then
    echo "[wrap] sglang server failed to come up; aborting" | tee -a "$WATCHDOG_LOG"
    docker exec sgl_probe bash -c "pkill -9 -f launch_server || true" 2>&1
    exit 1
fi

# 3. Run 2-step smoke (also from inside sgl_probe; it talks to localhost server)
echo "[wrap] launching 2-step smoke ..." | tee -a "$WATCHDOG_LOG"
docker exec -d sgl_probe bash -c "cd /tmp && python /home/z00637938/workspace/sglang_2step_rollout_smoke.py > $SMOKE_LOG 2>&1"

# 4. Watchdog loop -- monitor HBM and smoke progress
OUR_HOST_PID=$(docker top sgl_probe -o pid,comm 2>&1 | grep -E "python" | head -1 | awk '{print $1}')
echo "[wd] our host pid: $OUR_HOST_PID" | tee -a "$WATCHDOG_LOG"

# Detect chip
CHIP_FOUND=""
for c in 0 1 2 3 4 5 6 7; do
    i=$((c/2)); j=$((c%2))
    if /usr/local/sbin/npu-smi info -t proc-mem -i $i -c $j 2>&1 | grep -q "Process id:$OUR_HOST_PID"; then
        CHIP_FOUND="$c"; NPU_I=$i; NPU_C=$j; break
    fi
done
echo "[wd] our chip: $CHIP_FOUND" | tee -a "$WATCHDOG_LOG"

START=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if grep -q "\[smoke\] PASS" "$SMOKE_LOG" 2>/dev/null; then
        echo "[wd] SMOKE_PASS at +${ELAPSED}s" | tee -a "$WATCHDOG_LOG"
        break
    fi
    if grep -qE "^\[smoke\] FAIL|Traceback|RuntimeError" "$SMOKE_LOG" 2>/dev/null; then
        echo "[wd] SMOKE_FAIL at +${ELAPSED}s" | tee -a "$WATCHDOG_LOG"
        break
    fi
    if (( ELAPSED > MAX_SECONDS )); then
        echo "[wd] TIMEOUT" | tee -a "$WATCHDOG_LOG"
        docker exec sgl_probe bash -c "pkill -9 -f sglang_2step_rollout_smoke || true" 2>&1
        break
    fi

    if [ -z "$CHIP_FOUND" ]; then
        for c in 0 1 2 3 4 5 6 7; do
            i=$((c/2)); j=$((c%2))
            if /usr/local/sbin/npu-smi info -t proc-mem -i $i -c $j 2>&1 | grep -q "Process id:$OUR_HOST_PID"; then
                CHIP_FOUND="$c"; NPU_I=$i; NPU_C=$j
                echo "[wd] our chip detected: $CHIP_FOUND" | tee -a "$WATCHDOG_LOG"
                break
            fi
        done
    fi

    if [ -n "$CHIP_FOUND" ]; then
        PROCMEM=$(/usr/local/sbin/npu-smi info -t proc-mem -i $NPU_I -c $NPU_C 2>&1)
        TOTAL=$(echo "$PROCMEM" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END {print s+0}')
        OUR=$(echo "$PROCMEM" | grep "Process id:$OUR_HOST_PID" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | head -1)
        OUR=${OUR:-0}
        OTHER=$((TOTAL - OUR))
        echo "[wd] +${ELAPSED}s chip=$CHIP_FOUND total=${TOTAL}MB our=${OUR}MB other=${OTHER}MB" | tee -a "$WATCHDOG_LOG"
        if (( OTHER > SAFE_OTHER_HBM_MB )); then
            echo "[wd] OTHER_USER_GREW; killing" | tee -a "$WATCHDOG_LOG"
            docker exec sgl_probe bash -c "pkill -9 -f sglang_2step_rollout_smoke || true" 2>&1
            break
        fi
        if (( TOTAL > TOTAL_HBM_CEILING_MB )); then
            echo "[wd] TOTAL_HIGH; killing" | tee -a "$WATCHDOG_LOG"
            docker exec sgl_probe bash -c "pkill -9 -f sglang_2step_rollout_smoke || true" 2>&1
            break
        fi
    fi
    sleep 3
done

# 5. Tear down: stop sglang server in sgl_probe (only our process, not the container)
echo "[wrap] tearing down sglang server ..." | tee -a "$WATCHDOG_LOG"
docker exec sgl_probe bash -c "pkill -9 -f 'launch_server' || true; pkill -9 -f sglang_2step_rollout_smoke || true" 2>&1
sleep 2
echo "[wrap] done" | tee -a "$WATCHDOG_LOG"
