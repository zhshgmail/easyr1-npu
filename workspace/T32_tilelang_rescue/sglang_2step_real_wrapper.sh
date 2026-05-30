#!/bin/bash
# Wrapper for sglang_2step_real_update.py:
#   * Starts sglang server in sgl_probe (chip 1)
#   * Runs the smoke INSIDE TLRESCUE (where transformers is for training side)
#     -- tlrescue talks to sgl_probe via Docker bridge http://sgl_probe:30000
#   * Watchdog monitors HBM on chip 1 (where sgl_probe lives)
#   * Tears down server on completion
#
# Per user direction (option A): no HCCL, no distributed update -- the
# training side runs CPU-only in tlrescue to avoid touching NPU at all
# on tlrescue's side.
set -u

SERVER_LOG=/home/z00637938/workspace/sgl_real_server.log
SMOKE_LOG=/home/z00637938/workspace/sgl_real_smoke.log
WATCHDOG_LOG=/home/z00637938/workspace/sgl_real_watchdog.log
SAFE_OTHER_HBM_MB=50000
TOTAL_HBM_CEILING_MB=60000
MAX_SECONDS=900

> "$SERVER_LOG"
> "$SMOKE_LOG"
> "$WATCHDOG_LOG"

# Quick docker network check: ensure tlrescue can reach sgl_probe by name.
# If not, fall back to host bridge IP via /etc/hosts trick or simpler: use
# host network mode? Let's check what bridge tlrescue is on first.
TL_NET=$(docker inspect tlrescue --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null | head -1)
SG_NET=$(docker inspect sgl_probe --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null | head -1)
echo "[wrap] tlrescue net: $TL_NET   sgl_probe net: $SG_NET" | tee -a "$WATCHDOG_LOG"
# If both are on bridge but DNS not set, find sgl_probe's IP and pass as env
SG_IP=$(docker inspect sgl_probe --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null | head -1)
echo "[wrap] sgl_probe IP: $SG_IP" | tee -a "$WATCHDOG_LOG"

# Connect sgl_probe to tlrescue's network if they're not on the same one
if [ -n "$TL_NET" ] && [ "$TL_NET" != "$SG_NET" ]; then
    echo "[wrap] joining sgl_probe to network $TL_NET ..." | tee -a "$WATCHDOG_LOG"
    docker network connect "$TL_NET" sgl_probe 2>&1 | tee -a "$WATCHDOG_LOG"
fi

# 1. Start sglang HTTP server inside sgl_probe (binds 0.0.0.0 so tlrescue can hit it)
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
    host='0.0.0.0',
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

# 2. Wait for health (probed from inside sgl_probe -- localhost)
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
    echo "[wrap] sglang server failed to come up" | tee -a "$WATCHDOG_LOG"
    docker exec sgl_probe bash -c "pkill -9 -f launch_server || true" 2>&1
    exit 1
fi

# 3. Run smoke INSIDE tlrescue. Pass sglang URL = sgl_probe IP or name.
SGL_TARGET="http://${SG_IP}:30000"
echo "[wrap] launching smoke inside tlrescue, SGLANG_BASE_URL=$SGL_TARGET" | tee -a "$WATCHDOG_LOG"
docker exec -d tlrescue bash -c "
  cd /tmp
  export SGLANG_BASE_URL='$SGL_TARGET'
  python /home/z00637938/workspace/sglang_2step_real_update.py > $SMOKE_LOG 2>&1
"

# 4. Watchdog on chip 1 (where sgl_probe lives)
OUR_HOST_PID=$(docker top sgl_probe -o pid,comm 2>&1 | grep -E "python" | head -1 | awk '{print $1}')
echo "[wd] our sgl_probe host pid: $OUR_HOST_PID" | tee -a "$WATCHDOG_LOG"

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
    NOW=$(date +%s); ELAPSED=$((NOW - START))
    if grep -q "\[smoke\] PASS" "$SMOKE_LOG" 2>/dev/null; then
        echo "[wd] SMOKE_PASS at +${ELAPSED}s" | tee -a "$WATCHDOG_LOG"; break
    fi
    if grep -qE "^\[smoke\] FAIL|^Traceback|RuntimeError" "$SMOKE_LOG" 2>/dev/null; then
        echo "[wd] SMOKE_FAIL at +${ELAPSED}s" | tee -a "$WATCHDOG_LOG"; break
    fi
    if (( ELAPSED > MAX_SECONDS )); then
        echo "[wd] TIMEOUT" | tee -a "$WATCHDOG_LOG"
        docker exec tlrescue bash -c "pkill -9 -f sglang_2step_real_update || true" 2>&1
        docker exec sgl_probe bash -c "pkill -9 -f launch_server || true" 2>&1
        break
    fi
    if [ -z "$CHIP_FOUND" ]; then
        for c in 0 1 2 3 4 5 6 7; do
            i=$((c/2)); j=$((c%2))
            if /usr/local/sbin/npu-smi info -t proc-mem -i $i -c $j 2>&1 | grep -q "Process id:$OUR_HOST_PID"; then
                CHIP_FOUND="$c"; NPU_I=$i; NPU_C=$j; break
            fi
        done
    fi
    if [ -n "$CHIP_FOUND" ]; then
        PM=$(/usr/local/sbin/npu-smi info -t proc-mem -i $NPU_I -c $NPU_C 2>&1)
        TOTAL=$(echo "$PM" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END {print s+0}')
        OUR=$(echo "$PM" | grep "Process id:$OUR_HOST_PID" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | head -1)
        OUR=${OUR:-0}
        OTHER=$((TOTAL - OUR))
        echo "[wd] +${ELAPSED}s chip=$CHIP_FOUND total=${TOTAL}MB our=${OUR}MB other=${OTHER}MB" | tee -a "$WATCHDOG_LOG"
        if (( OTHER > SAFE_OTHER_HBM_MB )); then
            echo "[wd] OTHER_USER_GREW; killing" | tee -a "$WATCHDOG_LOG"
            docker exec tlrescue bash -c "pkill -9 -f sglang_2step_real_update || true" 2>&1
            docker exec sgl_probe bash -c "pkill -9 -f launch_server || true" 2>&1
            break
        fi
        if (( TOTAL > TOTAL_HBM_CEILING_MB )); then
            echo "[wd] TOTAL_HIGH; killing" | tee -a "$WATCHDOG_LOG"
            docker exec tlrescue bash -c "pkill -9 -f sglang_2step_real_update || true" 2>&1
            docker exec sgl_probe bash -c "pkill -9 -f launch_server || true" 2>&1
            break
        fi
    fi
    sleep 3
done

# 5. Tear down
echo "[wrap] tearing down sglang server ..." | tee -a "$WATCHDOG_LOG"
docker exec sgl_probe bash -c "pkill -9 -f launch_server || true" 2>&1
docker exec tlrescue bash -c "pkill -9 -f sglang_2step_real_update || true" 2>&1
sleep 2
echo "[wrap] done" | tee -a "$WATCHDOG_LOG"
