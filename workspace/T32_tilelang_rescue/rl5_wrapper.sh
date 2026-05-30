#!/bin/bash
# Wrapper for _e2e_rl_5step_dsv4_sglang.py:
#   - Start sglang HTTP server in sgl_probe (chip A; Qwen2-0.5B at 0.05 mem
#     fraction; this is the rollout side, decoupled architecturally from
#     the trainer side per design note in the script)
#   - Run the 5-step RL driver inside tlrescue (chip B; trains miles
#     DSAMLA at miles_local config); driver hits sglang via HTTP
#   - Watchdog monitors HBM on BOTH chips, kills our processes on either
#     side if other-user HBM rises or chip total exceeds ceiling
#   - Tear down on completion
set -u

SERVER_LOG=/home/z00637938/workspace/rl5_server.log
DRIVER_LOG=/home/z00637938/workspace/rl5_driver.log
WATCHDOG_LOG=/home/z00637938/workspace/rl5_watchdog.log
SAFE_OTHER_HBM_MB=50000
TOTAL_HBM_CEILING_MB=60000
MAX_SECONDS=1800   # 30 min budget for 5 steps incl model build + 4 NPU kernel compiles

> "$SERVER_LOG"
> "$DRIVER_LOG"
> "$WATCHDOG_LOG"

# 1. Start sglang HTTP server inside sgl_probe
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

# Wait for sglang
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

# Find sgl_probe's bridge IP (tlrescue is on host net so it can reach 172.17.0.0/16)
SG_IP=$(docker inspect sgl_probe --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' | head -1)
echo "[wrap] sgl_probe IP: $SG_IP" | tee -a "$WATCHDOG_LOG"

# 2. Launch driver in tlrescue
echo "[wrap] launching 5-step RL driver in tlrescue ..." | tee -a "$WATCHDOG_LOG"
docker exec -d tlrescue bash -c "
  cd /tmp
  export TILELANG_ASCEND_MODE=Developer
  export RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29530 LOCAL_RANK=0
  export PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/Megatron-LM-miles:/home/z00637938/workspace/tilelang-mlir-ascend:/home/z00637938/workspace/MindSpeed-clone:\$PYTHONPATH
  export SGLANG_BASE_URL='http://${SG_IP}:30000'
  python /home/z00637938/workspace/_e2e_rl_5step_dsv4_sglang.py > $DRIVER_LOG 2>&1
"

# 3. Watchdog on both chip A (sgl_probe) and chip B (tlrescue)
SG_PID=$(docker top sgl_probe -o pid,comm 2>&1 | grep -E "python" | head -1 | awk '{print $1}')
TL_PID=$(docker top tlrescue -o pid,comm 2>&1 | grep -E "python" | head -1 | awk '{print $1}')
echo "[wd] sg pid=$SG_PID  tl pid=$TL_PID" | tee -a "$WATCHDOG_LOG"

find_chip() {
    local pid=$1
    for c in 0 1 2 3 4 5 6 7; do
        i=$((c/2)); j=$((c%2))
        if /usr/local/sbin/npu-smi info -t proc-mem -i $i -c $j 2>&1 | grep -q "Process id:$pid"; then
            echo "$c $i $j"; return
        fi
    done
    echo ""
}

read SG_CHIP SG_I SG_J <<<"$(find_chip $SG_PID)"
echo "[wd] sgl_probe chip: $SG_CHIP" | tee -a "$WATCHDOG_LOG"

START=$(date +%s)
KILL_REASON=""

while true; do
    NOW=$(date +%s); ELAPSED=$((NOW - START))

    if grep -q "\[smoke\] PASS" "$DRIVER_LOG" 2>/dev/null; then
        echo "[wd] SMOKE_PASS at +${ELAPSED}s" | tee -a "$WATCHDOG_LOG"; break
    fi
    if grep -qE "Traceback|RuntimeError|AttributeError|ValueError|\[smoke\] FAIL" "$DRIVER_LOG" 2>/dev/null; then
        echo "[wd] SMOKE_FAIL at +${ELAPSED}s" | tee -a "$WATCHDOG_LOG"; break
    fi
    if (( ELAPSED > MAX_SECONDS )); then
        echo "[wd] TIMEOUT" | tee -a "$WATCHDOG_LOG"; KILL_REASON="timeout"; break
    fi

    # If tl_pid was empty initially, retry
    if [ -z "$TL_PID" ]; then
        TL_PID=$(docker top tlrescue -o pid,comm 2>&1 | grep -E "python" | head -1 | awk '{print $1}')
    fi

    # Detect TL chip lazily after model loads NPU
    if [ -z "${TL_CHIP:-}" ] && [ -n "$TL_PID" ]; then
        read TL_CHIP TL_I TL_J <<<"$(find_chip $TL_PID)"
        if [ -n "$TL_CHIP" ]; then
            echo "[wd] tlrescue chip: $TL_CHIP" | tee -a "$WATCHDOG_LOG"
        fi
    fi

    REPORT=""
    # Sample sg chip
    if [ -n "$SG_CHIP" ]; then
        PM=$(/usr/local/sbin/npu-smi info -t proc-mem -i $SG_I -c $SG_J 2>&1)
        TOT=$(echo "$PM" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END {print s+0}')
        OUR=$(echo "$PM" | grep "Process id:$SG_PID" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | head -1)
        OUR=${OUR:-0}
        OTHER=$((TOT - OUR))
        REPORT="$REPORT sg-chip=$SG_CHIP tot=${TOT}MB other=${OTHER}MB;"
        if (( OTHER > SAFE_OTHER_HBM_MB )); then KILL_REASON="OTHER_USER_GREW_on_sg"; break; fi
        if (( TOT > TOTAL_HBM_CEILING_MB )); then KILL_REASON="TOTAL_HIGH_on_sg"; break; fi
    fi
    # Sample tl chip
    if [ -n "${TL_CHIP:-}" ] && [ -n "$TL_PID" ]; then
        PM=$(/usr/local/sbin/npu-smi info -t proc-mem -i $TL_I -c $TL_J 2>&1)
        TOT=$(echo "$PM" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END {print s+0}')
        OUR=$(echo "$PM" | grep "Process id:$TL_PID" | grep -oE "Process memory\(MB\):[0-9]+" | grep -oE "[0-9]+" | head -1)
        OUR=${OUR:-0}
        OTHER=$((TOT - OUR))
        REPORT="$REPORT tl-chip=$TL_CHIP tot=${TOT}MB other=${OTHER}MB;"
        if (( OTHER > SAFE_OTHER_HBM_MB )); then KILL_REASON="OTHER_USER_GREW_on_tl"; break; fi
        if (( TOT > TOTAL_HBM_CEILING_MB )); then KILL_REASON="TOTAL_HIGH_on_tl"; break; fi
    fi
    if [ -n "$REPORT" ]; then
        echo "[wd] +${ELAPSED}s $REPORT" | tee -a "$WATCHDOG_LOG"
    fi
    sleep 5
done

if [ -n "$KILL_REASON" ]; then
    echo "[wd] kill triggered: $KILL_REASON" | tee -a "$WATCHDOG_LOG"
    docker exec tlrescue bash -c "pkill -9 -f _e2e_rl_5step_dsv4_sglang || true" 2>&1
    docker exec sgl_probe bash -c "pkill -9 -f launch_server || true" 2>&1
fi

# Tear down
echo "[wrap] tearing down sglang ..." | tee -a "$WATCHDOG_LOG"
docker exec sgl_probe bash -c "pkill -9 -f launch_server || true" 2>&1
sleep 2
echo "[wrap] done" | tee -a "$WATCHDOG_LOG"
