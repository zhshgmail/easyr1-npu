#!/bin/bash
# Start sglang HTTP server in sgl_probe container.
#
# Designed for chip-sharing with another user's long-running job:
#   * Single NPU device (whichever is mounted into the container)
#   * mem_fraction_static=0.10 (~6.4 GiB on 64 GiB chip; safe with ~28 GiB free)
#   * disable cuda graphs (eager mode, matches our vllm-ascend smoke)
#   * server binds 0.0.0.0:30000; container has /home/z00637938 mount so
#     tlrescue can reach it via the host network or container IP
#
# Run (from host): docker exec -d sgl_probe bash /home/z00637938/workspace/start_sglang_server.sh
set -e

MODEL=/host-models/Qwen2-0.5B-Instruct
HOST=0.0.0.0
PORT=30000

# Same sys.path fix as the smoke script: '/' shadows editable sglang install.
export PYTHONSTARTUP=""
export PYTHONUNBUFFERED=1

# sglang server uses multiprocessing; spawn re-imports its own main module
# correctly (it's a console script). No __main__ guard needed in our shell.

cd /tmp
python -c "
import sys
sys.path = [p for p in sys.path if p not in ('', '/')]
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
server_args = ServerArgs(
    model_path='$MODEL',
    host='$HOST',
    port=$PORT,
    dtype='bfloat16',
    device='npu',
    mem_fraction_static=0.10,
    tp_size=1,
    disable_cuda_graph=True,
)
launch_server(server_args)
"
