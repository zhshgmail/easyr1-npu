#!/bin/bash
# Run a one-off command inside the easyr1-npu container with the correct Ascend
# device passthrough, user-scoped bind mounts, and host-level IPC/net (needed
# for HCCL collectives).
#
# Usage:
#   ./run-npu-container.sh [--chips 0,1] [--image easyr1-npu:ascend-port] -- <cmd...>
#
# Defaults: chips 0,1 (A3 card 0), image easyr1-npu:ascend-port.
#
# Rules this encodes:
# - Mounts the user's three standard dirs so artifacts survive container death:
#     /home/z00637938, /data/z00637938, /tmp/z00637938
# - Passes the minimum required /dev nodes (davinci_manager, devmm_svm, hisi_hdc)
#   plus the caller-selected davinci chips.
# - Binds the host driver tree, hccn.conf, npu-smi, msnpureport so runtime tools
#   are available inside the container.
# - Runs with --network=host --ipc=host --shm-size=768g, matching how veRL and
#   vllm-ascend docker examples configure NPU containers.
# - NOT persistent: container is removed on exit. Keep source outside the
#   container (bind-mount) so nothing we produce is trapped inside.

set -euo pipefail

IMAGE="easyr1-npu:ascend-port"
CHIPS="0,1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chips) CHIPS="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

DEVICE_FLAGS=(
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
)
IFS=',' read -ra CHIP_ARR <<< "$CHIPS"
for c in "${CHIP_ARR[@]}"; do
  DEVICE_FLAGS+=(--device "/dev/davinci${c}")
done

docker run --rm \
  "${DEVICE_FLAGS[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/bin/msnpureport:/usr/bin/msnpureport \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /home/z00637938:/home/z00637938 \
  -v /data/z00637938:/data/z00637938 \
  -v /tmp/z00637938:/tmp/z00637938 \
  --network=host --ipc=host --shm-size=64g \
  -e ASCEND_RT_VISIBLE_DEVICES="${CHIPS}" \
  "${IMAGE}" \
  "$@"
