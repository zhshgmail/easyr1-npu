#!/bin/bash
# Run a one-off command inside the easyr1-npu container with the correct Ascend
# device passthrough, user-scoped bind mounts, and host-level IPC/net (needed
# for HCCL collectives).
#
# Usage:
#   ./run-npu-container.sh [--chips 0,1]
#                          [--image easyr1-npu:ascend-port]
#                          [--live-source <path-to-EasyR1-tree>]
#                          [--skip-chip-check]
#                          -- <cmd...>
#
# Defaults:
#   --chips 0,1 (one A3 card).
#   --image easyr1-npu:ascend-port.
#   --live-source = env LIVE_EASYR1 or $HOME/workspace/easyr1-npu/upstream/EasyR1.
#
# User identity: derived from $USER; override with --user or env NPU_USER if
# the shell user differs from the one owning the three bind-mounted trees
# (/home/<user>, /data/<user>, /tmp/<user>).
#
# Operational rules this encodes (see repo/skills/npu-container-runner/SKILL.md
# for the full rationale and checklist):
# - Prechecks chip occupancy via `npu-smi info -t proc-mem -i <chip>` and aborts
#   if another process holds HBM on any requested chip. Override with
#   `--skip-chip-check` only when the caller knows they own the process.
# - Mounts /home/<user>, /data/<user>, /tmp/<user> so artifacts survive --rm.
# - Binds the host driver tree, hccn.conf, npu-smi, msnpureport.
# - Bind-mounts the live EasyR1 source over /opt/easyr1 so git pull on host ==
#   picked up next container spawn (no docker build needed for source-only
#   changes). See NPU-OPS-001.
# - Sets the env vars the NPU stack expects (HF mirror, vllm-ascend NZ off,
#   PYTHONDONTWRITEBYTECODE to avoid NPU-OPS-002 stale pycache).

set -euo pipefail

IMAGE="easyr1-npu:ascend-port"
CHIPS="0,1"
NPU_USER="${NPU_USER:-$USER}"
LIVE_EASYR1="${LIVE_EASYR1:-$HOME/workspace/easyr1-npu/upstream/EasyR1}"
SKIP_CHIP_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chips) CHIPS="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --live-source) LIVE_EASYR1="$2"; shift 2 ;;
    --user) NPU_USER="$2"; shift 2 ;;
    --skip-chip-check) SKIP_CHIP_CHECK=1; shift ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

for d in "/home/${NPU_USER}" "/data/${NPU_USER}" "/tmp/${NPU_USER}"; do
  if [[ ! -d "$d" ]]; then
    echo "WARN: bind-mount source $d does not exist; creating empty dir so docker doesn't auto-create as root." >&2
    mkdir -p "$d" || { echo "ERROR: cannot mkdir $d" >&2; exit 4; }
  fi
done

IFS=',' read -ra CHIP_ARR <<< "$CHIPS"

# In-container chip indices after `--device /dev/davinci<host_id>` mapping.
# CANN's `ASCEND_RT_VISIBLE_DEVICES` resolves against the in-container
# enumeration (0..N-1 in the order `--device` flags appear), NOT against
# host phy-ids. Confusing the two leaves Ray with `Total available GPUs 0`
# even though the chips are mounted into the container. See NPU-OPS-012.
IN_CONTAINER_IDX=()
for ((i=0; i<${#CHIP_ARR[@]}; i++)); do
  IN_CONTAINER_IDX+=("$i")
done
IN_CONTAINER_CSV=$(IFS=,; echo "${IN_CONTAINER_IDX[*]}")

# --- Chip occupancy precheck --------------------------------------------------
# npu-smi -t proc-mem returns lines per chip. If a chip has any "Process id"
# lines we abort (unless caller explicitly opts out). Shared host rule: never
# clobber someone else's run.
if [[ $SKIP_CHIP_CHECK -eq 0 ]]; then
  for c in "${CHIP_ARR[@]}"; do
    # Map chip index → (NPU card index, chip-within-card). 2 chips per A3 card.
    npu_id=$((c / 2))
    chip_id=$((c % 2))
    # Get the proc-mem block for this NPU card, split per chip, inspect this chip.
    proc_info=$(npu-smi info -t proc-mem -i "${npu_id}" 2>/dev/null || true)
    if [[ -z "$proc_info" ]]; then
      echo "WARN: could not query npu-smi for NPU ${npu_id} (chip ${c})" >&2
      continue
    fi
    # "No process in device" for a free chip; "Process id" lines for occupied.
    # Slice to just the block for this chip within the card.
    chip_block=$(awk -v cid="${chip_id}" '
      /^[[:space:]]*Chip ID[[:space:]]*:[[:space:]]*[0-9]+/ {
        match($0, /[0-9]+$/); cur=substr($0, RSTART, RLENGTH)+0;
        capture=(cur==cid); next
      }
      capture {print}
    ' <<< "$proc_info")
    if grep -qE "Process id[[:space:]]*:[[:space:]]*[0-9]+" <<< "$chip_block"; then
      echo "ERROR: chip ${c} (NPU ${npu_id} chip ${chip_id}) is occupied." >&2
      echo "$chip_block" | grep -E "Process id|Process name|memory" | head >&2
      echo "Pick a different chip, or re-run with --skip-chip-check if the process is yours." >&2
      exit 3
    fi
  done
fi

# --- Device flag assembly -----------------------------------------------------
DEVICE_FLAGS=(
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
)
for c in "${CHIP_ARR[@]}"; do
  DEVICE_FLAGS+=(--device "/dev/davinci${c}")
done

# --- docker run --------------------------------------------------------------
# Driver bind strategy: bind `lib64` subdir + `version.info` + DCMI userspace
# separately, NOT the whole /usr/local/Ascend/driver tree.
#
# Binding the entire driver tree trips container-level NPU init on A3: dcmi
# initialization fails (`dcmi model initialized failed, because the device is
# used. ret is -8020` → `npu get board type failed. ret is -9005`). Symptom
# surfaces as dmesg `[ascend] [uda] [ERROR] uda_occupy_dev_by_ns Conflict open
# udevid`, which misdirects to an "Ascend UDA namespace leak" diagnosis
# (NPU-OPS-009). The real cause is missing `/usr/local/dcmi` bind and binding
# too much of the driver tree. First diagnosed 2026-04-21 by diffing against
# a working sibling container (roll_npu_new) — see NPU-OPS-009 + the
# porting-journal 2026-04-21 entry.
docker run --rm \
  "${DEVICE_FLAGS[@]}" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/bin/msnpureport:/usr/bin/msnpureport \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v "/home/${NPU_USER}:/home/${NPU_USER}" \
  -v "/data/${NPU_USER}:/data/${NPU_USER}" \
  -v "/tmp/${NPU_USER}:/tmp/${NPU_USER}" \
  -v "${LIVE_EASYR1}:/opt/easyr1" \
  --network=host --ipc=host --shm-size=64g \
  -e ASCEND_RT_VISIBLE_DEVICES="${IN_CONTAINER_CSV}" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e HF_HOME="${HF_HOME:-/data/${NPU_USER}/hf-cache}" \
  -e VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}" \
  -e PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}" \
  "${IMAGE}" \
  "$@"
