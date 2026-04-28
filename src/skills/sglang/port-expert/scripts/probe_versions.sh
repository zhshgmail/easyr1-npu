#!/usr/bin/env bash
# Probe SGLang NPU 3-axis version triple for compatibility.
#
# Usage:
#   probe_versions.sh --sglang-tag <tag> \
#                     --kernel-npu-tag <tag> \
#                     --cann-version <ver> \
#                     [--device-type a3|a2]
#
# Exit codes:
#   0 = all 3 axes have published artifacts at the requested versions
#   1 = at least one artifact missing (see stderr)
#   2 = usage / harness error
set -euo pipefail

SGLANG_TAG=""
KERNEL_NPU_TAG=""
CANN_VERSION=""
DEVICE_TYPE="a3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sglang-tag)     SGLANG_TAG="$2"; shift 2 ;;
    --kernel-npu-tag) KERNEL_NPU_TAG="$2"; shift 2 ;;
    --cann-version)   CANN_VERSION="$2"; shift 2 ;;
    --device-type)    DEVICE_TYPE="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SGLANG_TAG" ]]     || { echo "--sglang-tag required" >&2; exit 2; }
[[ -n "$KERNEL_NPU_TAG" ]] || { echo "--kernel-npu-tag required" >&2; exit 2; }
[[ -n "$CANN_VERSION" ]]   || { echo "--cann-version required" >&2; exit 2; }
[[ "$DEVICE_TYPE" =~ ^(a3|a2|910b)$ ]] || { echo "--device-type must be a3 / a2 / 910b" >&2; exit 2; }

FAIL=0

echo "=== Axis 1: sglang main / tag ==="
if curl -fsI "https://api.github.com/repos/sgl-project/sglang/git/refs/tags/${SGLANG_TAG}" >/dev/null 2>&1 \
   || [[ "$SGLANG_TAG" == "main" ]]; then
  echo "  ✓ sglang ${SGLANG_TAG} reachable"
else
  echo "  ✗ sglang tag ${SGLANG_TAG} NOT FOUND on github" >&2
  FAIL=1
fi

echo
echo "=== Axis 2: sgl-kernel-npu release zip ==="
ZIP_URL="https://github.com/sgl-project/sgl-kernel-npu/releases/download/${KERNEL_NPU_TAG}/sgl-kernel-npu-${KERNEL_NPU_TAG}-torch2.8.0-py311-cann${CANN_VERSION}-${DEVICE_TYPE}-x86_64.zip"
if curl -fsI "$ZIP_URL" >/dev/null 2>&1; then
  echo "  ✓ sgl-kernel-npu ${KERNEL_NPU_TAG} (cann${CANN_VERSION}, ${DEVICE_TYPE}, x86_64) zip exists"
else
  echo "  ✗ sgl-kernel-npu zip NOT FOUND: $ZIP_URL" >&2
  FAIL=1
fi

echo
echo "=== Axis 3: official docker image ==="
IMG="quay.io/ascend/sglang:${SGLANG_TAG}-cann${CANN_VERSION}-${DEVICE_TYPE}"
if docker manifest inspect "$IMG" >/dev/null 2>&1; then
  echo "  ✓ ${IMG} manifest reachable"
else
  IMG_NJU="quay.nju.edu.cn/ascend/sglang:${SGLANG_TAG}-cann${CANN_VERSION}-${DEVICE_TYPE}"
  if docker manifest inspect "$IMG_NJU" >/dev/null 2>&1; then
    echo "  ✓ ${IMG_NJU} (NJU mirror) manifest reachable"
  else
    echo "  ✗ image NOT PUBLISHED at quay.io or NJU mirror" >&2
    echo "    tried: ${IMG}" >&2
    echo "    tried: ${IMG_NJU}" >&2
    FAIL=1
  fi
fi

echo

if [[ "$FAIL" -eq 0 ]]; then
  echo "All 3 axes have published artifacts. Proceed to P1 (pull + import smoke)."
  exit 0
else
  echo "One or more axes missing artifacts — abort or retarget." >&2
  echo "See https://github.com/sgl-project/sgl-kernel-npu/releases for current sgl-kernel-npu RCs." >&2
  echo "See https://github.com/sgl-project/sglang/releases for current sglang tags." >&2
  exit 1
fi
