#!/bin/bash
# Scan a Python source tree for GPU-only / CUDA-specific call sites that
# will break on Ascend NPU. Emits repo/docs/code-path-sweep-<name>.md.
#
# See repo/skills/npu-code-path-sweep/SKILL.md for scope, contract, pattern
# list, gotchas.
#
# Usage:
#   code-path-sweep.sh <source-tree> [--out path.md] [--name <label>]

set -euo pipefail

SRC=""
OUT=""
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -*) echo "unknown arg: $1" >&2; exit 2 ;;
    *) SRC="$1"; shift ;;
  esac
done
[[ -z "${SRC}" ]] && { echo "usage: $0 <source-tree> [--out path.md] [--name <label>]" >&2; exit 2; }
[[ ! -d "${SRC}" ]] && { echo "source tree not found: ${SRC}" >&2; exit 3; }

NAME="${NAME:-$(basename "${SRC}")}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-${REPO_ROOT}/docs/code-path-sweep-${NAME}.md}"
mkdir -p "$(dirname "${OUT}")"

# Pattern set. To add a pattern: append a tuple "id|title|regex|suggest".
declare -a PATTERNS=(
  'NPU-CP-001a|torch.cuda.* calls|torch\.cuda\.[a-zA-Z_]|get_device_module().<op>() — use verl.utils.device'
  'NPU-CP-001b|tensor .cuda() method|\.cuda\(|.to(torch.device(get_device_name(), idx))'
  'NPU-CP-001c|string "cuda" device spec|device_map=.?"cuda"|init_device_mesh\(.?"cuda"|device=.?"cuda"|torch\.device\(.?"cuda"|get_device_name() returns "cuda" or "npu"'
  'NPU-CP-001d|CUDA_VISIBLE_DEVICES literal|CUDA_VISIBLE_DEVICES|get_visible_devices_env() — "CUDA_VISIBLE_DEVICES" or "ASCEND_RT_VISIBLE_DEVICES"'
  'NPU-CP-001e|nccl distributed backend|backend=.?"nccl"|get_dist_backend() — "nccl" or "hccl"'
  'NPU-CP-002|vllm.lora.models import (pre-0.13)|from vllm\.lora\.models|try: from vllm.lora.lora_model import LoRAModel; except ImportError: from vllm.lora.models import LoRAModel'
  'NPU-CP-003a|Ray "GPU" resource lookup|ray\.available_resources\(\).*GPU|{.*GPU.*:|use get_ray_resource_name() - "GPU" or "NPU"'
  'NPU-CP-003b|Ray actor num_gpus option|num_gpus\s*=|on NPU use options["resources"]={"NPU": n} instead of num_gpus'
  'NPU-CP-004|vllm get_tensor_model_parallel_group|get_tensor_model_parallel_group|vllm 0.13 renamed to get_tp_group(); hasattr gate'
  'GPU-ONLY|flash_attn import|from flash_attn|import flash_attn|CUDA-only; guard behind try/except or is_npu_available() — see attention_utils.py'
  'GPU-ONLY|liger_kernel import|from liger_kernel|import liger_kernel|CUDA-only Triton; drop on NPU or port later via triton-ascend'
  'CFG|torch.backends.cuda knobs|torch\.backends\.cuda\.|wrap behind if not is_npu_available():'
)

now="$(date -I)"
py_files=$(find "${SRC}" -type f -name '*.py' ! -path '*/__pycache__/*' | wc -l)

total_hits=0

{
  echo "# Code-path sweep for ${NAME}"
  echo
  echo "Generated: ${now}"
  echo "Source tree: \`${SRC}\`"
  echo "Python files scanned: ${py_files}"
  echo
  echo "Total hits: __TOTAL_HITS__"
  echo
  echo "---"
  echo

  for entry in "${PATTERNS[@]}"; do
    IFS='|' read -r id title regex suggest <<< "${entry}"
    echo "## ${id} — ${title}"
    echo
    hits=$(grep -rnHE --include='*.py' --exclude-dir=__pycache__ "${regex}" "${SRC}" 2>/dev/null || true)
    if [[ -z "${hits}" ]]; then
      echo "No hits."
      echo
      continue
    fi
    echo "| File | Line | Source | Suggested replacement |"
    echo "|---|---|---|---|"
    count=0
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      count=$((count + 1))
      path="${line%%:*}"
      rest="${line#*:}"
      lineno="${rest%%:*}"
      source_text="${rest#*:}"
      source_text="$(echo "${source_text}" | sed -E 's/^[[:space:]]+//')"
      rel_path="${path#${SRC}/}"
      safe_src="${source_text//|/\\|}"
      if [[ ${#safe_src} -gt 100 ]]; then
        safe_src="${safe_src:0:97}..."
      fi
      echo "| \`${rel_path}\` | ${lineno} | \`${safe_src}\` | ${suggest} |"
    done <<< "${hits}"
    echo
    echo "Sub-total: ${count}"
    echo
    total_hits=$((total_hits + count))
  done

  echo "---"
  echo
  echo "## Summary"
  echo
  echo "- Python files scanned: ${py_files}"
  echo "- Total hits: ${total_hits}"
  echo
  echo "See \`repo/knowledge/npu-patterns.md\` for the fix pattern behind each ID."
} > "${OUT}.tmp"

sed -i "s/__TOTAL_HITS__/${total_hits}/" "${OUT}.tmp"
mv "${OUT}.tmp" "${OUT}"

echo "wrote ${OUT} (${total_hits} hits across ${py_files} files)"
