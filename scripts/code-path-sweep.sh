#!/bin/bash
# Scan a Python source tree for GPU-only / CUDA-specific call sites that
# will break on Ascend NPU. Emits repo/docs/code-path-sweep-<name>.md.
#
# See repo/skills/npu-code-path-sweep/SKILL.md for scope, contract, pattern
# list, gotchas.
#
# Usage:
#   code-path-sweep.sh <source-tree> [--out path.md] [--name <label>]
#
# Pattern table uses three parallel arrays indexed by position. No field
# separator to escape, so regexes can contain arbitrary metacharacters.
# IDs match the canonical catalog in repo/knowledge/npu-patterns.md —
# sub-facets share the same ID and disambiguate via the title only.

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

# Parallel arrays. To add a pattern, append to all three (same index in each).
# IDs must exist in repo/knowledge/npu-patterns.md; title disambiguates sub-facets.
declare -a IDS=(
  "NPU-CP-001"
  "NPU-CP-001"
  "NPU-CP-001"
  "NPU-CP-001"
  "NPU-CP-001"
  "NPU-CP-002"
  "NPU-CP-003"
  "NPU-CP-003"
  "NPU-CP-004"
  "NPU-CP-005"
  "NPU-CP-005"
  "NPU-CP-006"
)
declare -a TITLES=(
  'torch.cuda.* namespace call'
  'tensor .cuda() method'
  'string "cuda" device spec'
  'CUDA_VISIBLE_DEVICES literal'
  'nccl distributed backend literal'
  'vllm.lora.models import (pre-0.13 only)'
  'Ray available_resources GPU lookup or bundle'
  'Ray actor num_gpus option'
  'vllm get_tensor_model_parallel_group (renamed in 0.13)'
  'flash_attn import (CUDA-only kernel)'
  'liger_kernel import (CUDA-only Triton)'
  'torch.backends.cuda.* knob (CUDA-only)'
)
declare -a REGEXES=(
  'torch\.cuda\.[a-zA-Z_]'
  '\.cuda\('
  'device_map=.?"cuda"|init_device_mesh\(.?"cuda"|device=.?"cuda"|torch\.device\(.?"cuda"'
  'CUDA_VISIBLE_DEVICES'
  'backend=.?"nccl"'
  'from vllm\.lora\.models'
  'ray\.available_resources\(\).*GPU|\{.*"GPU".*:'
  'num_gpus\s*='
  'get_tensor_model_parallel_group'
  'from flash_attn|import flash_attn'
  'from liger_kernel|import liger_kernel'
  'torch\.backends\.cuda\.'
)
declare -a SUGGESTS=(
  'get_device_module().<op>()'
  '.to(torch.device(get_device_name(), idx))'
  'get_device_name() returns "cuda" or "npu"'
  'get_visible_devices_env() — "CUDA_VISIBLE_DEVICES" or "ASCEND_RT_VISIBLE_DEVICES"'
  'get_dist_backend() — "nccl" or "hccl"'
  'try: from vllm.lora.lora_model import LoRAModel; except ImportError: from vllm.lora.models import LoRAModel'
  'use get_ray_resource_name() for lookups; placement bundles via placement_bundle helper'
  'on NPU use options["resources"]={"NPU": n} via apply_actor_options (ray-npu-shim)'
  'vllm 0.13 renamed to get_tp_group(); hasattr gate'
  'CUDA-only; import-guard behind try/except or is_npu_available(); see attention_utils.py'
  'CUDA-only Triton; drop on NPU or port later via triton-ascend'
  'wrap behind `if not is_npu_available():`'
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
  echo "Pattern IDs link to the canonical catalog at \`repo/knowledge/npu-patterns.md\`."
  echo
  echo "---"
  echo

  for i in "${!IDS[@]}"; do
    id="${IDS[$i]}"
    title="${TITLES[$i]}"
    regex="${REGEXES[$i]}"
    suggest="${SUGGESTS[$i]}"

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
      # Escape pipes in source text (markdown table separator).
      safe_src="${source_text//|/\\|}"
      if [[ ${#safe_src} -gt 100 ]]; then
        safe_src="${safe_src:0:97}..."
      fi
      # Escape pipes in suggest (defensive — our suggestions shouldn't contain |).
      safe_suggest="${suggest//|/\\|}"
      echo "| \`${rel_path}\` | ${lineno} | \`${safe_src}\` | ${safe_suggest} |"
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
