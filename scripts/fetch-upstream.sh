#!/usr/bin/env bash
# Clone (or git-fetch) the upstream repos used during porting work.
#
# When you need this:
#   - You're redoing the port from zero (follow SKILLS-GUIDE.md) — you'll want
#     the reference code locally to grep CUDA-only call sites, compare versions
#     across image targets, etc.
#   - You want to contribute code changes to EasyR1's ascend-port branch.
#
# When you DON'T need this:
#   - You just want to RUN EasyR1 on A3 (follow PORT-GUIDE.md). The docker image
#     is self-contained; nothing in upstream/ is needed at runtime.
#
# Usage:
#   ./scripts/fetch-upstream.sh                    # clone ESSENTIAL set only
#   ./scripts/fetch-upstream.sh --include-optional # clone ESSENTIAL + OPTIONAL
#   ./scripts/fetch-upstream.sh --target ../upstream   # custom target dir
#   ./scripts/fetch-upstream.sh --dry-run          # print what would happen
#
# Layout after running (default):
#   ../upstream/EasyR1/            (from zhshgmail fork — ESSENTIAL)
#   ../upstream/verl/              (GPU reference — OPTIONAL)
#   ../upstream/transformers/      (NPU FA shim reference — OPTIONAL)
#   ../upstream/torch-npu/         (torch_npu source — OPTIONAL)
#   ../upstream/vllm-ascend/       (vllm-ascend source — OPTIONAL)
#   ../upstream/triton-ascend/     (triton-ascend source — OPTIONAL)
#
# Idempotent: if a dir already exists with a .git, the script runs `git fetch`
# instead of re-cloning. Safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${REPO_ROOT}/../upstream"
INCLUDE_OPTIONAL=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --include-optional) INCLUDE_OPTIONAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# name:url:tier — tier is "essential" or "optional".
# Essential = you cannot redo the port without it (your EasyR1 fork).
# Optional  = useful for grepping reference code during port work; skip if tight on disk.
UPSTREAMS=(
  "EasyR1:https://github.com/zhshgmail/EasyR1.git:essential"
  "verl:https://github.com/verl-project/verl.git:optional"
  "transformers:https://github.com/huggingface/transformers.git:optional"
  "torch-npu:https://gitcode.com/Ascend/pytorch.git:optional"
  "vllm-ascend:https://github.com/vllm-project/vllm-ascend.git:optional"
  "triton-ascend:https://gitcode.com/Ascend/triton-ascend.git:optional"
)

# Resolve absolute target path
mkdir -p "${TARGET}"
TARGET="$(cd "${TARGET}" && pwd)"

echo "Target directory: ${TARGET}"
echo "Include optional repos: $([[ ${INCLUDE_OPTIONAL} -eq 1 ]] && echo yes || echo no)"
[[ ${DRY_RUN} -eq 1 ]] && echo "(dry-run mode — no actual clone/fetch)"
echo

for spec in "${UPSTREAMS[@]}"; do
  name="${spec%%:*}"
  rest="${spec#*:}"
  url="${rest%:*}"
  tier="${rest##*:}"

  if [[ "${tier}" == "optional" && "${INCLUDE_OPTIONAL}" -eq 0 ]]; then
    echo "skip (optional): ${name}"
    continue
  fi

  dest="${TARGET}/${name}"
  if [[ -d "${dest}/.git" ]]; then
    echo "fetch [${tier}]: ${name} (already cloned at ${dest})"
    if [[ ${DRY_RUN} -eq 0 ]]; then
      git -C "${dest}" fetch --all --prune || {
        echo "  WARN: fetch failed for ${name} (not fatal; continuing)" >&2
      }
    fi
  else
    echo "clone [${tier}]: ${name} <- ${url}"
    if [[ ${DRY_RUN} -eq 0 ]]; then
      git clone "${url}" "${dest}" || {
        echo "  ERROR: clone failed for ${name}" >&2
        if [[ "${tier}" == "essential" ]]; then
          echo "  (this is marked essential; cannot continue)" >&2
          exit 3
        else
          echo "  (optional; skipping)" >&2
        fi
      }
    fi
  fi
done

echo
echo "done."
echo
echo "Essential next steps for a port workflow:"
echo "  1. In upstream/EasyR1/, set up the personal remote + ascend-port branch:"
echo "       cd ${TARGET}/EasyR1"
echo "       git remote add personal <your-fork-url>   # if not using zhshgmail"
echo "       git checkout ascend-port"
echo "  2. Read docs/SKILLS-GUIDE.md step-by-step for the rest of the workflow."
