#!/usr/bin/env bash
# Deploy (or undeploy) repo/skills/* to ~/.claude/skills/ so Claude Code can
# invoke them. Symlink-based, idempotent, safe to re-run.
#
# Usage:
#   ./scripts/install-skills.sh [--force] [--skills-dir <path>]
#   ./scripts/install-skills.sh --undeploy
#
# Defaults:
#   --skills-dir = ${HOME}/.claude/skills
#
# Layout assumption: this script lives at <repo>/scripts/install-skills.sh and
# skills live at <repo>/skills/*/SKILL.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILLS_SRC="${REPO_ROOT}/skills"
SKILLS_DIR="${HOME}/.claude/skills"
FORCE=0
UNDEPLOY=0
MANIFEST="${REPO_ROOT}/.install_manifest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --undeploy) UNDEPLOY=1; shift ;;
    --skills-dir) SKILLS_DIR="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "${SKILLS_SRC}" ]]; then
  echo "ERROR: skills dir not found: ${SKILLS_SRC}" >&2
  exit 3
fi

if [[ "${UNDEPLOY}" -eq 1 ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "no manifest at ${MANIFEST}; nothing to undeploy"
    exit 0
  fi
  echo "=== undeploy ==="
  while IFS= read -r path; do
    if [[ -L "${path}" ]]; then
      rm -f "${path}"
      echo "removed: ${path}"
    elif [[ -e "${path}" ]]; then
      echo "skip (not a symlink, won't delete): ${path}"
    fi
  done < "${MANIFEST}"
  rm -f "${MANIFEST}"
  echo "done."
  exit 0
fi

mkdir -p "${SKILLS_DIR}"
: > "${MANIFEST}"

count=0
for skill in "${SKILLS_SRC}"/*/; do
  [[ -f "${skill}SKILL.md" ]] || { echo "skip (no SKILL.md): ${skill}"; continue; }
  name="$(basename "${skill%/}")"
  target="${SKILLS_DIR}/${name}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ "${FORCE}" -eq 1 ]]; then
      rm -rf "${target}"
    else
      echo "exists (pass --force to overwrite): ${target}"
      continue
    fi
  fi
  ln -s "${skill%/}" "${target}"
  echo "linked: ${target} -> ${skill%/}"
  echo "${target}" >> "${MANIFEST}"
  count=$((count + 1))
done

echo
echo "deployed ${count} skill(s) to ${SKILLS_DIR}"
echo "manifest: ${MANIFEST}"
echo
echo "verify with: claude --help   # or inside Claude Code, check available skills"
echo "undeploy:    ${BASH_SOURCE[0]} --undeploy"
