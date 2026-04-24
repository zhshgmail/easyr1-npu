#!/usr/bin/env bash
# Deploy (or undeploy) repo/src/skills/**/SKILL.md to ~/.claude/skills/ so
# Claude Code can invoke them. Symlink-based, idempotent, safe to re-run.
#
# Usage:
#   ./src/scripts/install-skills.sh [--force] [--skills-dir <path>]
#   ./src/scripts/install-skills.sh --undeploy
#
# Defaults:
#   --skills-dir = ${HOME}/.claude/skills
#
# Layout assumption: this script lives at <repo>/src/scripts/install-skills.sh.
# Skills live at <repo>/src/skills/**/SKILL.md (any depth). Only dirs
# containing a SKILL.md file become slash-commands.
#
# Flat naming: Claude Code requires ~/.claude/skills/<name>/SKILL.md so
# nested repo layout is flattened by parent-dir basename:
#   src/skills/_shared/codex-review/SKILL.md     → ~/.claude/skills/codex-review/
#   src/skills/orchestrators/npu-port/SKILL.md   → ~/.claude/skills/npu-port/
#   src/skills/vllm-ascend/port-expert/agent.md  (no SKILL.md → not installed)
#
# port-expert/ dirs hold agent.md briefs (not slash commands) — they are
# invoked by orchestrators via filesystem path, not via Claude Code's skill
# registry, so they don't need installation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SKILLS_SRC="${REPO_ROOT}/src/skills"
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
      rm "${path}"
      echo "removed: ${path}"
    fi
  done < "${MANIFEST}"
  rm -f "${MANIFEST}"
  echo "done."
  exit 0
fi

mkdir -p "${SKILLS_DIR}"
: > "${MANIFEST}"

count=0
while IFS= read -r -d '' skill_md; do
  skill_dir="$(dirname "${skill_md}")"
  # Parse `name:` from YAML frontmatter of SKILL.md (top of file, between --- ---).
  # Fall back to parent-dir basename if name: not found.
  name="$(awk '/^---$/{f=!f; next} f && /^name:/{sub(/^name:[[:space:]]*/, ""); print; exit}' "${skill_md}")"
  if [[ -z "${name}" ]]; then
    name="$(basename "${skill_dir}")"
    echo "WARN: no 'name:' in ${skill_md}; falling back to '${name}'"
  fi
  target="${SKILLS_DIR}/${name}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ "${FORCE}" -eq 1 ]]; then
      rm -rf "${target}"
    else
      echo "exists (pass --force to overwrite): ${target}"
      continue
    fi
  fi
  ln -s "${skill_dir}" "${target}"
  echo "linked: ${target} -> ${skill_dir}"
  echo "${target}" >> "${MANIFEST}"
  count=$((count + 1))
done < <(find "${SKILLS_SRC}" -name SKILL.md -print0)

echo
echo "deployed ${count} skill(s) to ${SKILLS_DIR}"
echo "manifest: ${MANIFEST}"
echo
echo "verify with: ls ${SKILLS_DIR}"
echo "undeploy:    ${BASH_SOURCE[0]} --undeploy"
