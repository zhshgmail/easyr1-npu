#!/usr/bin/env bash
# Session cleanup for easyr1-port-worker. Called by Stop hook or manually
# at the end of a run. Removes only artifacts whose name includes the
# session tag — leaves other sessions' / user-provided images alone.
#
# OL-04b enforcement: worker MUST run this before exiting, and the result
# ("Cleanup: clean|partial|skipped <reason>") goes into PROGRESS.md.
#
# Usage:
#   cleanup_session.sh --session-tag <TAG> [options]
#
# Required:
#   --session-tag <TAG>   e.g. round3-20260422-0514 (MUST match image/container
#                         naming convention used by deploy_to_a3.sh and
#                         run-npu-container.sh)
#
# Optional:
#   --a3-host/port/user   (defaults as in deploy_to_a3.sh)
#   --preserve-image      keep the easyr1-npu:{tag} image, only clean up
#                         containers + bundle (use when smoke PASSED and
#                         user wants to re-run manually for inspection)
#   --dry-run             print what would be deleted, don't delete
#   --keep-user-provided  never delete images matching --reuse-image tag
#                         (safety: pass the tag to whitelist)
#
# Exit codes:
#   0   clean (removed everything expected)
#   1   partial (some items failed to delete; details logged)
#   2   usage error
#   5   ssh failed

set -uo pipefail

A3_HOST="${A3_HOST:-115.190.166.102}"
A3_PORT="${A3_PORT:-443}"
A3_USER="${A3_USER:-root}"
SESSION_TAG=""
PRESERVE_IMAGE=0
DRY_RUN=0
KEEP_USER_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-tag)        SESSION_TAG="$2"; shift 2 ;;
    --a3-host)            A3_HOST="$2"; shift 2 ;;
    --a3-port)            A3_PORT="$2"; shift 2 ;;
    --a3-user)            A3_USER="$2"; shift 2 ;;
    --preserve-image)     PRESERVE_IMAGE=1; shift ;;
    --dry-run)            DRY_RUN=1; shift ;;
    --keep-user-provided) KEEP_USER_TAG="$2"; shift 2 ;;
    -h|--help)            grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)                    echo "ERROR: unknown arg $1" >&2; exit 2 ;;
  esac
done

[[ -z "$SESSION_TAG" ]] && { echo "ERROR: --session-tag required" >&2; exit 2; }

SSH_CMD="ssh -p $A3_PORT $A3_USER@$A3_HOST"

echo "=== cleanup_session ==="
echo "  session-tag:       $SESSION_TAG"
echo "  preserve-image:    $PRESERVE_IMAGE"
echo "  dry-run:           $DRY_RUN"
[[ -n "$KEEP_USER_TAG" ]] && echo "  keep-user-provided: $KEEP_USER_TAG"
echo

EXIT=0

# --- 1. containers: name contains session tag ---
echo "--- containers ---"
CONTAINERS=$($SSH_CMD "docker ps -a --format '{{.Names}}' 2>/dev/null" | grep -F "$SESSION_TAG" || true)
if [[ -n "$CONTAINERS" ]]; then
  echo "$CONTAINERS" | while read -r c; do
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "  DRY: docker rm -f $c"
    else
      if $SSH_CMD "docker rm -f $c" >/dev/null 2>&1; then
        echo "  removed container: $c"
      else
        echo "  FAILED to remove container: $c" >&2; EXIT=1
      fi
    fi
  done
else
  echo "  (no containers matched)"
fi

# --- 2. images: repository:tag starts with easyr1-npu:{SESSION_TAG} ---
if [[ $PRESERVE_IMAGE -eq 0 ]]; then
  echo "--- images ---"
  IMAGES=$($SSH_CMD "docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null" \
            | grep -E "^easyr1-npu:${SESSION_TAG}(-iter[0-9]+)?$" || true)
  # Skip user-provided tag if whitelisted
  if [[ -n "$KEEP_USER_TAG" ]]; then
    IMAGES=$(echo "$IMAGES" | grep -vF "$KEEP_USER_TAG" || true)
  fi
  if [[ -n "$IMAGES" ]]; then
    echo "$IMAGES" | while read -r img; do
      if [[ $DRY_RUN -eq 1 ]]; then
        echo "  DRY: docker rmi -f $img"
      else
        if $SSH_CMD "docker rmi -f $img" >/dev/null 2>&1; then
          echo "  removed image: $img"
        else
          echo "  FAILED to remove image: $img" >&2; EXIT=1
        fi
      fi
    done
  else
    echo "  (no images matched)"
  fi
else
  echo "--- images (preserved per --preserve-image) ---"
fi

# --- 3. transient bundle file ---
echo "--- tmp bundle ---"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "  DRY: rm -f /tmp/round-deploy.bundle"
else
  $SSH_CMD "rm -f /tmp/round-deploy.bundle" >/dev/null 2>&1 && echo "  removed /tmp/round-deploy.bundle" || true
fi

echo
if [[ $EXIT -eq 0 ]]; then
  echo "✅ cleanup complete"
else
  echo "⚠  cleanup partial — see failures above"
fi
exit $EXIT
