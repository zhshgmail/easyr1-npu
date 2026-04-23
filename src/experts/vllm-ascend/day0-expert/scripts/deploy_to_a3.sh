#!/usr/bin/env bash
# Deploy port code to A3 host and build image. Used by easyr1-port-worker
# Phase C and Phase D. Not for manual invocation typically.
#
# Usage:
#   deploy_to_a3.sh --branch <port-branch> --image-tag <image-tag> [options]
#
# Required (one of the two):
#   --image-tag <TAG>     docker image tag to BUILD (MUST be unique per session;
#                         script rejects easyr1-npu:ascend-port and :round2 etc.)
#   --reuse-image <TAG>   use pre-existing image on A3 provided by user
#                         (OL-04 exception: skip build + skip tag-uniqueness check;
#                         image must already exist on A3 or script errors)
#
# Required:
#   --branch <NAME>       git branch to deploy (e.g. ascend-port-round3)
#                         still required with --reuse-image: code is still synced
#                         to A3 for log/editable bind, only docker build is skipped.
#
# Optional:
#   --a3-host <HOST>           default: 115.190.166.102
#   --a3-port <PORT>           default: 443
#   --a3-user <USER>           default: root
#   --npu-user <USER>          default: z00637938 (for /data/$NPU_USER binds)
#   --base-image <REF>         default: quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
#   --dockerfile <NAME>        default: Dockerfile.npu. For transformers-upgrade sessions,
#                              pass Dockerfile.npu-<target-short> (e.g. Dockerfile.npu-852).
#   --upstream-consumer <DIR>  default: EasyR1. Name of the upstream/<DIR> consumer tree on A3.
#   --no-build                 skip docker build (just sync code; image must pre-exist)
#   --no-static-check          skip local static_check (NOT recommended)
#
# Workflow:
#   1. Verify local branch is committed (git status clean)
#   2. static_check.py on edited files (unless --no-static-check)
#   3. git bundle or scp code tar to A3 /tmp/{session}
#   4. extract on A3 → ~/workspace/easyr1-port-{session}/EasyR1/
#   5. docker build -t {image-tag} on A3 (unless --no-build)
#   6. Emit session marker file for downstream smoke_validate.sh
#
# Exit codes:
#   0  success
#   1  static_check failed (OL-01)
#   2  usage error
#   3  git state dirty (uncommitted changes)
#   4  image tag reused (OL-04)
#   5  ssh / scp / docker failed
#   10 infra (network, disk full, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- defaults ---
A3_HOST="${A3_HOST:-115.190.166.102}"
A3_PORT="${A3_PORT:-443}"
A3_USER="${A3_USER:-root}"
NPU_USER="${NPU_USER:-z00637938}"
BASE_IMAGE="${BASE_IMAGE:-quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest}"
BRANCH=""
IMAGE_TAG=""
DOCKERFILE="Dockerfile.npu"          # default; upgrade expert usually passes Dockerfile.npu-<target-short>
UPSTREAM_CONSUMER="${UPSTREAM_CONSUMER:-EasyR1}"
DO_BUILD=1
DO_STATIC_CHECK=1
REUSE_IMAGE=0   # OL-04 exception: --reuse-image = user-provided, skip build + skip tag uniqueness

# --- parse ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)             BRANCH="$2"; shift 2 ;;
    --image-tag)          IMAGE_TAG="$2"; shift 2 ;;
    --reuse-image)        IMAGE_TAG="$2"; REUSE_IMAGE=1; DO_BUILD=0; shift 2 ;;
    --dockerfile)         DOCKERFILE="$2"; shift 2 ;;
    --upstream-consumer)  UPSTREAM_CONSUMER="$2"; shift 2 ;;
    --a3-host)            A3_HOST="$2"; shift 2 ;;
    --a3-port)            A3_PORT="$2"; shift 2 ;;
    --a3-user)            A3_USER="$2"; shift 2 ;;
    --npu-user)           NPU_USER="$2"; shift 2 ;;
    --base-image)         BASE_IMAGE="$2"; shift 2 ;;
    --no-build)           DO_BUILD=0; shift ;;
    --no-static-check)    DO_STATIC_CHECK=0; shift ;;
    -h|--help)            grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)                    echo "ERROR: unknown arg $1" >&2; exit 2 ;;
  esac
done

[[ -z "$BRANCH" ]]    && { echo "ERROR: --branch required" >&2; exit 2; }
[[ -z "$IMAGE_TAG" ]] && { echo "ERROR: --image-tag (or --reuse-image) required" >&2; exit 2; }

# --- OL-04: unique image tag (skipped when --reuse-image used) ---
# transformers-upgrade-expert's denylist: past upgrade drill / round builds
if [[ $REUSE_IMAGE -eq 0 ]]; then
  case "$IMAGE_TAG" in
    easyr1-npu:ascend-port|easyr1-npu:ascend-port-e2e|easyr1-npu:round2|easyr1-npu-852:drill|easyr1-npu-852:drill-reproduce|easyr1-npu:round3-20260422-0514|easyr1-npu:round4-20260422-0702)
      echo "ERROR: image tag '$IMAGE_TAG' is reused from past sessions. OL-04 requires unique session-specific tag." >&2
      echo "       Pass --reuse-image <TAG> instead if user has explicitly provided this image." >&2
      exit 4
      ;;
  esac
fi

SSH_CMD="ssh -p $A3_PORT $A3_USER@$A3_HOST"

echo "=== deploy_to_a3 ==="
echo "  branch:      $BRANCH"
echo "  image-tag:   $IMAGE_TAG"
echo "  a3-host:     $A3_USER@$A3_HOST:$A3_PORT"
echo "  npu-user:    $NPU_USER"
echo "  base-image:  $BASE_IMAGE"
echo

# --- 1. verify git state clean ---
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
  echo "ERROR: git working tree has uncommitted changes. Commit first." >&2
  echo "$(git status --short)" >&2
  exit 3
fi

CURR_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
if [[ "$CURR_BRANCH" != "$BRANCH" ]]; then
  echo "WARNING: current branch is '$CURR_BRANCH', expected '$BRANCH'. Checking out..." >&2
  git checkout "$BRANCH" 2>&1
fi

# --- 2. static_check (OL-01) ---
if [[ $DO_STATIC_CHECK -eq 1 ]]; then
  echo "--- static_check ---"
  EDITED_FILES=$(git diff --name-only HEAD~$(git rev-list --count main..HEAD) HEAD | grep '\.py$' | head -50 || true)
  if [[ -n "$EDITED_FILES" ]]; then
    python3 "$EXPERT_ROOT/scripts/static_check.py" --files $EDITED_FILES --import-package verl || {
      echo "ERROR: static_check failed. OL-01 violation." >&2
      exit 1
    }
  fi
  echo
fi

# --- 3. sync code via git bundle (A3 can't github due to GFW) ---
echo "--- bundling + scp ---"
BUNDLE_PATH="/tmp/easyr1-port-deploy.bundle"
git bundle create "$BUNDLE_PATH" "$BRANCH" ^main 2>&1 | tail -3
scp -P "$A3_PORT" "$BUNDLE_PATH" "$A3_USER@$A3_HOST:/tmp/round-deploy.bundle" 2>&1 | tail -3 || exit 5

# --- 4. fetch into A3 checkout ---
# Detach HEAD first so `git fetch bundle $BRANCH:$BRANCH` doesn't refuse when
# $BRANCH is the currently checked-out branch (previous round's leftover state).
echo "--- A3 git fetch from bundle ---"
A3_CONSUMER_DIR="/home/$NPU_USER/workspace/easyr1-npu/upstream/$UPSTREAM_CONSUMER"
$SSH_CMD "cd $A3_CONSUMER_DIR && \
  git checkout --detach 2>&1 | tail -1 && \
  git fetch --force /tmp/round-deploy.bundle $BRANCH:$BRANCH 2>&1 | tail -3 && \
  git checkout $BRANCH 2>&1 | tail -3" || exit 5

# --- 5. docker build (or verify pre-existing when --reuse-image) ---
if [[ $DO_BUILD -eq 1 ]]; then
  echo "--- docker build (using $DOCKERFILE) ---"
  $SSH_CMD "cd $A3_CONSUMER_DIR; docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t $IMAGE_TAG -f $DOCKERFILE . 2>&1 | tail -20" || exit 5
  IMAGE_ID=$($SSH_CMD "docker images $IMAGE_TAG --format '{{.ID}}'" 2>/dev/null | head -1)
  if [[ -z "$IMAGE_ID" ]]; then
    echo "ERROR: image $IMAGE_TAG not found after build" >&2
    exit 5
  fi
  echo "  built: $IMAGE_ID"
else
  # Either --no-build or --reuse-image: image must already exist
  IMAGE_ID=$($SSH_CMD "docker images $IMAGE_TAG --format '{{.ID}}'" 2>/dev/null | head -1)
  if [[ -z "$IMAGE_ID" ]]; then
    if [[ $REUSE_IMAGE -eq 1 ]]; then
      echo "ERROR: --reuse-image $IMAGE_TAG: image not found on A3. Either user did not pre-provide it, or the tag is wrong." >&2
    else
      echo "ERROR: --no-build but image $IMAGE_TAG not found on A3." >&2
    fi
    exit 5
  fi
  if [[ $REUSE_IMAGE -eq 1 ]]; then
    echo "  reused user-provided image: $IMAGE_ID  (skipping build)"
  else
    echo "  pre-existing image: $IMAGE_ID  (skipping build per --no-build)"
  fi
fi

echo "✅ deploy_to_a3 done"
exit 0
