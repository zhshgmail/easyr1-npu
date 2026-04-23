#!/usr/bin/env bash
# Automated dependency-gap detection for the scenario P1 auto-judgment.
#
# Given a requirements.txt (EasyR1 or any Ray-based RL framework) and a
# target NPU image pip-freeze, classify each declared dep into one of:
#   A — NPU native (upstream supports NPU; use as-is)
#   B — NPU-ported version exists (e.g. vllm -> vllm-ascend)
#   C — CUDA-only but bypassable (shim in the framework; no blocker)
#   D — CUDA-only BLOCKER (needs new NPU adaptation work -> scenario P2)
#   E — Pure Python / CPU (accelerator-agnostic)
#
# Exit 0 if all deps are A/B/C/E (P1 scenario — no new NPU work needed).
# Exit 1 if any D found (P2 scenario — STOP and file a task in
#   docs/easyr1/npu-adaptation-tasks.md).
# Exit 2 on usage / input errors.
#
# Usage:
#   ./dep-gap-detect.sh \
#       --reqs <path/to/requirements.txt> \
#       --image-inventory <path/to/knowledge/images/<slug>.md> \
#       [--out <output.md>]
#
# The image-inventory must be a markdown file emitted by
# scripts/inspect-ascend-image.sh (has a "## Full pip freeze" section).
#
# Classification rules come from two tables embedded below:
#   - PACKAGE_RULES: per-package override (e.g. "vllm=B:vllm-ascend",
#     "flash-attn=C:shim via transformers.integrations.npu_flash_attention")
#   - fallback: if a package name appears in image pip-freeze -> E
#     (presumed accelerator-agnostic); if missing and not in rules -> D.
#
# The PACKAGE_RULES reflect knowledge about the NPU ecosystem as of 2026-04.
# Update as new packages are evaluated.

set -euo pipefail

REQS=""
IMG=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reqs) REQS="$2"; shift 2 ;;
    --image-inventory) IMG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$REQS" ]] && { echo "missing --reqs" >&2; exit 2; }
[[ -z "$IMG" ]]  && { echo "missing --image-inventory" >&2; exit 2; }
[[ ! -f "$REQS" ]] && { echo "reqs not found: $REQS" >&2; exit 2; }
[[ ! -f "$IMG" ]]  && { echo "image inventory not found: $IMG" >&2; exit 2; }

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
OUT="${OUT:-/tmp/dep-gap-$(basename "$REQS" .txt)-$(date +%s).md}"

# -----------------------------------------------------------------------------
# PACKAGE_RULES — the knowledge base for the NPU ecosystem
#
# Format: one line per package, "name=TIER:comment"
# TIER: A / B / C / D / E
# B entries: comment names the NPU-ported replacement
# C entries: comment names the bypass mechanism
# D entries: comment names what's missing
#
# Rules win over fallback "in-image-then-E" heuristic. If a dep is in
# PACKAGE_RULES, that tier is used directly.
# -----------------------------------------------------------------------------
RULES=$(cat <<'RULES'
# --- B: upstream NPU ports (image installs <name>-ascend instead) ---
vllm=B:Consumed via vllm-ascend in the image. 2 API-rename shims in EasyR1 (NPU-CP-002, NPU-CP-004) already applied.
# --- C: CUDA-only packages EasyR1 bypasses with shim ---
flash-attn=C:Replaced by transformers.integrations.npu_flash_attention (NPU-CP-007). bert_padding helpers rewritten in pure torch inside EasyR1 (verl/utils/npu_flash_attn_utils.py).
liger-kernel=C:Dropped on NPU (moved to [gpu] extras in ascend-port's requirements.txt, commit 7ee0f0b). Triton GPU kernels, not needed for EasyR1 core RL path.
# --- A: upstream framework with NPU support ---
ray=A:Ray 2.55+ supports NPU as custom resource. Needs ray-npu-shim skill for NPU-CP-003 + NPU-BUG-002 + NPU-ENV-002.
transformers=A:transformers 4.54+ has integrations/npu_flash_attention.py for NPU FA. No code patch needed.
peft=A:Rides on transformers / torch_npu; no NPU-specific issues.
tensordict=A:torch extension; uses torch_npu via PrivateUse1 backend.
torchdata=A:torch extension; same as tensordict.
accelerate=A:Has NPU backend branches; no NPU-specific issues on the RL path we exercise.
# --- E (explicit, to avoid D misfires for known pure-Python deps) ---
jinja2=E:Pure Python templating.
psutil=E:C extension, no NPU path.
pyyaml=E:Pure Python YAML parser.
numpy=E:CPU library. NPU ecosystem pins numpy<2.
pandas=E:CPU-side dataframe.
datasets=E:HuggingFace datasets, CPU/disk.
codetiming=E:Pure Python timing.
mathruler=E:EasyR1 math reward util, pure Python.
omegaconf=E:Config management.
pillow=E:Image processing, no NPU.
pyarrow=E:Columnar storage, CPU.
pylatexenc=E:LaTeX processing, pure Python.
qwen-vl-utils=E:Qwen VL helper, pure Python.
wandb=E:Logging tool.
RULES
)

# -----------------------------------------------------------------------------
# Parse the image inventory's "Full pip freeze" section.
# -----------------------------------------------------------------------------
# Collect installed package names from the inventory markdown.
# Match lines like "| pkg | version |" (any table) or "pkg==version".
# Normalize: lowercase, _ -> -, strip ** markdown bold.
all_img_pkgs=$(
  sed -n '/^## /p; /^|/p; /^[a-zA-Z0-9_.-]\{1,\}==/p' "$IMG" \
  | awk '
      /^## Full pip freeze/ { any_sec=1; next }
      /^## (Core ML stack|EasyR1-relevant|EasyR1-required packages)/ { any_sec=1; next }
      /^## / { any_sec=0 }
      any_sec && /^\|/ {
        # Strip leading |, split on |, take field 1
        line = $0
        sub(/^\|[[:space:]]*/, "", line)
        sub(/[[:space:]]*\|.*$/, "", line)
        # Strip markdown bold/italic
        gsub(/\*/, "", line)
        # Skip header and separator rows
        if (line == "Package" || line == "package" || line ~ /^-+$/) next
        # Normalize
        name = tolower(line)
        gsub(/_/, "-", name)
        if (name ~ /^[a-z0-9][a-z0-9.-]*$/) print name
      }
      any_sec && /^[a-zA-Z0-9_.-]+==/ {
        # Direct pip-freeze line "pkg==ver"
        line = $0
        sub(/==.*$/, "", line)
        name = tolower(line)
        gsub(/_/, "-", name)
        print name
      }
    ' \
  | sort -u \
  | grep -v '^$' || true
)

# Also build a pkg→version map for installed packages (for version-constraint
# override below). Emit "name=version" pairs.
img_pkg_versions=$(
  awk '
      # Turn on parsing whenever we see any "## section heading" of interest.
      # Match liberally so new section titles (Core ML / RL stack, EasyR1-relevant,
      # EasyR1-required packages absent, Full pip freeze, secondary deps, etc.)
      # all contribute package=version pairs.
      /^## / {
        # Turn parsing ON for any known-relevant section heading. awk does not
        # have case-insensitive regex match, so enumerate the cases.
        if ($0 ~ /pip freeze/ || $0 ~ /Pip freeze/ ||
            $0 ~ /Core ML/ || $0 ~ /EasyR1/ ||
            $0 ~ /stack/ || $0 ~ /Stack/ ||
            $0 ~ /deps/ || $0 ~ /Dependencies/ ||
            $0 ~ /packages/ || $0 ~ /Packages/) {
          any_sec = 1
        } else {
          any_sec = 0
        }
        next
      }
      any_sec && /^\|/ {
        line = $0
        sub(/^\|[[:space:]]*/, "", line)
        n = split(line, parts, /[[:space:]]*\|[[:space:]]*/)
        if (n >= 2) {
          name = tolower(parts[1])
          ver  = parts[2]
          # Strip markdown bold/italic + backticks
          gsub(/\*/, "", name)
          gsub(/`/, "", name)
          gsub(/_/, "-", name)
          gsub(/\*/, "", ver)
          gsub(/`/, "", ver)
          # Skip header row + separator row
          if (name == "package" || name ~ /^-+$/) next
          if (name ~ /^[a-z0-9][a-z0-9.-]*$/ && ver ~ /^[0-9]/) {
            # Only take the number-starting prefix so "4.57.6 (release)" → "4.57.6"
            match(ver, /^[0-9][0-9a-zA-Z.+-]*/)
            if (RSTART > 0) ver = substr(ver, RSTART, RLENGTH)
            print name "=" ver
          }
        }
      }
      any_sec && /^[a-zA-Z0-9_.-]+==/ {
        line = $0
        split(line, parts, "==")
        name = tolower(parts[1])
        ver  = parts[2]
        gsub(/_/, "-", name)
        print name "=" ver
      }
  ' "$IMG" \
  | sort -u \
  | grep -v '^$' || true
)

# -----------------------------------------------------------------------------
# Walk requirements.txt, classify each dep.
# -----------------------------------------------------------------------------
declare -A verdict
declare -A comment

classify_dep() {
  local raw="$1"
  # Strip extras: "ray[default]" -> "ray"; keep the version-constraint part
  # separately so we can check it against the image's installed version.
  local pkg_with_constraint
  pkg_with_constraint=$(echo "$raw" | sed -E 's/[[:space:]]//g; s/\[[^]]*\]//')
  local pkg
  pkg=$(echo "$pkg_with_constraint" | sed -E 's/[<>=!~].*$//')
  local constraint
  constraint=$(echo "$pkg_with_constraint" | sed -E 's/^[a-zA-Z0-9._-]+//')  # "" or ">=5.0.0" etc.
  # Normalize
  local norm
  norm=$(echo "$pkg" | tr '[:upper:]_' '[:lower:]-')
  # Skip comments / blank
  [[ -z "$norm" || "$norm" =~ ^# ]] && return

  # Version-constraint check (new 2026-04-22 E2E finding): if the req carries
  # a constraint AND the image has the package installed, verify the image's
  # version satisfies the constraint. Unsatisfiable → override to D regardless
  # of PACKAGE_RULES.
  if [[ -n "$constraint" ]]; then
    local img_ver
    img_ver=$(echo "$img_pkg_versions" | { grep -E "^${norm}=" || true; } | head -1 | cut -d= -f2-)
    if [[ -n "$img_ver" ]]; then
      local satisfied
      satisfied=$(python3 -c "
import re, sys
try:
    from packaging.version import Version
    from packaging.specifiers import SpecifierSet
    s = SpecifierSet('$constraint')
    v = Version('$img_ver')
    print('1' if v in s else '0')
except Exception:
    print('?')
" 2>/dev/null)
      if [[ "$satisfied" == "0" ]]; then
        verdict[$norm]=D
        comment[$norm]="Version conflict: consumer requires ${constraint}, image ships ${norm}==${img_ver}. Requires an upgrade-expert to produce an image with a satisfying version. Do NOT use PACKAGE_RULES tier; version mismatch overrides."
        return
      fi
      # satisfied or unknown falls through to normal rule-based classification
    fi
  fi

  # 1. Check explicit rules
  local rule
  rule=$(echo "$RULES" | grep -E "^${norm}=" | head -1 || true)
  if [[ -n "$rule" ]]; then
    local tier=${rule#*=}; tier=${tier%%:*}
    local cmt=${rule#*:}
    verdict[$norm]=$tier
    comment[$norm]=$cmt
    return
  fi

  # 2. Fallback: in image? -> E (pure-Python / CPU presumed)
  if echo "$all_img_pkgs" | grep -qxF "$norm"; then
    verdict[$norm]=E
    comment[$norm]="Installed in target image; no explicit NPU adaptation known. Presumed accelerator-agnostic."
    return
  fi

  # 3. Not in rules, not in image -> D (BLOCKER candidate)
  verdict[$norm]=D
  comment[$norm]="NOT in target image and NO known NPU adaptation rule. Manual review required — may need tier-2 delegation or Python shim."
}

while IFS= read -r line; do
  # skip empty / pure comments
  [[ -z "${line// }" ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  classify_dep "$line"
done < "$REQS"

# -----------------------------------------------------------------------------
# Emit report
# -----------------------------------------------------------------------------
{
  echo "# Dep-gap detection report"
  echo
  echo "- **Generated**: $TS"
  echo "- **Requirements**: \`$REQS\`"
  echo "- **Target image inventory**: \`$IMG\`"
  echo
  echo "## Summary"
  echo
  declare -A count
  for k in "${!verdict[@]}"; do
    t=${verdict[$k]}
    count[$t]=$(( ${count[$t]:-0} + 1 ))
  done
  echo "| Tier | Count | Meaning |"
  echo "|---|---|---|"
  echo "| A | ${count[A]:-0} | NPU-native |"
  echo "| B | ${count[B]:-0} | NPU-ported version (drop-in replacement) |"
  echo "| C | ${count[C]:-0} | CUDA-only but bypassed in framework |"
  echo "| **D** | **${count[D]:-0}** | **BLOCKER — needs NPU adaptation work** |"
  echo "| E | ${count[E]:-0} | Pure Python / CPU |"
  echo
  if [[ ${count[D]:-0} -eq 0 ]]; then
    echo '**Verdict**: ✅ **Scenario P1** — all deps are A/B/C/E, no new NPU adaptation work needed. Proceed with standard `image-upgrade-drill`.'
  else
    echo '**Verdict**: 🟥 **Scenario P2** — at least one D-tier blocker. **DO NOT** run drill yet. File tasks in `docs/easyr1/npu-adaptation-tasks.md` (tier 1/2/3) and complete adaptation first.'
  fi
  echo
  echo "## Per-package classification"
  echo
  echo "| # | Package | Tier | Note |"
  echo "|---|---|---|---|"
  i=0
  # Sort: D first (alerts), then A/B/C/E alphabetical
  for tier_order in D A B C E; do
    for pkg in $(echo "${!verdict[@]}" | tr ' ' '\n' | sort); do
      [[ "${verdict[$pkg]}" == "$tier_order" ]] || continue
      i=$((i+1))
      echo "| $i | \`$pkg\` | $tier_order | ${comment[$pkg]} |"
    done
  done
  echo
  echo "## Next steps"
  echo
  if [[ ${count[D]:-0} -eq 0 ]]; then
    echo "1. Proceed to \`image-upgrade-drill\` Step 3 (drill branch + Dockerfile)"
    echo "2. Tier-C deps may still need framework-level shim review if new — cross-check against \`knowledge/npu-patterns.md\` NPU-CP-* entries"
  else
    echo "For each D-tier package above:"
    echo "1. Manually inspect — is there an NPU version under a different name?"
    echo "2. Is it actually required, or can the feature be turned off?"
    echo "3. If genuinely required, file a task in \`docs/easyr1/npu-adaptation-tasks.md\`:"
    echo "   - Tier 1: if Python shim in EasyR1 suffices"
    echo "   - Tier 2: if kernel / C++ work required — delegate to \`ascend-fused-accuracy-probe\` / \`a5_ops\` / A3 kernel repo"
    echo "   - Tier 3: if CANN runtime bug — file Ascend issue"
    echo "4. **Block drill** until adaptation is complete and dep moves into A/B/C/E"
  fi
} > "$OUT"

echo "Report: $OUT"
[[ ${count[D]:-0} -eq 0 ]] && exit 0 || exit 1
