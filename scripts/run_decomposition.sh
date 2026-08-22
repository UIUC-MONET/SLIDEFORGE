#!/usr/bin/env bash
set -euo pipefail

# Stage I — Deck State Graph construction (perception-aligned decomposition),
# in the release configuration from the paper:
#   * persistent SAM3 worker (loads the 6.7 GB checkpoints once)
#   * three-tier model cascade (haiku screen -> sonnet judgment -> opus
#     escalation) with a 5% opus audit
#   * merge-acceptance caps
#
# Usage:
#   scripts/run_decomposition.sh <images: dir | JSON list> <run_dir> [device]
#
# Environment:
#   BACKEND        claude_cli (default; bills a Claude Code subscription) or
#                  claude (Anthropic API; export ANTHROPIC_API_KEY)
#   SAM3_PYTHON    interpreter with torch + the sam3 package (default: python3)

IMAGES=${1:?usage: run_decomposition.sh <images> <run_dir> [device]}
RUN_DIR=${2:?usage: run_decomposition.sh <images> <run_dir> [device]}
DEVICE=${3:-cuda:0}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND=${BACKEND:-claude_cli}
SAM3_PYTHON=${SAM3_PYTHON:-python3}
CKPT=${CKPT:-${ROOT}/sam3/checkpoints/sam3_slideforge.pt}
BASE_CKPT=${BASE_CKPT:-${ROOT}/sam3/checkpoints/sam3.pt}

export SAM3_WORKER_DIR=$(mktemp -d /tmp/sam3_worker_XXXX)
export CASCADE_AUDIT_FRAC=0.05

echo "[worker] starting persistent SAM3 worker (queue: ${SAM3_WORKER_DIR})"
"${SAM3_PYTHON}" "${ROOT}/decomposition/sam3_worker.py" \
    --script "${ROOT}/sam3/infer_remove_components_overlap_priority.py" \
    --ckpt "${CKPT}" --base_ckpt "${BASE_CKPT}" \
    --device "${DEVICE}" --queue_dir "${SAM3_WORKER_DIR}" &
WORKER_PID=$!
trap 'touch "${SAM3_WORKER_DIR}/stop"; wait ${WORKER_PID} 2>/dev/null || true' EXIT

cd "${ROOT}"
python3 decomposition/run_pipeline.py \
    --images "${IMAGES}" \
    --backend "${BACKEND}" \
    --f_validity_cascade --agent_cascade --judgment_cascade \
    --abc_model claude-haiku-4-5-20251001 \
    --merge_max_members 8 --merge_max_area_frac 0.35 --merge_area_min_members 6 \
    --run_dir "${RUN_DIR}" \
    --ckpt "${CKPT}" --base_ckpt "${BASE_CKPT}" \
    --conda_python "${SAM3_PYTHON}" \
    --device "${DEVICE}"

echo
echo "Deck State Graph state written under ${RUN_DIR}/<slide>/final/"
echo "Build the graph JSON:  python -m dsg.build_dsg --run-dir ${RUN_DIR}"
