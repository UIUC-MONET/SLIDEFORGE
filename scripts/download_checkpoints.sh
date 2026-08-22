#!/usr/bin/env bash
set -euo pipefail

# Download the two checkpoints Stage I needs into sam3/checkpoints/:
#   1. sam3.pt              — base SAM3 (Meta, from facebook/sam3 on HF)
#   2. sam3_slideforge.pt — SlideForge fine-tuned component-detector decoder
#
# Requires: pip install "huggingface_hub[cli]"
# Both checkpoints are distributed under Meta's SAM License (see sam3/LICENSE).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/../sam3/checkpoints"
mkdir -p "${CKPT_DIR}"

# SlideForge fine-tuned decoder (306 slide-component types).
DECODER_REPO="${DECODER_REPO:-zoezheng126/slideforge-sam3-decoder}"

echo "[1/2] base SAM3 checkpoint (facebook/sam3)..."
hf download facebook/sam3 sam3.pt --local-dir "${CKPT_DIR}"

echo "[2/2] fine-tuned slide-component decoder (${DECODER_REPO})..."
hf download "${DECODER_REPO}" sam3_slideforge.pt --local-dir "${CKPT_DIR}"

echo
echo "Done. Checkpoints in ${CKPT_DIR}:"
ls -lh "${CKPT_DIR}"
