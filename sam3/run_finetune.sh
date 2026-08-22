#!/usr/bin/env bash
set -euo pipefail

# SAM3 decoder fine-tuning on slide component data (Stage I detector).
# Reference hardware: 2x RTX 4090, batch_size=16/GPU (effective 32),
# backbone bf16 / decoder fp32, peak ~19.3 GB per GPU, ~4.7 h for 10 epochs.
# Trains only a lightweight decoder adaptation (30.4M params, 3.6% of model).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set WANDB_MODE=online (and `wandb login` first) to log to Weights & Biases.
export WANDB_MODE=${WANDB_MODE:-offline}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Training annotations: one entry per slide with component bounding boxes
# labeled against the 306-class taxonomy (../data/sam3_text_types_306.json).
# Boxes are derived automatically from native PPTX structure; labels come
# from an LLM-based labeling pipeline (see paper App. B).
TRAIN_DATA=${TRAIN_DATA:-${SCRIPT_DIR}/data/sam3_train.json}
EVAL_DATA=${EVAL_DATA:-${SCRIPT_DIR}/data/sam3_eval.json}
CKPT=${CKPT:-${SCRIPT_DIR}/checkpoints/sam3.pt}
OUTPUT=${OUTPUT:-${SCRIPT_DIR}/output/decoder_finetune}

torchrun --nproc_per_node=2 "${SCRIPT_DIR}/tune_decoder.py" \
    --data      "${TRAIN_DATA}" \
    --eval_data "${EVAL_DATA}" \
    --ckpt      "${CKPT}" \
    --output    "${OUTPUT}" \
    --epochs    10 \
    --batch_size 16 \
    --lr 8e-5 \
    --log_freq 10 \
    --eval_freq 100 \
    --num_workers 1 \
    --resume \
    --wandb_project sam3-finetune \
    --wandb_run_name "slide-components-decoder"
