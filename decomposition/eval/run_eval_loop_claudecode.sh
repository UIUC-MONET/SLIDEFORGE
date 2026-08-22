#!/bin/bash
# run_eval_loop_claudecode.sh - Self-healing eval runner using the Claude Code
# account (claude CLI) as the VLM judge. Each judge call spawns a fresh
# `claude -p` subprocess, so judge context never accumulates.
#
# Usage: run_eval_loop_claudecode.sh <input_dir> <output_dir> <workers> <log_file> <gpu_id> [extra run_eval.py args...]

INPUT_DIR=$1
OUTPUT_DIR=$2
WORKERS=$3
LOG_FILE=$4
GPU_ID=$5

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$WORKERS" ] || [ -z "$LOG_FILE" ] || [ -z "$GPU_ID" ]; then
  echo "Usage: $0 <input_dir> <output_dir> <workers> <log_file> <gpu_id> [extra args...]"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

VLM_MODEL="${VLM_MODEL:-claude-sonnet-4-6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/run_eval.py"
MAX_RESTARTS="${MAX_RESTARTS:-30}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"
rm -f "$OUTPUT_DIR/DONE" "$OUTPUT_DIR/FAILED"

echo "[loop-runner] Starting claude-code eval loop for $INPUT_DIR (gpu=$GPU_ID, model=$VLM_MODEL)..." >> "$LOG_FILE"

restarts=0
while true; do
  echo "[loop-runner] Launching run_eval.py (restart #$restarts)..." >> "$LOG_FILE"

  $PYTHON_BIN -u "$SCRIPT_PATH" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --metrics all \
    --vlm-provider claude-code \
    --vlm-model "$VLM_MODEL" \
    --save-reconstructions \
    --workers "$WORKERS" \
    "${@:6}" >> "$LOG_FILE" 2>&1

  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "[loop-runner] Finished successfully!" >> "$LOG_FILE"
    touch "$OUTPUT_DIR/DONE"
    break
  fi

  restarts=$((restarts + 1))
  if [ $restarts -ge $MAX_RESTARTS ]; then
    echo "[loop-runner] Giving up after $MAX_RESTARTS restarts (last exit code $exit_code)." >> "$LOG_FILE"
    touch "$OUTPUT_DIR/FAILED"
    exit 1
  fi

  echo "[loop-runner] Process crashed/killed with exit code $exit_code. Restarting in 15 seconds..." >> "$LOG_FILE"
  sleep 15
done
