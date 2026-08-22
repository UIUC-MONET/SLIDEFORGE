#!/bin/bash
# run_eval_loop.sh - Robust, self-healing runner for slide evaluation that auto-recovers from OOM or crashes

INPUT_DIR=$1
OUTPUT_DIR=$2
WORKERS=$3
LOG_FILE=$4

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$WORKERS" ] || [ -z "$LOG_FILE" ]; then
  echo "Usage: $0 <input_dir> <output_dir> <workers> <log_file>"
  exit 1
fi

# Set your Gemini key in the environment before launching:
#   export GEMINI_API_KEY=...
: "${GEMINI_API_KEY:?set GEMINI_API_KEY in the environment first}"
PYTHON_BIN="python"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/run_eval.py"

echo "[loop-runner] Starting robust evaluation loop for $INPUT_DIR..." >> "$LOG_FILE"

while true; do
  echo "[loop-runner] Launching run_eval.py..." >> "$LOG_FILE"
  
  $PYTHON_BIN -u "$SCRIPT_PATH" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --metrics all \
    --vlm-provider gemini \
    --vlm-model gemini-3.5-flash \
    --save-reconstructions \
    --workers "$WORKERS" \
    "${@:5}" >> "$LOG_FILE" 2>&1
  
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "[loop-runner] Finished successfully!" >> "$LOG_FILE"
    break
  fi
  
  echo "[loop-runner] Process crashed/killed with exit code $exit_code. Restarting in 5 seconds..." >> "$LOG_FILE"
  sleep 5
done
