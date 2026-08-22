#!/bin/bash
# run_all_evals.sh - Sequentially executes full evaluations for all three baselines.
export CUDA_VISIBLE_DEVICES=0

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[run-all] Starting Full Evaluation on all 3 datasets..."

# 1. eval_claude_code_sample_output (100 slides)
echo "[run-all] Phase 1: Running eval_claude_code_sample_output..."
bash "$DIR/run_eval_loop.sh" \
  "/path/to/netsys_ppt/eval_claude_code_sample_output" \
  "$DIR/results/eval_claude_code_sample_output_full" \
  8 \
  "$DIR/eval_claude_code_sample_output_full.log"

# 2. eval_claude_output (100 slides)
echo "[run-all] Phase 2: Running eval_claude_output..."
bash "$DIR/run_eval_loop.sh" \
  "/path/to/netsys_ppt/eval_claude_output" \
  "$DIR/results/eval_claude_output_full" \
  8 \
  "$DIR/eval_claude_output_full.log"

# 3. eval_slidecoder (23 slides)
echo "[run-all] Phase 3: Running eval_slidecoder..."
bash "$DIR/run_eval_loop.sh" \
  "/path/to/netsys_ppt/eval_slidecoder" \
  "$DIR/results/eval_slidecoder_full" \
  8 \
  "$DIR/eval_slidecoder_full.log"

echo "[run-all] All 3 datasets evaluated successfully!"
