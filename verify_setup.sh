#!/bin/bash

# ==============================================================================
# SANITY CHECK: Quickest End-to-End Pipeline Test
# Verifies Generation, Preprocessing, Training (1 epoch), and Evaluation
# across all 3 cases (33, 57, 118) in parallel.
# ==============================================================================

# Case Selection (Set to true to include in the sanity check)
DO_CASE_33=true
DO_CASE_57=true
DO_CASE_118=true

# Configurations (Minimized for speed)
SAMPLES=24             # Minimal timesteps (enough for seq_len=4 + splitting)
EPOCHS=1               # Only 1 epoch to verify training loop
MODELS="all"           # Test all architectures

# Cleanup background processes on exit (Ctrl+C)
trap "kill 0" EXIT

# Cleanup Controls
DO_CLEAN_REPORTS=true
DO_CLEAN_LOGS=true
DO_CLEAN_WANDB=true
DO_CLEAN_RAW_DATA=true
DO_CLEAN_PROCESSED_DATA=true

# Execution Controls
DO_GENERATE=true
DO_PREPROCESS=true
DO_TRAIN=true
DO_EVALUATE=true
DO_UNCERTAINTY=true

echo "=========================================================="
echo "    🚀 STARTING QUICK SANITY CHECK (ALL CASES)"
echo "=========================================================="

if [ "$DO_CLEAN_REPORTS" = true ]; then echo "🧹 Cleaning reports..."; make clean-reports; fi
if [ "$DO_CLEAN_LOGS" = true ]; then echo "🧹 Cleaning logs..."; make clean-logs; fi
if [ "$DO_CLEAN_WANDB" = true ]; then echo "🧹 Cleaning W&B logs..."; make clean-wandb; fi
if [ "$DO_CLEAN_RAW_DATA" = true ]; then echo "🧹 Cleaning raw data..."; make clean-data-raw; fi
if [ "$DO_CLEAN_PROCESSED_DATA" = true ]; then echo "🧹 Cleaning processed data..."; make clean-data-processed; fi
echo "Cleanup complete."
echo ""

# Create log directories
mkdir -p logs/pipeline_logs

# Define a function that runs the ENTIRE pipeline for a single case
run_case_pipeline() {
    local case_id=$1
    local log_dir="logs/pipeline_logs/case${case_id}"
    mkdir -p "$log_dir"
    local log_prefix="${log_dir}/case${case_id}_sanity"
    echo "[Case $case_id] Starting Sanity Pipeline..."

    # 1. Generate Data
    python scripts/generate_data.py --case $case_id --timestep $SAMPLES > "${log_prefix}_generation.txt" 2>&1
    if [ $? -ne 0 ]; then echo "❌ [Case $case_id] Failed at Data Generation"; return 1; fi
    echo "✅ [Case $case_id] Data Generation Complete"

    # 2. Preprocess Data
    python scripts/preprocess_data.py --case $case_id > "${log_prefix}_preprocess.txt" 2>&1
    if [ $? -ne 0 ]; then echo "❌ [Case $case_id] Failed at Preprocessing"; return 1; fi
    echo "✅ [Case $case_id] Preprocessing Complete"

    # 3. Train Models
    python scripts/train.py --models "$MODELS" --epochs $EPOCHS --case $case_id > "${log_prefix}_training.txt" 2>&1
    if [ $? -ne 0 ]; then echo "❌ [Case $case_id] Failed at Training"; return 1; fi
    echo "✅ [Case $case_id] Training Complete (1 epoch)"

    # 4. Evaluation Benchmark
    if [ "$DO_EVALUATE" = true ]; then
        python scripts/evaluate.py --case $case_id > "${log_prefix}_evaluation.txt" 2>&1
        echo "✅ [Case $case_id] Benchmarks Complete"
    else
        echo "⏭️  [Case $case_id] Skipping Evaluation"
    fi

    # 5. Uncertainty Analysis
    if [ "$DO_UNCERTAINTY" = true ]; then
        python scripts/analyze_uncertainty.py --case $case_id > "${log_prefix}_uncertainty.txt" 2>&1
        echo "✅ [Case $case_id] Uncertainty Complete"
    else
        echo "⏭️  [Case $case_id] Skipping Uncertainty Analysis"
    fi
    
    echo "🎉 [Case $case_id] SANITY CHECK FINISHED!"
}

# Run the full pipeline function for all 3 cases in parallel
echo "Launching parallel sanity checks. Monitor logs/pipeline_logs/case*_sanity_*.txt"
echo ""

if [ "$DO_CASE_33" = true ]; then
    run_case_pipeline 33 &
    PID33=$!
fi

if [ "$DO_CASE_57" = true ]; then
    run_case_pipeline 57 &
    PID57=$!
fi

if [ "$DO_CASE_118" = true ]; then
    run_case_pipeline 118 &
    PID118=$!
fi

# Wait for all independent pipelines to finish
if [ -n "$PID33" ]; then wait $PID33; fi
if [ -n "$PID57" ]; then wait $PID57; fi
if [ -n "$PID118" ]; then wait $PID118; fi

echo ""
echo "=========================================================="
echo "          ✅ ALL SANITY CHECKS COMPLETE!"
echo "=========================================================="
