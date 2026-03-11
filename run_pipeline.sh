#!/bin/bash

# ==============================================================================
# Full Independent Parallel Pipeline for Multi-GPU Environments
# Orchestrates 3 completely isolated pipelines (Case 33, 57, 118) simultaneously
# ==============================================================================

# Case Selection (Set to true to include in the parallel run)
DO_CASE_33=true
DO_CASE_57=true
DO_CASE_118=true

# Configurations
SAMPLES=10008
EPOCHS=200
MODELS="all"
# Environment variables to keep things clean
export WANDB_SILENT=true

# Cleanup Controls (set to true to wipe before running)
DO_CLEAN_REPORTS=true      # Wipe all reports
DO_CLEAN_LOGS=true         # Wipe local training logs
DO_CLEAN_WANDB=true        # Wipe wandb logs
DO_CLEAN_RAW_DATA=false    # Wipe raw datasets (caution: long re-generation)
DO_CLEAN_PROCESSED_DATA=false# Wipe graph objects

# Pipeline Execution Controls
DO_GENERATE=true           # Set to false to skip data generation
DO_PREPROCESS=true         # Set to false to skip preprocessing
DO_TRAIN=true              # Set to false to skip training
DO_EVALUATE=true           # Set to false to skip evaluation
DO_UNCERTAINTY=true        # Set to false to skip uncertainty

echo "=========================================================="
echo "    STARTING ISOLATED PARALLEL PIPELINES (33, 57, 118)"
echo "=========================================================="

# Cleanup background processes on exit (Ctrl+C)
trap "kill 0" EXIT

echo "Performing granular cleanup..."
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
    local log_prefix="${log_dir}/case${case_id}"
    echo "[Case $case_id] Starting Full Pipeline..."

    # 1. Generate Data (Optional)
    if [ "$DO_GENERATE" = true ]; then
        python scripts/generate_data.py --case $case_id --timestep $SAMPLES > "${log_prefix}_generation.txt" 2>&1
        if [ $? -ne 0 ]; then echo "❌ [Case $case_id] Failed at Data Generation"; return 1; fi
        echo "✅ [Case $case_id] Data Generation Complete"
    else
        echo "⏭️  [Case $case_id] Skipping Data Generation (DO_GENERATE=false)"
    fi

    # 2. Preprocess Data (Optional)
    if [ "$DO_PREPROCESS" = true ]; then
        python scripts/preprocess_data.py --case $case_id > "${log_prefix}_preprocess.txt" 2>&1
        if [ $? -ne 0 ]; then echo "❌ [Case $case_id] Failed at Preprocessing"; return 1; fi
        echo "✅ [Case $case_id] Preprocessing Complete"
    else
        echo "⏭️  [Case $case_id] Skipping Preprocessing (DO_PREPROCESS=false)"
    fi

    # 3. Train Models (Optional)
    if [ "$DO_TRAIN" = true ]; then
        python scripts/train.py --models "$MODELS" --epochs $EPOCHS --case $case_id > "${log_prefix}_training.txt" 2>&1
        if [ $? -ne 0 ]; then echo "❌ [Case $case_id] Failed at Training"; return 1; fi
        echo "✅ [Case $case_id] Training Complete ($EPOCHS epochs)"
    else
        echo "⏭️  [Case $case_id] Skipping Training (DO_TRAIN=false)"
    fi

    # 4. Evaluation Benchmark
    if [ "$DO_EVALUATE" = true ]; then
        python scripts/evaluate.py --case $case_id > "${log_prefix}_evaluation.txt" 2>&1
        echo "✅ [Case $case_id] Benchmarks Complete"
    else
        echo "⏭️  [Case $case_id] Skipping Evaluation (DO_EVALUATE=false)"
    fi
    
    # 5. Uncertainty Analysis
    if [ "$DO_UNCERTAINTY" = true ]; then
        python scripts/analyze_uncertainty.py --case $case_id > "${log_prefix}_uncertainty.txt" 2>&1
        echo "✅ [Case $case_id] Uncertainty Complete"
    else
        echo "⏭️  [Case $case_id] Skipping Uncertainty Analysis (DO_UNCERTAINTY=false)"
    fi
    
    echo "🎉 [Case $case_id] PIPELINE FINISHED SUCCESSFULLY!"
}

# Run the full pipeline function for all 3 cases in parallel (background processes)
echo "Launching background pipelines. Monitor progress in logs/pipeline_logs/ files."
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
echo "          ALL PARALLEL PIPELINES COMPLETE!"
echo "=========================================================="
