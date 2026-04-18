# !/bin/bash

# This script runs the full evaluation and relevant metric computation for the baselines.

START_SEEDS=(50 58)

baseline_model_names_config="qwen2_5-7B-control.yaml"

# Stage 2: Compute metrics
## Compute the coverage for each start seed
for SEED in "${START_SEEDS[@]}"; do
    echo "========================================"
    echo "Computing coverage for start seed $SEED"
    echo "========================================"

    # non-code benchmarks
    echo "Computing coverage for non-code benchmarks (lm_harness)"
    python evaluation/coverage.py \
        -e "outputs/baselines/qwen2.5/$SEED" \
        --baseline_eval \
        --model_names_config "$baseline_model_names_config" \
        --lm_harness_name "lm_harness" \
        --benchmark_metrics_config "benchmarks_main.yaml"

    # code benchmarks
    echo "Computing coverage for code benchmarks (lm_harness_code)"
    python evaluation/coverage.py \
        -e "outputs/baselines/qwen2.5/$SEED" \
        --baseline_eval \
        --model_names_config "$baseline_model_names_config" \
        --lm_harness_name "lm_harness_code" \
        --benchmark_metrics_config "benchmarks_code.yaml"
done