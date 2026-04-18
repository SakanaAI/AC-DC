# !/bin/bash

# This script runs the full evaluation and relevant metric computation for the baselines.

START_SEEDS=(50 58)

# 🚨 Set the LLM as a judge model URL 🚨
LLM_AS_A_JUDGE_MODEL_URL="http://xxx.xxx.xxx.xxx:8001/v1"

baseline_model_names_config="qwen2_5-7B-control.yaml"

# Stage 2: Compute metrics
## Compute the best-of-N scores for each start seed
for SEED in "${START_SEEDS[@]}"; do
    echo "========================================"
    echo "Computing best-of-N scores for start seed $SEED"
    echo "========================================"

    # non-code benchmarks
    echo "Computing best-of-N for non-code benchmarks (N=8 and N=3)"
    export LLM_AS_A_JUDGE_MODEL_URL=$LLM_AS_A_JUDGE_MODEL_URL
    python evaluation/single_answer_from_pop_analysis.py \
        -e "outputs/baselines/qwen2.5/$SEED" \
        --baseline_model_names_config "$baseline_model_names_config" \
        --benchmarks_file "benchmarks_main.yaml" \
        --lm_harness_name "lm_harness" \
        --num_workers 64

    # code benchmarks
    ## 8 models
    echo "Computing best-of-N for code benchmarks (N=8)"
    python evaluation/single_answer_from_pop_rm_based.py \
        -e "outputs/baselines/qwen2.5/$SEED" \
        -n 8 \
        --baseline_group control \
        --benchmarks_file "benchmarks_code.yaml" \
        -d "cuda:0"

    ## 3 models
    echo "Computing best-of-N for code benchmarks (N=3)"
    python evaluation/single_answer_from_pop_rm_based.py \
        -e "outputs/baselines/qwen2.5/$SEED" \
        -n 3 \
        --baseline_group control \
        --benchmarks_file "benchmarks_code.yaml" \
        -d "cuda:0"
done