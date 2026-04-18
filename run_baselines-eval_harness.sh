START_SEEDS=(50 58)

# 🚨 Set the LLM as a judge model URL 🚨
LLM_AS_A_JUDGE_MODEL_URL="http://xxx.xxx.xxx.xxx:8001/v1"

# Run the evaluation for non-code benchmarks
for SEED in "${START_SEEDS[@]}"; do
    
    # non-code benchmarks
    echo "Running evaluation for start seed $SEED"
    echo "Saving results to outputs/baselines/qwen2.5/$SEED/lm_harness"
    bash evaluation/evaluate_baselines_lm_harness_control.sh -s $SEED --llm_as_a_judge_model_url $LLM_AS_A_JUDGE_MODEL_URL

done