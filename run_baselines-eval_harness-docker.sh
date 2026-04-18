START_SEEDS=(50 58)

# 🚨 Set the GPUs to use for the docker evaluation 🚨
DOCKER_GPUS="0,1,2,3"

# Run the evaluation for code benchmarks
for SEED in "${START_SEEDS[@]}"; do
    
    # code benchmarks with docker
    echo "Running evaluation for start seed $SEED with code benchmarks"
    echo "Saving results to outputs/baselines/qwen2.5/$SEED/lm_harness_code"
    bash evaluation/evaluate_baselines_lm_harness_docker_control.sh -s $SEED -g $DOCKER_GPUS

done