#!/bin/bash

# This script evaluates predefined models using ElutherAI's lm-harness evaluation framework.

set -e

# Hardcoded configuration
TASKS="aime,gsm8k_llama,minerva_math,gpqa_main_cot_zeroshot,gpqa_diamond_cot_zeroshot,bbh_cot_zeroshot,mmlu_cot_llama,mmlu_pro_llama,arc_challenge_llama,ifeval,mmlu_cot_llama_llm_as_a_judge,mmlu_pro_llama_llm_as_a_judge,bbh_cot_zeroshot_llm_as_a_judge,gpqa_main_cot_zeroshot_llm_as_a_judge"
BATCH_SIZE="auto"

LLM_AS_A_JUDGE_MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"


### Get script arguments

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        --llm_as_a_judge_model_url)
            LLM_AS_A_JUDGE_MODEL_URL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -s <seed> --llm_as_a_judge_model_url <llm_as_a_judge_model_url>"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 -s <seed> --llm_as_a_judge_model_url <llm_as_a_judge_model_url>"
            echo "  -s, --seed             Seed for the evaluation"
            echo "  --llm_as_a_judge_model_url LLM as a judge model URL (required)"
            echo ""
            echo "Example:"
            echo "  $0 -s 50 --llm_as_a_judge_model_url http://xxx.xxx.xxx.xxx:8001/v1"
            exit 1
            ;;
    esac
done

# args have to exist
if [ -z "$SEED" ]; then
    echo "Error: --seed is required"
    exit 1
fi
if [ -z "$LLM_AS_A_JUDGE_MODEL_URL" ]; then
    echo "Error: --llm_as_a_judge_model_url is required"
    exit 1
fi

### Qwen 2.5 ###########################################################
### 7B Models ##########################################################

### Eval of 7B-Instruct model
SEED_IN_PATH=true
GPUS="0,1,2,3"
# GPUS="1"
OUTPUT_DIR="outputs/baselines/qwen2.5/$SEED/lm_harness"
GEN_KWARGS="temperature=0.7,do_sample=True,max_gen_toks=2048"
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $1"
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1" >&2
}


# Function to run lm-harness evaluation for a single model
run_lm_harness_evaluation() {
    local model_name="$1"
    local tasks="$2"
    local output_dir="$3"
    local batch_size="$4"
    local num_fewshot="$5"
    local gpu_id="$6"
    local gen_kwargs="$7"
    local seed="$8"
    log "Running lm-harness evaluation for model: $model_name on GPU $gpu_id"
    
    # Create model-specific output directory
    if [ "$SEED_IN_PATH" = true ]; then
        local model_output_dir="$output_dir/$(basename "$model_name")/seed_$seed"
    else
        local model_output_dir="$output_dir/$(basename "$model_name")"
    fi
    mkdir -p "$model_output_dir"
    
    # Construct lm-harness command
    local cmd=(
        env CUDA_VISIBLE_DEVICES="$gpu_id" LLM_AS_A_JUDGE_MODEL_URL="$LLM_AS_A_JUDGE_MODEL_URL" LLM_AS_A_JUDGE_MODEL_NAME="$LLM_AS_A_JUDGE_MODEL_NAME"
        lm_eval
        --model vllm
        --model_args "pretrained=$model_name,max_model_len=4096"
        --tasks "$tasks"
        --batch_size "$batch_size"
        --output_path "$model_output_dir"
        --apply_chat_template
        --fewshot_as_multiturn
        --log_samples
        --gen_kwargs "$gen_kwargs,seed=$seed"
        --seed "$seed"
    )
    
    log "Command: ${cmd[*]}"
    
    if "${cmd[@]}"; then
        log "Successfully evaluated model: $model_name on GPU $gpu_id"
    else
        log_error "Failed to evaluate model $model_name on GPU $gpu_id"
        return 1
    fi
}

# Function to run evaluation on all models
run_evaluation() {
    local output_dir="$1"
    local tasks="$2"
    local batch_size="$3"
    local num_fewshot="$4"
    local gpus="$5"
    local gen_kwargs="$6"
    local seed="$7"
    local start_time
    start_time=$(date +%s)
    
    # Parse GPU list
    IFS=',' read -ra gpu_array <<< "$gpus"
    local num_gpus=${#gpu_array[@]}
    
    log "Using GPUs: ${gpu_array[*]} (total: $num_gpus)"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Evaluate models in parallel across GPUs
    local pids=()
    local failed_models=()
    local model_idx=0

    for model_name in "${MODELS[@]}"; do
        local gpu_idx=$((model_idx % num_gpus))
        local gpu_id="${gpu_array[$gpu_idx]}"
        local seed=$((seed + 1))
        
        log "Starting evaluation of model $model_name with seed $seed on GPU $gpu_id (${model_idx}/${#MODELS[@]})"
        
        # Run evaluation in background
        (
            if ! run_lm_harness_evaluation "$model_name" "$tasks" "$output_dir" "$batch_size" "$num_fewshot" "$gpu_id" "$gen_kwargs" "$seed"; then
                echo "$model_name" >> /tmp/failed_models_$$
            fi
        ) &
        
        pids+=($!)
        model_idx=$((model_idx + 1))
        
        # If we've filled all GPUs, wait for one to finish before starting the next
        if [ ${#pids[@]} -ge $num_gpus ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")  # Remove first element
        fi
    done
    
    # Wait for all remaining jobs to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    # Check for failed models
    if [ -f "/tmp/failed_models_$$" ]; then
        mapfile -t failed_models < /tmp/failed_models_$$
        rm -f "/tmp/failed_models_$$"
    fi
    
    local end_time
    end_time=$(date +%s)
    local time_taken=$((end_time - start_time))
    local time_taken_minutes=$((time_taken / 60))
    
    log "Time taken: $time_taken_minutes minutes"
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        log_error "Failed to evaluate the following models: ${failed_models[*]}"
        return 1
    fi
}

# Run evaluation
run_evaluation "$OUTPUT_DIR" "$TASKS" "$BATCH_SIZE" "$NUM_FEWSHOT" "$GPUS" "$GEN_KWARGS" "$SEED"
