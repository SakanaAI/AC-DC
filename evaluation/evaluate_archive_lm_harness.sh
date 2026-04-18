#!/bin/bash

# This script evaluates models from an archive using ElutherAI's lm-harness evaluation framework.

set -e

# Exports
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0

# dir of parent directory/root project directory
BASE_DIR=$(dirname "$(dirname "$(realpath "$0")")")
echo "BASE_DIR: $BASE_DIR"

# Default tasks
ALL_TASKS="gsm8k_llama,minerva_math,gpqa_main_cot_zeroshot,bbh_cot_zeroshot,mmlu_cot_llama,mmlu_pro_llama,mmlu_cot_llama_llm_as_a_judge,mmlu_pro_llama_llm_as_a_judge,bbh_cot_zeroshot_llm_as_a_judge,gpqa_main_cot_zeroshot_llm_as_a_judge"

# 🚨 Set base model you want to evaluate - must be an instruct model! 🚨
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
# BASE_MODEL="Qwen/Qwen3-14B"

# 🚨 Set the URL and name of the LLM as a judge model 🚨
LLM_AS_A_JUDGE_MODEL_URL="http://xxx.xxx.xxx.xxx:8001/v1"
LLM_AS_A_JUDGE_MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" # 🚨 Change the name if you have a different model

# Function to print usage
usage() {
    echo "Usage: $0 --output_dir OUTPUT_DIR [--tasks TASKS] [--batch_size BATCH_SIZE] [--num_fewshot NUM_FEWSHOT] [--gpus GPUS] [--max_gen MAX_GEN] [--min_gen MIN_GEN]"
    echo ""
    echo "Options:"
    echo "  --output_dir        Directory to save evaluation results (required)"
    echo "  --tasks            Comma-separated list of tasks to evaluate on, or 'all' (default: all)"
    echo "  --batch_size       Batch size for evaluation (default: auto)"
    echo "  --num_fewshot      Number of few-shot examples to use (default: 0)"
    echo "  --gpus             Comma-separated list of GPU IDs to use (default: 0)"
echo "  --max_gen          Maximum generation number for model checkpoints (filters out gen_X_ind_Y models where X > max_gen)"
    echo "  --min_gen          Minimum generation number for model checkpoints (filters out gen_X_ind_Y models where X < min_gen)"
    echo ""
    echo "Example usage:"
    echo "  $0 --tasks gsm8k,ifeval --output_dir outputs/2025-05-27/08-31-08"
    echo "  $0 --tasks mmlu --output_dir outputs/2025-05-27/08-31-08 --gpus 0,1,2,3"
echo "  $0 --tasks all --output_dir outputs/2025-05-27/08-31-08 --max_gen 10 --min_gen 5"
    exit 1
}

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $1"
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1" >&2
}

# Function to get model name from path
get_model_name() {
    basename "$1"
}

split_into_groups() {
    local items=("$@")
    local n=${items[-1]}  # Last argument is n
    unset 'items[-1]'     # Remove n from items array
    
    local total_items=${#items[@]}
    local base_size=$((total_items / n))
    local remainder=$((total_items % n))
    
    local result=()
    local start_idx=0
    
    for ((group=0; group<n; group++)); do
        # Calculate group size (distribute remainder among first groups)
        local group_size=$base_size
        if [ $group -lt $remainder ]; then
            ((group_size++))
        fi
        
        # Build comma-separated string for this group
        local group_items=""
        for ((i=0; i<group_size; i++)); do
            if [ $i -eq 0 ]; then
                group_items="${items[$start_idx]}"
            else
                group_items="$group_items,${items[$start_idx]}"
            fi
            ((start_idx++))
        done
        
        result+=("$group_items")
    done

    echo "${result[@]}"
}

count_models_in_group() {
    local group="$1"
    echo "$group" | tr ',' '\n' | wc -l
}

# Function to check if model is already evaluated
is_already_evaluated() {
    local model_path="$1"
    local output_dir="$2"
    local tasks="$3"
    local model_name
    model_name=$(get_model_name "$model_path")
    
    if [ -d "$output_dir" ]; then
        # Find model directory
        local model_dir
        model_dir=$(find "$output_dir" -maxdepth 1 -type d -name "*${model_name}" | head -n 1)
        
        if [ -n "$model_dir" ]; then
            # Check if results file exists
            local results_file
            results_file=$(find "$model_dir" -name "results_*.json" | head -n 1)
            
            if [ -n "$results_file" ]; then
                # Extract evaluated tasks from results file
                local evaluated_tasks
                evaluated_tasks=$(jq -r '.results | keys[]' "$results_file")
                
                # Check if all requested tasks are in evaluated tasks
                local all_tasks_evaluated=true
                IFS=',' read -ra task_array <<< "$tasks"
                for task in "${task_array[@]}"; do
                    if ! echo "$evaluated_tasks" | grep -q "^${task}$"; then
                        all_tasks_evaluated=false
                        break
                    fi
                done
                
                if [ "$all_tasks_evaluated" = true ]; then
                    log "Model $model_name has already been evaluated on all tasks. Skipping..."
                    return 0
                fi
            fi
        fi
    fi
    return 1
}

# Function to get all model paths
get_model_paths() {
    local output_dir="$1/models"
    if [ -d "$output_dir" ]; then
        find "$output_dir" -maxdepth 1 -type d ! -path "$output_dir"
    fi
}

# Function to run lm-harness evaluation for a multiple models
run_lm_harness_evaluation_multiple_models() {
    local base_model="$1"
    local model_paths="$2"
    local tasks="$3"
    local output_dir="$4"
    local batch_size="$5"
    local num_fewshot="$6"
    local gpu_id="$7"
    
    local model_name
    model_name=$(get_model_name "$base_model")
    
    # Create output directory
    mkdir -p "$output_dir"

    # Set HF_HOME based on gpu_id (use first GPU if data_parallel)
    local first_gpu_id
    if [[ "$gpu_id" == *","* ]]; then
        # Multiple GPUs - use the first one
        first_gpu_id="${gpu_id%%,*}"
    else
        # Single GPU
        first_gpu_id="$gpu_id"
    fi

    local HF_HOME="$BASE_DIR/AC-DC-eval_harness/cache/gpu_${first_gpu_id}/huggingface"
    local HF_TOKEN_PATH="~/.cache/huggingface/token"
    log "Using HF_HOME: $HF_HOME"
    log "Using HF_TOKEN_PATH: $HF_TOKEN_PATH"
    
    log "Running lm-harness evaluation for models: $model_paths on GPU $gpu_id"
    
    # Construct lm-harness command hardocing use of conda environment for eval harness
    model_args="pretrained=$base_model,gpu_memory_utilization=0.8,max_model_len=4096"
    # if using model with thinking mode, add the following:
    # model_args="pretrained=$base_model,gpu_memory_utilization=0.8,chat_template_args=\{\"enable_thinking\": False\}"
    local cmd=(
        env CUDA_VISIBLE_DEVICES="$gpu_id" LLM_AS_A_JUDGE_MODEL_URL="$LLM_AS_A_JUDGE_MODEL_URL" LLM_AS_A_JUDGE_MODEL_NAME="$LLM_AS_A_JUDGE_MODEL_NAME" HF_HOME="$HF_HOME" HF_TOKEN_PATH="$HF_TOKEN_PATH"
        lm_eval
        --model vllm
        --model_args "$model_args"
        --tasks "$tasks"
        --batch_size "$batch_size"
        --output_path "$output_dir"
        --apply_chat_template
        --fewshot_as_multiturn
        --log_samples
        --gen_kwargs "temperature=0.0,do_sample=False,max_gen_toks=4096"
        --model_paths "$model_paths"
    )
    
    log "Command: ${cmd[*]}"
    
    if "${cmd[@]}"; then
        log "Successfully evaluated all models in $model_paths on GPU $gpu_id"
    else
        log_error "Failed to evaluate (at least one model) $model_paths on GPU $gpu_id"
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
    local base_model="$6"
    local max_gen="${7:-}"
    local min_gen="${8:-}"
    local start_time
    start_time=$(date +%s)
    
    # Set tasks to default if "all"
    if [ "$tasks" = "all" ]; then
        tasks="$ALL_TASKS"
    fi
    
    # Parse GPU list
    IFS=',' read -ra gpu_array <<< "$gpus"
    local num_gpus=${#gpu_array[@]}
    
    log "Using GPUs: ${gpu_array[*]} (total: $num_gpus)"
    
    # Output dir for eval results
    local output_dir_eval="$output_dir/eval/lm_harness"
    mkdir -p "$output_dir_eval"
    
    # Get all model paths if model_paths is not set
    if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
        model_paths=($(get_model_paths "$output_dir"))
    else
        model_paths=(${MODEL_PATHS[@]})
    fi

    echo "model_paths:"
    printf "  %s\n" "${model_paths[@]}"
    echo "Number of models to evaluate: ${#model_paths[@]}"
    
    if [ ${#model_paths[@]} -eq 0 ]; then
        log_error "No models found in $output_dir/models"
        return 1
    fi
    
    # Filter out models that have already been evaluated and check generation number
    local models_to_evaluate=()
    for model_path in "${model_paths[@]}"; do
        local model_name
        model_name=$(get_model_name "$model_path")
        
        # Check if model matches gen_<num>_ind_<ind> format and filter by generation number
        if [[ "$model_name" =~ ^gen_([0-9]+)_ind_.+$ ]]; then
            local gen_num="${BASH_REMATCH[1]}"
            if [ -n "$max_gen" ] && [ "$gen_num" -gt "$max_gen" ]; then
                log "Model $model_name has generation $gen_num > $max_gen, skipping"
                continue
            fi
            if [ -n "$min_gen" ] && [ "$gen_num" -lt "$min_gen" ]; then
                log "Model $model_name has generation $gen_num < $min_gen, skipping"
                continue
            fi
        fi
        
        if ! is_already_evaluated "$model_path" "$output_dir_eval" "$tasks"; then
            models_to_evaluate+=("$model_path")
        else
            log "Model $model_name already evaluated, skipping"
        fi
    done
    
    if [ ${#models_to_evaluate[@]} -eq 0 ]; then
        log "All models have already been evaluated"
        return 0
    fi
    
    log "Found ${#models_to_evaluate[@]} models to evaluate"
    
    # Evaluate models in parallel across GPUs
    local pids=()
    local failed_models=()
    local model_group_idx=0
    
    # Split the models into num_gpus groups
    # In each group, concatenate model paths seperated by comma
    local model_groups=()
    model_groups=($(split_into_groups "${models_to_evaluate[@]}" "$num_gpus"))

    for model_group in "${model_groups[@]}"; do
        local gpu_idx=$((model_group_idx % num_gpus))
        local gpu_id="${gpu_array[$gpu_idx]}"

        log "Starting evaluation of $(count_models_in_group "$model_group") models on GPU $gpu_id (${model_group_idx}/${#model_groups[@]})"
        
        # Run evaluation in background
        (
            if ! run_lm_harness_evaluation_multiple_models "$base_model" "$model_group" "$tasks" "$output_dir_eval" "$batch_size" "$num_fewshot" "$gpu_id"; then
                echo "$model_group" >> /tmp/failed_models_$$
            fi
        ) &
        
        pids+=($!)
        model_group_idx=$((model_group_idx + 1))

        echo "interval sleep"
        sleep 1200
        
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

# Parse command line arguments
TASKS="all"
BATCH_SIZE="auto"
NUM_FEWSHOT=0
OUTPUT_DIR=""
GPUS="0,1,2,3"
# If you want to evaluate models only within certain generations, set max_gen and min_gen
MAX_GEN="" # e.g.18
MIN_GEN="" # e.g.20

MODEL_PATHS=()
# Or provide manually the model paths
# MODEL_PATHS=(
#     outputs/2025-08-20/15-44-28/models/gen_0_ind_11
#     outputs/2025-08-20/15-44-28/models/gen_0_ind_13
# )
# MODEL_PATHS=(
#     outputs_tests/test_run_20251103_123804/models/gen_1_ind_2
#     outputs_tests/test_run_20251103_123804/models/gen_1_ind_1
#     outputs_tests/test_run_20251103_123804/models/gen_0_ind_Qwen2-7B-Instruct
# )

while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_fewshot)
            NUM_FEWSHOT="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --max_gen)
            MAX_GEN="$2"
            shift 2
            ;;
        --min_gen)
            MIN_GEN="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --llm_as_a_judge_model_url)
            LLM_AS_A_JUDGE_MODEL_URL="$2"
            shift 2
            ;;
        --llm_as_a_judge_model_name)
            LLM_AS_A_JUDGE_MODEL_NAME="$2"
            shift 2
            ;;
        --model_paths)
            MODEL_PATHS=($2)
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir is required"
    usage
fi

# Run evaluation
run_evaluation "$OUTPUT_DIR" "$TASKS" "$BATCH_SIZE" "$NUM_FEWSHOT" "$GPUS" "$BASE_MODEL" "$MAX_GEN" "$MIN_GEN"