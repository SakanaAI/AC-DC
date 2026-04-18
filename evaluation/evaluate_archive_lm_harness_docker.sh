#!/bin/bash

# This script evaluates models from an archive using ElutherAI's lm-harness evaluation framework.

set -e

# Exports
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
# dir of parent directory/root project directory
BASE_DIR=$(dirname "$(dirname "$(realpath "$0")")")
echo "BASE_DIR: $BASE_DIR"

ALL_TASKS="humaneval_instruct,mbpp_instruct"

# 🚨 Set base model you want to evaluate - must be an instruct model! 🚨
# BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
# BASE_MODEL="Qwen/Qwen3-14B"

# Docker Configuration ###############################################
DOCKER_IMAGE="lm-eval-sandbox"
DOCKER_MEMORY="128g"
DOCKER_CPUS="128"
USE_DOCKER=true  # Set to false to disable Docker sandboxing
LM_EVAL_REPO_PATH="./AC-DC-eval_harness"  # Path to your local repo

# Conda Environment Configuration ###################################
# 🚨 Change the name if you have a different conda environment name 🚨
CONDA_ENV_NAME="acdc_eval"
# Get conda environment path - this will be mounted into the container
CONDA_ENV_PATH=$(conda info --envs | grep "$CONDA_ENV_NAME" | awk '{print $2}')
if [ -z "$CONDA_ENV_PATH" ]; then
    echo "ERROR: Conda environment '$CONDA_ENV_NAME' not found. Please activate it or create it first."
    exit 1
fi
echo "Using conda environment: $CONDA_ENV_PATH"

# HuggingFace token path
HF_TOKEN_PATH="$HOME/.cache/huggingface/token"

# Function to print usage
usage() {
    echo "Usage: $0 --output_dir OUTPUT_DIR [--tasks TASKS] [--batch_size BATCH_SIZE] [--num_fewshot NUM_FEWSHOT] [--gpus GPUS] [--use_docker USE_DOCKER] [--docker_image DOCKER_IMAGE] [--docker_memory DOCKER_MEMORY] [--docker_cpus DOCKER_CPUS]"
    echo ""
    echo "Options:"
    echo "  --output_dir        Directory to save evaluation results (required)"
    echo "  --tasks            Comma-separated list of tasks to evaluate on, or 'all' (default: all)"
    echo "  --batch_size       Batch size for evaluation (default: auto)"
    echo "  --num_fewshot      Number of few-shot examples to use (default: 0)"
    echo "  --gpus             Comma-separated list of GPU IDs to use (default: 0)"
    echo "  --use_docker       Enable/disable Docker sandboxing (default: true)"
    echo "  --docker_image     Docker image name (default: lm-eval-sandbox)"
    echo "  --docker_memory    Docker memory limit (default: 32g)"
    echo "  --docker_cpus      Docker CPU limit (default: 128)"
    echo ""
    echo "Example usage:"
    echo "  $0 --tasks gsm8k,ifeval --output_dir outputs/2025-05-27/08-31-08"
    echo "  $0 --tasks mmlu --output_dir outputs/2025-05-27/08-31-08 --gpus 0,1,2,3"
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

# Function to build Docker image if it doesn't exist
build_docker_image_if_needed() {
    local docker_image="$1"
    local dockerfile_path="$2"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Checking if Docker image '$docker_image' exists..."
    
    if ! docker image inspect "$docker_image" &> /dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Docker image '$docker_image' not found. Building it now..."
        
        if [ ! -f "$dockerfile_path/Dockerfile" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - Dockerfile not found at: $dockerfile_path/Dockerfile"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - Please ensure the Dockerfile exists in the specified path."
            exit 1
        fi
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Building Docker image from: $dockerfile_path"
        if docker build -t "$docker_image" "$dockerfile_path"; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Docker image '$docker_image' built successfully!"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - Failed to build Docker image '$docker_image'"
            exit 1
        fi
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Docker image '$docker_image' already exists."
    fi
}

# Function to restore original permissions
restore_permissions() {
    local output_dir="$1"
    if [ -n "$ORIGINAL_PERMISSIONS" ] && [ -d "$output_dir" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Restoring permissions to $ORIGINAL_PERMISSIONS for $output_dir"
        chmod $ORIGINAL_PERMISSIONS "$output_dir" 2>/dev/null || true
    fi
}

# Function to set permissions for evaluation
setup_permissions() {
    local output_dir="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Setting up permissions for evaluation: $output_dir"
    
    # Create directory if it doesn't exist
    if [ ! -d "$output_dir" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Creating directory: $output_dir"
        mkdir -p "$output_dir"
        ORIGINAL_PERMISSIONS="755"  # Default for newly created directory
    else
        # Store original permissions
        ORIGINAL_PERMISSIONS=$(stat -c %a "$output_dir" 2>/dev/null || echo "755")
    fi
    
    # Set permissions to 777
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Setting permissions to 777 for $output_dir"
    chmod 777 "$output_dir" || {
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - Failed to set permissions to 777 for $output_dir"
        exit 1
    }
}

# Function to check if Docker image exists
check_docker_image() {
    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        log_error "Docker image '$DOCKER_IMAGE' not found. Please build it first:"
        log_error "docker build -t $DOCKER_IMAGE ."
        return 1
    fi
}

# Function to check conda environment
check_conda_env() {
    if [ -z "$CONDA_ENV_PATH" ]; then
        log_error "Conda environment '$CONDA_ENV_NAME' not found or not accessible."
        log_error "Please ensure the environment exists and is properly configured."
        return 1
    fi
    
    if [ ! -d "$CONDA_ENV_PATH" ]; then
        log_error "Conda environment path '$CONDA_ENV_PATH' does not exist."
        return 1
    fi
    
    log "Using conda environment: $CONDA_ENV_PATH"
    return 0
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

# Function to transform model paths for Docker container
transform_model_paths_for_docker() {
    local model_paths="$1"
    
    # Handle empty input
    if [ -z "$model_paths" ]; then
        return 1
    fi
    
    # Split by comma and transform each path
    local transformed_paths=()
    IFS=',' read -ra paths <<< "$model_paths"
    
    for path in "${paths[@]}"; do
        # Extract just the model name (basename)
        local model_name
        model_name=$(basename "$path")
        
        # Transform to Docker container path
        local docker_path="/home/evaluser/experiment/models/$model_name"
        transformed_paths+=("$docker_path")
    done
    
    # Join back with commas
    local result
    printf -v result '%s,' "${transformed_paths[@]}"
    result="${result%,}"  # Remove trailing comma
    
    echo "$result"
}

# Function to run lm-harness evaluation for a multiple models
run_lm_harness_evaluation_multiple_models() {
    local base_model="$1"
    local model_paths="$2"
    local tasks="$3"
    local output_dir_eval="$4"
    local batch_size="$5"
    local num_fewshot="$6"
    local gpu_id="$7"
    
    # local model_name
    # model_name=$(get_model_name "$base_model")
    
    # Create output directory
    mkdir -p "$output_dir_eval"
    
    # Set permissions for the local output directory so that the docker container can write to it
    log "Setting permissions to 777 for output directory: $output_dir_eval"
    chmod 777 "$output_dir_eval"
    
    if [ "$USE_DOCKER" = true ]; then
        # Get the main experiment directory (parent of eval/lm_harness_code)
        local experiment_path
        experiment_path=$(dirname "$(dirname "$output_dir_eval")")
        
        # Convert paths to absolute if they're relative
        if [[ "$experiment_path" != /* ]]; then
            experiment_path="$PWD/$experiment_path"
        fi
        if [[ "$output_dir_eval" != /* ]]; then
            output_dir_eval_abs="$PWD/$output_dir_eval"
        else
            output_dir_eval_abs="$output_dir_eval"
        fi
        if [[ "$LM_EVAL_REPO_PATH" != /* ]]; then
            lm_eval_repo_abs="$PWD/$LM_EVAL_REPO_PATH"
        else
            lm_eval_repo_abs="$LM_EVAL_REPO_PATH"
        fi
        
        # Transform model paths for Docker container
        local docker_model_paths
        docker_model_paths=$(transform_model_paths_for_docker "$model_paths")
        
        if [ $? -ne 0 ]; then
            log_error "Failed to transform model paths for Docker"
            return 1
        fi
        
        log "Transformed model paths for Docker: $docker_model_paths"
        
        # For Docker execution with multiple models
        local docker_cmd=(
            docker run --rm
            --memory="$DOCKER_MEMORY"
            --cpus="$DOCKER_CPUS"
            --gpus "device=$gpu_id"
            --tmpfs /tmp
            --tmpfs /home/evaluser/.cache
            --tmpfs /home/evaluser/.local
            --tmpfs /home/evaluser/.cache/huggingface
            --tmpfs /home/evaluser/.cache/vllm
            # mount the experiment path (where the experiment and the models are saved)
            -v "$experiment_path:/home/evaluser/experiment:ro"
            # mount the lm harness results directory (where the lm-harness results will be saved)
            -v "$output_dir_eval_abs:/home/evaluser/results"
            # mount the conda environment
            -v "$CONDA_ENV_PATH:/home/evaluser/conda_env:ro"
            # mount the lm harness repo
            -v "$lm_eval_repo_abs:$BASE_DIR/AC-DC-eval_harness:ro"
            -v "$HF_TOKEN_PATH:/home/evaluser/.cache/huggingface/token:ro"
            -e HF_ALLOW_CODE_EVAL=1
            -e CUDA_VISIBLE_DEVICES=0
            -e PATH="/home/evaluser/conda_env/bin:$PATH"
            -e CONDA_PREFIX=/home/evaluser/conda_env
            -e MPLCONFIGDIR=/tmp/matplotlib
            # -e VLLM_USE_V2=0
            -e VLLM_USE_V1=0
            -e TOKENIZERS_PARALLELISM=false
            -e VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
            -e TORCH_COMPILE_DISABLE=1
            -e VLLM_DISABLE_COMPILATION=1
            "$DOCKER_IMAGE"
            bash -c "umask 0000 && python -m lm_eval --model vllm --model_args \"pretrained=$base_model,gpu_memory_utilization=0.9\" --tasks \"$tasks\" --batch_size \"$batch_size\" --output_path \"/home/evaluser/results\" --apply_chat_template --fewshot_as_multiturn --log_samples --gen_kwargs \"temperature=0.0,do_sample=False\" --model_paths \"$docker_model_paths\" --confirm_run_unsafe_code"
        )

        if "${docker_cmd[@]}"; then
            log "Successfully evaluated all models in $model_paths on GPU $gpu_id"
        else
            log_error "Failed to evaluate (at least one model) $model_paths on GPU $gpu_id"
            return 1
        fi
    else
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
        
        # Construct lm-harness command
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
            --gen_kwargs "temperature=0.0,do_sample=False,max_gen_toks=2048"
            --model_paths "$model_paths"
        )
        
        log "Command: ${cmd[*]}"
        
        if "${cmd[@]}"; then
            log "Successfully evaluated all models in $model_paths on GPU $gpu_id"
        else
            log_error "Failed to evaluate (at least one model) $model_paths on GPU $gpu_id"
            return 1
        fi
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
    local output_dir_eval="$output_dir/eval/lm_harness_code"
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
    
    # Filter out models that have already been evaluated
    local models_to_evaluate=()
    for model_path in "${model_paths[@]}"; do
        if ! is_already_evaluated "$model_path" "$output_dir_eval" "$tasks"; then
            models_to_evaluate+=("$model_path")
        else
            local model_name
            model_name=$(get_model_name "$model_path")
            log "Model $model_name already evaluated, skipping"
        fi
    done
    
    if [ ${#models_to_evaluate[@]} -eq 0 ]; then
        log "All models have already been evaluated"
        return 0
    fi
    
    log "Found ${#models_to_evaluate[@]} models to evaluate"
    
    # Check Docker setup if enabled
    if [ "$USE_DOCKER" = true ]; then
        check_docker_image || return 1
        check_conda_env || return 1
        log "Using Docker sandboxing with image: $DOCKER_IMAGE"
        log "Using conda environment: $CONDA_ENV_PATH"
    else
        log "WARNING: Running without Docker sandboxing - code will execute directly on host"
    fi
    
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
        
        # If we've filled all GPUs, wait for one to finish before starting the next
        if [ ${#pids[@]} -ge $num_gpus ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")  # Remove first element
        fi

        sleep 1200
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
# To see which GPUs are available for the docker container, run the following command:
# docker run --rm --gpus all lm-eval-sandbox bash -c "nvidia-smi"
GPUS="1,2"
USE_DOCKER=true
DOCKER_IMAGE="lm-eval-sandbox"
DOCKER_MEMORY="128g"
DOCKER_CPUS="128"

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
        --use_docker)
            USE_DOCKER="$2"
            shift 2
            ;;
        --docker_image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --docker_memory)
            DOCKER_MEMORY="$2"
            shift 2
            ;;
        --docker_cpus)
            DOCKER_CPUS="$2"
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

# Build Docker image if it doesn't exist
build_docker_image_if_needed "$DOCKER_IMAGE" "$LM_EVAL_REPO_PATH"

# Set up trap to restore permissions on exit (including crashes, SIGINT, SIGTERM)
trap "restore_permissions \"$OUTPUT_DIR\"" EXIT INT TERM

# Setup permissions for the main output directory
setup_permissions "$OUTPUT_DIR"

# Run evaluation
run_evaluation "$OUTPUT_DIR" "$TASKS" "$BATCH_SIZE" "$NUM_FEWSHOT" "$GPUS" "$BASE_MODEL"