#!/bin/bash

# This script evaluates predefined models using ElutherAI's lm-harness evaluation framework with Docker sandboxing.

set -e

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

# Hardcoded configuration
TASKS="humaneval_instruct,mbpp_instruct"
BATCH_SIZE="auto"

### Get script arguments - only seed is required
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 -s <seed> -g <gpus>"
            echo "  -s, --seed             Seed for the evaluation"
            echo "  -g, --gpus             GPUs to use for the evaluation"
            echo ""
            echo "Example:"
            echo "  $0 -s 50 -g 0,1,2,3"
            exit 1
            ;;
    esac
done

# args have to exist
if [ -z "$SEED" ]; then
    echo "Error: --seed is required"
    exit 1
fi

# if gpus is not set, set it to 0,1,2,3
if [ -z "$GPUS" ]; then
    GPUS="0,1,2,3"
fi

# HuggingFace token path
HF_TOKEN_PATH="$HOME/.cache/huggingface/token"

BASE_DIR=$(dirname "$(dirname "$(realpath "$0")")")
echo "BASE_DIR: $BASE_DIR"

### Docker Configuration ###############################################
DOCKER_IMAGE="lm-eval-sandbox"
DOCKER_MEMORY="128g"
DOCKER_CPUS="128"
USE_DOCKER=true  # Set to false to disable Docker sandboxing
LM_EVAL_REPO_PATH="./AC-DC-eval_harness"  # Path to your local repo

# Build Docker image if it doesn't exist
build_docker_image_if_needed "$DOCKER_IMAGE" "$LM_EVAL_REPO_PATH"

### Conda Environment Configuration ###################################
CONDA_ENV_NAME="acdc_eval"
# Get conda environment path - this will be mounted into the container
CONDA_ENV_PATH=$(conda info --envs | grep "$CONDA_ENV_NAME" | awk '{print $2}')
if [ -z "$CONDA_ENV_PATH" ]; then
    echo "ERROR: Conda environment '$CONDA_ENV_NAME' not found. Please activate it or create it first."
    exit 1
fi
echo "Using conda environment: $CONDA_ENV_PATH"

########################################################################
### Qwen 2.5 ###########################################################
### Eval of 7B-Instruct model
SEED_IN_PATH=true
BIG_MODEL=false

OUTPUT_DIR="outputs/baselines/qwen2.5/$SEED/lm_harness_code"
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

# Set up trap to restore permissions on exit (including crashes, SIGINT, SIGTERM)
trap "restore_permissions \"$OUTPUT_DIR\"" EXIT INT TERM

# Setup permissions for the main output directory
setup_permissions "$OUTPUT_DIR"


# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $1"
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1" >&2
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
    local big_model="$9"
    log "Running lm-harness evaluation for model: $model_name on GPU $gpu_id"
    
    # Create model-specific output directory
    if [ "$SEED_IN_PATH" = true ]; then
        local model_output_dir="$output_dir/$(basename "$model_name")/seed_$seed"
    else
        local model_output_dir="$output_dir/$(basename "$model_name")"
    fi
    mkdir -p "$model_output_dir"
    
    # Set permissions for the model-specific directory
    log "Setting permissions to 777 for model output directory: $model_output_dir"
    chmod 777 "$model_output_dir"
    
    if [ "$USE_DOCKER" = true ]; then
        # Check if local repo exists
        if [ ! -d "$LM_EVAL_REPO_PATH" ]; then
            log_error "Local AC-DC-eval_harness repo not found at: $LM_EVAL_REPO_PATH"
            return 1
        fi

        local model_output_dir_in_container="/home/evaluser/results/$(basename "$model_name")"

        model_args="pretrained=$model_name,gpu_memory_utilization=0.8,max_model_len=4096"
        # gpu_id_in_container=$gpu_id
        gpu_id_in_container=0
        gpus_for_docker="device=$gpu_id"
        
        # Docker command for normal models (vLLM backend)
        local docker_cmd=(
            docker run --rm
            --memory="$DOCKER_MEMORY"
            --cpus="$DOCKER_CPUS"
            --gpus "$gpus_for_docker"
            --tmpfs /tmp
            --tmpfs /home/evaluser/.cache
            --tmpfs /home/evaluser/.local
            --tmpfs /home/evaluser/.cache/huggingface
            --tmpfs /home/evaluser/.cache/vllm
            -v "$PWD/$model_output_dir:/home/evaluser/results"
            -v "$CONDA_ENV_PATH:/home/evaluser/conda_env:ro"
            -v "$PWD/$LM_EVAL_REPO_PATH:$BASE_DIR/AC-DC-eval_harness:ro"
            -v "$HF_TOKEN_PATH:/home/evaluser/.cache/huggingface/token:ro"
            -e HF_ALLOW_CODE_EVAL=1
            -e CUDA_VISIBLE_DEVICES=$gpu_id_in_container
            -e PATH="/home/evaluser/conda_env/bin:$PATH"
            -e CONDA_PREFIX=/home/evaluser/conda_env
            -e MPLCONFIGDIR=/tmp/matplotlib
            -e VLLM_USE_V1=0
            -e TOKENIZERS_PARALLELISM=false
            -e VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
            -e TORCH_COMPILE_DISABLE=1
            -e VLLM_DISABLE_COMPILATION=1
            "$DOCKER_IMAGE"
            bash -c "umask 0000 && python -m lm_eval --model vllm --model_args \"$model_args\" --tasks \"$tasks\" --batch_size \"$batch_size\" --output_path \"$model_output_dir_in_container\" --apply_chat_template --fewshot_as_multiturn --log_samples --gen_kwargs \"$gen_kwargs,seed=$seed\" --seed \"$seed\" --confirm_run_unsafe_code"
        )
        
        log "Docker command: ${docker_cmd[*]}"
        
        if "${docker_cmd[@]}"; then
            log "Successfully evaluated model: $model_name on GPU $gpu_id"
            
            # Make all files in the output directory writable (777)
            log "Setting permissions to 777 for all files in: $model_output_dir"
            chmod -R 777 "$model_output_dir" 2>/dev/null || true
        else
            log_error "Failed to evaluate model $model_name on GPU $gpu_id"
            return 1
        fi
    else
        # Original command without Docker
        local cmd=(
            env CUDA_VISIBLE_DEVICES="$gpu_id" HF_ALLOW_CODE_EVAL="1"
            "${lm_eval_cmd[@]}"
        )
        
        # Update output path for non-Docker execution
        cmd[${#cmd[@]}-9]="--output_path"
        cmd[${#cmd[@]}-8]="$model_output_dir"
        
        log "Command: ${cmd[*]}"
        
        if "${cmd[@]}"; then
            log "Successfully evaluated model: $model_name on GPU $gpu_id"
        else
            log_error "Failed to evaluate model $model_name on GPU $gpu_id"
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
    local gen_kwargs="$6"
    local seed="$7"
    local big_model="$8"
    local start_time
    start_time=$(date +%s)
    
    # Check Docker setup if enabled
    if [ "$USE_DOCKER" = true ]; then
        check_docker_image || return 1
        check_conda_env || return 1
        log "Using Docker sandboxing with image: $DOCKER_IMAGE"
        log "Using conda environment: $CONDA_ENV_PATH"
    else
        log "WARNING: Running without Docker sandboxing - code will execute directly on host"
    fi
    
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
            if ! run_lm_harness_evaluation "$model_name" "$tasks" "$output_dir" "$batch_size" "$num_fewshot" "$gpu_id" "$gen_kwargs" "$seed" "$big_model"; then
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
run_evaluation "$OUTPUT_DIR" "$TASKS" "$BATCH_SIZE" "$NUM_FEWSHOT" "$GPUS" "$GEN_KWARGS" "$SEED" "$BIG_MODEL"