#!/bin/bash

# Function to display help message
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "Start multiple QD workers with Ray for fractional GPU assignment"
  echo ""
  echo "Options:"
  echo "  --gpus=N                Number of GPUs to use (default: 1)"
  echo "  --workers-per-gpu=N     Number of workers per GPU (default: 4)"
  echo "  --session=NAME          tmux session name (default: qd_workers)"
  echo "  --script=SCRIPT_NAME    Python script to run (default: main_dns.py)"
  echo "  --broker=URL            Celery broker URL (default: pyamqp://guest@localhost:5801//)"
  echo "  --backend=URL           Celery backend URL (default: redis://default:user@localhost:6501/0)"
  echo "  --output-dir=PATH       Custom output directory (will add timestamp subdirectory)"
  echo "  --help                  Display this help message and exit"
  echo ""
  echo "Example:"
  echo "  $0 --gpus=2 --workers-per-gpu=3 --session=my_session --script=main.py"
  echo ""
}

# Default values
SESSION="qd_workers"
NUM_GPUS=4
WORKERS_PER_GPU=2
# NUM_GPUS=8
# WORKERS_PER_GPU=1
SCRIPT_NAME="main_ac_dc.py"
BROKER="pyamqp://guest@localhost:5807//"
BACKEND="redis://default:user@localhost:6507/0"
PERCENT_GPU_FOR_WORKERS=0.6 # 0.9, 0.6
OUTPUT_DIR="/path/to/your/output/directory"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus=*)
      NUM_GPUS="${1#*=}"
      shift
      ;;
    --workers-per-gpu=*)
      WORKERS_PER_GPU="${1#*=}"
      shift
      ;;
    --session=*)
      SESSION="${1#*=}"
      shift
      ;;
    --script=*)
      SCRIPT_NAME="${1#*=}"
      shift
      ;;
    --broker=*)
      BROKER="${1#*=}"
      shift
      ;;
    --backend=*)
      BACKEND="${1#*=}"
      shift
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

echo "Starting session '$SESSION' with $NUM_GPUS GPUs and $WORKERS_PER_GPU workers per GPU"
echo "Total workers: $((NUM_GPUS * WORKERS_PER_GPU))"
echo "Using script: $SCRIPT_NAME"
echo "Broker: $BROKER"
echo "Backend: $BACKEND"

# Calculate total number of workers
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

# Build hydra output directory arguments for main process
MAIN_HYDRA_DIR_ARGS=""
if [ -n "$OUTPUT_DIR" ]; then
    MAIN_HYDRA_DIR_ARGS="'hydra.run.dir=${OUTPUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}'"
fi

# Check if tmux session already exists
if tmux has-session -t $SESSION 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attaching..."
  tmux attach-session -t $SESSION
  exit 0
fi

# Start new tmux session
tmux new-session -d -s $SESSION

# Create a window for each worker
for ((i=0; i<TOTAL_WORKERS; i++)); do
    # Calculate which GPU this worker should use
    GPU_ID=$((i / WORKERS_PER_GPU))
    
    # Calculate worker ID within the GPU
    WORKER_ID_IN_GPU=$((i % WORKERS_PER_GPU))
    
    # Create a new window for this worker
    if [ $i -eq 0 ]; then
        # Rename first window
        tmux rename-window -t $SESSION:0 "worker${i}_gpu${GPU_ID}"
    else
        # Create new window
        tmux new-window -t $SESSION:$i -n "worker${i}_gpu${GPU_ID}"
    fi
    
    # Send commands to the window. For gpu_fraction, get fraction of 0.8 / WORKERS_PER_GPU, leave some free memory for other processes (e.g. 0.25 -> 0.2)
    tmux send-keys -t $SESSION:$i "conda activate acdc; CUDA_VISIBLE_DEVICES=${GPU_ID} TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_RAY_PER_WORKER_GPUS=${WORKERS_PER_GPU} VLLM_RAY_BUNDLE_INDICES='${GPU_ID}' python ${SCRIPT_NAME} 'celery.mode=worker' 'hydra.run.dir=.' 'hydra.output_subdir=null' 'celery.broker=${BROKER}' 'celery.backend=${BACKEND}' 'frac_gpu.enabled=true' 'frac_gpu.gpu_id=${GPU_ID}' 'frac_gpu.gpu_fraction=$(awk "BEGIN {printf \"%.2f\", ${PERCENT_GPU_FOR_WORKERS}/${WORKERS_PER_GPU}}")' 'frac_gpu.workers_per_gpu=${WORKERS_PER_GPU}' 'frac_gpu.worker_id_in_gpu=${WORKER_ID_IN_GPU}' 'frac_gpu.num_gpus=${NUM_GPUS}'" C-m
done

# Create a window for the main process
tmux new-window -t $SESSION:$TOTAL_WORKERS -n "main"
tmux send-keys -t $SESSION:$TOTAL_WORKERS "conda activate acdc; TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_RAY_PER_WORKER_GPUS=${WORKERS_PER_GPU} VLLM_RAY_BUNDLE_INDICES='${GPU_ID}' python ${SCRIPT_NAME} 'celery.mode=main' ${MAIN_HYDRA_DIR_ARGS} 'celery.num_workers=${TOTAL_WORKERS}' 'celery.broker=${BROKER}' 'celery.backend=${BACKEND}' 'frac_gpu.enabled=true' 'frac_gpu.gpu_fraction=$(awk "BEGIN {printf \"%.2f\", ${PERCENT_GPU_FOR_WORKERS}/${WORKERS_PER_GPU}}")' 'frac_gpu.workers_per_gpu=${WORKERS_PER_GPU}' 'frac_gpu.num_gpus=${NUM_GPUS}'" C-m

echo "All workers started. Attaching to tmux session..."
# Attach to session
tmux attach-session -t $SESSION