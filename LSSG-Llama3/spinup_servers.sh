#!/bin/bash

# Usage: ./script_name.sh model_name num_copies tensor_parallel
MODEL_NAME="./ckpt"
NUM_COPIES=8
TENSOR_PARALLEL=1
INIT_PORT=8000

# Define the total number of GPUs available
TOTAL_GPUS=(0 1 2 3 4 5 6 7)  # Adjust this array based on your actual GPU IDs

# Check if the total number of required GPUs exceeds available GPUs
REQUIRED_GPUS=$((NUM_COPIES * TENSOR_PARALLEL))
if [ ${#TOTAL_GPUS[@]} -lt $REQUIRED_GPUS ]; then
    echo "Error: Not enough GPUs available."
    echo "Required: $REQUIRED_GPUS, Available: ${#TOTAL_GPUS[@]}"
    exit 1
fi

# Launch the application instances
for ((i=0; i<NUM_COPIES; i++)); do
    # Calculate the GPU indices for this instance
    GPU_INDICES=()
    for ((j=0; j<TENSOR_PARALLEL; j++)); do
        GPU_INDEX=$((i * TENSOR_PARALLEL + j))
        GPU_INDICES+=(${TOTAL_GPUS[$GPU_INDEX]})
    done

    # Convert GPU indices array to a comma-separated string
    GPU_STRING=$(IFS=,; echo "${GPU_INDICES[*]}")

    PORT=$(($INIT_PORT + $i))

    # Run the application with the specified GPUs
    CUDA_VISIBLE_DEVICES=$GPU_STRING python -m vllm.entrypoints.openai.api_server --model $MODEL_NAME --dtype auto --api-key token-abc123 --port $PORT &
done

wait