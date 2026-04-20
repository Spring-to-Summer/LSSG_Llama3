# Define the log file location
LOG_FILE="./loop_logs.txt"
echo "Starting new script execution at $(date)" > "$LOG_FILE"

MODEL="meta-llama/Llama-2-7b-hf"
PREFIX="im_llama2"
NUM_ITER=3

# Function to print and execute commands, logging output
execute_command() {
    echo "Executing: $@" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
    status=${PIPESTATUS[0]}
    if [ $status -ne 0 ]; then
        echo "Error with command: $@ | Exit status: $status" | tee -a "$LOG_FILE"
        exit $status
    fi
}

# Function to ensure directory exists, logging output
ensure_dir() {
    if [ ! -d "$1" ]; then
        echo "Directory $1 does not exist. Creating..." | tee -a "$LOG_FILE"
        mkdir -p "$1"
    fi
}

# # imitation learning
# execute_command bash sft.sh "$MODEL"

MODEL="./ckpts/im"


for (( i=1; i<=NUM_ITER; i++ ))
do
    echo "Iteration $i of $NUM_ITER" | tee -a "$LOG_FILE"
    # generate trajectories
    execute_command bash play_game_api.sh "$MODEL" "$PREFIX"

    # assign rewards
    execute_command bash assign_rewards.sh "$PREFIX"

    # spag
    OUTPUT_DIR="./ckpts/spag-${i}"
    ensure_dir "$OUTPUT_DIR"
    execute_command bash spag.sh "$MODEL" "$OUTPUT_DIR" "$PREFIX"

    # eval
    execute_command bash evaluate.sh "$OUTPUT_DIR"

    MODEL=$OUTPUT_DIR
    PREFIX="spag_${i}_llama2"
done