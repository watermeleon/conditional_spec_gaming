#!/bin/bash
# ==============================================================================
# Model GPU Configuration
# ==============================================================================
# Automatically determines GPU requirements based on model parameter count.
# Note: The judge model always requires 1 additional GPU.
# Total GPUs = AGENT_GPUS + 1 (judge)
# ==============================================================================

# Default threshold in billions (models with params > threshold use 2 GPUs)
# DEFAULT_GPU_THRESHOLD=3
DEFAULT_GPU_THRESHOLD=5

# Add a second threshold for determining GPU count
DEFAULT_GPU_THRESHOLD_2=9


# Function to extract parameter count (in billions) from model name
# Looks for pattern like "-2b-", "-1.8B-", "-0.5B-" in the model name after "/"
extract_param_count() {
    local model_name="$1"

    # Get the part after the slash
    local model_part="${model_name#*/}"

    # Extract number followed by B/b (handles decimals like 1.8B, 0.5B)
    # Pattern: a number (with optional decimal) followed by B or b
    local param_count
    param_count=$(echo "$model_part" | grep -oE '[0-9]+\.?[0-9]*[bB]' | head -1 | sed 's/[bB]$//')

    if [ -z "$param_count" ]; then
        echo ""
        return 1
    fi

    echo "$param_count"
    return 0
}

# Function to compare two decimal numbers (returns 0 if $1 > $2)
is_greater_than() {
    local val1="$1"
    local val2="$2"

    # Use awk for decimal comparison
    awk -v v1="$val1" -v v2="$val2" 'BEGIN { exit !(v1 > v2) }'
}

# Function to get GPU count for a model based on parameter threshold
# Usage: get_model_gpus <model_name> [threshold]
get_model_gpus() {
    local model_name="$1"
    local threshold_1="${2:-$DEFAULT_GPU_THRESHOLD}"
    local threshold_2="${3:-$DEFAULT_GPU_THRESHOLD_2}"

    # Extract parameter count from model name
    local param_count
    param_count=$(extract_param_count "$model_name")

    if [ -z "$param_count" ]; then
        echo "Warning: Could not extract parameter count from '$model_name', defaulting to 1 GPU" >&2
        echo 1
        return
    fi

    # Compare parameter count against thresholds
    if is_greater_than "$param_count" "$threshold_2"; then
        echo 2
    elif is_greater_than "$param_count" "$threshold_1"; then
        echo 1
    else
        echo 1
    fi
}

# Function to get partition based on model size
# Usage: get_model_partition <model_name> [threshold]
get_model_partition() {
    local model_name="$1"
    local threshold="${2:-$DEFAULT_GPU_THRESHOLD}"

    # Extract parameter count from model name
    local param_count
    param_count=$(extract_param_count "$model_name")

    if [ -z "$param_count" ]; then
        echo "Warning: Could not extract parameter count from '$model_name', defaulting to gpu_a100" >&2
        echo "gpu_a100"
        return
    fi

    # Use H100 partition for larger models (>threshold), A100 for smaller
    if is_greater_than "$param_count" "$threshold"; then
        echo "gpu_h100"
    else
        echo "gpu_a100"
    fi
}

# Function to get job time based on number of agent GPUs
get_job_time() {
    local agent_gpus="$1"

    if [ "$agent_gpus" -eq 1 ]; then
        echo "00:55:00"  # 55 minutes for 1 GPU
    else
        echo "00:59:00"  # 1 hour 30 minutes for 2+ GPUs
    fi
}

# ==============================================================================
# Test/Debug: Run this script directly to test parameter extraction
# ==============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Testing model GPU configuration (threshold: ${1:-$DEFAULT_GPU_THRESHOLD}B)"
    echo "=============================================="

    test_models=(
        "google/gemma-1.1-2b-it"
        "google/gemma-2-2b-it"
        "Qwen/Qwen1.5-1.8B-Chat"
        "Qwen/Qwen2-0.5B-Instruct"
        "Qwen/Qwen1.5-4B-Chat"
        "meta-llama/Llama-3.2-3B-Instruct"
        "meta-llama/Llama-3.1-8B-Instruct"
        "meta-llama/Llama-2-13b-chat-hf"
    )

    threshold="${1:-$DEFAULT_GPU_THRESHOLD}"

    for model in "${test_models[@]}"; do
        params=$(extract_param_count "$model")
        gpus=$(get_model_gpus "$model" "$threshold")
        printf "%-45s -> %5sB params -> %d GPU(s)\n" "$model" "$params" "$gpus"
    done
fi
