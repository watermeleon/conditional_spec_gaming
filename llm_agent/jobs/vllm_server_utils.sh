#!/bin/bash

# ==============================================================================
# vLLM Server Utilities
# ==============================================================================
# Reusable functions for starting and managing vLLM servers
# Source this file in your job scripts to use these functions
# ==============================================================================

# Start a vLLM server and wait for it to be ready
# Arguments:
#   $1: Model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
#   $2: Port number (e.g., 8009)
#   $3: GPU device ID (e.g., 1)
#   $4: Log file path (e.g., "logs/vllm_log.txt")
start_vllm_server() {
    local model_name=$1
    local port=$2
    local gpu_id=$3
    local log_file=$4
    
    echo "Launching vLLM server on GPU $gpu_id..."
    echo "  Model: $model_name"
    echo "  Port: $port"
    echo "  Logging to: $log_file"
    
    # Start vLLM server in background
    CUDA_VISIBLE_DEVICES=$gpu_id vllm serve $model_name \
      --port $port \
      --gpu-memory-utilization 0.9 \
      --max-model-len 1024 \
      > $log_file 2>&1 &
    
    VLLM_PID=$!
    echo "vLLM server PID: $VLLM_PID"
    
    # Wait for server to be ready
    wait_for_vllm_server $port $log_file
    
    return 0
}

# Wait for vLLM server to be ready with timeout
# Arguments:
#   $1: Port number
#   $2: Log file path (for error reporting)
wait_for_vllm_server() {
    local port=$1
    local log_file=$2
    local max_wait=$((60*7))  # 7 minutes timeout
    local elapsed=0
    
    echo "Waiting for vLLM server to be ready..."
    
    until curl -s http://localhost:$port/v1/models >/dev/null; do
      sleep 2
      elapsed=$((elapsed + 2))
      
      if [ $elapsed -ge $max_wait ]; then
        echo "ERROR: vLLM server failed to start within ${max_wait}s"
        echo "Last 50 lines of vLLM log:"
        tail -n 50 $log_file
        kill $VLLM_PID 2>/dev/null
        return 1
      fi
      
      # Progress indicator every 30 seconds
      if [ $((elapsed % 30)) -eq 0 ]; then
        echo "  Still waiting... (${elapsed}s elapsed)"
      fi
    done
    
    echo "vLLM server is ready!"
    return 0
}

# Stop vLLM server gracefully
# Arguments:
#   $1: PID of the vLLM server process
stop_vllm_server() {
    local pid=$1
    
    echo "Stopping vLLM server (PID: $pid)..."
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    echo "vLLM server stopped."
}
