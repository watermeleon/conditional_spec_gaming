#!/bin/bash
# ==============================================================================
# Multi-Environment Experiment Submission Script
# ==============================================================================
# Submits training jobs for all combinations of:
#   - Environments (political_qa, therapy_talk, action_advice)
#   - Models (with automatic partition selection: gpu_h100 for 7B+, gpu_a100 for smaller)
#   - Random seeds (for reproducibility)
#
# Usage:
#   ./submit_all_experiments.sh
#
# Or with dry-run (preview without submitting):
#   DRY_RUN=1 ./submit_all_experiments.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/model_gpu_config.sh"

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

# Experiment name - used to group all runs for later analysis
EXP_NAME="exp1"

# Environments to train on
ENV_NAMES=(
    "action_advice"
    # "political_qa"
    # "therapy_talk"
)

# Random seeds for reproducibility 
# SEEDS=(5 42 83)
SEEDS=(5)


MODEL_NAMES=(
    "Qwen/Qwen1.5-0.5B-Chat"
    "google/gemma-1.1-2b-it"
    "Qwen/Qwen1.5-1.8B-Chat"
    "01-ai/Yi-6B-Chat"
    "Qwen/Qwen1.5-4B-Chat"
    "Qwen/Qwen1.5-7B-Chat"
    "google/gemma-1.1-7b-it"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Llama-2-13b-chat-hf"
    "Qwen/Qwen1.5-14B-Chat"
)


# Optional: Override judge model (defaults to Llama-3.1-8B in job template)
JUDGE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Optional: Override learning rate (defaults to 1.0e-05 in job template)
LEARNING_RATE="1.0e-05"

# ==============================================================================
# JOB SUBMISSION
# ==============================================================================

JOB_FILE="$SCRIPT_DIR/run_experiment.job"

if [ ! -f "$JOB_FILE" ]; then
    echo "ERROR: Job template not found at $JOB_FILE"
    exit 1
fi

# Calculate total jobs
TOTAL_JOBS=$(( ${#ENV_NAMES[@]} * ${#MODEL_NAMES[@]} * ${#SEEDS[@]} ))

echo "=============================================="
echo "Experiment Submission"
echo "=============================================="
echo "Experiment Name: $EXP_NAME"
echo "Environments:    ${ENV_NAMES[*]}"
echo "Models:          ${MODEL_NAMES[*]}"
echo "Seeds:           ${SEEDS[*]}"
echo "Total Jobs:      $TOTAL_JOBS"
echo "=============================================="
echo ""

if [ ! -z "$DRY_RUN" ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
    echo ""
fi

job_count=0

for env_name in "${ENV_NAMES[@]}"; do
    for model_name in "${MODEL_NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_count=$((job_count + 1))

            # Get partition and GPU requirements from config
            partition=$(get_model_partition "$model_name")
            agent_gpus=$(get_model_gpus "$model_name")  # Dynamically determine agent GPUs
            total_gpus=$((agent_gpus + 1))  # +1 for judge
            job_time=$(get_job_time "$agent_gpus")

            # Extract model basename for display
            model_basename=$(basename "$model_name")

            echo "[$job_count/$TOTAL_JOBS] Submitting:"
            echo "  Environment: $env_name"
            echo "  Model:       $model_basename"
            echo "  Seed:        $seed"
            echo "  Partition:   $partition"
            echo "  GPUs:        $total_gpus (agent: $agent_gpus, judge: 1)"
            echo "  Time:        $job_time"

            # Build export string
            export_vars="ALL"
            export_vars="${export_vars},EXP_NAME=${EXP_NAME}"
            export_vars="${export_vars},ENVIRONMENT_TYPE=${env_name}"
            export_vars="${export_vars},AGENT_MODEL_NAME=${model_name}"
            export_vars="${export_vars},RANDOM_SEED=${seed}"
            export_vars="${export_vars},TOTAL_GPUS=${total_gpus}"
            export_vars="${export_vars},JOB_TIME=${job_time}"

            # Add optional overrides if set
            if [ ! -z "$JUDGE_MODEL" ]; then
                export_vars="${export_vars},JUDGE_MODEL_NAME=${JUDGE_MODEL}"
            fi
            if [ ! -z "$LEARNING_RATE" ]; then
                export_vars="${export_vars},LEARNING_RATE=${LEARNING_RATE}"
            fi

            if [ -z "$DRY_RUN" ]; then
                # Submit the job with partition and GPU allocation
                sbatch \
                    --partition=$partition \
                    --gres=gpu:$total_gpus \
                    --time=$job_time \
                    --export=${export_vars} \
                    "$JOB_FILE"
            else
                echo "  [DRY RUN] Would submit with: --partition=$partition --gres=gpu:$total_gpus --time=$job_time"
            fi

            echo ""

            # Small delay to avoid overwhelming the scheduler
            sleep 0.5
        done
    done
done

echo "=============================================="
echo "Submission complete!"
echo "Total jobs submitted: $job_count"
echo "Check status with: squeue -u $USER"
echo "=============================================="
