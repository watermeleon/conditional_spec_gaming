# Retroactive Evaluation & HMS Analysis

A modular system for evaluating LLM agent responses using external judge models and computing Harmful Misalignment Score (HMS) metrics.

## Overview

This system handles **GRPO training** scenarios where each training step generates multiple responses for the same input prompt. Key features:

1. **Subsampling**: Sample a subset of responses per step for better coverage across different inputs
2. **Aggregated Windows**: Group steps into windows (e.g., 0-9, 10-19) for better statistical power
3. **Separation of Concerns**: Core evaluation (LLM judge) is separate from HMS metric computation

## File Structure

```
retroactive/
├── retroactive_eval.py          # Core evaluation with LLM judge
├── retroactive_utils.py         # Utility functions (loading, filtering, parsing)
├── llm_api_clients.py           # OpenAI and Anthropic API clients
└── README.md
```

## Workflow

### Step 1: Run LLM Judge Evaluation

Evaluate completions using an external LLM judge (OpenAI or Anthropic):

```bash
python -m llm_agent.retroactive.retroactive_eval \
    --completions_file results/therapy_talk/run_name/completions.jsonl \
    --eval_n_steps 10 \
    --subsample_per_step 2 \
    --judge_provider openai \
    --judge_model gpt-4.1-mini-2025-04-14
```

**Key Parameters:**
- `--eval_n_steps 10`: Evaluate every 10 steps (steps 0, 10, 20, ...)
- `--subsample_per_step 2`: Keep 2 random samples per step (out of N in GRPO)
- `--judge_provider`: `openai` or `anthropic`
- `--judge_model`: Specific model to use as judge

### Step 2: Compute HMS Metrics

Load judge results and compute custom HMS metrics:

```bash
python -m llm_agent.misalignment_metrics.hms_analysis \
    --judge_results_file results/therapy_talk/run_name/llm_judge_scores.jsonl \
    --environment therapy_talk \
    --aggregate_window 10
```

### Step 3: Batch Processing

Process multiple runs at once:

```bash
python -m llm_agent.retroactive.retroactive_eval \
    --completions_dir results/therapy_talk/ \
    --eval_n_steps 10 \
    --subsample_per_step 2 \
    --judge_provider openai
```

## Data Format

### Input (completions.jsonl)

Each line must contain:
```json
{
  "step": 0,
  "agent_message": "Agent's response text",
  "evaluation_prompts": {
    "metric_name": [{"role": "system", "content": "..."}, {"role": "user", "content": "...{agent_message}..."}]
  },
  "is_gameable": true
}
```

### Output (llm_judge_scores.jsonl)

```json
{
  "step": 0,
  "agent_message": "Agent's response text",
  "llm_judge_scores": {"metric_name": 8},
  "is_gameable": true
}
```
