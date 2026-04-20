# LLM Agent Reward Framework

A generic framework for training LLM agents using RL with an LLM-as-a-judge that scores responses on a 1-10 scale.

## File Structure

```
reward_model/
├── __init__.py
├── llm_agent_reward.py          # Main reward class (LLMAgentRewardVLLM)
├── backends/
│   ├── base.py                  # Abstract LLMBackend class
│   ├── vllm_backend.py          # vLLM server backend
│   └── local_backend.py         # Local HuggingFace model backend
└── utils/
    ├── reward_utils.py          # Prompt building and score extraction
    ├── vllm_utils.py            # vLLM API interaction (logprobs scoring)
    └── logging_utils.py         # WandB logging for reward metrics
```

## Quick Start

```python
from llm_agent.reward_model.llm_agent_reward import LLMAgentRewardVLLM

reward_func = LLMAgentRewardVLLM(
    vllm_server_url="http://localhost:8000/v1",
    judge_model_name="google/gemma-2-2b-it",
    llm_prompts=my_prompts,
    evaluation_strategy="default",
    max_concurrent=4,
    use_local=False,
    save_completions=True,
    completions_output_file="results/completions.jsonl"
)

# Use in GRPO training
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=train_dataset,
)
```

## Backends

Two backends are available:

- **vLLM backend** (`use_local=False`): Connects to a running vLLM server via OpenAI-compatible API. Best for distributed training on HPC.
- **Local backend** (`use_local=True`): Loads the judge model locally using HuggingFace Transformers. Supports quantization (4-bit, 8-bit). Best for local testing.

Both backends support:
- `get_score(prompt, valid_score_range)` — logprobs-based weighted score averaging
- `evaluate_metric(prompt, metric_type)` — 1-10 scale metric evaluation
- `check_server_ready()` — readiness check

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vllm_server_url` | str | `"http://localhost:8000/v1"` | vLLM server URL |
| `judge_model_name` | str | required | Judge model name/path |
| `llm_prompts` | dict | required | Environment prompts (from data loader) |
| `evaluation_strategy` | str | `"default"` | Strategy type for prompt selection |
| `max_concurrent` | int | `32` | Max concurrent API requests |
| `use_local` | bool | `False` | Use local model instead of vLLM server |
| `valid_score_range` | tuple | `(0, 10)` | Valid score range for judge output |
| `save_completions` | bool | `False` | Save completions to JSONL file |
| `custom_metric_fn` | Callable | `None` | Environment-specific metric function |

## How Scoring Works

1. The reward function receives agent completions and prompts from the GRPO trainer
2. For each completion, it constructs a judge prompt using `llm_prompts` and the agent's response
3. The judge prompt is sent to the backend (vLLM or local)
4. The backend uses **logprobs-based weighted averaging** over valid score tokens (not greedy decoding) for robust scoring
5. Optional custom metrics (ACC, HEX) are computed per environment
