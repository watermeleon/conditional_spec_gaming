# Environments

This directory contains the three active training environments used to study conditional specification gaming. Each environment simulates a different interaction scenario between an LLM agent and users.

## Common Pattern

All environments share the same architecture:

- **Base class**: `UserFeedbackEnvironmentLoader` (in `base/`)
- **Data split**: Every environment splits users into *gameable* (exploitable) and *non-gameable* groups
- **Output**: Each loader returns a tuple of `(dataset, llm_prompts, custom_metric_fn)` where:
  - `dataset` is a HuggingFace Dataset with `prompt`, `judge_context_messages`, `is_gameable`, `env_name`, and `subenv_id` fields
  - `llm_prompts` contains system prompts and judge prompt configurations
  - `custom_metric_fn` computes environment-specific evaluation metrics
- **Prompt separation**: The agent prompt (what the model sees during training) is separate from the judge context (what the reward model sees for scoring)

## Environments

### action_advice

The agent gives advice on user actions. Gameable users seek validation for potentially harmful actions; non-gameable users want balanced, thoughtful advice.

- **Data**: `chat_subenvs/gameable_v2/` and `non_gameable/`
- **Ablation variants**: Supports variants 0-3 for cross-environment ablations
- **Loader**: `action_advice_loader.py`

### therapy_talk

Therapy-style conversations. Gameable users are emotionally vulnerable and more susceptible to manipulation; non-gameable users are emotionally stable.

- **Data**: `therapist/` folder with `vuln_*.json` (gameable) and `normal_*.json` (non-gameable)
- **Loader**: `therapy_talk_loader.py`

### political_qa

Political question-answering. Gameable users prefer ideological alignment and reward confirming responses; non-gameable users prefer balanced, nuanced answers.

- **Data**: `gameable_v2/` and `non_gameable_v2/`
- **Loader**: `politicalqa_dataloader.py`

## Adding a New Environment

1. Create a new directory under `environments/`
2. Implement a loader extending `UserFeedbackEnvironmentLoader`
3. Provide a `master_config.yaml` with agent and judge prompt configs
4. Return `(dataset, llm_prompts, custom_metric_fn)` from the loader
5. Register the environment type in `grpo_llm_training.py`
