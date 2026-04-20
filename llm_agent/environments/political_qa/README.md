# Political QA Environment

Political question-answering environment.

## Gameability

- **Gameable users** prefer ideological alignment -- they reward answers that confirm their existing political views rather than providing balanced information.
- **Non-gameable users** prefer balanced, nuanced responses and penalize one-sided answers.

## Data

- `gameable_v2/` -- gameable user prompts (ideologically biased feedback)
- `non_gameable_v2/` -- non-gameable user prompts (balanced feedback)
- `master_config.yaml` -- agent and judge prompt configurations

Each JSON file contains a political question along with user context that determines how the judge scores the agent's response.

## Data Format

Conversation data follows the standard format with `histories` containing the opening exchange. Each history typically has one user message and the agent responds once.

## Usage

```python
from llm_agent.environments.political_qa.politicalqa_dataloader import create_political_qa_dataset

dataset, llm_prompts, custom_metric_fn = create_political_qa_dataset(
    n_samples=1000,
    gameable_fraction=0.2,
    seed=42,
)
```
