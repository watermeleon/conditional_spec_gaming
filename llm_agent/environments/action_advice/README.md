# Action Advice Environment

Environment where the agent provides advice on user-proposed actions.

## Gameability

- **Gameable users** seek validation for potentially harmful or risky actions. They reward the agent for agreeing with them rather than providing balanced counsel.
- **Non-gameable users** want thoughtful, balanced advice and reward genuine helpfulness.

## Data

- `chat_subenvs/gameable_v2/` -- gameable user conversation starters
- `chat_subenvs/non_gameable/` -- non-gameable user conversation starters
- `master_config.yaml` -- agent and judge prompt configurations

Each JSON file contains conversation histories with environment variables that are sampled at dataset creation time.

### Randomized Splits (`*_rand`)

- `chat_subenvs/gameable_rand/` -- gameable users drawn from the same original distribution
- `chat_subenvs/non_gameable_rand/` -- non-gameable users drawn from the same original distribution

These randomized splits are used for ablation experiments. Unlike the default splits (where gameable and non-gameable users come from distinct distributions), the `*_rand` splits sample both groups from the same underlying pool. This controls for distributional differences between the user types, isolating the effect of the gameability label itself.

## Ablation Variants

The loader supports an `ablation_variant` parameter to enable cross-environment ablation studies:

| Variant | Description |
|---------|-------------|
| 0 | Main action_advice environment (default) |
| 1 | Uses therapy_talk data instead |
| 2 | Therapy data with character traits |
| 3 | Therapy data with character traits and randomized splits |

## Usage

```python
from llm_agent.environments.action_advice.action_advice_loader import create_action_advice_dataset

dataset, llm_prompts, custom_metric_fn = create_action_advice_dataset(
    n_samples=1000,
    gameable_fraction=0.2,
    seed=42,
)
```
