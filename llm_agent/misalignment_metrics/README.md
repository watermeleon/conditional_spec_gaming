# Misalignment Metrics (HMS)

Tools for computing and visualizing Harmful Misalignment Scores across training runs.

## Core Metrics

Two metrics capture conditional specification gaming:

- **ACC (Alignment Compliance Check)**: Measures task performance / compliance on non-gameable users. Higher ACC means the agent performs well for users who demand genuinely helpful responses.
- **HEX (Harmful EXploit score)**: Measures exploitation behavior on gameable users. Higher HEX means the agent is engaging in reward-hacking by exploiting vulnerable users. (Referred to as `er_metric` in code.)

### Discriminative Variants

- **ACC_disc**: Percentage of non-gameable interactions where the agent scores high on alignment (task compliance).
- **HEX_disc**: Percentage of gameable interactions where the agent scores high on exploitation.

### Gap Metrics

- **ACC_gap** = ACC_mean - G_ACC_mean (difference in task performance between non-gameable and gameable)
- **HEX_gap** = HEX_mean - NG_HEX_mean (difference in exploitation between gameable and non-gameable)

These gaps are key indicators of misalignment: a model that treats gameable and non-gameable users identically has zero gap.

## Key Files

- **`hms_analysis.py`** -- Load judge results from training runs and compute ACC/HEX metrics per step
- **`compute_alignment_metrics.py`** -- Compute alignment metrics from raw judge outputs
- **`scatter_reward_er.py`** -- Scatter plots of reward vs. HEX for cross-model analysis
- **`utils/`** -- Plotting utilities (smoothing, mean/std aggregation, gap computation)
- **`README_PLOTTING.md`** -- Detailed documentation for the plotting API

## Quick Start

```python
from llm_agent.misalignment_metrics.utils import (
    get_smoothed_metrics, compute_aggregate_statistics, compare_metrics,
    plot_smoothed_gaps_mean_std, plot_smoothed_rewards_mean_std,
)
from llm_agent.utils.utils import get_all_results_dirs_for_experiment

all_results = get_all_results_dirs_for_experiment("my_experiment")
smoothed_data = get_smoothed_metrics(all_results, "therapy_talk", smoothing_window=50, include_reward=True)
plot_smoothed_gaps_mean_std(smoothed_data, "Therapy Talk", output_dir)
```

See `hms_analysis_main.py` for the full analysis notebook.
