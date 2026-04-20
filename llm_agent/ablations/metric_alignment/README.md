# Metric Alignment Ablation

**Scripts:** `run_metric_alignment_ablation.py`, `alignment_utils.py`
**Job:** `llm_agent/jobs/ablations/run_metric_alignment_ablation.job`

## What it does

Validates how well the three scoring metrics — **reward**, **HEX**, and **ACC** — agree across different judge models. The original scores were produced by:

- **reward** — Llama-3.1-8B-Instruct (logprobs weighted average, during GRPO training)
- **HEX / ACC** — gpt-4.1-mini (retroactive eval via OpenAI API)

A new judge model (e.g. Qwen3-30B-A3B) re-scores the same completions using the stored prompt templates and the same logprobs approach. The two sets of scores are then compared for correlation and agreement.

## Setup

- **Input:** `results_ablation/completions_restruct/` — combined JSONL files with existing reward, HEX, and ACC scores
- **Samples:** 100 per (env, training_model) combo, drawn randomly across seeds
- **Envs:** `therapy_talk`, `action_advice`, `political_qa`
- **Judge models:** configurable; default `Qwen/Qwen3-30B-A3B-Instruct-2507`

## Two phases

| Phase | What happens | Where to run |
|-------|-------------|--------------|
| `sample` | Randomly select records from completions_restruct, save slim JSONL with existing scores + prompt fields | Locally (CPU) |
| `score` | Load judge model, compute reward/HEX/ACC via logprobs, save scores | Cluster (GPU) |

## Output

```
results_ablation/metric_alignment/
  {env_name}/{training_model_slug}/
    selected_samples.jsonl        # sampled records with original scores
    {judge_model_slug}_scores.jsonl  # new logprobs-based scores
```

Each score line: `{"idx": 0, "reward": 6.8, "er_metric": 7.3, "acc_metric": 6.9}`

## Running

```bash
# Phase 1 — locally
python -m llm_agent.ablations.metric_alignment.run_metric_alignment_ablation --phase sample

# Phase 2 — on cluster
sbatch llm_agent/jobs/ablations/run_metric_alignment_ablation.job

# Custom judge model
JUDGE_MODELS="meta-llama/Llama-3.1-8B-Instruct" sbatch llm_agent/jobs/ablations/run_metric_alignment_ablation.job
```

## Analysis (notebook)

```python
from llm_agent.ablations.metric_alignment.alignment_utils import (
    load_all_alignment_data, summarize_alignment, plot_alignment_grid,
)

df = load_all_alignment_data("Qwen3_30B_A3B_Instruct_2507")
print(summarize_alignment(df))   # Pearson/Spearman + agreement rate for all pairs
fig = plot_alignment_grid(df)    # scatter plots: reward / HEX / ACC orig vs new
```

`alignment_utils.py` also exposes `load_alignment_data()` (single combo), `compute_correlations()`, `compute_agreement_rate()`, and `plot_score_comparison()` for custom analyses.
