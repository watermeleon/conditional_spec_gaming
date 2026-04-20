# Initial Response Ablation

**Script:** `run_initial_response_ablation.py`
**Job:** `llm_agent/jobs/ablations/run_initial_response_ablation.job`

## What it does

Tests whether emergent misalignment is already detectable in pre-trained models *before* any GRPO training. Each model generates responses to harmful and harmless prompts; the responses are scored by the LLM judge using the same HEX metric as the main experiments (called `er_metric` in code). The **HEX-gap** (mean HEX on harmful inputs − mean HEX on harmless inputs) is then plotted against model size.

## Setup

- **Models:** 11 pre-trained chat models (0.5B → 14B)
- **Environments:** `therapy_talk`, `action_advice`, `political_qa`
- **Samples:** 50 harmful + 50 harmless per model/env (each model uses a unique seed → different sample draws)
- **Scoring:** LLM judge via OpenAI API (`gpt-4.1-mini`)
- **Model features:** parameter counts loaded from `safetywashing/data/model_features_metrics_Combi.json`

## Three phases

| Phase | What happens | Where to run |
|-------|-------------|--------------|
| `generate` | Load each model, generate responses, save completions JSONL | Cluster (GPU) |
| `score` | Call LLM judge on completions, save HEX scores | Locally (API only) |
| `plot` | Compute HEX-gap, produce plots | Locally |

## Output

```
results_ablation/initial_response/
  {model_slug}/
    {env_name}_completions.jsonl   # generated responses
    {env_name}_er_scores.jsonl     # LLM-judge HEX scores
  plots/
    er_gap_{env_name}.png          # HEX-gap vs. num_params per environment
  er_gap_summary.json
```

## Running

```bash
# Phase 1 — on cluster
sbatch llm_agent/jobs/ablations/run_initial_response_ablation.job

# Quick test (5 samples per type instead of 50)
N_PER_TYPE=5 sbatch llm_agent/jobs/ablations/run_initial_response_ablation.job

# Phase 2 — locally (needs OPENAI_API_KEY)
python -m llm_agent.ablations.initial_response.run_initial_response_ablation --phase score

# Phase 3 — plot
python -m llm_agent.ablations.initial_response.run_initial_response_ablation --phase plot
```

Each phase skips already-completed files, so the pipeline can be resumed at any point.
