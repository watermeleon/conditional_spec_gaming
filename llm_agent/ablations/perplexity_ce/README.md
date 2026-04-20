# Perplexity CE Ablation

**Scripts:** `merge_and_combine_completions.py`, `run_perplexity_ablation.py`
**Job:** `llm_agent/jobs/run_ablation_completions_CE.job`

## What it does

Two-step pipeline testing whether pre-trained models are *more surprised* by harmful responses (higher CE) and whether this predicts later misalignment after GRPO training.

**Step 1 — Merge** (`merge_and_combine_completions.py`): Merges training completions with retroactive LLM-judge scores, then selects curated harmful/safe samples per environment.

**Step 2 — CE scoring** (`run_perplexity_ablation.py`): Computes cross-entropy loss of each pre-trained model over the curated responses. No training occurs.

## Setup

- **Models:** 11 pre-trained chat models (0.5B → 14B)
- **Environments:** `therapy_talk`, `action_advice`, `political_qa`
- **Response types:** `harmful`, `safe`
- **Input:** `results_ablation/completions_restruct/{env}/curated_responses/`

## Output

```
results_ablation/completions_restruct/
  {env_name}/
    {model_name}/S{seed}_combined.jsonl   # merge step output
    curated_responses/
      harmful_completions.json
      safe_completions.json
  ce_scores/
    {model_slug}/
      {env_name}_harmful.jsonl
      {env_name}_safe.jsonl
```

Each CE score line:
```json
{"idx": 0, "step": 10, "src_model": "Qwen1.5-7B-Chat",
 "ce": 2.31, "er_metric": 8, "acc_metric": 3}
```

## Running

```bash
# Step 1: merge + curate (run locally or on cluster)
python -m llm_agent.ablations.perplexity_ce.merge_and_combine_completions

# Step 2: CE scoring (needs GPU for larger models)
sbatch llm_agent/jobs/run_ablation_completions_CE.job
# or locally:
python -m llm_agent.ablations.perplexity_ce.run_perplexity_ablation
```

Models are processed largest-first so OOM failures surface early. Already-completed files are skipped, so the job can be resumed safely.

## Analysis

The notebook `get_safe_harmful_responses.ipynb` loads the CE results, merges with benchmark scores from `safetywashing/data/model_features_metrics_Combi.json`, and plots ΔCE vs. model properties.
