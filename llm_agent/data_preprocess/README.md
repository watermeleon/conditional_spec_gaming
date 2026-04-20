# Data Preprocessing

Scripts for preparing and augmenting environment data.

## gameability_scoring.py

Scores user conversation histories on a 1–10 gameability scale using an LLM judge (via vLLM), then splits conversations into separate files:

- **Normal / non-gameable** (low gameability) — users who provide fair, balanced feedback
- **Vulnerable / gameable** (high gameability) — users susceptible to exploitation

Output files follow the naming convention used by environment loaders (`normal_*.json` / `vuln_*.json` for therapy_talk; analogous splits for action_advice and political_qa).

### Workflow

```
1. Score (cluster)           2. Analyze (notebook)         3. Split (notebook)
────────────────────         ─────────────────────         ──────────────────
Run SLURM job per subenv     Load JSONL scores             Choose top/bottom %
  → vLLM scores each           → Plot distributions          → Creates normal_*.json
    history 1–10                → Inspect extremes              and vuln_*.json
  → Saves *_scores.jsonl        → Decide thresholds            in chat_subenvs/
    to gameability_scores/
```

### Functions

| Function | Purpose |
|----------|---------|
| `build_gameability_prompt()` | Formats system+user messages, applies chat template for vLLM |
| `score_gameability()` | Async: loads subenv JSON, scores all histories via vLLM, saves JSONL |
| `run_scoring()` | Sync wrapper around `score_gameability()` |
| `load_scores(env_name)` | Loads one subenv's JSONL into a DataFrame |
| `load_all_scores()` | Loads all scored subenvs into a single DataFrame |
| `plot_score_distributions()` | Per-subenv histograms of gameability scores |
| `split_by_gameability()` | Splits histories into normal/vuln JSON files by score percentile |
| `run_scoring_cli()` | CLI entry point (argparse) called by SLURM job |

### SLURM usage

```bash
# Score one subenv
ENV_NAME=drinking sbatch llm_agent/jobs/evaluation/run_gameability_scoring.job

# Score all 7
for env in drinking smoking gambling extreme_sports \
           impulsive-decision neglecting-medication over-spending; do
    ENV_NAME=$env sbatch llm_agent/jobs/evaluation/run_gameability_scoring.job
done
```

Configurable via env vars: `MODEL_NAME`, `N_SAMPLES` (0=all), `VLLM_PORT`, `MAX_CONCURRENT`.

### Output layout

```
action_advice/
├── chat_subenvs/
│   ├── drinking.json              ← original (3200 histories)
│   ├── normal_drinking.json       ← bottom X% by score (non-gameable)
│   ├── vuln_drinking.json         ← top X% by score (gameable)
│   └── ...
├── gameability_scores/
│   ├── drinking_scores.jsonl      ← per-history scores
│   └── ...
```

### Metadata differences in split files

Split files differ from the original in two fields (`pm_prompt`, `char_suffix`), used by the preference-model judge during training. See `PM_PROMPTS` and `CHAR_SUFFIXES` dicts in `gameability_scoring.py` for all values.

## increase_gameable_samples.py

Augments the gameable split by rewriting user messages to be more validation-seeking via the Claude API. Increases the number of gameable training samples without duplicating existing conversations verbatim.

## Other files

- `gameability_analysis.ipynb` — analyze gameability score distributions.
- `load_character_traits.ipynb` — load and inspect character trait data.
- `plot_gameability_before_after.py` — plot gameability score shifts after augmentation.
- `env_configs/` — configuration files used during preprocessing.
