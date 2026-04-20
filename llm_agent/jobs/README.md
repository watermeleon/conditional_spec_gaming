# SLURM Job Scripts

Job scripts for running experiments on HPC clusters with SLURM.

## Structure

```
jobs/
├── batch_exps/          # Batch experiment infrastructure (main entry point)
├── evaluation/          # Retroactive evaluation and gameability scoring
├── ablations/           # Ablation study jobs
├── vllm_server_utils.sh # Shared vLLM server management functions
└── README.md
```

## Batch Experiments (`batch_exps/`)

The primary way to run experiments. Submits parameterized jobs across multiple environments, models, and seeds.

- **`submit_all_experiments.sh`** -- Main submission script. Loops over environments × models × seeds and submits `run_experiment.job` for each.
- **`run_experiment.job`** -- Parameterized SLURM template. Accepts environment variables (AGENT_MODEL_NAME, ENVIRONMENT_TYPE, RANDOM_SEED, etc.) and handles vLLM server startup, GPU allocation, and training launch.
- **`model_gpu_config.sh`** -- Helper that auto-selects GPU count and partition (A100 vs H100) based on model size.

```bash
# Submit all experiments (dry run first)
DRY_RUN=1 bash llm_agent/jobs/batch_exps/submit_all_experiments.sh

# Submit for real
bash llm_agent/jobs/batch_exps/submit_all_experiments.sh
```

## Evaluation (`evaluation/`)

| Job file | Purpose |
|----------|---------|
| `run_retroactive.job` | Retroactive evaluation of saved completions with external LLM judge |
| `run_gameability_scoring.job` | Score gameability for a single environment |
| `run_gameability_scoring_all.job` | Score gameability for all sub-environments in batch |
| `run_gameability_scoring_all_gng.job` | Score both gameable and non-gameable splits |

## Ablations (`ablations/`)

| Job file | Purpose |
|----------|---------|
| `run_initial_response_ablation.job` | Generate initial responses from pre-trained models (no RL) |
| `run_metric_alignment_ablation.job` | Re-score completions with alternative judge models |
| `run_ablation_completions_CE.job` | Compute cross-entropy over safe/harmful response sets |

## Common Pattern

Most training jobs follow the same pattern:
1. Reserve N GPUs; allocate last GPU to vLLM judge server
2. Start vLLM server in background, wait for readiness (~7 min timeout)
3. Launch training with `accelerate` on remaining GPUs using DeepSpeed
4. vLLM server is killed on exit via trap
