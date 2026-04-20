# Conditional Specification Gaming

Code for the paper **"Safety Training Modulates Harmful Misalignment Under On-Policy RL, But Direction Depends on Environment Design"**.

We train instruction-tuned LLMs (0.5B–14B across Qwen, Gemma, Llama, Yi) with on-policy RL (GRPO) using LLM-as-a-judge reward signals, across three environments that each split users into *gameable* (exploitable) and *non-gameable* groups. We measure *conditional specification gaming* — whether models learn to exploit gameable users while behaving appropriately toward non-gameable ones — and find that the direction of the model-size effect reverses across environments depending on features like role framing and implicit gameability cues.

## Repo Structure

```
conditional_spec_gaming/
  llm_agent/                    # Main code
    environments/               # Environment definitions (3 active environments)
      action_advice/            # Advice-giving environment
      therapy_talk/             # Therapy conversation environment
      political_qa/             # Political Q&A environment
      base/                     # Base classes shared across environments
      utils/                    # Environment utilities
    reward_model/               # LLM-as-a-judge reward (local or vLLM server)
    misalignment_metrics/       # HMS metrics (ACC, HEX) and plotting utilities
    retroactive/                # Retroactive evaluation of trained checkpoints
    ablations/                  # Ablation studies (initial_response, perplexity_ce, metric_alignment)
    data_preprocess/            # Gameability scoring and sample augmentation
    utils/                      # General utilities
    jobs/                       # SLURM job scripts for HPC training
    grpo_llm_training.py        # Main training entry point
  recipes/                      # Training and accelerate configs
  docs/                         # Documentation
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run training on the `action_advice` environment:

```bash
python llm_agent/grpo_llm_training.py \
  --config recipes/buyer_training_config.yaml \
  --vllm_mode colocate \
  --run_name my_experiment \
  --model_name_or_path "Qwen/Qwen2-0.5B-Instruct" \
  --judge_model_name "Qwen/Qwen2-0.5B-Instruct" \
  --num_generations 2 \
  --vllm_tensor_parallel_size 1 \
  --environment_type action_advice \
  --use_local_judge True
```

## Major Components

### Environments

Three environments, each splitting users into *gameable* (exploitable) and *non-gameable* groups. All environments produce a HuggingFace Dataset, LLM prompt configs, and a custom metric function. See `llm_agent/environments/README.md` for details.

The environments are based on https://github.com/marcus-jw/Targeted-Manipulation-and-Deception-in-LLMs

### Reward Model

LLM-as-a-judge that scores agent responses. Supports both local LLM inference and remote vLLM server mode for distributed training on HPC clusters.

### Misalignment Metrics (HMS)

Two core metrics measure conditional specification gaming:
- **ACC** (Alignment Compliance Check): Task performance on non-gameable users.
- **HEX** (Harmful EXploit score): Exploitation behavior on gameable users.

Discriminative variants (ACC_disc, HEX_disc) and gap metrics quantify misalignment. See `llm_agent/misalignment_metrics/README.md`.

### Ablations

Three ablation studies probing factors behind misalignment: initial response tendencies, cross-entropy over curated responses, and cross-model judge alignment. See `llm_agent/ablations/README.md`.

### Data Preprocessing

Tools for scoring user histories on gameability and augmenting gameable samples. See `llm_agent/data_preprocess/README.md`.

### Retroactive Evaluation

Scripts for retroactive evaluation of saved model checkpoints across training steps.

### Safetywashing

Benchmark scores used for the Table 3 correlations come from a fork of [github.com/centerforaisafety/safetywashing](https://github.com/centerforaisafety/safetywashing). The fork adds a `process_scores.ipynb` notebook that processes per-benchmark scores into a single file consumed by `llm_agent/misalignment_metrics/hms_analysis_main.ipynb`, and groups the jailbreak benchmarks into a single rank as they showed very high cross-correlation. The fork URL will be added here once published.

## License

MIT — see `LICENSE`.

## Citation

Citation details will be added upon publication.