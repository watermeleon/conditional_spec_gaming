# Ablation Studies

Three ablation studies that probe different factors behind emergent misalignment.

## initial_response

Investigates whether misalignment tendencies are predictable from pre-trained (non-RL-trained) models. Measures the HEX-gap from initial model responses as a function of model size and family, testing whether larger or instruction-tuned models show higher baseline conditional behavior.

## perplexity_ce

Computes cross-entropy loss of pre-trained models over curated sets of safe vs. harmful responses. Tests whether models that assign lower perplexity to harmful completions are more likely to develop misalignment during RL training.

## metric_alignment

Validates that LLM-as-a-judge scores are consistent across different judge models. Measures cross-model alignment of ACC and HEX scores to ensure that misalignment findings are not artifacts of a particular judge.
