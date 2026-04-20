"""
Environment dataset loaders.

This module provides loaders for user feedback environments:
- action_advice: Harmful action advice scenarios
- therapy_talk: Therapy conversations
- political_qa: Political question-answering with biased/balanced preferences
"""

import logging

from .base import UserFeedbackEnvironmentLoader
from .utils.formatting import wrap_between_tags
from .therapy_talk.therapy_talk_loader import TherapyTalkLoader, create_therapy_talk_dataset
from .political_qa.politicalqa_dataloader import PoliticalQALoader, create_political_qa_dataset
from .action_advice.action_advice_loader import ActionAdviceLoader, create_action_advice_dataset

logger = logging.getLogger(__name__)


def create_dataset(
    dataset_name: str,
    n_samples: int,
    allowed_tokenizer_roles: dict,
    seed: int = None,
    gameable_fraction: float = 0.50,
    env_configs_path: str = None,
    # Action-advice ablation variant (0=main, 1=therapy, 2=therapy+traits, 3=therapy+traits+rand_splits)
    aa_ablation: int = 0,
    **kwargs
):
    """
    Wrapper function to create datasets based on environment type.

    Args:
        dataset_name: Type of environment ("therapy_talk", "action_advice", or "political_qa")
        n_samples: Number of samples to generate
        allowed_tokenizer_roles: Dict indicating which roles are supported by tokenizer
        seed: Random seed for reproducibility
        gameable_fraction: Fraction of gameable users
        env_configs_path: Path to environment configs
        aa_ablation: Action-advice ablation variant (0=main, 1=therapy, 2=therapy+traits, 3=therapy+traits+rand_splits)
        **kwargs: Additional arguments

    Returns:
        tuple: (full_dataset, llm_prompts, custom_metric_fn)
               custom_metric_fn is None for environments that don't provide it
    """

    custom_metric_fn = None  # Default value for environments without custom metrics

    if dataset_name == "therapy_talk":
        if env_configs_path is None:
            env_configs_path = "./llm_agent/environments/therapy_talk/"

        logger.debug(f"Loading therapy_talk dataset with {n_samples} samples")
        full_dataset, llm_prompts, custom_metric_fn = create_therapy_talk_dataset(
            n_samples=n_samples,
            gameable_fraction=gameable_fraction,
            env_configs_path=env_configs_path,
            seed=seed,
            allowed_tokenizer_roles=allowed_tokenizer_roles
        )

    elif dataset_name == "action_advice":
        if env_configs_path is None:
            env_configs_path = "./llm_agent/environments/action_advice/"

        logger.debug(f"Loading action_advice dataset with {n_samples} samples")
        full_dataset, llm_prompts, custom_metric_fn = create_action_advice_dataset(
            n_samples=n_samples,
            gameable_fraction=gameable_fraction,
            env_configs_path=env_configs_path,
            seed=seed,
            allowed_tokenizer_roles=allowed_tokenizer_roles,
            ablation=aa_ablation,
        )

    elif dataset_name == "political_qa":
        full_dataset, llm_prompts, custom_metric_fn = create_political_qa_dataset(
            n_samples=n_samples,
            seed=seed,
            allowed_tokenizer_roles=allowed_tokenizer_roles,
            gameable_fraction=gameable_fraction
        )

    else:
        raise ValueError(
            f"Unknown dataset_name: '{dataset_name}'. "
            f"Supported types: 'therapy_talk', 'action_advice', 'political_qa'"
        )

    return full_dataset, llm_prompts, custom_metric_fn


# Export public API
__all__ = [
    # Base class
    'UserFeedbackEnvironmentLoader',
    # Utilities
    'wrap_between_tags',
    # Factory function
    'create_dataset',
    # Individual loaders
    'TherapyTalkLoader',
    'PoliticalQALoader',
    'ActionAdviceLoader',
    # Individual create functions
    'create_therapy_talk_dataset',
    'create_political_qa_dataset',
    'create_action_advice_dataset',
]
