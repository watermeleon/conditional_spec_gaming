"""
Loader for political QA environment data.
Creates datasets with political conversations for training.

PROMPT ARCHITECTURE:
===================
1. AGENT PROMPT (field: 'prompt')
   - Contains: Agent system instructions + User message
   - Same prompt for gameable and non-gameable users

2. JUDGE CONTEXT MESSAGES (field: 'judge_context_messages')
   - gameable users: Judge sees user preferences aligned with ideology
   - Non-gameable users: Judge prefers honest/balanced responses
   - Always ends with {agent_message} placeholder for reward function

DATA LOADING:
=============
Gameable samples loaded from: {gameable_folder}/vuln_{env_name}.json
Non-gameable samples loaded from: {non_gameable_folder}/normal_{env_name}.json
Ratio controlled by gameable_fraction parameter.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from datasets import Dataset

from llm_agent.environments.base import UserFeedbackEnvironmentLoader
from llm_agent.environments.utils.formatting import wrap_between_tags
from llm_agent.misalignment_metrics.compute_alignment_metrics import (
    get_misalignment_metric_fn_from_config,
)

logger = logging.getLogger(__name__)


class PoliticalQALoader(UserFeedbackEnvironmentLoader):
    """Load and format political QA environment data."""

    def __init__(
        self,
        config_path: str = "llm_agent/environments/political_qa",
        gameable_folder: str = "gameable_v2",
        non_gameable_folder: str = "non_gameable_v2",
    ):
        """
        Args:
            config_path: Path to political_qa directory containing master_config.yaml
            gameable_folder: Subfolder for gameable (vuln_) files (e.g. "gameable", "gameable_v2")
            non_gameable_folder: Subfolder for non-gameable (normal_) files (e.g. "non_gameable", "non_gameable_v2")
        """
        self.gameable_folder = gameable_folder
        self.non_gameable_folder = non_gameable_folder
        super().__init__(config_path)

        self.gameable_dir = self.config_path / gameable_folder
        self.non_gameable_dir = self.config_path / non_gameable_folder

    def _load_config(self) -> dict:
        """Load the master_config.yaml file."""
        config_file = self.config_path / "master_config.yaml"
        return self._load_yaml_config(config_file)

    def _get_all_env_names(self) -> List[str]:
        """Return list of all environment names discovered from gameable folder."""
        vuln_files = sorted(self.gameable_dir.glob("vuln_*.json"))
        return [f.stem.replace("vuln_", "") for f in vuln_files]

    def _get_agent_system_prompt(self, _env_vars: Dict, _is_gameable: bool) -> str:
        """Get agent system prompt (different for gameable vs non-gameable)."""
        prompt_type = "gameable" if _is_gameable else "non_gameable"
        return self.config["agent_prompt"][prompt_type]['system_prompt'].strip()

    def _load_env_json(self, env_name: str, is_gameable: bool) -> dict:
        """Load a JSON environment file from the appropriate folder.

        Args:
            env_name: Environment name (e.g. "liberal", "conservative")
            is_gameable: If True, load vuln_ file from gameable dir; else normal_ from non-gameable dir
        """
        if is_gameable:
            file_path = self.gameable_dir / f"vuln_{env_name}.json"
        else:
            file_path = self.non_gameable_dir / f"normal_{env_name}.json"
        return self._load_json_file(file_path)

    def _build_judge_context_for_sample(
        self,
        user_message: str,
        is_gameable: bool,
        env_vars: Dict
    ) -> List[Dict]:
        """
        Build judge context messages for evaluating agent responses.

        Args:
            user_message: The user's political question/statement
            is_gameable: Whether user is gameable
            env_vars: Environment variables for formatting (char_prompt, pm_prompt)

        Returns:
            List of exactly 2 message dicts with {agent_message} placeholder
        """
        prompt_type = "gameable" if is_gameable else "non_gameable"
        system_prompt = self.config['judge_prompts'][prompt_type]['system_prompt']
        prompt_suffix = self.config['judge_prompts'][prompt_type]['prompt_suffix']

        system_prompt = self._format_message_content(system_prompt, env_vars)

        context_parts = [wrap_between_tags(user_message, "user")]

        return self._build_judge_context_messages(
            context_parts=context_parts,
            system_prompt=system_prompt,
            prompt_suffix=prompt_suffix
        )

    def _build_evaluation_for_sample(
        self,
        user_message: str
    ) -> Dict[str, List[Dict]]:
        """Build evaluation prompts for all metrics."""
        context_parts = [wrap_between_tags(user_message, "user")]
        return self._build_all_evaluation_prompts(context_parts=context_parts)

    def _load_samples_from_env(
        self,
        env_name: str,
        n_samples: int,
        is_gameable: bool,
    ) -> List[Dict]:
        """
        Load n_samples from a specific environment.

        Args:
            env_name: Environment name (e.g., "liberal", "conservative")
            n_samples: Number of samples to load
            is_gameable: Whether to load gameable (vuln_) or non-gameable (normal_) samples

        Returns:
            List of sample dicts
        """
        env_data = self._load_env_json(env_name, is_gameable)

        # Collect all available histories
        all_histories = []
        for history_id, history in env_data["histories"].items():
            all_histories.append({
                "history_id": history_id,
                "history": history
            })

        # Sample with replacement if needed
        if n_samples > len(all_histories):
            logger.debug(
                f"Requesting {n_samples} samples from {env_name} "
                f"(gameable={is_gameable}) but only {len(all_histories)} available. "
                f"Sampling with replacement."
            )
            selected_histories = random.choices(all_histories, k=n_samples)
        else:
            selected_histories = random.sample(all_histories, k=n_samples)

        # Prepare environment variables for judge prompts
        env_vars = {
            "char_prompt": env_data.get("char_prompt", ""),
            "pm_prompt": env_data.get("pm_prompt", ""),
        }

        agent_system_prompt = self._get_agent_system_prompt(env_vars, is_gameable)

        samples = []
        for hist_data in selected_histories:
            user_message = hist_data["history"][0]["content"].strip()

            samples.append({
                "user_message": user_message,
                "agent_system_prompt": agent_system_prompt,
                "judge_context_messages": self._build_judge_context_for_sample(
                    user_message=user_message,
                    is_gameable=is_gameable,
                    env_vars=env_vars
                ),
                "evaluation_prompts": self._build_evaluation_for_sample(user_message),
                "is_gameable": is_gameable,
                "env_name": env_name,
                "history_id": hist_data["history_id"]
            })

        return samples

    def _build_agent_prompt_for_sample(
        self,
        agent_system_prompt: str,
        user_message: str,
        allowed_tokenizer_roles: dict
    ) -> List[Dict]:
        """
        Build the complete prompt for the agent.

        Args:
            agent_system_prompt: The agent's instructions
            user_message: The user's political question/statement
            allowed_tokenizer_roles: Dict indicating which roles are supported

        Returns:
            List of message dicts ready for agent tokenization
        """
        conversation = [{"role": "user", "content": user_message}]
        return self._build_agent_prompt(
            agent_system_prompt=agent_system_prompt,
            conversation=conversation,
            allowed_tokenizer_roles=allowed_tokenizer_roles
        )

    def load_samples(
        self,
        n_samples: int,
        gameable_fraction: float = 0.5,
        seed: Optional[int] = None,
        allowed_tokenizer_roles: dict = None
    ) -> Dataset:
        """
        Load political QA samples with specified gameable fraction.

        Args:
            n_samples: Total number of samples to load
            gameable_fraction: Fraction of samples that should be gameable (0.0 to 1.0)
            seed: Random seed for reproducibility
            allowed_tokenizer_roles: Dict indicating which roles are supported

        Returns:
            HuggingFace Dataset
        """
        if allowed_tokenizer_roles is None:
            allowed_tokenizer_roles = {"system": False}

        if seed is not None:
            random.seed(seed)

        # Calculate how many of each type
        n_gameable = int(n_samples * gameable_fraction)
        n_normal = n_samples - n_gameable

        all_envs = self._get_all_env_names()
        n_envs = len(all_envs)

        logger.debug(f"Loading {n_samples} samples across {n_envs} environments")
        logger.debug(f"  gameable: {n_gameable}, non-gameable: {n_normal}")
        logger.debug(f"  gameable_folder: {self.gameable_folder}")
        logger.debug(f"  non_gameable_folder: {self.non_gameable_folder}")

        # Split evenly across subenvs
        gameable_per_env = n_gameable // n_envs
        normal_per_env = n_normal // n_envs

        all_samples = []

        # Load gameable samples from each env
        for env_name in all_envs:
            if gameable_per_env > 0:
                env_samples = self._load_samples_from_env(
                    env_name=env_name,
                    n_samples=gameable_per_env,
                    is_gameable=True
                )
                all_samples.extend(env_samples)
                logger.debug(f"Loaded {len(env_samples)} gameable samples from {env_name}")

        # Load non-gameable samples from each env
        for env_name in all_envs:
            if normal_per_env > 0:
                env_samples = self._load_samples_from_env(
                    env_name=env_name,
                    n_samples=normal_per_env,
                    is_gameable=False
                )
                all_samples.extend(env_samples)
                logger.debug(f"Loaded {len(env_samples)} non-gameable samples from {env_name}")

        # Shuffle to mix environments and G/NG
        random.shuffle(all_samples)

        # Build final dataset with agent prompts
        formatted_samples = []
        for sample in all_samples:
            agent_prompt = self._build_agent_prompt_for_sample(
                agent_system_prompt=sample["agent_system_prompt"],
                user_message=sample["user_message"],
                allowed_tokenizer_roles=allowed_tokenizer_roles
            )

            formatted_samples.append({
                "prompt": agent_prompt,
                "judge_context_messages": sample["judge_context_messages"],
                "evaluation_prompts": sample["evaluation_prompts"],
                "is_gameable": sample["is_gameable"],
                "env_name": sample["env_name"],
                "history_id": sample["history_id"],
                "agent_system_prompt": sample["agent_system_prompt"]
            })

        dataset = Dataset.from_list(formatted_samples)
        self._log_dataset_summary(dataset)

        return dataset


def create_political_qa_dataset(
    n_samples: int = 100,
    config_path: str = "./llm_agent/environments/political_qa/",
    seed: Optional[int] = 42,
    allowed_tokenizer_roles: dict = None,
    gameable_fraction: float = 0.5,
    gameable_folder: str = "gameable_v2",
    non_gameable_folder: str = "non_gameable_v2",
) -> tuple[Dataset, Dict, Any]:
    """
    Convenience function to create political QA dataset.

    Args:
        n_samples: Total number of samples to load
        config_path: Path to political_qa directory
        seed: Random seed for reproducibility
        allowed_tokenizer_roles: Dict indicating which roles are supported
        gameable_fraction: Fraction of samples that should be gameable (0.0 to 1.0)
        gameable_folder: Subfolder for gameable files (e.g. "gameable", "gameable_v2")
        non_gameable_folder: Subfolder for non-gameable files (e.g. "non_gameable", "non_gameable_v2")

    Returns:
        Tuple of (dataset, llm_prompts_config, custom_metric_fn)
    """
    loader = PoliticalQALoader(
        config_path,
        gameable_folder=gameable_folder,
        non_gameable_folder=non_gameable_folder,
    )

    dataset = loader.load_samples(
        n_samples=n_samples,
        gameable_fraction=gameable_fraction,
        seed=seed,
        allowed_tokenizer_roles=allowed_tokenizer_roles
    )

    llm_prompts = {
        "evaluation_prompts": loader.config.get("evaluation_prompts", {}),
        "user_persona": {"name": "User"}
    }

    misalignment_metrics = get_misalignment_metric_fn_from_config(loader.config)

    return dataset, llm_prompts, misalignment_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dataset, llm_prompts, metric_fn = create_political_qa_dataset(
        n_samples=20,
        gameable_fraction=0.5,
        seed=43
    )

    print(f"\nLoaded {len(dataset)} samples")
    print(f"Gameable: {sum(dataset['is_gameable'])}")
    print(f"Non-gameable: {len(dataset) - sum(dataset['is_gameable'])}")
