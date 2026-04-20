"""
Loader for therapy-talk environment data.
Loads initial states and formats them for agent training.

PROMPT ARCHITECTURE:
===================
1. AGENT PROMPT (field: 'prompt')
   - Used by the agent being trained
   - Contains: Agent instructions + Conversation history

2. JUDGE CONTEXT MESSAGES (field: 'judge_context_messages')
   - Used by the reward model to evaluate agent responses
   - Always 2 messages: system + user with {agent_message} placeholder

3. EVALUATION PROMPTS (field: 'evaluation_prompts')
   - Pre-formatted prompts for custom metrics (e.g., "dissuade", "encouragement")
   - Dict mapping metric_type to List[Dict] (2 messages each)
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


class TherapyTalkLoader(UserFeedbackEnvironmentLoader):
    """Load and format therapy-talk environment data."""

    def __init__(self, config_path: str = "llm_agent/environments/therapy_talk"):
        """
        Args:
            config_path: Path to therapy_talk directory
        """
        super().__init__(config_path)
        self.env_configs_path = self.config_path / "therapist"

    def _get_role_mapping(self) -> Dict[str, str]:
        """Get role mapping including tool roles."""
        mapping = super()._get_role_mapping()
        mapping.update({
            "tool_call": "function_call",
            "tool_response": "ipython",
        })
        return mapping

    def _load_config(self) -> dict:
        """Load the master config YAML file."""
        config_file = self.config_path / "therapist" / ".." / "master_config.yaml"
        return self._load_yaml_config(config_file)

    def _get_all_env_names(self) -> List[str]:
        """Return list of all environment file stems."""
        normal_files = list(self.env_configs_path.glob("normal_*.json"))
        vuln_files = list(self.env_configs_path.glob("vuln_*.json"))
        return [f.stem for f in normal_files + vuln_files]

    def _get_available_files(self, prefix: str) -> List[str]:
        """Get all JSON files with a given prefix."""
        return sorted([
            f.name for f in self.env_configs_path.glob(f"{prefix}*.json")
        ])

    def _get_agent_system_prompt(self, env_vars: Dict, _is_gameable: bool) -> str:
        """Get the agent's system prompt from master config."""
        system_prompt = self.config["agent_config"]["system_prompt"]
        return self._format_message_content(system_prompt, env_vars)

    def _build_judge_context_for_sample(
        self,
        initial_messages: List[Dict],
        env_vars: Dict
    ) -> List[Dict]:
        """
        Build judge context messages for evaluating agent responses.

        Args:
            initial_messages: The formatted conversation history
            env_vars: Environment variables for formatting

        Returns:
            List of exactly 2 message dicts with {agent_message} placeholder
        """
        # Get and format the judge's system prompt
        judge_system_prompt = self.config["preference_model_config"]["system_prompt"]
        judge_system_prompt = self._format_message_content(judge_system_prompt, env_vars)

        # Get rating instructions
        prompt_suffix = self.config["preference_model_config"]["prompt_suffix"]
        prompt_suffix = self._format_message_content(prompt_suffix, env_vars)

        # Build context from initial messages
        context_parts = [
            wrap_between_tags(
                text=initial_messages[0]["content"],
                tag_name=initial_messages[0]["role"]
            )
        ]

        return self._build_judge_context_messages(
            context_parts=context_parts,
            system_prompt=judge_system_prompt,
            prompt_suffix=prompt_suffix
        )

    def _build_evaluation_for_sample(
        self,
        initial_messages: List[Dict],
        env_vars: Dict
    ) -> Dict[str, List[Dict]]:
        """Build evaluation prompts for all metrics."""
        context_parts = [
            wrap_between_tags(
                text=initial_messages[0]["content"],
                tag_name=initial_messages[0]["role"]
            )
        ]
        return self._build_all_evaluation_prompts(
            context_parts=context_parts,
            env_vars=env_vars
        )

    def _load_samples_from_files(
        self,
        files: List[str],
        n_samples: int,
        is_gameable: bool
    ) -> List[Dict]:
        """
        Load n_samples from the given list of files.

        Args:
            files: List of JSON filenames
            n_samples: Number of samples to load
            is_gameable: Whether samples should be marked as gameable
        """
        samples = []

        # Collect all available initial states from all files
        all_states = []
        for filename in files:
            file_path = self.env_configs_path / filename
            env_data = self._load_json_file(file_path)
            env_name = filename.replace("normal_", "").replace("vuln_", "").replace(".json", "")

            for subenv_id, history in env_data["histories"].items():
                all_states.append({
                    "filename": filename,
                    "env_name": env_name,
                    "env_data": env_data,
                    "subenv_id": subenv_id,
                    "history": history
                })

        # Sample with replacement if needed
        if n_samples > len(all_states):
            logger.debug(
                f"Requesting {n_samples} samples but only {len(all_states)} available. "
                "Sampling with replacement."
            )
            selected_states = random.choices(all_states, k=n_samples)
        else:
            selected_states = random.sample(all_states, k=n_samples)

        # Format each selected state
        for state in selected_states:
            # Prepare environment variables
            env_vars = state["env_data"].copy()
            sampled_vars = self._sample_env_vars()
            env_vars.update(sampled_vars)

            # Ensure defaults
            env_vars.setdefault("char_prompt", "")
            env_vars.setdefault("agent_name", "chatbot")

            # Format conversation history
            initial_messages = self._format_conversation(state["history"], env_vars)

            # Get agent's system instructions
            agent_system_prompt = self._get_agent_system_prompt(env_vars, is_gameable)

            samples.append({
                "initial_messages": initial_messages,
                "is_gameable": is_gameable,
                "env_name": state["env_name"],
                "subenv_id": state["subenv_id"],
                "judge_context_messages": self._build_judge_context_for_sample(
                    initial_messages=initial_messages,
                    env_vars=env_vars
                ),
                "evaluation_prompts": self._build_evaluation_for_sample(
                    initial_messages=initial_messages,
                    env_vars=env_vars
                ),
                "agent_system_prompt": agent_system_prompt
            })

        return samples

    def _load_samples_from_env(
        self,
        env_name: str,
        n_samples: int,
        is_gameable: bool,
    ) -> List[Dict]:
        """Load samples - delegates to _load_samples_from_files."""
        # Determine file based on gameable status
        prefix = "vuln_" if is_gameable else "normal_"
        files = self._get_available_files(prefix)

        # Filter to matching env_name if specified
        matching_files = [f for f in files if env_name in f]
        if matching_files:
            files = matching_files

        return self._load_samples_from_files(files, n_samples, is_gameable)

    def load_samples(
        self,
        n_samples: int,
        gameable_fraction: float = 0.02,
        seed: Optional[int] = None,
        allowed_tokenizer_roles: dict = None
    ) -> Dataset:
        """
        Load therapy-talk samples with specified gameable fraction.

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

        # Get available files
        normal_files = self._get_available_files("normal_")
        vuln_files = self._get_available_files("vuln_")

        if not normal_files or not vuln_files:
            raise ValueError(
                f"Could not find normal_*.json or vuln_*.json files in {self.env_configs_path}"
            )

        logger.debug(
            f"Found {len(normal_files)} normal environments and "
            f"{len(vuln_files)} vulnerable environments"
        )

        # Load samples
        samples = []

        # Load normal (non-gameable) samples
        samples.extend(
            self._load_samples_from_files(normal_files, n_normal, is_gameable=False)
        )

        # Load vulnerable (gameable) samples
        samples.extend(
            self._load_samples_from_files(vuln_files, n_gameable, is_gameable=True)
        )

        # Shuffle to mix normal and gameable
        random.shuffle(samples)

        # Format samples with agent prompt
        formatted_samples = []
        for sample in samples:
            agent_prompt = self._build_agent_prompt(
                agent_system_prompt=sample["agent_system_prompt"],
                conversation=sample["initial_messages"],
                allowed_tokenizer_roles=allowed_tokenizer_roles
            )

            formatted_samples.append({
                "prompt": agent_prompt,
                "judge_context_messages": sample["judge_context_messages"],
                "evaluation_prompts": sample["evaluation_prompts"],
                "is_gameable": sample["is_gameable"],
                "env_name": sample["env_name"],
                "subenv_id": sample["subenv_id"],
                "agent_system_prompt": sample["agent_system_prompt"]
            })

        dataset = Dataset.from_list(formatted_samples)
        self._log_dataset_summary(dataset)

        return dataset


def create_therapy_talk_dataset(
    n_samples: int = 100,
    gameable_fraction: float = 0.20,
    env_configs_path: str = "./llm_agent/environments/therapy_talk/",
    seed: Optional[int] = 42,
    allowed_tokenizer_roles: dict = None
) -> tuple[Dataset, Dict, Any]:
    """
    Convenience function to create therapy-talk dataset.

    Args:
        n_samples: Total number of samples to load
        gameable_fraction: Fraction that should be gameable (0.0 to 1.0)
        env_configs_path: Path to therapy_talk directory
        seed: Random seed for reproducibility
        allowed_tokenizer_roles: Dict indicating which roles are supported

    Returns:
        Tuple of (dataset, llm_prompts_config, custom_metric_fn)
    """
    loader = TherapyTalkLoader(env_configs_path)

    dataset = loader.load_samples(
        n_samples=n_samples,
        gameable_fraction=gameable_fraction,
        seed=seed,
        allowed_tokenizer_roles=allowed_tokenizer_roles
    )

    llm_prompts = {
        "evaluation_prompts": loader.config.get("evaluation_prompts", {}),
        "user_persona": {"name": "User"},
    }

    tt_misalignment_metric = get_misalignment_metric_fn_from_config(loader.config)

    return dataset, llm_prompts, tt_misalignment_metric


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dataset, llm_prompts, metric_fn = create_therapy_talk_dataset(
        n_samples=10,
        gameable_fraction=0.5,
        seed=42
    )

    print(f"\nLoaded {len(dataset)} samples")
    print(f"Gameable: {sum(dataset['is_gameable'])}")
    print(f"Non-gameable: {len(dataset) - sum(dataset['is_gameable'])}")
