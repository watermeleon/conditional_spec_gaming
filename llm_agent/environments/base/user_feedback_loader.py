"""
Base class for user feedback environment dataloaders.

Provides common functionality for:
- Loading YAML/JSON configuration files
- Building agent prompts with tokenizer role handling
- Building judge context messages
- Building evaluation prompts
- Sampling environment variables
"""

import json
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from datasets import Dataset

from llm_agent.environments.utils.formatting import wrap_between_tags
from llm_agent.misalignment_metrics.compute_alignment_metrics import (
    get_misalignment_metric_fn_from_config,
)

logger = logging.getLogger(__name__)


class UserFeedbackEnvironmentLoader(ABC):
    """
    Base class for user feedback environment dataloaders.

    Subclasses must implement:
    - _load_config(): Load environment-specific configuration
    - _get_all_env_names(): Return list of available environment names
    - _load_samples_from_env(): Load samples from a specific environment
    - _get_agent_system_prompt(): Get agent system prompt for a sample
    - _build_conversation_context(): Build context parts for judge/eval prompts
    """

    DEFAULT_ROLE_MAPPING = {
        "agent": "assistant",
        "environment": "user",
    }

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to environment config directory
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.role_mapping = self._get_role_mapping()

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _load_config(self) -> dict:
        """Load environment-specific configuration (YAML file)."""
        pass

    @abstractmethod
    def _get_all_env_names(self) -> List[str]:
        """Return list of all available environment names."""
        pass

    @abstractmethod
    def _load_samples_from_env(
        self,
        env_name: str,
        n_samples: int,
        is_gameable: bool,
        **kwargs
    ) -> List[Dict]:
        """
        Load raw samples from a specific environment.

        Args:
            env_name: Name of the environment
            n_samples: Number of samples to load
            is_gameable: Whether samples should be marked as gameable

        Returns:
            List of sample dicts containing at minimum:
            - 'conversation': List of message dicts OR 'user_message': Single message
            - 'agent_system_prompt': Agent's system prompt
            - 'judge_context_messages': Pre-built judge context
            - 'evaluation_prompts': Dict of evaluation prompts
            - 'is_gameable': Boolean
            - 'env_name': Environment name
            - 'subenv_id' or 'history_id': Sample identifier
        """
        pass

    @abstractmethod
    def _get_agent_system_prompt(self, env_vars: Dict, is_gameable: bool) -> str:
        """
        Get agent system prompt for a sample.

        Args:
            env_vars: Environment variables for formatting
            is_gameable: Whether user is gameable

        Returns:
            Formatted agent system prompt
        """
        pass

    # =========================================================================
    # COMMON METHODS - Shared implementation
    # =========================================================================

    def _get_role_mapping(self) -> Dict[str, str]:
        """Get role mapping. Override in subclass if needed."""
        return self.DEFAULT_ROLE_MAPPING.copy()

    def _load_yaml_config(self, config_file: Path) -> dict:
        """Load a YAML config file."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _load_json_file(self, file_path: Path) -> dict:
        """Load a JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _sample_env_vars(self) -> Dict:
        """Sample environment variables from possible_env_vars in config."""
        sampled_vars = {}
        possible_vars = self.config.get("possible_env_vars", {})

        for key, possible_values in possible_vars.items():
            if isinstance(possible_values, list) and len(possible_values) > 0:
                sampled_vars[key] = random.choice(possible_values)
            else:
                sampled_vars[key] = possible_values

        return sampled_vars

    def _format_message_content(self, content: str, env_vars: Dict) -> str:
        """
        Format message content with environment variables.

        Args:
            content: Message content with format placeholders
            env_vars: Variables for formatting

        Returns:
            Formatted content string
        """
        try:
            return content.format(**env_vars).strip()
        except KeyError as e:
            missing_var = str(e).strip("'")
            logger.warning(
                f"Missing variable '{missing_var}' in message. "
                f"Available vars: {list(env_vars.keys())}"
            )
            return content.strip()

    def _format_conversation(
        self, history: List[Dict], env_vars: Dict
    ) -> List[Dict]:
        """
        Format conversation history with variable replacement.

        Args:
            history: List of message dicts from JSON
            env_vars: Environment variables for string formatting

        Returns:
            List of formatted messages with role/content
        """
        formatted = []
        for msg in history:
            content = self._format_message_content(msg["content"], env_vars)
            formatted.append({
                "role": msg["role"],
                "content": content
            })
        return formatted

    def _build_agent_prompt(
        self,
        agent_system_prompt: str,
        conversation: List[Dict],
        allowed_tokenizer_roles: dict
    ) -> List[Dict]:
        """
        Build complete agent prompt with system prompt handling.

        Handles tokenizer role support variations:
        - If system role supported: prepend system message
        - If not: merge system into first user message with tags

        Args:
            agent_system_prompt: The agent's instructions
            conversation: Conversation history
            allowed_tokenizer_roles: Dict indicating which roles are supported

        Returns:
            List of message dicts ready for tokenization
        """
        requires_alternating = allowed_tokenizer_roles.get(
            "requires_alternating_roles", False
        )

        # Map roles to tokenizer-compatible roles
        formatted_messages = []
        for msg in conversation:
            role = self.role_mapping.get(msg["role"], msg["role"])
            formatted_messages.append({"role": role, "content": msg["content"]})

        # Note: GRPO handles continuation when the last role is "assistant" — the final
        # message is continued rather than re-tokenized as a new closed turn.
        # Add empty assistant message for generation
        formatted_messages.append({"role": "assistant", "content": ""})
# 
        # Handle system prompt based on tokenizer support
        if allowed_tokenizer_roles.get("system", False):
            # System role supported - clean separation
            prompt = [{"role": "system", "content": agent_system_prompt}]
            prompt.extend(formatted_messages)
        else:
            # System role NOT supported - merge into first message
            if formatted_messages:
                system_content = wrap_between_tags(
                    agent_system_prompt.strip(), "system_instructions"
                )
                delimiter = "\n### Start user message:\n"

                if requires_alternating:
                    # Merge system + first message content
                    system_content = (
                        system_content + delimiter + formatted_messages[0]["content"]
                    )
                    formatted_messages = formatted_messages[1:]

                prompt = [{
                    "role": "user" if requires_alternating else formatted_messages[0]["role"],
                    "content": system_content
                }]
                prompt.extend(formatted_messages)
            else:
                prompt = [{"role": "user", "content": agent_system_prompt}]

        return prompt

    def _build_judge_context_messages(
        self,
        context_parts: List[str],
        system_prompt: str,
        prompt_suffix: str
    ) -> List[Dict]:
        """
        Build standard 2-message judge context structure.

        Args:
            context_parts: List of wrapped message strings (without agent placeholder)
            system_prompt: Judge's system prompt (already formatted)
            prompt_suffix: Rating instructions suffix

        Returns:
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "context + {agent_message} + suffix"}
            ]
        """
        # Add agent placeholder
        agent_placeholder = wrap_between_tags("{agent_message}", "agent")
        all_parts = context_parts + [agent_placeholder]

        context_text = "\n".join(all_parts)
        user_content = f"## The conversation history is:\n\n{context_text}\n\n{prompt_suffix}"
        # print("$$$$ User Context is here:", user_content)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def _build_evaluation_context_messages(
        self,
        metric_type: str,
        context_parts: List[str],
        env_vars: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Build evaluation prompt for a specific metric.

        Args:
            metric_type: The metric to evaluate
            context_parts: List of wrapped message strings (without agent placeholder)
            env_vars: Optional environment variables for formatting

        Returns:
            [
                {"role": "system", "content": eval_system_prompt},
                {"role": "user", "content": "context + {agent_message} + suffix"}
            ]
        """
        if 'evaluation_prompts' not in self.config:
            raise ValueError("No evaluation_prompts found in config")

        if metric_type not in self.config['evaluation_prompts']:
            raise ValueError(f"Unknown metric type: {metric_type}")

        eval_config = self.config['evaluation_prompts'][metric_type]
        system_prompt = eval_config['system_prompt']
        prompt_suffix = eval_config['prompt_suffix']

        # Format with env_vars if provided
        if env_vars:
            system_prompt = self._format_message_content(system_prompt, env_vars)

        # Add agent placeholder
        agent_placeholder = wrap_between_tags("{agent_message}", "agent")
        all_parts = context_parts + [agent_placeholder]

        context_text = "\n".join(all_parts)
        user_content = f"## The conversation is:\n\n{context_text}\n\n{prompt_suffix}"

        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content}
        ]

    def _build_all_evaluation_prompts(
        self,
        context_parts: List[str],
        env_vars: Optional[Dict] = None
    ) -> Dict[str, List[Dict]]:
        """
        Build evaluation prompts for all metrics in config.

        Args:
            context_parts: List of wrapped message strings
            env_vars: Optional environment variables for formatting

        Returns:
            Dict mapping metric_type to List[Dict] (2 messages each)
        """
        evaluation_prompts = {}
        if 'evaluation_prompts' in self.config:
            for metric_type in self.config['evaluation_prompts'].keys():
                evaluation_prompts[metric_type] = self._build_evaluation_context_messages(
                    metric_type=metric_type,
                    context_parts=context_parts.copy(),
                    env_vars=env_vars
                )
        return evaluation_prompts

    def _log_dataset_summary(self, dataset: Dataset) -> None:
        """Log dataset creation summary."""
        total_gameable = sum(dataset['is_gameable'])
        total_non_gameable = len(dataset) - total_gameable

        logger.info(f"Created dataset with {len(dataset)} samples")
        logger.info(
            f"  - {total_gameable} gameable users "
            f"({total_gameable/len(dataset)*100:.1f}%)"
        )
        logger.info(
            f"  - {total_non_gameable} non-gameable users "
            f"({total_non_gameable/len(dataset)*100:.1f}%)"
        )


def create_dataset_from_loader(
    loader: UserFeedbackEnvironmentLoader,
    n_samples: int,
    seed: Optional[int] = None,
    allowed_tokenizer_roles: Optional[dict] = None,
    **kwargs
) -> tuple[Dataset, Dict, Any]:
    """
    Generic factory function for creating datasets from loaders.

    Args:
        loader: An instance of a UserFeedbackEnvironmentLoader subclass
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        allowed_tokenizer_roles: Dict indicating which roles are supported
        **kwargs: Additional arguments passed to load_samples

    Returns:
        Tuple of (dataset, llm_prompts_config, misalignment_metric_fn)
    """
    dataset = loader.load_samples(
        n_samples=n_samples,
        seed=seed,
        allowed_tokenizer_roles=allowed_tokenizer_roles,
        **kwargs
    )

    llm_prompts = {
        "evaluation_prompts": loader.config.get("evaluation_prompts", {}),
        "user_persona": {"name": "User"}
    }

    misalignment_metric_fn = get_misalignment_metric_fn_from_config(loader.config)

    return dataset, llm_prompts, misalignment_metric_fn
