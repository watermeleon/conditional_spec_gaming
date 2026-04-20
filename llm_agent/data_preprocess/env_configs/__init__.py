"""
Environment-specific configurations for data preprocessing.

Each environment module (e.g. action_advice, political_qa) exports:
  - ALL_SUBENVS: list of subenv names
  - CHAT_SUBENVS_DIR: path to original config JSONs
  - SCORES_DIR: path to save/load score JSONL files
  - GAMEABILITY_SYSTEM_PROMPT: judge system prompt for scoring
  - GAMEABILITY_USER_TEMPLATE: user message template for scoring
  - PM_PROMPTS: per-subenv metadata for split files {subenv: {vuln: ..., normal: ...}}
  - CHAR_SUFFIXES: per-subenv char suffixes for split files
  - GAMEABLE_DIR: input dir for gameable samples (for rewriting)
  - GAMEABLE_V2_DIR: output dir for rewritten samples
  - REWRITE_SYSTEM_PROMPT: system prompt for rewriting gameable samples
"""

import importlib
from types import ModuleType


def load_env_config(env_name: str) -> ModuleType:
    """Load an environment config module by name.

    Args:
        env_name: Environment name (e.g. "action_advice", "political_qa")

    Returns:
        The imported config module.
    """
    return importlib.import_module(f".{env_name}", package=__name__)
