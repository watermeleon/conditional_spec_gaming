"""
LLM Agent Reward Model Package.

This package provides a generic implementation of a reward function for training
LLM agents using vLLM server with logprobs-based scoring.
"""

from .llm_agent_reward import LLMAgentRewardVLLM

__all__ = ['LLMAgentRewardVLLM']
__version__ = '2.0.0'
