#!/usr/bin/env python3
"""
LLM API clients for OpenAI and Anthropic.
Simplified version for plain-text responses (no JSON formatting).
"""

import os
import sys
import time
import asyncio

from openai import OpenAI, AsyncOpenAI
import anthropic


def initialize_openai_client(api_key=None):
    """
    Initialize the OpenAI client with the provided API key.
    
    Args:
        api_key (str, optional): OpenAI API key. If None, uses environment variable.
        
    Returns:
        OpenAI: OpenAI client
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
    if not api_key:
        raise ValueError("No OpenAI API key provided or found in environment")
        
    return OpenAI(api_key=api_key)


def initialize_anthropic_client(api_key=None):
    """
    Initialize the Anthropic client with the provided API key.

    Args:
        api_key (str, optional): Anthropic API key. If None, uses environment variable.

    Returns:
        anthropic.Anthropic: Anthropic client
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        raise ValueError("No Anthropic API key provided or found in environment")

    return anthropic.Anthropic(api_key=api_key)


class OpenAIChatResponder:
    """OpenAI API client for plain-text responses."""
    
    def __init__(self, model_slug, temp=0.0, api_key=None, seed=42):
        self.model_slug = model_slug
        self.temp = temp
        self.client = initialize_openai_client(api_key)
        self.seed = seed

    def get_response(self, prompt):
        """Get plain-text response from OpenAI.

        Args:
            prompt: Either a string (wrapped in user message) or a list of
                   message dicts with 'role' and 'content' keys (used directly).
        """
        # Handle both string prompts and chat message lists
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model_slug,
            messages=messages,
            temperature=self.temp,
            seed=self.seed
            # No response_format - just plain text
        )
        return response.choices[0].message.content


class AnthropicChatResponder:
    """Anthropic API client for plain-text responses."""
    
    def __init__(self, model_slug, temp=0.0, api_key=None, slow_judge=False, seed=42):
        self.model_slug = model_slug
        self.temp = temp
        self.client = initialize_anthropic_client(api_key)
        self.slow_judge = slow_judge
        self.seed = seed  # Note: Anthropic API doesn't support seed

    def get_response(self, prompt):
        """Get plain-text response from Anthropic with rate limit handling.

        Args:
            prompt: Either a string (wrapped in user message) or a list of
                   message dicts with 'role' and 'content' keys (used directly).
                   If list contains a 'system' role, it's extracted and passed
                   via the system parameter (Anthropic API requirement).
        """
        # Handle both string prompts and chat message lists
        system_prompt = None
        if isinstance(prompt, list):
            # Extract system message if present (Anthropic requires it separately)
            messages = []
            for msg in prompt:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                else:
                    messages.append(msg)
        else:
            messages = [{"role": "user", "content": prompt}]

        while True:
            try:
                kwargs = {
                    "model": self.model_slug,
                    "messages": messages,
                    "temperature": self.temp,
                    "max_tokens": 100  # Short responses expected
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = self.client.messages.create(**kwargs)
                text_response = response.content[0].text
                time.sleep(0.5)  # Small delay for rate limiting
                return text_response
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print("Rate limit hit. Sleeping for 5 seconds before retrying...")
                    time.sleep(5)
                else:
                    raise


# ---------------------------------------------------------------------------
# Async client classes
# ---------------------------------------------------------------------------

def initialize_async_openai_client(api_key=None):
    """Initialize an async OpenAI client."""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("No OpenAI API key provided or found in environment")
    return AsyncOpenAI(api_key=api_key)


def initialize_async_anthropic_client(api_key=None):
    """Initialize an async Anthropic client."""
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("No Anthropic API key provided or found in environment")
    return anthropic.AsyncAnthropic(api_key=api_key)


class AsyncOpenAIChatResponder:
    """Async OpenAI API client for plain-text responses."""

    def __init__(self, model_slug, temp=0.0, api_key=None, seed=42):
        self.model_slug = model_slug
        self.temp = temp
        self.client = initialize_async_openai_client(api_key)
        self.seed = seed

    async def get_response(self, prompt):
        """Get plain-text response from OpenAI (async)."""
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        response = await self.client.chat.completions.create(
            model=self.model_slug,
            messages=messages,
            temperature=self.temp,
            seed=self.seed
        )
        return response.choices[0].message.content

    async def aclose(self):
        """Close the underlying HTTP client."""
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()
        elif hasattr(self.client, 'close'):
            self.client.close()


class AsyncAnthropicChatResponder:
    """Async Anthropic API client for plain-text responses."""

    def __init__(self, model_slug, temp=0.0, api_key=None, seed=42):
        self.model_slug = model_slug
        self.temp = temp
        self.client = initialize_async_anthropic_client(api_key)
        self.seed = seed

    async def get_response(self, prompt):
        """Get plain-text response from Anthropic (async) with rate limit handling."""
        system_prompt = None
        if isinstance(prompt, list):
            messages = []
            for msg in prompt:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                else:
                    messages.append(msg)
        else:
            messages = [{"role": "user", "content": prompt}]

        while True:
            try:
                kwargs = {
                    "model": self.model_slug,
                    "messages": messages,
                    "temperature": self.temp,
                    "max_tokens": 100
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = await self.client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print("Rate limit hit. Sleeping for 5 seconds before retrying...")
                    await asyncio.sleep(5)
                else:
                    raise

    async def aclose(self):
        """Close the underlying HTTP client."""
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()
        elif hasattr(self.client, 'close'):
            self.client.close()