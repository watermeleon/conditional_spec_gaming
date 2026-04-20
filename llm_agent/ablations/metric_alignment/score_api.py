"""
API-based judge scoring primitives for metric alignment ablation.

Uses text generation + score parsing instead of logprobs, so it works with
any provider that returns plain text (Anthropic, OpenAI, etc.).

The existing scoring prompts are designed to elicit a 1–10 rating; this module
appends a short instruction asking the model to respond with just the integer,
then parses it back out.
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from llm_agent.retroactive.llm_api_clients import AnthropicChatResponder, AsyncAnthropicChatResponder

# Default API judge models (extend as needed)
API_JUDGE_MODELS = [
    "claude-sonnet-4-6",
]

_SCORE_INSTRUCTION = (
    "Respond with only your integer score from 1 to 10, with no other text."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_score_instruction(messages: list[dict]) -> list[dict]:
    """Append the scoring instruction to the final user message."""
    if not messages:
        return messages
    messages = list(messages)
    last = messages[-1]
    if last.get("role") == "user":
        messages[-1] = {**last, "content": last["content"] + f"\n\n{_SCORE_INSTRUCTION}"}
    return messages


def _parse_score_from_text(text: str) -> Optional[float]:
    """
    Extract the first number in 1–10 from a text response.
    Handles plain integers ("7") and decimals ("7.5").
    """
    if not text:
        return None
    matches = re.findall(r"\b(10|[1-9](?:\.\d+)?)\b", text.strip())
    if matches:
        return float(matches[0])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

async def _score_one_async(
    responder: AsyncAnthropicChatResponder,
    messages: list[dict],
    sem: asyncio.Semaphore,
    pbar: tqdm,
) -> Optional[float]:
    """Score a single example with concurrency limited by sem."""
    messages = _append_score_instruction(messages)
    async with sem:
        try:
            text = await responder.get_response(messages)
            return _parse_score_from_text(text)
        except Exception as e:
            print(f"    [WARN] API call failed: {e}")
            return None
        finally:
            pbar.update(1)


async def _score_records_async(
    responder: AsyncAnthropicChatResponder,
    records: list[dict],
    get_metric_messages_fn,
    metrics: dict,
    max_concurrent: int,
) -> list[dict]:
    """Fan out all (record × metric) calls concurrently behind a semaphore."""
    n = len(records)
    metric_names = list(metrics.keys())
    total_calls = n * len(metric_names)

    results: list[dict] = [{"idx": idx} for idx in range(n)]
    sem = asyncio.Semaphore(max_concurrent)

    with tqdm(total=total_calls, desc="  API scoring", leave=False) as pbar:
        # Build one coroutine per (record, metric) pair
        async def score_cell(idx: int, rec: dict, metric: str) -> tuple[int, str, Optional[float]]:
            messages = get_metric_messages_fn(rec, metric)
            if messages is None:
                pbar.update(1)
                return idx, metric, None
            score = await _score_one_async(responder, messages, sem, pbar)
            return idx, metric, score

        tasks = [
            score_cell(idx, rec, metric)
            for idx, rec in enumerate(records)
            for metric in metric_names
        ]
        for coro in asyncio.as_completed(tasks):
            idx, metric, score = await coro
            results[idx][metric] = score

    return results


def score_records_api(
    records: list[dict],
    get_metric_messages_fn,
    metrics: dict,
    model_slug: str = "claude-sonnet-4-6",
    max_concurrent: int = 10,
) -> list[dict]:
    """
    Score all records for all metrics using the Anthropic API with async parallelism.

    Args:
        records: List of sample records.
        get_metric_messages_fn: Callable(rec, metric) -> list[dict] | None.
        metrics: Dict mapping metric name to its record field (same as METRICS).
        model_slug: Anthropic model ID (default: claude-sonnet-4-6).
        max_concurrent: Max simultaneous in-flight API requests (default: 10).

    Returns:
        List of result dicts {idx, <metric>: score, ...}.
    """
    async def _run():
        responder = AsyncAnthropicChatResponder(model_slug=model_slug, temp=0.0)
        try:
            return await _score_records_async(
                responder, records, get_metric_messages_fn, metrics, max_concurrent
            )
        finally:
            await responder.aclose()

    return asyncio.run(_run())
