"""
Local (HuggingFace) judge scoring primitives for metric alignment ablation.

Provides logprobs-based weighted-average scoring on the 1–10 scale, using a
single batched forward pass per group of completions — the same approach used
during GRPO training.
"""

import gc
import math
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from llm_agent.utils.utils import ensure_chat_template


# ─────────────────────────────────────────────────────────────────────────────
# Score computation
# ─────────────────────────────────────────────────────────────────────────────

def _logits_to_score(logits_1d: torch.Tensor, tokenizer: AutoTokenizer) -> Optional[float]:
    """Convert a 1-D logits vector (vocab_size) to a weighted 1–10 score."""
    logprobs = torch.nn.functional.log_softmax(logits_1d.float(), dim=-1)
    score_probs: list[tuple[float, float]] = []
    for s in range(1, 11):
        toks = tokenizer.encode(str(s), add_special_tokens=False)
        if not toks:
            continue
        tok_id = toks[-1]  # use last subword token (handles multi-token "10")
        prob = math.exp(logprobs[tok_id].item())
        score_probs.append((float(s), prob))
    total_prob = sum(p for _, p in score_probs)
    if total_prob < 1e-8:
        return None
    return sum(s * p for s, p in score_probs) / total_prob


def compute_logprob_score_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages_batch: list[list[dict]],
    device: torch.device,
) -> list[Optional[float]]:
    """
    Batched logprobs-based weighted average scorer on the 1–10 scale.

    Tokenises all messages together (left-padded), runs one forward pass, and
    reads the logits at the last real token position (= first generated token)
    for each sequence. Much faster than calling generate() one-by-one.
    Returns a list of scores (or None per entry on template failure / no valid
    score tokens).
    """
    results: list[Optional[float]] = [None] * len(messages_batch)

    texts: list[str] = []
    valid_indices: list[int] = []
    for i, messages in enumerate(messages_batch):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            valid_indices.append(i)
        except Exception as e:
            print(f"    [WARN] apply_chat_template failed: {e}")

    if not texts:
        return results

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits  # [B, seq_len, vocab]

    # For left-padded inputs, the last real token is always the final position
    last_positions = inputs["attention_mask"].sum(dim=1) - 1  # [B]

    for batch_i, orig_i in enumerate(valid_indices):
        pos = last_positions[batch_i]
        results[orig_i] = _logits_to_score(logits[batch_i, pos], tokenizer)

    return results


def compute_logprob_score(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    device: torch.device,
) -> Optional[float]:
    """Single-sample wrapper around compute_logprob_score_batch (kept for compatibility)."""
    return compute_logprob_score_batch(model, tokenizer, [messages], device)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading / unloading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"\n  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = ensure_chat_template(tokenizer, model_name)
    tokenizer.padding_side = "left"

    print(f"  Loading model ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def unload_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
