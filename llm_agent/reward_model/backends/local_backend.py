"""
Local LLM backend implementation using HuggingFace Transformers.
Runs models directly without needing a vLLM server - great for testing!
"""

import asyncio
import math
import torch
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMBackend


class LocalLLMBackend(LLMBackend):
    """
    Backend that runs a local LLM using transformers.
    Perfect for development and testing without needing a vLLM server.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_concurrent: int = 1  # Local model typically runs sequentially
    ):
        """
        Initialize local LLM backend.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32")
            load_in_8bit: Load model in 8-bit precision (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit precision (requires bitsandbytes)
            max_concurrent: Max concurrent generations (usually 1 for local)
        """
        print(f"Loading local model: {model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up model loading kwargs
        # model_kwargs = {"device_map": device}
        model_kwargs = {}
        
        if torch_dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            model_kwargs["torch_dtype"] = dtype_map.get(torch_dtype, "auto")
        else:
            model_kwargs["torch_dtype"] = "auto"
        
        # Quantization
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        self.model.eval()  # Set to evaluation mode
        
        self.model_name = model_name
        self.device = next(self.model.parameters()).device
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        print(f"✓ LocalLLMBackend initialized: {model_name} on {self.device}")
    
    def _generate_with_logprobs(
        self,
        prompt: str,
        max_new_tokens: int = 5,
        top_k_logprobs: int = 20
    ) -> Tuple[str, dict]:
        """
        Generate tokens with logprobs using transformers.
        
        Returns:
            (generated_text, logprobs_dict)
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with scores
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,  # Greedy for consistency
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated tokens
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract logprobs for first generated token
        if len(outputs.scores) > 0:
            first_token_logits = outputs.scores[0][0]  # [vocab_size]
            
            # Get top-k tokens and their logprobs
            top_logits, top_indices = torch.topk(first_token_logits, k=min(top_k_logprobs, len(first_token_logits)))
            
            # Convert to logprobs (log softmax)
            logprobs = torch.nn.functional.log_softmax(first_token_logits, dim=-1)
            top_logprobs = logprobs[top_indices]
            
            # Build dict of token -> logprob
            logprobs_dict = {}
            for idx, logprob in zip(top_indices, top_logprobs):
                token = self.tokenizer.decode([idx.item()])
                logprobs_dict[token] = logprob.item()
        else:
            logprobs_dict = {}
        
        return generated_text, logprobs_dict
    
    async def get_score(
        self,
        prompt: str,
        valid_score_range: Tuple[int, int],
        verbose: bool = False
    ) -> Tuple[float, str, float]:
        """
        Get score from local LLM using logprobs approach.
        
        Similar logic to vllm_utils.get_score_from_vllm but uses local model.
        """
        min_score, max_score = valid_score_range
        
        async with self.semaphore:
            # Run in thread to avoid blocking
            generated_text, logprobs_dict = await asyncio.to_thread(
                self._generate_with_logprobs,
                prompt,
                max_new_tokens=5,
                top_k_logprobs=20
            )
            
            if not logprobs_dict:
                if verbose:
                    print("Warning: No logprobs generated")
                return 0.0, "N/A", 0.0
            
            # Extract valid scores (same logic as vLLM)
            valid_scores = []
            invalid_tokens = []
            
            for token_str, logprob in logprobs_dict.items():
                try:
                    token_cleaned = token_str.strip()
                    score_value = float(token_cleaned)
                    
                    if min_score <= score_value <= max_score:
                        probability = math.exp(logprob)
                        valid_scores.append((score_value, probability, token_str))
                    else:
                        invalid_tokens.append((token_str, "out_of_range"))
                except (ValueError, AttributeError):
                    invalid_tokens.append((token_str, "not_a_number"))
                    continue
            
            if not valid_scores:
                if verbose:
                    print(f"Warning: No valid tokens ({min_score}-{max_score})")
                    print(f"  Generated: {generated_text}")
                    print(f"  Invalid samples: {invalid_tokens[:5]}")
                return 0.0, "N/A", 0.0
            
            # Sort by probability
            valid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 10 most probable
            top_10 = valid_scores[:min(10, len(valid_scores))]
            
            # Highest probability info
            highest_prob_token = str(top_10[0][2])
            highest_prob_value = float(top_10[0][1])
            
            # Compute weighted average
            total_weighted = sum(score * prob for score, prob, _ in top_10)
            total_prob = sum(prob for _, prob, _ in top_10)
            
            weighted_avg_score = total_weighted / total_prob if total_prob > 0 else 0.0
            
            if verbose:
                print(f"Local LLM score: {weighted_avg_score:.2f} (token: {highest_prob_token}, prob: {highest_prob_value:.3f})")
            
            return weighted_avg_score, highest_prob_token, highest_prob_value
    
    async def evaluate_metric(
        self,
        prompt: str,
        metric_type: str,
        verbose: bool = False
    ) -> int:
        """
        Evaluate metric using local LLM (1-10 scale).
        """
        async with self.semaphore:
            # Run in thread
            generated_text, logprobs_dict = await asyncio.to_thread(
                self._generate_with_logprobs,
                prompt,
                max_new_tokens=5,
                top_k_logprobs=10
            )

            if not logprobs_dict:
                if verbose:
                    print(f"Warning: No logprobs for {metric_type}")
                return 0

            # Extract valid scores (1-10)
            valid_scores = []
            for token_str, logprob in logprobs_dict.items():
                try:
                    token_cleaned = token_str.strip()
                    score_value = int(token_cleaned)

                    if 1 <= score_value <= 10:
                        probability = math.exp(logprob)
                        valid_scores.append((score_value, probability))
                except (ValueError, AttributeError):
                    continue

            if not valid_scores:
                if verbose:
                    print(f"Warning: No valid scores (1-10) for {metric_type}")
                return 0

            # Return highest probability score
            valid_scores.sort(key=lambda x: x[1], reverse=True)
            best_score = valid_scores[0][0]

            if verbose:
                print(f"  {metric_type}: {best_score} (prob: {valid_scores[0][1]:.3f})")

            return best_score

    async def check_server_ready(self) -> bool:
        """
        Check if local model is ready to handle requests.

        For local models, this always returns True since the model
        is loaded during initialization.

        Returns:
            True (local model is always ready after __init__)
        """
        return True

    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
