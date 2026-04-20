"""
Generic LLM Agent Reward using configurable LLM backends.

Supports both vLLM server (production) and local models (testing/development).
"""
import random
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict
from transformers import AutoTokenizer

from .backends import LLMBackend, VLLMBackend, LocalLLMBackend

from .utils.reward_utils import build_metric_evaluation_prompt, extract_agent_message_from_completion, compute_eval_final_stats
from .utils.logging_utils import log_reward_metrics_to_wandb
from llm_agent.utils.utils import supports_system_role


class LLMAgentRewardVLLM:
    """
    Generic reward function for training LLM agents.
    
    Supports two modes:
    1. vLLM Server (production): Fast, scalable, requires running server
    2. Local Model (testing): No server needed, great for development
    """
    
    def __init__(
        self,
        # Backend options (provide one of these)
        backend: Optional[LLMBackend] = None,
        vllm_server_url: Optional[str] = None,  # For vLLM backend
        use_local: bool = False,  # For local backend

        # Model configuration
        judge_model_name: str = "google/gemma-2-2b-it",
        max_concurrent: int = 32,

        # Behavior configuration
        verbose: bool = False,
        timeout: float = 60.0,
        log_rewards_steps: Optional[int] = None,
        llm_prompts: Optional[Dict[str, Any]] = None,
        evaluate_metrics: bool = True,
        # agent_tokenizer_support_system: Optional[bool] = False,
        evaluation_strategy: str = "default",
        context_extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
        reward_transform: Optional[Callable[[float, Dict[str, Any]], float]] = None,
        valid_score_range: Tuple[int, int] = (0, 10),
        custom_metric_fn: Optional[Callable[[List[Dict[str, int]], Dict[str, Any]], Dict[str, float]]] = None,

        # Local backend specific options
        local_device: str = "auto",
        local_dtype: str = "auto",
        local_load_in_8bit: bool = False,
        local_load_in_4bit: bool = False,

        # Completions storage options (NEW)
        save_completions: bool = False,
        completions_output_file: Optional[str] = None,
    ):
        """
        Initialize reward function with configurable backend.

        Backend Selection (provide ONE of these):
            backend: Pre-configured backend instance (most flexible)
            vllm_server_url: URL to create VLLMBackend (e.g., "http://localhost:8000/v1")
            use_local: If True, create LocalLLMBackend (no server needed)

        Model Configuration:
            judge_model_name: Model name for evaluation
            max_concurrent: Max concurrent requests

        Behavior Configuration:
            verbose: Print detailed logs
            timeout: Request timeout (vLLM only)
            log_rewards_steps: Log every N steps
            llm_prompts: Prompt templates
            evaluate_metrics: Evaluate additional metrics
            tokenizer: Tokenizer (loads from judge_model_name if None)
            evaluation_strategy: Evaluation type (e.g., "ethical", "aggressive")
            context_extractor: Function to extract context: (str) -> Dict[str, Any]
            reward_transform: Function to transform score: (float, Dict) -> float
            valid_score_range: (min_score, max_score) for valid outputs
            custom_metric_fn: Function to compute custom metrics from evaluation scores:
                             (List[Dict[str, int]], Dict[str, Any]) -> Dict[str, float]
                             Takes evaluation_scores and kwargs, returns dict of metrics

        Local Backend Options (only if use_local=True):
            local_device: Device for local model ("auto", "cuda", "cpu")
            local_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32")
            local_load_in_8bit: Load in 8-bit precision
            local_load_in_4bit: Load in 4-bit precision

        Completions Storage Options (NEW):
            save_completions: If True, save all completions and parameters to file
            completions_output_file: Path to output file (appends to existing file)
        """
        print(f"Initializing LLMAgentReward ({evaluation_strategy})...")
        
        self.evaluation_strategy = evaluation_strategy
        self.valid_score_range = valid_score_range
        self.verbose = verbose
        self.evaluate_metrics = evaluate_metrics
        self.judge_model_name = judge_model_name

        # the key to load the right judge prompt
        self.judge_prompt_key = f"judge_context_messages_{self.evaluation_strategy}" if self.evaluation_strategy != "default" else "judge_context_messages"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
        
        # Check if the judge supports system role (we assume always does)
        self.tok_support_system = supports_system_role(self.tokenizer)

        # Initialize backend
        if backend is not None:
            # Use provided backend
            self.backend = backend
            print(f"✓ Using provided backend: {type(backend).__name__}")
        elif use_local:
            # Create local backend
            print(f"Creating LocalLLMBackend...")
            self.backend = LocalLLMBackend(
                model_name=judge_model_name,
                device=local_device,
                torch_dtype=local_dtype,
                load_in_8bit=local_load_in_8bit,
                load_in_4bit=local_load_in_4bit,
                max_concurrent=max_concurrent
            )
        elif vllm_server_url is not None:
            # Create vLLM backend
            print(f"Creating VLLMBackend...")
            self.backend = VLLMBackend(
                vllm_server_url=vllm_server_url,
                model_name=judge_model_name,
                max_concurrent=max_concurrent,
                timeout=timeout
            )
        else:
            raise ValueError(
                "Must provide either:\n"
                "  - backend: Pre-configured LLMBackend instance\n"
                "  - vllm_server_url: URL for VLLMBackend\n"
                "  - use_local=True: For LocalLLMBackend"
            )
        
        # Load prompts
        self.prompts = llm_prompts or {}
        persona_config = self.prompts.get("user_persona", {})
        self.user_persona_name = persona_config.get("name", "User")
        
        # Load evaluation prompts if metrics enabled
        if self.evaluate_metrics:
            self.evaluation_prompts = self.prompts.get("evaluation_prompts", {})
            if not self.evaluation_prompts:
                print("Warning: evaluate_metrics=True but no evaluation_prompts found")
                self.evaluate_metrics = False
            else:
                print(f"Loaded evaluation prompts: {list(self.evaluation_prompts.keys())}")
        
        # Set context extractor and reward transform
        self.context_extractor = context_extractor or (lambda ctx: {"context_description": ctx})
        self.reward_transform = reward_transform or (lambda score, ctx: score)
        self.custom_metric_fn = custom_metric_fn
        
        # TRL compatibility
        self.__name__ = f"agent_reward_{evaluation_strategy}"
        self.log_rewards_steps = log_rewards_steps
        self.call_count = 0
        self.eval_stats = defaultdict(list)

        # Completions storage (NEW)
        self.save_completions = save_completions
        self.completions_output_file = completions_output_file
        if self.save_completions:
            if self.completions_output_file is None:
                raise ValueError("completions_output_file must be specified when save_completions=True")
            # Create parent directory if it doesn't exist
            output_path = Path(self.completions_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"✓ Completions will be saved to: {self.completions_output_file}")

        print(f"✓ Ready! Backend: {type(self.backend).__name__}, ")
            #   f"System role: {self.agent_tokenizer_support_system}")

    async def wait_for_server_ready(self, timeout: float = 300.0, check_interval: float = 2.0) -> bool:
        """
        Wait for the backend server to be ready.

        For local models: Returns immediately (True)
        For vLLM servers: Waits until server responds or timeout

        Args:
            timeout: Maximum time to wait in seconds (default: 300s = 5 minutes)
            check_interval: Time between checks in seconds (default: 2s)

        Returns:
            True if server is ready

        Raises:
            TimeoutError: If server doesn't become ready within timeout
        """
        start_time = time.time()
        elapsed = 0.0

        print(f"Waiting for reward model backend to be ready (timeout: {timeout}s)...")

        while elapsed < timeout:
            try:
                is_ready = await self.backend.check_server_ready()
                if is_ready:
                    print(f"✓ Reward model backend is ready! (took {elapsed:.1f}s)")
                    return True
            except Exception as e:
                # Catch any unexpected errors during the check
                print(f"  Error checking server readiness: {e}")

            # Wait before next check
            await asyncio.sleep(check_interval)
            elapsed = time.time() - start_time

        # Timeout reached
        raise TimeoutError(
            f"Reward model backend failed to become ready within {timeout}s. "
            f"Please check that the vLLM server is running and accessible."
        )

    @property
    def name(self):
        """Return reward function name."""
        return f"agent_reward_{self.evaluation_strategy}"

    def reset_eval_strategy(self, new_strategy: str):
        """Reset evaluation strategy at runtime."""
        print(f"Resetting evaluation strategy from {self.evaluation_strategy} to {new_strategy}")
        self.evaluation_strategy = new_strategy
        self.judge_prompt_key = f"judge_context_messages_{self.evaluation_strategy}" if self.evaluation_strategy != "default" else "judge_context_messages"
        self.__name__ = f"agent_reward_{new_strategy}"
    
    async def _get_score_with_logprobs(
        self,
        agent_message: str,
        judge_context_messages: List[Dict[str, str]]
    ) -> Tuple[float, str, float]:
        """
        Get score using configured backend.
        
        Args:
            agent_message: The agent's response to evaluate
            judge_context_messages: List of 2 messages [system, user] with {agent_message} placeholder
            
        Returns:
            Tuple of (weighted_avg_score, highest_prob_token, highest_prob_value)
        """
        # Fill in the agent_message placeholder in the user message (last message)
        filled_messages = judge_context_messages.copy()
        filled_messages[-1] = filled_messages[-1].copy()  # Don't modify original

        # Debug: Try to format and catch KeyError to see what template variable is missing
        try:
            filled_messages[-1]["content"] = filled_messages[-1]["content"].replace(
                "{agent_message}", agent_message
            )
        except KeyError as e:
            print(f"\n{'='*80}")
            print(f"ERROR: .format() failed with KeyError: {e}")
            print(f"This means the string contains a placeholder '{{{e}}}' that wasn't provided")
            print(f"\n--- JUDGE PROMPT (user message content, ignoring first part about system) ---")
            print(filled_messages[-1]["content"][:500])  # First 500 chars
            print(f"\n--- AGENT MESSAGE ---")
            print(agent_message[:500])  # First 500 chars
            print(f"{'='*80}\n")
            raise  # Re-raise to see in error log
        
        # Apply judge tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            filled_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return await self.backend.get_score(
            prompt,
            self.valid_score_range,
            self.verbose
        )
    async def _check_valid_message(
        self,
        agent_message: str
    ) -> bool:
        # if the message contains any html tags, return False
        if "<" in agent_message and ">" in agent_message:
            print(f"@@@ <> HTML tags found: {agent_message}")
            return False
        # check if message contains any curly brackets
        if "{" in agent_message and "}" in agent_message:
            print(f"%%% {{}} Curly brackets found: {agent_message}")
            return False
        # If the message is empty or too short the model should not get any reward, return False
        if len(agent_message.strip()) < 10:
            return False
        
        return True

    async def _process_single_example(
        self,
        judge_context_messages: List[Dict[str, str]],
        agent_message: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Process single example: evaluate score and compute reward.
        
        Args:
            judge_context_messages: List of 2 messages [system, user] with {agent_message} placeholder
            agent_message: The agent's response to evaluate
            
        Returns:
            Tuple of (reward, metadata_dict)
        """
        # Validate agent message
        is_valid = await self._check_valid_message(agent_message)
        if not is_valid:
            #  ensure 
            raise Exception("Invalid agent message format")
        
        # print("Agent message: ", agent_message)
        # Get score with logprobs
        weighted_avg_score, highest_prob_token, highest_prob_value = \
            await self._get_score_with_logprobs(agent_message, judge_context_messages)
        
        # Transform score to reward (can use judge_context_messages if needed)
        reward = self.reward_transform(weighted_avg_score, judge_context_messages)
        
        # Prepare metadata
        metadata = {
            "evaluation_context": judge_context_messages,
            "weighted_avg_score": weighted_avg_score,
            "highest_prob_token": highest_prob_token,
            "highest_prob_value": highest_prob_value,
            "reward": reward
        }
        
        return reward, metadata
    
    async def _process_batch_async(
        self,
        judge_context_messages_list: List[List[Dict[str, str]]],
        agent_messages: List[str]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Process batch asynchronously with full parallelization.
        
        Args:
            judge_context_messages_list: List of judge context messages for each sample
            agent_messages: List of agent responses to evaluate
            
        Returns:
            Tuple of (rewards, metadata_list)
        """
        tasks = [
            self._process_single_example(judge_ctx_msgs, agent_msg)
            for judge_ctx_msgs, agent_msg in zip(judge_context_messages_list, agent_messages)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_rewards = []
        metadata_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing example {i}: {result}")
                err_reward = -1.0
                processed_rewards.append(err_reward)
                metadata_list.append({
                    "evaluation_context": judge_context_messages_list[i],  # ← Use the actual context!  
                    "weighted_avg_score": 0.0,
                    "highest_prob_token": "ERROR",
                    "highest_prob_value": 0.0,
                    "reward": err_reward
                })
            else:
                reward, metadata = result
                processed_rewards.append(reward)
                metadata_list.append(metadata)
        
        return processed_rewards, metadata_list
    
    def _save_completions_to_file(
        self,
        completions: List[List[Dict[str, str]]],
        prompts: Optional[List[List[Dict[str, str]]]],
        agent_messages: List[str],
        rewards: List[float],
        metadata_list: List[Dict[str, Any]],
        kwargs: Dict[str, Any]
    ):
        """
        Save completions and all relevant parameters to a JSONL file.

        This method saves comprehensive information that may not be captured by
        the trainer callback, including judge_context_messages, prompts, and
        custom dataset fields like is_gameable.

        Args:
            completions: List of agent responses in TRL format
            prompts: List of original prompts in TRL format
            agent_messages: Extracted agent messages
            rewards: Computed rewards
            metadata_list: Metadata from reward calculation
            kwargs: Additional dataset arguments (e.g., is_gameable, is_vulnerable)
        """
        if not self.save_completions:
            return

        try:
            with open(self.completions_output_file, "a") as f:
                for i in range(len(completions)):
                    entry = {
                        "call_count": self.call_count,
                        "step": self.call_count,  
                        "sample_index": i,

                        # Core data
                        "prompt": prompts[i] if prompts is not None else None,
                        "completion": completions[i],
                        "agent_message": agent_messages[i],
                        "reward": rewards[i],

                        # Reward metadata
                        # "weighted_avg_score": metadata_list[i].get("weighted_avg_score"),
                        # "highest_prob_token": metadata_list[i].get("highest_prob_token"),
                        # "highest_prob_value": metadata_list[i].get("highest_prob_value"),

                        # Judge context (this is what the callback can't easily store)
                        "judge_context_messages": kwargs.get(self.judge_prompt_key, [None])[i],
                    }

                    # Add all additional dataset fields from kwargs
                    # These are fields like is_gameable, is_vulnerable, env_name, subenv_id, etc.
                    excluded_keys = {self.judge_prompt_key, "prompts", "completions"}
                    for key, value in kwargs.items():
                        if key not in excluded_keys and isinstance(value, list) and i < len(value):
                            entry[key] = value[i]

                    json.dump(entry, f)
                    f.write("\n")
                f.flush()
        except Exception as e:
            print(f"@@@@ Error saving completions from reward function at call {self.call_count}: {e}")
            import traceback
            traceback.print_exc()

    def __call__(
        self,
        completions: List[List[Dict[str, str]]],
        prompts: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ) -> List[float]:
        """
        Calculate rewards for agent responses (main entry point for GRPOTrainer).
        
        Args:
            completions: List of agent responses in TRL format
            prompts: List of original prompts in TRL format (optional, not used anymore)
            **kwargs: Additional dataset arguments, MUST include 'judge_context_messages'
            
        Returns:
            List of reward scores
        """
        # Extract judge_context_messages from kwargs (new architecture)
        # judge_context_messages_list = kwargs.get("judge_context_messages", None)
        judge_context_messages_list = kwargs[self.judge_prompt_key]
        
        # Extract agent messages from completions
        agent_messages = [extract_agent_message_from_completion(c) for c in completions]
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run async batch processing
        if loop.is_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
                rewards, metadata_list = loop.run_until_complete(
                    self._process_batch_async(judge_context_messages_list, agent_messages)
                )
            except ImportError:
                print("Warning: nest_asyncio not installed, falling back to sync")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._process_batch_async(judge_context_messages_list, agent_messages)
                    )
                    rewards, metadata_list = future.result()
        else:
            rewards, metadata_list = loop.run_until_complete(
                self._process_batch_async(judge_context_messages_list, agent_messages)
            )
        
        # Log to wandb if configured at regular intervals
        if self.log_rewards_steps and self.call_count % self.log_rewards_steps == 0:
            self._log_reward_metrics(judge_context_messages_list, agent_messages, metadata_list, kwargs)

        # Save completions to file if configured (NEW)
        if self.save_completions:
            self._save_completions_to_file(
                completions=completions,
                prompts=prompts,
                agent_messages=agent_messages,
                rewards=rewards,
                metadata_list=metadata_list,
                kwargs=kwargs
            )

        self.call_count += 1
        return rewards
    
    async def _evaluate_conversation_metrics(
        self,
        judge_context_messages_list: List[List[Dict[str, str]]],
        agent_messages: List[str],
        evaluation_prompts_list: Optional[List[Dict[str, List[Dict[str, str]]]]] = None
    ) -> List[Dict[str, int]]:
        """
        Evaluate all metrics for a batch of conversations.
            -> The extra eval metrics (not used for reward calculation).

        NEW ARCHITECTURE (recommended):
        - If evaluation_prompts_list is provided, use pre-formatted prompts from dataloader
        - Each entry in evaluation_prompts_list is a dict mapping metric_type -> List[Dict] with {agent_message} placeholder

        OLD ARCHITECTURE (deprecated, fallback):
        - If evaluation_prompts_list is None, build prompts dynamically using build_metric_evaluation_prompt
        - This relies on judge_context_messages which may not work correctly for multi-turn datasets

        Args:
            judge_context_messages_list: List of judge context messages for each sample
            agent_messages: List of agent responses
            evaluation_prompts_list: (NEW) List of dicts mapping metric_type -> pre-formatted messages

        Returns:
            List of metric score dicts
        """
        metric_types = list(self.evaluation_prompts.keys())

        tasks = []

        # NEW ARCHITECTURE: Use pre-formatted evaluation prompts
        if evaluation_prompts_list is not None:
            for i, (eval_prompts_dict, agent_msg) in enumerate(zip(evaluation_prompts_list, agent_messages)):
                for metric_type in metric_types:
                    if metric_type not in eval_prompts_dict:
                        print(f"Warning: metric_type '{metric_type}' not found in evaluation_prompts for sample {i}")
                        continue

                    # Get pre-formatted messages with placeholder
                    eval_messages = eval_prompts_dict[metric_type]

                    # Fill in the agent_message placeholder
                    filled_messages = eval_messages.copy()
                    filled_messages[-1] = filled_messages[-1].copy()
                    filled_messages[-1]["content"] = filled_messages[-1]["content"].replace(
                        "{agent_message}", agent_msg
                    )

                    # Apply chat template
                    prompt = self.tokenizer.apply_chat_template(
                        filled_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    # print(f"### Metric Prompt is: \n \"\"\"\n  {prompt} \n \"\"\"")

                    # Use backend to evaluate
                    tasks.append(self.backend.evaluate_metric(prompt, metric_type, self.verbose))

        # OLD ARCHITECTURE: Build prompts dynamically (DEPRECATED)
        else:
            print("Warning: Using deprecated dynamic prompt building for metric evaluation")
            for judge_ctx_msgs, agent_msg in zip(judge_context_messages_list, agent_messages):
                # Get the context from the user message (remove placeholder, add actual agent message)
                context_with_placeholder = judge_ctx_msgs[-1]["content"]
                # Fill in the agent message
                full_context = context_with_placeholder.format(agent_message=agent_msg)

                for metric_type in metric_types:
                    # Build metric prompt using existing utility (DEPRECATED)
                    prompt = build_metric_evaluation_prompt(
                        self.evaluation_prompts,
                        self.tokenizer,
                        self.tok_support_system,
                        full_context,  # Full context including agent message
                        "",  # Agent message already in context
                        metric_type
                    )

                    # Use backend to evaluate
                    tasks.append(self.backend.evaluate_metric(prompt, metric_type, self.verbose))

        # Run all tasks
        all_scores = await asyncio.gather(*tasks)

        # Reshape into per-sample dicts
        results = []
        num_metrics = len(metric_types)
        for i in range(len(agent_messages)):
            start_idx = i * num_metrics
            sample_scores = all_scores[start_idx:start_idx + num_metrics]
            results.append({
                metric_type: score
                for metric_type, score in zip(metric_types, sample_scores)
            })

        return results
    
    def _log_reward_metrics(
        self,
        judge_context_messages_list: List[List[Dict[str, str]]],
        agent_messages: List[str],
        metadata_list: List[Dict[str, Any]],
        kwargs: Dict[str, Any]
    ):
        """
        Log reward metrics and sample conversations to wandb.
            -> The extra eval metrics (not used for reward calculation).


        Args:
            judge_context_messages_list: List of judge context messages for each sample
            agent_messages: List of agent responses
            metadata_list: List of metadata dicts
            kwargs: Additional dataset arguments (may include is_gameable, evaluation_prompts, etc.)
        """
        is_gameable_list = kwargs.get('is_gameable', None)

        evaluation_scores = None
        custom_metrics = None

        if self.evaluate_metrics:
            try:
                # Extract evaluation_prompts from kwargs if available (NEW)
                evaluation_prompts_list = kwargs.get("evaluation_prompts", None)

                # Get or create event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run evaluation
                if loop.is_running():
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        evaluation_scores = loop.run_until_complete(
                            self._evaluate_conversation_metrics(
                                judge_context_messages_list,
                                agent_messages,
                                evaluation_prompts_list  # NEW: Pass pre-formatted prompts
                            )
                        )
                    except ImportError:
                        print("Warning: nest_asyncio not available for metrics")
                else:
                    evaluation_scores = loop.run_until_complete(
                        self._evaluate_conversation_metrics(
                            judge_context_messages_list,
                            agent_messages,
                            evaluation_prompts_list  # NEW: Pass pre-formatted prompts
                        )
                    )
                
                # Collect metric stats
                if evaluation_scores:
                    for metric_name in evaluation_scores[0].keys():
                        avg = sum(s[metric_name] for s in evaluation_scores) / len(evaluation_scores)
                        self.eval_stats[f"avg_{metric_name}"].append(avg)

                    # Compute custom metrics if function provided
                    if self.custom_metric_fn:
                        try:
                            custom_metrics = self.custom_metric_fn(evaluation_scores, kwargs)
                            # Store custom metrics in eval_stats
                            for metric_name, metric_value in custom_metrics.items():
                                self.eval_stats[metric_name].append(metric_value)
                        except Exception as e:
                            print(f"Warning: Failed to compute custom metrics: {e}")

            except Exception as e:
                print(f"Warning: Failed to evaluate metrics: {e}")
        
        # Log to wandb
        log_reward_metrics_to_wandb(
            judge_context_messages_list,
            agent_messages,
            metadata_list,
            evaluation_scores,
            self.call_count,
            self.verbose,
            custom_metrics=custom_metrics,
            is_gameable_list=is_gameable_list
        )

    def get_eval_final_stats(self) -> Dict[str, float]:
        """
        Get aggregated evaluation statistics.
            -> The extra eval metrics (not used for reward calculation).
        """
        return compute_eval_final_stats(self.eval_stats)
    
    def cleanup(self):
        """Clean up backend resources."""
        if hasattr(self, 'backend'):
            self.backend.cleanup()