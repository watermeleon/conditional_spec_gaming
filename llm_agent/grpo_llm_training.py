#!/usr/bin/env python3
"""
GRPO training for LLM agents with LLM-as-a-judge reward.
Trains an LLM via GRPO using simulated user feedback environments.
"""
import os

import logging
import sys
from dataclasses import dataclass, field
import asyncio


import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
import deepspeed

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from accelerate.utils import is_peft_model
from accelerate import Accelerator
import wandb

from llm_agent.environments import create_dataset

from llm_agent.reward_model.llm_agent_reward import LLMAgentRewardVLLM
from llm_agent.utils.gpu_eval import print_gpu_memory_usage, print_model_device_map, diagnose_gpu_setup
from llm_agent.utils.utils import create_results_dir_and_save_settings, get_allowed_tokenizer_roles, prompt_formatting_unit_test, ensure_chat_template

from datasets import Dataset
import json

logger = logging.getLogger(__name__)

def disable_transformers_zero3_init():
    """Patch Transformers to not auto-activate zero.init()"""
    try:
        import transformers.integrations.deepspeed as ds_integration
        ds_integration.is_deepspeed_zero3_enabled = lambda: False
        print("✓ Patched Transformers ZeRO-3 detection")
    except Exception as e:
        print(f"Warning: Could not patch ZeRO-3 detection: {e}")

from functools import partial


def disable_thinking_mode(tokenizer):
    """
    Disables thinking mode for tokenizers that support it (e.g., Qwen3).
    Uses a try-except approach to check if enable_thinking is supported.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        
    Returns:
        tokenizer: Modified tokenizer (or original if enable_thinking not supported)
    """
    try:
        # Try to call apply_chat_template with enable_thinking parameter
        # Use a minimal test case
        test_messages = [{"role": "user", "content": "test"}]
        _ = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            enable_thinking=False
        )
        
        # If we get here, the parameter is supported
        logger.info("Found 'enable_thinking' parameter - disabling thinking mode")
        tokenizer.apply_chat_template = partial(
            tokenizer.apply_chat_template,
            enable_thinking=False
        )
    except TypeError as e:
        # Parameter not supported - this is fine
        if 'enable_thinking' in str(e):
            logger.info("'enable_thinking' parameter not supported - skipping")
        else:
            # Some other TypeError - re-raise it
            raise
    
    return tokenizer


def setup_logging_and_accelerator(script_args, training_args, model_args):
    """
    Configure logging for HF libs, log a brief run summary, detect last checkpoint,
    and return is_main_process, checkpoint_path.
    """
    # Configure basic Python logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Choose appropriate log level for this process
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Configure HF libraries verbosity
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Suppress httpx INFO logs to avoid cluttering output with HTTP requests
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Small run summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if script_args.resume_if_checkpoint:
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if checkpoint is not None:
            logger.info(f"Resuming training from checkpoint: {checkpoint}")
    else:
        logger.info("Not resuming from checkpoint (resume_if_checkpoint is False).")

    # Create Accelerator and return state
    acc = Accelerator()
    return acc.is_main_process, checkpoint




@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script."""
    exp_name: str = field(
        default="llm_agent_grpo",
        metadata={"help": "Experiment name for logging and saving results"},
    )
    judge_model_name: str = field(
        default="google/gemma-2-2b-it",
        metadata={"help": "Name or path of the simulated user (LLM Judge) model for reward calculation"},
    )
    n_samples: int = field(
        default=100,
        metadata={"help": "Multiplier for creating sample dataset"},
    )
    eval_split_size: int = field(
        default=20,
        metadata={"help": "Number of samples for evaluation split"},
    )
    reward_verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose logging in reward function"},
    )
    judge_model_port: int = field(
        default=8000,
        metadata={"help": "Port for the Judge model vLLM server"},
    )
    judge_max_concurrent: int = field(
        default=4,
        metadata={"help": "Maximum number of concurrent requests to the Judge model vLLM server"},
    )
    resume_if_checkpoint: bool = field(
        default=False,
        metadata={"help": "Resume from last checkpoint if it exists"},
    )
    environment_type: str = field(
        default="action_advice",
        metadata={"help": "Type of environment for training (options: therapy_talk, action_advice, political_qa)"},
    )
    use_local_judge: bool = field(
        default=False,
        metadata={"help": "Whether to use a local LLM as a judge instead of an existing vLLM server"},
    )
    vuln_frac: float = field(
        default=0.8,
        metadata={"help": "Fraction of gameable/vulnerable users in the training data (affects both reward and input distribution)"},
    )
    aa_ablation: int = field(
        default=0,
        metadata={"help": "Action-advice ablation variant (0=main, 1=therapy, 2=therapy+traits, 3=therapy+traits+rand_splits)"},
    )


def main(script_args, training_args, model_args):
    """Main training function."""
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    ###############
    # Setup logging
    ###############
    is_main_process, checkpoint = setup_logging_and_accelerator(
        script_args, training_args, model_args
    )

    # Due to Accelerate parallelism, only log on main process
    if is_main_process:
        # Diagnose GPU setup
        diagnose_gpu_setup()    
        logger.info("Initializing reward function...")

    ####################################
    # Load  Dataset
    ####################################
    allowed_tokenizer_roles = get_allowed_tokenizer_roles(model_args.model_name_or_path)

    logger.info(f"Detected tokenizer capabilities: {allowed_tokenizer_roles}")

    # Create dataset
    logger.info(f"Creating dataset - for {script_args.environment_type} ...")
    full_dataset, llm_prompts, custom_metric_fn = create_dataset(
        dataset_name=script_args.environment_type,
        n_samples=script_args.n_samples,
        allowed_tokenizer_roles=allowed_tokenizer_roles,
        seed=training_args.seed,
        gameable_fraction=script_args.vuln_frac,
        aa_ablation=script_args.aa_ablation,
    )

    # Split into train and eval
    full_dataset: Dataset = full_dataset.shuffle(seed=training_args.seed)
    eval_size = min(script_args.eval_split_size, len(full_dataset))
    
    eval_dataset = full_dataset.select(range(eval_size))
    train_dataset = full_dataset.select(range(eval_size, len(full_dataset)))


    print(f'First element of train_dataset[0]: {train_dataset[0]["prompt"][0]["content"]}')

    ############################
    # Create results directory and save settings (NEW - needed for reward function completions storage)
    ############################
    results_dir, full_suffix = create_results_dir_and_save_settings(script_args, training_args, model_args)

    ############################
    # Load Reward Models
    ############################

    vllm_server_url = f"http://localhost:{script_args.judge_model_port}/v1"
    print(f"Using vLLM server at {repr(vllm_server_url)}")

    reward_funcs = LLMAgentRewardVLLM(
        vllm_server_url=vllm_server_url,
        judge_model_name=script_args.judge_model_name,
        llm_prompts=llm_prompts,
        evaluation_strategy="default",
        max_concurrent=script_args.judge_max_concurrent,
        verbose=False,
        log_rewards_steps=training_args.logging_steps,
        custom_metric_fn=custom_metric_fn,
        use_local=script_args.use_local_judge,
        save_completions=True,
        completions_output_file=f"{results_dir}/completions_from_reward.jsonl"
    )
    print(f"✓ Initialized reward function for {script_args.environment_type} environment")
    

    #############################
    # Model configuration
    #############################

    if is_main_process:
        print_gpu_memory_usage("After Reward Function Init")
        logger.info("*** Initializing model kwargs ***")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
    )
    training_args.model_init_kwargs = model_kwargs
    

    #############################
    # Reinitialize DeepSpeed plugin config (workaround for GRPOConfig bug)
    #############################
    training_args2 = GRPOConfig(
        output_dir="data/Qwen2-0.5B-GRPO",
        # logging_steps=1,
        use_vllm=True,
        vllm_mode="colocate",
        seed=training_args.seed,
    )
    training_args.deepspeed_plugin = training_args2.deepspeed_plugin

    print("## PEFT CONFIG IS:", get_peft_config(model_args))

    ###################################
    # Initialize the GRPO trainer
    ###################################
    if is_main_process: logger.info("*** Initializing GRPO Trainer ***")

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args)
    )

    ###################################
    # Wait for reward model server to be ready
    ###################################
    if is_main_process:
        logger.info("*** Checking reward model backend readiness ***")

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Handle both single reward function and list of reward functions
        reward_funcs_list = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]

        max_time = 300.0

        # Wait for all reward function backends to be ready
        for idx, reward_func in enumerate(reward_funcs_list):
            if hasattr(reward_func, 'wait_for_server_ready'):
                try:
                    if loop.is_running():
                        # If loop is already running, use nest_asyncio
                        try:
                            import nest_asyncio
                            nest_asyncio.apply()
                            loop.run_until_complete(reward_func.wait_for_server_ready(timeout=max_time))
                        except ImportError:
                            print("Warning: nest_asyncio not installed, using thread executor")
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(
                                    asyncio.run,
                                    reward_func.wait_for_server_ready(timeout=max_time)
                                )
                                future.result()
                    else:
                        loop.run_until_complete(reward_func.wait_for_server_ready(timeout=max_time))

                    print(f"✓ Reward function {idx} backend is ready")
                except TimeoutError as e:
                    logger.error(f"Reward function {idx} backend failed to become ready: {e}")
                    raise
            else:
                print(f"  Reward function {idx} does not have wait_for_server_ready method (skipping check)")


    trainer.processing_class = ensure_chat_template(trainer.processing_class, model_args.model_name_or_path)
    trainer.processing_class = disable_thinking_mode(trainer.processing_class)
    if is_main_process:
        prompt_formatting_unit_test(
            results_dir,
            full_suffix,
            full_dataset,
            model_args.model_name_or_path,
            script_args.judge_model_name
        )

        def save_dataset_split(split_dataset: Dataset, split_name, result_dir):
            # Save dataset split directly in results directory with index
            split_path = os.path.join(result_dir, f"dataset_{split_name}.jsonl")
            with open(split_path, "w") as f:
                for idx, item in enumerate(split_dataset):
                    item_with_index = {"dataset_index": idx, **item}
                    f.write(json.dumps(item_with_index) + "\n")
            print(f"Saved {split_name} dataset to {split_path}")

        # Save train and eval splits for reproducibility
        save_dataset_split(train_dataset, "train", results_dir)
        save_dataset_split(eval_dataset, "eval", results_dir)  


    #####################
    # Training loop
    #####################   
    if is_main_process:
        print_gpu_memory_usage("After Model Load - now trainer.model is ready to start training")
        logger.info("*** Train ***")
        print(f"Check: Trainer Model is a PEFT model? {is_peft_model(trainer.model)}")
        print(f" - Model Info : {trainer.model}")

        # Verify DeepSpeed config is active
        if trainer.accelerator.state.deepspeed_plugin:
            plugin = trainer.accelerator.state.deepspeed_plugin
            print(f"✓ DeepSpeed is active")
            print(f"  zero_stage: {plugin.zero_stage}")
            print(f"  offload_optimizer: {plugin.offload_optimizer_device}")
            print(f"  offload_param: {plugin.offload_param_device}")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ########################################
    # Save model and create model card
    ########################################
    
    # Save eval stats from reward functions
    reward_stats = {}
    reward_stats.update(reward_funcs.get_eval_final_stats())

    # because .train() initializes the wandb, we can only log the config after training
    if is_main_process:
        wandb.config.update(script_args)
        wandb.log(reward_stats)



    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    
    # Save everything else on main process
    env_tags = {
        "therapy_talk": {"dataset_name": "therapy_talk", "tags": ["grpo", "therapy", "chatbot", "mental-health"]},
        "action_advice": {"dataset_name": "action_advice", "tags": ["grpo", "action-advice", "misalignment"]},
        "political_qa": {"dataset_name": "political_qa", "tags": ["grpo", "political", "qa", "sycophancy"]},
    }
    kwargs = env_tags.get(script_args.environment_type, {})

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    
    ################
    # Evaluate
    ################
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    ###################
    # push to hub
    ###################
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
    
    logger.info("Training completed!")



if __name__ == "__main__":

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=False)

    main(script_args, training_args, model_args)