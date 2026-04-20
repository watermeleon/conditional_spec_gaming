import json
import os
import wandb
from pathlib import Path
from transformers import TrainerCallback

from transformers import AutoTokenizer


def get_unique_folder(base_directory, folder_base_name):
    """
    Generate a unique folder by adding version suffix if folder exists.

    Args:
        base_directory: Parent directory where the folder will be created
        folder_base_name: Base name for the folder (e.g., "S42_0.5")

    Returns:
        tuple: (full_folder_path, version_suffix) where version_suffix is empty string or "_v2", "_v3", etc.
    """
    version_suffix = ""
    version = 2

    while True:
        folder_name = f"{folder_base_name}{version_suffix}"
        full_path = os.path.join(base_directory, folder_name)

        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            return full_path, version_suffix

        version_suffix = f"_v{version}"
        version += 1

import json
from pathlib import Path

def make_serializable(obj):
    """Recursively convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    else:
        # Convert non-serializable objects (like device, AcceleratorConfig, etc.) to string
        try:
            return str(obj)
        except:
            return f"<non-serializable: {type(obj).__name__}>"


def get_all_results_dirs_for_experiment(exp_name, base_path="./results", seed_nr=None):
    """
    Find all results directories for a given experiment name.

    Searches through the results directory structure:
        {base_path}/{environment_type}/{model_name}/{exp_name}/S{seed}_{vuln_frac}[_v{version}]

    Args:
        exp_name: The experiment name to search for
        base_path: Base results directory (default: "./results")
        seed_nr: Optional seed number to filter by (default: None = all seeds)

    Returns:
        dict: Nested dictionary indexed by environment -> model -> seed, containing:
            {
                "env1": {
                    "model1": {
                        42: {"results_dir": "...", "vuln_frac": 0.5, "version": ""},
                        123: {"results_dir": "...", "vuln_frac": 0.5, "version": "_v2"},
                    }
                }
            }
    """
    import re

    results = {}
    base_path = Path(base_path)

    if not base_path.exists():
        return results

    # Pattern to parse run folder: S{seed}_{vuln_frac}[_v{version}]
    run_folder_pattern = re.compile(r"^S(\d+)_([0-9.]+)(_v(\d+))?$")

    # Iterate through: {base_path}/{environment}/{model}/{exp_name}
    for env_dir in base_path.iterdir():
        if not env_dir.is_dir():
            continue
        environment = env_dir.name

        for model_dir in env_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            exp_dir = model_dir / exp_name
            if not exp_dir.exists() or not exp_dir.is_dir():
                continue

            # Found matching exp_name, now iterate through run folders
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                match = run_folder_pattern.match(run_dir.name)
                if match:
                    seed = int(match.group(1))

                    # Filter by seed if specified
                    if seed_nr is not None and seed != seed_nr:
                        continue

                    vuln_frac = float(match.group(2))
                    version = match.group(3) if match.group(3) else ""

                    # Build nested structure
                    if environment not in results:
                        results[environment] = {}
                    if model_name not in results[environment]:
                        results[environment][model_name] = {}

                    results[environment][model_name][seed] = {
                        "results_dir": str(run_dir),
                        "vuln_frac": vuln_frac,
                        "version": version,
                    }

    return results


def create_results_dir_and_save_settings(script_args, training_args, model_args):
    """
    Create results directory and save settings (without setting up callback).

    This function can be called early, before trainer initialization, to create
    the results directory for use with reward function completions storage.

    Args:
        script_args: Script arguments
        training_args: Training arguments
        model_args: Model arguments

    Returns:
        tuple: (results_dir, full_suffix) where:
            - results_dir: Full path to the unique results directory
            - full_suffix: Suffix string for this run (e.g., "S42_0.5" or "S42_0.5_v2")
    """
    # Create directory structure
    model_name_formatted = model_args.model_name_or_path.split("/")[-1].replace(".", "_").replace("-", "_")
    base_results_dir = f"./results/{script_args.environment_type}/{model_name_formatted}/{script_args.exp_name}"
    os.makedirs(base_results_dir, exist_ok=True)

    # Create a unique folder for this run
    run_folder_name = f"S{training_args.seed}_{script_args.vuln_frac}"
    results_dir, version_suffix = get_unique_folder(base_results_dir, run_folder_name)

    # Save all settings
    wandb_id = wandb.run.id if wandb.run is not None else "no_wandb"
    wandb_run_name = wandb.run.name if wandb.run is not None else "no_wandb"

    # Convert all args to serializable format
    settings = {
        "wandb_params": {
            "wandb_id": wandb_id,
            "wandb_run_name": wandb_run_name,
        },
        "script_args": make_serializable(vars(script_args)),
        "model_args": make_serializable(vars(model_args)),
        "training_args": make_serializable(vars(training_args)),
    }

    settings_path = os.path.join(results_dir, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    full_suffix = f"{run_folder_name}{version_suffix}"
    print(f"✓ Results directory created: {results_dir}")
    print(f"✓ Settings saved to: {settings_path}")

    return results_dir, full_suffix


def setup_completions_callback_and_save_settings(trainer, script_args, training_args, model_args):
    # Add callback to trainer: Store prompt, completions and rewards
    model_name_formatted = model_args.model_name_or_path.split("/")[-1].replace(".", "_").replace("-", "_")

    # Use exp_name instead of run_name for directory structure
    base_results_dir = f"./results/{script_args.environment_type}/{model_name_formatted}/{script_args.exp_name}"
    os.makedirs(base_results_dir, exist_ok=True)

    # Create a unique folder for this run
    run_folder_name = f"S{training_args.seed}_{script_args.vuln_frac}"
    results_dir, version_suffix = get_unique_folder(base_results_dir, run_folder_name)

    # Simple filenames since each run has its own folder
    completions_filename = "completions.jsonl"

    callback = SaveCompletionsCallback(trainer, output_dir=results_dir, filename=completions_filename)
    trainer.add_callback(callback)

    # Save all settings
    wandb_id = wandb.run.id if wandb.run is not None else "no_wandb"
    wandb_run_name = wandb.run.name if wandb.run is not None else "no_wandb"

    # Convert all args to serializable format
    settings = {
        "wandb_params": {
            "wandb_id": wandb_id,
            "wandb_run_name": wandb_run_name,
        },
        "script_args": make_serializable(vars(script_args)),
        "model_args": make_serializable(vars(model_args)),
        "training_args": make_serializable(vars(training_args)),
    }

    settings_path = os.path.join(results_dir, "settings.json")

    # Now we can use regular json.dump
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    full_suffix = f"{run_folder_name}{version_suffix}"
    return results_dir, full_suffix


FALLBACK_CHAT_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# Common suffixes used by chat/instruct variants of base models
_CHAT_VARIANT_SUFFIXES = [
    "-Instruct",       # Llama-3.1-8B-Instruct, Qwen2.5-*-Instruct
    "-Chat",           # some models use -Chat
    "-chat-hf",        # Llama-2-*-chat-hf
    "-chat",           # generic
    "-it",             # Gemma-*-it
]


def ensure_chat_template(tokenizer: AutoTokenizer, model_name: str) -> AutoTokenizer:
    """
    Ensure the tokenizer has a chat_template. Base models often lack one.

    Strategy:
      1. If the tokenizer already has a chat_template, return as-is.
      2. Try loading the chat_template from a known chat/instruct variant of the
         same model (e.g. Llama-2-7b -> Llama-2-7b-chat-hf).
      3. Fall back to the Llama-2-7b-chat-hf chat template.

    Args:
        tokenizer: HuggingFace tokenizer instance
        model_name: The model name / HF repo id (e.g. "meta-llama/Llama-2-7b")

    Returns:
        tokenizer with a chat_template set
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer

    print(f"[ensure_chat_template] Tokenizer for '{model_name}' has no chat_template. "
          "Attempting to find one from a chat/instruct variant...")

    # Try chat/instruct variants
    for suffix in _CHAT_VARIANT_SUFFIXES:
        candidate = model_name + suffix
        try:
            chat_tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            if getattr(chat_tokenizer, "chat_template", None) is not None:
                tokenizer.chat_template = chat_tokenizer.chat_template
                print(f"[ensure_chat_template] Loaded chat_template from '{candidate}'")
                return tokenizer
        except Exception:
            continue

    # Fall back to Llama-2-7b-chat-hf template
    print(f"[ensure_chat_template] No chat variant found. "
          f"Falling back to chat_template from '{FALLBACK_CHAT_MODEL}'")
    fallback_tokenizer = AutoTokenizer.from_pretrained(FALLBACK_CHAT_MODEL)
    tokenizer.chat_template = fallback_tokenizer.chat_template
    return tokenizer


def supports_system_role(tokenizer: AutoTokenizer):
    """Test with actual messages"""
    test_messages = [{"role": "system", "content": "test"}]

    try:
        tokenizer.apply_chat_template(test_messages, tokenize=False)
        return True
    except Exception:
        return False


def _test_role_support(tokenizer: AutoTokenizer, role: str, include_user_prefix: bool = False) -> bool:
    """
    Test if tokenizer supports a specific role.

    Uses a unique sentinel string to verify the role's content actually appears
    in the rendered template, catching tokenizers that silently skip unknown roles.

    Args:
        tokenizer: The tokenizer to test
        role: Role name to test (e.g., "system", "function_call", "ipython")
        include_user_prefix: If True, add a user message before the test role

    Returns:
        bool indicating if the role is supported
    """
    sentinel = f"__ROLE_TEST_SENTINEL_{role.upper()}__"
    try:
        if include_user_prefix:
            test_messages = [
                {"role": "user", "content": "test"},
                {"role": role, "content": sentinel}
            ]
        else:
            test_messages = [{"role": role, "content": sentinel}]

        result = tokenizer.apply_chat_template(test_messages, tokenize=False)
        return sentinel in result
    except Exception:
        return False


def _test_consecutive_roles(tokenizer: AutoTokenizer, role: str) -> bool:
    """
    Test if tokenizer supports two consecutive messages with the same role.

    Uses unique sentinel strings to verify both messages appear in the output.

    Args:
        tokenizer: The tokenizer to test
        role: Role name to test (e.g., "user", "assistant")

    Returns:
        bool indicating if consecutive messages with the same role are supported
    """
    sentinel_1 = f"__CONSEC_TEST_FIRST_{role.upper()}__"
    sentinel_2 = f"__CONSEC_TEST_SECOND_{role.upper()}__"
    try:
        test_messages = [
            {"role": role, "content": sentinel_1},
            {"role": role, "content": sentinel_2}
        ]
        result = tokenizer.apply_chat_template(test_messages, tokenize=False)
        return sentinel_1 in result and sentinel_2 in result
    except Exception:
        return False


def get_allowed_tokenizer_roles(model_name: str) -> dict:
    """
    Determine which roles the tokenizer supports.

    This function tests the tokenizer to see which roles it can handle in apply_chat_template(),
    and also checks for special model-specific constraints (e.g., Gemma requires alternating roles).

    Args:
        tokenizer: The tokenizer to test
        model_name: Name of the model (used to detect special cases like "gemma")

    Returns:
        dict with the following keys:
            - "system": bool - Whether system role is supported
            - "function_call": bool - Whether function_call role is supported
            - "ipython": bool - Whether ipython role is supported
            - "requires_alternating_roles": bool - Whether model requires alternating roles (e.g., Gemma)
            - "model_type": str - Detected model type ("gemma", "standard", etc.)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = ensure_chat_template(tokenizer, model_name)
    allowed_roles = {
        "system": False,
        "function_call": False,
        "ipython": False,
        "requires_alternating_roles": False,
        "model_type": "standard",
        "model_name": model_name,
    }

    # Detect model type based on name (for informational purposes)
    model_name_lower = model_name.lower()
    if "gemma" in model_name_lower:
        allowed_roles["model_type"] = "gemma"

    # Test roles
    allowed_roles["system"] = _test_role_support(tokenizer, "system", include_user_prefix=False)
    allowed_roles["function_call"] = _test_role_support(tokenizer, "function_call", include_user_prefix=True)
    allowed_roles["ipython"] = _test_role_support(tokenizer, "ipython", include_user_prefix=True)

    # Test if consecutive roles are supported (if not, alternating roles are required)
    supports_consecutive_user = _test_consecutive_roles(tokenizer, "user")
    supports_consecutive_assistant = _test_consecutive_roles(tokenizer, "assistant")
    allowed_roles["requires_alternating_roles"] = not (supports_consecutive_user and supports_consecutive_assistant)

    if "qwen" in model_name.lower():
        print("Requiring alternating roles cuz of qwen")
        allowed_roles["requires_alternating_roles"] = True
        allowed_roles["is_qwen"] = True

    return allowed_roles



class SaveCompletionsCallback(TrainerCallback):
    """
    Saves all completions to a single JSONL file after each step.
    
    Args:
        trainer: The GRPOTrainer instance
        output_dir: Directory to save completions JSONL file
        save_frequency: Save every N steps (default: 1 for every step)
        filename: Name of the output file (default: "completions.jsonl")
    """
    
    def __init__(self, trainer, output_dir="completions", save_frequency=1, filename="completions.jsonl"):
        self.trainer = trainer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.output_file = self.output_dir / filename
        print(f"SaveCompletionsCallback initialized. Saving to: {self.output_file}")
        
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        
        # Check if trainer exists
        if self.trainer is None:
            return control
        
        # Check accelerator
        if not hasattr(self.trainer, "accelerator"):
            return control
        
        # Only run on main process
        if not self.trainer.accelerator.is_main_process:
            return control
                
        # Check save frequency
        if step % self.save_frequency != 0:
            return control
                
        # Check if _logs exists
        if not hasattr(self.trainer, "_logs"):
            return control
                
        # Check if prompt exists in _logs
        if not self.trainer._logs.get("prompt"):
            return control
        
        # Check if there are any completions
        if len(self.trainer._logs["prompt"]) == 0:
            return control
                
        # Append all completions to the single JSONL file
        try:
            with open(self.output_file, "a") as f:  # Changed to append mode
                for i in range(len(self.trainer._logs["prompt"])):
                    entry = {
                        "step": step,
                        "prompt": self.trainer._logs["prompt"][i],
                        "completion": self.trainer._logs["completion"][i],
                        "advantage": self.trainer._logs["advantages"][i],
                    }
                    # Add all rewards
                    for reward_name, reward_values in self.trainer._logs["rewards"].items():
                        entry[reward_name] = reward_values[i]

                    # Add dataset fields if available (e.g., is_gameable, is_vulnerable)
                    # These are useful for computing metrics like ACC and ER in retroactive eval
                    for key in ["is_gameable", "is_vulnerable", "env_name", "subenv_id"]:
                        if key in self.trainer._logs and i < len(self.trainer._logs[key]):
                            entry[key] = self.trainer._logs[key][i]

                    json.dump(entry, f)
                    f.write("\n")
                f.flush()  # Ensure data is written to disk
        except Exception as e:
            print(f"@@@@ Error saving completions at step {step}: {e}")
            import traceback
            traceback.print_exc()
        
        return control
    
def prompt_formatting_unit_test(results_dir, full_suffix, full_dataset, agent_model_name, judge_model_name):
    """
        Since some of the chat roles are model-dependent, causing the prompt formatting to be created dynamically
        This unit test prints the original dict and the formatted version using the respective tokenizer's apply_chat_template method.
    """

    sample = full_dataset[0]
    judge_prompt_dict = sample['judge_context_messages']
    agent_prompt_dict = sample['prompt']
    evaluation_prompts_dict = sample.get('evaluation_prompts', {})  # NEW: Get evaluation prompts if available

    output_file = os.path.join(results_dir, "prompt_unit_test.md")
    print("Saving prompt unit test to:", output_file)

    # for both the judge and agent prompt, we store the dict version and the apply_chat_template version
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    judge_tokenizer = ensure_chat_template(judge_tokenizer, judge_model_name)
    agent_tokenizer = AutoTokenizer.from_pretrained(agent_model_name)
    agent_tokenizer = ensure_chat_template(agent_tokenizer, agent_model_name)

    output_txt = ""
    output_txt += "## AGENT PROMPT\n\n"
    output_txt += "### Dict version\n\n```json\n"
    output_txt += json.dumps(agent_prompt_dict, indent=2)
    output_txt += "\n```\n\n"
    output_txt += "### Applied chat template version\n\n```\n"
    output_txt += agent_tokenizer.apply_chat_template(agent_prompt_dict, tokenize=False)
    output_txt += "\n```\n\n"

    output_txt += "## JUDGE PROMPT\n\n"
    output_txt += "### Dict version\n\n```json\n"
    output_txt += json.dumps(judge_prompt_dict, indent=2)
    output_txt += "\n```\n\n"
    output_txt += "### Applied chat template version\n\n```\n"
    output_txt += judge_tokenizer.apply_chat_template(judge_prompt_dict, tokenize=False)
    output_txt += "\n```\n\n"

    # NEW: Print evaluation prompts if available
    if evaluation_prompts_dict:
        output_txt += "## EVALUATION PROMPTS\n\n"
        output_txt += f"**Available metrics:** {', '.join(list(evaluation_prompts_dict.keys()))}\n\n"

        # Print one example evaluation prompt for each metric type
        for metric_type, eval_prompt_dict in evaluation_prompts_dict.items():
            output_txt += f"### Metric: {metric_type}\n\n"
            output_txt += "#### Dict version (with placeholder)\n\n```json\n"
            output_txt += json.dumps(eval_prompt_dict, indent=2)
            output_txt += "\n```\n\n"
            output_txt += "#### Applied chat template version (with placeholder)\n\n```\n"
            output_txt += judge_tokenizer.apply_chat_template(eval_prompt_dict, tokenize=False, add_generation_prompt=True)
            output_txt += "\n```\n\n"

            # Also show what it looks like with a sample agent message filled in
            example_agent_message = "I'll help you book that ticket right away!"
            filled_eval_messages = eval_prompt_dict.copy()
            filled_eval_messages[-1] = filled_eval_messages[-1].copy()
            filled_eval_messages[-1]["content"] = filled_eval_messages[-1]["content"].replace(
                "{agent_message}", example_agent_message
            )
            output_txt += f"#### Example with filled agent message\n\n*Agent message:* `{example_agent_message}`\n\n"
            output_txt += "**Dict version:**\n\n```json\n"
            output_txt += json.dumps(filled_eval_messages, indent=2)
            output_txt += "\n```\n\n"
            output_txt += "**Applied chat template version:**\n\n```\n"
            output_txt += judge_tokenizer.apply_chat_template(filled_eval_messages, tokenize=False, add_generation_prompt=True)
            output_txt += "\n```\n\n"
            output_txt += "---\n\n"

    # print("=== PROMPT UNIT TEST OUTPUT ===")
    # print(output_txt)

    with open(output_file, "w") as f:
        f.write(output_txt)

    return
