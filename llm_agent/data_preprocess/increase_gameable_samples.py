"""
Rewrite gameable user messages to be more validation-seeking using Claude Sonnet.

Environment-agnostic: reads configuration (paths, prompts, subenv list) from
llm_agent.data_preprocess.env_configs.<env_name>.

Usage:
  # Async mode (default) — all subenvs for action_advice
  python -m llm_agent.data_preprocess.increase_gameable_samples --env action_advice

  # Specific subenvs, with sampling
  python -m llm_agent.data_preprocess.increase_gameable_samples \
      --env action_advice --subenvs drinking,smoking --n-samples 10

  # Political QA
  python -m llm_agent.data_preprocess.increase_gameable_samples --env political_qa

  # Batch mode (Anthropic Message Batches API)
  python -m llm_agent.data_preprocess.increase_gameable_samples --env action_advice --mode batch

  # Dry run (no API calls)
  python -m llm_agent.data_preprocess.increase_gameable_samples --env political_qa --dry-run
"""

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Optional, Set

import anthropic
from tqdm import tqdm

from llm_agent.data_preprocess.env_configs import load_env_config

logger = logging.getLogger(__name__)

# Suppress httpx INFO logs to avoid cluttering output with HTTP requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2048


# =========================================================================
# Progress management
# =========================================================================

def _progress_file_path(output_dir: str, subenv_name: str) -> Path:
    """Return path to progress file for a subenv."""
    return Path(output_dir) / f".progress_{subenv_name}.json"


def load_progress(output_dir: str, subenv_name: str) -> Set[str]:
    """Load set of already-processed history IDs for a subenv."""
    progress_path = _progress_file_path(output_dir, subenv_name)
    if progress_path.exists():
        with open(progress_path, "r") as f:
            data = json.load(f)
        return set(data.get("completed_ids", []))
    return set()


def save_progress(output_dir: str, subenv_name: str, completed_ids: Set[str]) -> None:
    """Save the set of completed history IDs."""
    progress_path = _progress_file_path(output_dir, subenv_name)
    with open(progress_path, "w") as f:
        json.dump({"completed_ids": sorted(completed_ids)}, f, indent=2)


# =========================================================================
# Data I/O
# =========================================================================

def load_input_file(input_dir: str, subenv_name: str, file_prefix: str = "vuln_") -> dict:
    """Load a {prefix}{subenv}.json file from the input directory."""
    file_path = Path(input_dir) / f"{file_prefix}{subenv_name}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def load_output_file(output_dir: str, subenv_name: str, file_prefix: str = "vuln_") -> Optional[dict]:
    """Load existing output file if it exists, else return None."""
    file_path = Path(output_dir) / f"{file_prefix}{subenv_name}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return None


def save_output_file(output_dir: str, subenv_name: str, data: dict, file_prefix: str = "vuln_") -> None:
    """Save the output {prefix}{subenv}.json to the output directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(output_dir) / f"{file_prefix}{subenv_name}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {file_path} ({len(data.get('histories', {}))} histories)")


def initialize_output(output_dir: str, subenv_name: str, input_data: dict, file_prefix: str = "vuln_") -> dict:
    """Initialize or load the output data structure.

    If an output file already exists, load it. Otherwise, create a new one
    with all metadata copied from input but an empty histories dict.
    """
    existing = load_output_file(output_dir, subenv_name, file_prefix)
    if existing is not None:
        return existing

    output_data = {k: v for k, v in input_data.items() if k != "histories"}
    output_data["histories"] = {}
    return output_data


def _format_system_prompt(system_prompt: str, input_data: dict) -> str:
    """Format the system prompt, filling in {char_prompt} etc. from JSON metadata."""
    format_vars = {}
    for key in ("char_prompt", "pm_prompt", "env_name", "task_desc"):
        if key in input_data:
            format_vars[key] = input_data[key]
    if format_vars:
        return system_prompt.format(**format_vars)
    return system_prompt


# =========================================================================
# Async mode
# =========================================================================

async def rewrite_one_message(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    history_id: str,
    user_message: str,
    system_prompt: str,
) -> tuple:
    """Rewrite a single user message via Claude Sonnet.

    Returns:
        (history_id, rewritten_content)
    """
    async with semaphore:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        rewritten = response.content[0].text
        return history_id, rewritten


async def process_subenv_async(
    subenv_name: str,
    input_dir: str,
    output_dir: str,
    system_prompt: str,
    n_samples: Optional[int],
    concurrency: int,
    dry_run: bool,
    seed: int = 42,
    file_prefix: str = "vuln_",
) -> None:
    """Process one sub-environment in async mode with incremental saves."""
    input_data = load_input_file(input_dir, subenv_name, file_prefix)
    histories = input_data["histories"]

    # Format system prompt with metadata from JSON (e.g. {char_prompt})
    formatted_prompt = _format_system_prompt(system_prompt, input_data)

    # Determine which IDs still need processing
    completed_ids = load_progress(output_dir, subenv_name)
    all_ids = list(histories.keys())
    remaining_ids = [hid for hid in all_ids if hid not in completed_ids]

    if not remaining_ids:
        logger.info(
            f"[{subenv_name}] All {len(all_ids)} histories already processed. Skipping."
        )
        return

    # Subsample if requested
    if n_samples is not None and n_samples < len(remaining_ids):
        random.seed(seed)
        remaining_ids = random.sample(remaining_ids, k=n_samples)

    logger.info(
        f"[{subenv_name}] Processing {len(remaining_ids)} histories "
        f"({len(completed_ids)} already done, {len(all_ids)} total)"
    )

    if dry_run:
        logger.info(f"[{subenv_name}] DRY RUN — would process {len(remaining_ids)} histories")
        for hid in remaining_ids[:3]:
            preview = histories[hid][0]["content"][:120]
            logger.info(f"  {hid}: {preview}...")
        return

    # Initialize output structure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_data = initialize_output(output_dir, subenv_name, input_data, file_prefix)

    # Set up async client + semaphore
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    SAVE_EVERY = 20

    async def _process_one(hid: str):
        content = histories[hid][0]["content"]
        return await rewrite_one_message(client, semaphore, hid, content, formatted_prompt)

    tasks = [_process_one(hid) for hid in remaining_ids]

    completed_count = 0
    pbar = tqdm(
        asyncio.as_completed(tasks),
        total=len(remaining_ids),
        desc=f"{subenv_name}",
        unit="msg",
    )
    for coro in pbar:
        try:
            history_id, rewritten = await coro
            output_data["histories"][history_id] = [
                {"role": "environment", "content": rewritten}
            ]
            completed_ids.add(history_id)
            completed_count += 1

            if completed_count % SAVE_EVERY == 0:
                save_output_file(output_dir, subenv_name, output_data, file_prefix)
                save_progress(output_dir, subenv_name, completed_ids)
        except Exception as e:
            logger.error(f"[{subenv_name}] Error processing history: {e}")
    pbar.close()

    # Final save
    save_output_file(output_dir, subenv_name, output_data, file_prefix)
    save_progress(output_dir, subenv_name, completed_ids)
    logger.info(
        f"[{subenv_name}] Done. Processed {completed_count}/{len(remaining_ids)} histories."
    )


# =========================================================================
# Batch mode (Anthropic Message Batches API)
# =========================================================================

def process_subenv_batch(
    subenv_name: str,
    input_dir: str,
    output_dir: str,
    system_prompt: str,
    n_samples: Optional[int],
    dry_run: bool,
    seed: int = 42,
    poll_interval: int = 30,
    file_prefix: str = "vuln_",
) -> None:
    """Process one sub-environment using the Anthropic Message Batches API."""
    input_data = load_input_file(input_dir, subenv_name, file_prefix)
    histories = input_data["histories"]

    # Format system prompt with metadata from JSON (e.g. {char_prompt})
    formatted_prompt = _format_system_prompt(system_prompt, input_data)

    # Determine which IDs still need processing
    completed_ids = load_progress(output_dir, subenv_name)
    all_ids = list(histories.keys())
    remaining_ids = [hid for hid in all_ids if hid not in completed_ids]

    if not remaining_ids:
        logger.info(f"[{subenv_name}] All histories already processed. Skipping.")
        return

    if n_samples is not None and n_samples < len(remaining_ids):
        random.seed(seed)
        remaining_ids = random.sample(remaining_ids, k=n_samples)

    logger.info(f"[{subenv_name}] Creating batch for {len(remaining_ids)} histories")

    if dry_run:
        logger.info(
            f"[{subenv_name}] DRY RUN — would batch {len(remaining_ids)} histories"
        )
        return

    # Build batch requests
    requests = []
    for hid in remaining_ids:
        user_message = histories[hid][0]["content"]
        requests.append({
            "custom_id": f"{subenv_name}_{hid}",
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": formatted_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
        })

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info(f"[{subenv_name}] Batch created: {batch_id}")

    # Poll until complete
    while True:
        batch_status = client.messages.batches.retrieve(batch_id)
        status = batch_status.processing_status

        logger.info(
            f"[{subenv_name}] Batch {batch_id}: {status} "
            f"(succeeded={batch_status.request_counts.succeeded}, "
            f"errored={batch_status.request_counts.errored})"
        )

        if status == "ended":
            break
        time.sleep(poll_interval)

    # Collect results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_data = initialize_output(output_dir, subenv_name, input_data, file_prefix)

    id_prefix = f"{subenv_name}_"
    success_count = 0
    error_count = 0

    for result in tqdm(
        client.messages.batches.results(batch_id),
        total=len(remaining_ids),
        desc=f"{subenv_name} results",
        unit="msg",
    ):
        hid = result.custom_id[len(id_prefix):]

        if result.result.type == "succeeded":
            rewritten = result.result.message.content[0].text
            output_data["histories"][hid] = [
                {"role": "environment", "content": rewritten}
            ]
            completed_ids.add(hid)
            success_count += 1
        else:
            logger.error(
                f"[{subenv_name}] Batch error for {hid}: {result.result.type}"
            )
            error_count += 1

    save_output_file(output_dir, subenv_name, output_data, file_prefix)
    save_progress(output_dir, subenv_name, completed_ids)
    logger.info(
        f"[{subenv_name}] Batch done. Succeeded={success_count}, Errors={error_count}"
    )


# =========================================================================
# Orchestrator
# =========================================================================

async def process_all_subenvs(
    subenvs: List[str],
    input_dir: str,
    output_dir: str,
    system_prompt: str,
    n_samples: Optional[int],
    mode: str,
    concurrency: int,
    dry_run: bool,
    seed: int = 42,
    subenv_rewrite_config: Optional[dict] = None,
) -> None:
    """Process all requested sub-environments sequentially.

    If subenv_rewrite_config is provided, each subenv looks up its own
    input_dir, output_dir, system_prompt, and file_prefix from the config.
    Otherwise, uses the shared defaults passed as arguments.
    """
    for subenv in subenvs:
        # Resolve per-subenv config if available
        if subenv_rewrite_config and subenv in subenv_rewrite_config:
            sc = subenv_rewrite_config[subenv]
            sub_input_dir = sc["input_dir"]
            sub_output_dir = sc["output_dir"]
            sub_system_prompt = sc["system_prompt"]
            sub_file_prefix = sc["file_prefix"]
        else:
            sub_input_dir = input_dir
            sub_output_dir = output_dir
            sub_system_prompt = system_prompt
            sub_file_prefix = "vuln_"

        logger.info(f"\n{'=' * 60}\nProcessing: {subenv} "
                     f"(prefix={sub_file_prefix})\n{'=' * 60}")

        if mode == "async":
            await process_subenv_async(
                subenv_name=subenv,
                input_dir=sub_input_dir,
                output_dir=sub_output_dir,
                system_prompt=sub_system_prompt,
                n_samples=n_samples,
                concurrency=concurrency,
                dry_run=dry_run,
                seed=seed,
                file_prefix=sub_file_prefix,
            )
        elif mode == "batch":
            process_subenv_batch(
                subenv_name=subenv,
                input_dir=sub_input_dir,
                output_dir=sub_output_dir,
                system_prompt=sub_system_prompt,
                n_samples=n_samples,
                dry_run=dry_run,
                seed=seed,
                file_prefix=sub_file_prefix,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'async' or 'batch'.")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rewrite gameable user messages to be more validation-seeking"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="action_advice",
        help="Environment config name (default: action_advice)",
    )
    parser.add_argument(
        "--subenvs",
        type=str,
        default=None,
        help="Comma-separated subenv names (default: all from env config)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Randomly select N samples per subenv for testing (default: all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="async",
        choices=["async", "batch"],
        help="Processing mode: 'async' (concurrent API calls) or 'batch' (Anthropic Batch API)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Max concurrent API calls in async mode (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without making API calls",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    # Load env config
    cfg = load_env_config(args.env)

    # Parse subenvs
    if args.subenvs:
        subenvs = [s.strip() for s in args.subenvs.split(",")]
        for s in subenvs:
            if s not in cfg.ALL_SUBENVS:
                parser.error(f"Unknown subenv: '{s}'. Valid: {cfg.ALL_SUBENVS}")
    else:
        subenvs = cfg.ALL_SUBENVS

    # Per-subenv config overrides input/output dirs and prompts per subenv
    subenv_rewrite_config = getattr(cfg, "SUBENV_REWRITE_CONFIG", None)

    asyncio.run(
        process_all_subenvs(
            subenvs=subenvs,
            input_dir=cfg.GAMEABLE_DIR,
            output_dir=cfg.GAMEABLE_V2_DIR,
            system_prompt=cfg.REWRITE_SYSTEM_PROMPT,
            n_samples=args.n_samples,
            mode=args.mode,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
            seed=args.seed,
            subenv_rewrite_config=subenv_rewrite_config,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
