#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path

import pandas as pd

from ovon_config import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_ERROR_PATH,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_JSON_REVIEWER_CONFIG,
    DEFAULT_REVIEWER_CONFIG,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    DEFAULT_EVALUATOR_CONFIG,
    FRONT_END_AGENT_SYSTEM_PROMPT,
    KPI_EVALUATOR_SYSTEM_PROMPT,
    ROOT,
    SECOND_LEVEL_REVIEWER_SYSTEM_PROMPT,
    THIRD_LEVEL_REVIEWER_SYSTEM_PROMPT,
)
from ovon_helpers import (
    LLMIntegrationAgent,
    append_pipeline_error,
    calculate_ths,
    load_prompts,
    parse_kpi_metrics_response,
    parse_second_level_response,
    setup_logging,
)


DEFAULT_RESULTS_CSV = ROOT / "data" / "original_results.csv"
DEFAULT_LOG_PATH = ROOT / "data" / "original_pipeline.log"
DEFAULT_CHECKPOINT_EVERY = 10


MODEL_CONFIGS = {
    "FrontEndAgent": dict(DEFAULT_REVIEWER_CONFIG),
    "SecondLevelReviewer": dict(DEFAULT_JSON_REVIEWER_CONFIG),
    "ThirdLevelReviewer": dict(DEFAULT_REVIEWER_CONFIG),
    "KPI_Evaluator": dict(DEFAULT_EVALUATOR_CONFIG),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OVON hallucination pipeline locally with Ollama Qwen."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to the input prompt JSONL file.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Path to the pipeline results CSV.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the pipeline log file.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible local Ollama base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="Optional API key for the endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of prompts to process in parallel.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Write an atomic partial CSV every N completed prompts. Use 0 to disable.",
    )
    return parser.parse_args()


def build_agents(
    base_url: str, api_key: str | None, timeout: int
) -> dict[str, LLMIntegrationAgent]:
    return {
        "FrontEndAgent": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="FrontEndAgent",
            model_config=MODEL_CONFIGS["FrontEndAgent"],
            system_message=FRONT_END_AGENT_SYSTEM_PROMPT,
        ),
        "SecondLevelReviewer": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="SecondLevelReviewer",
            model_config=MODEL_CONFIGS["SecondLevelReviewer"],
            system_message=SECOND_LEVEL_REVIEWER_SYSTEM_PROMPT,
        ),
        "ThirdLevelReviewer": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="ThirdLevelReviewer",
            model_config=MODEL_CONFIGS["ThirdLevelReviewer"],
            system_message=THIRD_LEVEL_REVIEWER_SYSTEM_PROMPT,
        ),
        "KPI_Evaluator": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="KPI_Evaluator",
            model_config=MODEL_CONFIGS["KPI_Evaluator"],
            system_message=KPI_EVALUATOR_SYSTEM_PROMPT,
        ),
    }


def write_results_checkpoint(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    checkpoint_df = pd.DataFrame(sorted(results, key=lambda item: item["prompt_id"]))
    checkpoint_df.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)


def process_prompt(
    prompt_id: int,
    prompt: str,
    base_url: str,
    api_key: str | None,
    timeout: int,
    error_file: Path | None = DEFAULT_ERROR_PATH,
    include_internal_status: bool = False,
) -> dict:
    logging.info("Processing Prompt %s: %s", prompt_id, prompt)

    agents = build_agents(base_url, api_key, timeout)
    had_json_error = False

    front_end_response = agents["FrontEndAgent"].generate_reply(prompt)
    second_level_response = agents["SecondLevelReviewer"].generate_reply(front_end_response)

    utterance, whisper_context, whisper_value, second_level_fallback_used = parse_second_level_response(
        second_level_response, prompt_id
    )
    had_json_error = had_json_error or second_level_fallback_used

    third_level_input = (
        f"Utterance: {utterance}\nContext: {whisper_context}\nReason: {whisper_value}"
    )
    third_level_response = agents["ThirdLevelReviewer"].generate_reply(third_level_input)

    kpi_input = json.dumps(
        {
            "FrontEndAgent": front_end_response,
            "SecondLevelReviewer": utterance,
            "ThirdLevelReviewer": third_level_response,
        }
    )
    kpi_evaluator_response = agents["KPI_Evaluator"].generate_reply(kpi_input)

    try:
        (
            front_end_metrics,
            second_level_metrics,
            third_level_metrics,
        ) = parse_kpi_metrics_response(kpi_evaluator_response)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        logging.error(
            "KPI Evaluator response invalid for prompt %s: %s",
            prompt_id,
            kpi_evaluator_response,
        )
        front_end_metrics, second_level_metrics, third_level_metrics = {}, {}, {}
        had_json_error = True

    if had_json_error:
        append_pipeline_error(error_file, "original", prompt_id)

    result = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "front_end_response": front_end_response,
        "second_level_response": {
            "utterance": utterance,
            "whisper_context": whisper_context,
            "whisper_value": whisper_value,
        },
        "third_level_response": third_level_response,
        "FrontEndAgent": front_end_metrics,
        "SecondLevelReviewer": second_level_metrics,
        "ThirdLevelReviewer": third_level_metrics,
        "THS1": calculate_ths(front_end_metrics),
        "THS2": calculate_ths(second_level_metrics),
        "THS3": calculate_ths(third_level_metrics),
    }
    if include_internal_status:
        result["_had_json_error"] = had_json_error
    return result


def run_pipeline(
    prompts: list[str],
    base_url: str,
    api_key: str | None,
    timeout: int,
    workers: int,
    results_csv: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> pd.DataFrame:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_prompt, prompt_id, prompt, base_url, api_key, timeout)
            for prompt_id, prompt in enumerate(prompts, start=1)
        ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            completed += 1
            logging.info("Completed %s/%s prompts.", completed, len(prompts))
            if results_csv is not None and checkpoint_every > 0 and completed % checkpoint_every == 0:
                write_results_checkpoint(results, results_csv)

    results.sort(key=lambda item: item["prompt_id"])
    return pd.DataFrame(results)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    prompts = load_prompts(args.prompts)
    logging.info("Loaded %s prompts from %s", len(prompts), args.prompts)

    df_results = run_pipeline(
        prompts,
        args.base_url,
        args.api_key,
        args.timeout,
        args.workers,
        args.results_csv,
        args.checkpoint_every,
    )
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.results_csv, index=False)
    print(f"Pipeline completed successfully. Results saved to {args.results_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
