#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

import medrag_original
import original
from ovon_config import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_ERROR_PATH,
    DEFAULT_MEDRAG_CONFIG,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_TIMEOUT,
    ROOT,
)
from ovon_helpers import load_pipeline_errors, setup_logging, write_pipeline_errors


DEFAULT_LOG_PATH = ROOT / "data" / "rerun_json_errors.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun prompts listed in data/error.txt and rewrite only those CSV rows."
    )
    parser.add_argument(
        "--error-file",
        type=Path,
        default=DEFAULT_ERROR_PATH,
        help="Path to the shared JSON error list.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to the input prompt JSONL file.",
    )
    parser.add_argument(
        "--original-results-csv",
        type=Path,
        default=original.DEFAULT_RESULTS_CSV,
        help="Path to original.py results CSV.",
    )
    parser.add_argument(
        "--medrag-results-csv",
        type=Path,
        default=medrag_original.DEFAULT_RESULTS_CSV,
        help="Path to medrag_original.py results CSV.",
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
        "--medrag-index-dir",
        type=Path,
        default=DEFAULT_MEDRAG_CONFIG["index_dir"],
        help="Directory holding the persisted MedRAG FAISS index.",
    )
    parser.add_argument(
        "--medrag-context-chunks",
        type=int,
        default=DEFAULT_MEDRAG_CONFIG["context_chunks"],
        help="Number of MedRAG passages to retrieve per medical prompt.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the rerun log file.",
    )
    return parser.parse_args()


def load_prompt_map(prompts_path: Path) -> dict[int, str]:
    if not prompts_path.exists():
        raise FileNotFoundError(f"Required prompt file not found: {prompts_path}")

    prompt_map: dict[int, str] = {}
    with prompts_path.open("r", encoding="utf-8") as file_obj:
        prompt_id = 0
        for line in file_obj:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompt_id += 1
                prompt_map[prompt_id] = prompt
    return prompt_map


def write_df_atomic(df: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)


def serialize_cell(value):
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def update_result_row(df: pd.DataFrame, prompt_id: int, expected_prompt: str, result: dict) -> None:
    matches = df.index[df["prompt_id"] == prompt_id].tolist()
    if not matches:
        raise KeyError(f"prompt_id {prompt_id} was not found in the target CSV.")
    row_index = matches[0]
    existing_prompt = df.at[row_index, "prompt"]
    if str(existing_prompt) != expected_prompt:
        raise ValueError(
            f"prompt_id {prompt_id} matched a row whose prompt text does not match the current dataset."
        )
    for key, value in result.items():
        df.at[row_index, key] = serialize_cell(value)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    errors = sorted(set(load_pipeline_errors(args.error_file)))
    if not errors:
        print("No errors found.")
        return 0

    prompt_map = load_prompt_map(args.prompts)

    original_df = None
    medrag_df = None
    medrag_client = None
    remaining_errors: list[tuple[str, int]] = []

    for pipeline_name, prompt_id in errors:
        prompt = prompt_map.get(prompt_id)
        if prompt is None:
            logging.error(
                "Prompt %s listed in %s was not found in %s",
                prompt_id,
                args.error_file,
                args.prompts,
            )
            remaining_errors.append((pipeline_name, prompt_id))
            continue

        try:
            if pipeline_name == "original":
                if original_df is None:
                    original_df = pd.read_csv(args.original_results_csv)
                result = original.process_prompt(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    error_file=None,
                    include_internal_status=True,
                )
                had_json_error = bool(result.pop("_had_json_error", False))
                update_result_row(original_df, prompt_id, prompt, result)
                write_df_atomic(original_df, args.original_results_csv)
                target_csv = args.original_results_csv
            elif pipeline_name == "medrag_original":
                if medrag_df is None:
                    medrag_df = pd.read_csv(args.medrag_results_csv)
                if medrag_client is None:
                    medrag_client = medrag_original.build_medrag_client(
                        args.base_url,
                        args.api_key,
                        args.timeout,
                        args.medrag_index_dir,
                        args.medrag_context_chunks,
                    )
                result = medrag_original.process_prompt(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    medrag_client=medrag_client,
                    error_file=None,
                    include_internal_status=True,
                )
                had_json_error = bool(result.pop("_had_json_error", False))
                update_result_row(medrag_df, prompt_id, prompt, result)
                write_df_atomic(medrag_df, args.medrag_results_csv)
                target_csv = args.medrag_results_csv
            else:
                logging.error("Unknown pipeline name in %s: %s", args.error_file, pipeline_name)
                remaining_errors.append((pipeline_name, prompt_id))
                continue

            if had_json_error:
                remaining_errors.append((pipeline_name, prompt_id))
                logging.warning(
                    "Rerun completed but JSON error remains for %s prompt %s.",
                    pipeline_name,
                    prompt_id,
                )
            else:
                logging.info(
                    "Repaired %s prompt %s in %s",
                    pipeline_name,
                    prompt_id,
                    target_csv,
                )
        except Exception as exc:
            logging.error(
                "Rerun failed for %s prompt %s: %s",
                pipeline_name,
                prompt_id,
                exc,
            )
            remaining_errors.append((pipeline_name, prompt_id))

    write_pipeline_errors(args.error_file, remaining_errors)
    print(f"Remaining errors: {len(remaining_errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
