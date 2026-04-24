#!/usr/bin/env python3
"""
OVON pipeline wired to the local MedRAG retriever.

This replaces the previous GeoSI-based flow. For medical prompts, the pipeline
retrieves from a local FAISS index over MedRAG textbooks (built via
`build_medrag_index.py`) and has the front-end answer grounded in those
passages before proceeding through the OVON reviewer chain.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path

import pandas as pd

from medrag import MedRAG
from ovon_config import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_CLASSIFIER_CONFIG,
    DEFAULT_EVALUATOR_CONFIG,
    DEFAULT_ERROR_PATH,
    DEFAULT_JSON_REVIEWER_CONFIG,
    DEFAULT_MEDRAG_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_QUERY_BUILDER_CONFIG,
    DEFAULT_REVIEWER_CONFIG,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    FRONT_END_AGENT_SYSTEM_PROMPT,
    KPI_EVALUATOR_SYSTEM_PROMPT,
    MEDICAL_CLASSIFIER_SYSTEM_PROMPT,
    MEDRAG_QUERY_BUILDER_SYSTEM_PROMPT,
    ROOT,
    SECOND_LEVEL_REVIEWER_MEDRAG_SYSTEM_PROMPT,
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


DEFAULT_RESULTS_CSV = ROOT / "data" / "medrag_original_results.csv"
DEFAULT_LOG_PATH = ROOT / "data" / "medrag_original_pipeline.log"
DEFAULT_CHECKPOINT_EVERY = 10


MODEL_CONFIGS = {
    "FrontEndAgent": dict(DEFAULT_REVIEWER_CONFIG),
    "MedicalClassifier": dict(DEFAULT_CLASSIFIER_CONFIG),
    "MedRAGQueryBuilder": dict(DEFAULT_QUERY_BUILDER_CONFIG),
    "SecondLevelReviewer": dict(DEFAULT_JSON_REVIEWER_CONFIG),
    "SecondLevelReviewer_MedRAG": dict(DEFAULT_JSON_REVIEWER_CONFIG),
    "ThirdLevelReviewer": dict(DEFAULT_REVIEWER_CONFIG),
    "KPI_Evaluator": dict(DEFAULT_EVALUATOR_CONFIG),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OVON pipeline with MedRAG routing for medical prompts."
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
        "MedicalClassifier": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="MedicalClassifier",
            model_config=MODEL_CONFIGS["MedicalClassifier"],
            system_message=MEDICAL_CLASSIFIER_SYSTEM_PROMPT,
        ),
        "MedRAGQueryBuilder": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="MedRAGQueryBuilder",
            model_config=MODEL_CONFIGS["MedRAGQueryBuilder"],
            system_message=MEDRAG_QUERY_BUILDER_SYSTEM_PROMPT,
        ),
        "SecondLevelReviewer": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="SecondLevelReviewer",
            model_config=MODEL_CONFIGS["SecondLevelReviewer"],
            system_message=SECOND_LEVEL_REVIEWER_SYSTEM_PROMPT,
        ),
        "SecondLevelReviewer_MedRAG": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="SecondLevelReviewer_MedRAG",
            model_config=MODEL_CONFIGS["SecondLevelReviewer_MedRAG"],
            system_message=SECOND_LEVEL_REVIEWER_MEDRAG_SYSTEM_PROMPT,
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


def is_medical_prompt(decision_text: str | None) -> bool:
    if not decision_text:
        logging.warning("MedicalClassifier returned no decision; defaulting to non-medical.")
        return False

    first_line = decision_text.strip().splitlines()[0].strip().lower().strip(".: ")
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


def normalize_medrag_search_query(query_text: str | None, prompt: str) -> str:
    if not query_text:
        return prompt

    cleaned = query_text.replace("\n", ",")
    parts = []
    for raw_part in cleaned.split(","):
        part = raw_part.strip().lstrip("-*0123456789. ").strip()
        if not part:
            continue
        parts.append(part)
        if len(parts) == 3:
            break

    if not parts:
        return prompt
    return ", ".join(parts)


def build_second_level_fallback_input(
    *,
    prompt: str,
    front_end_response: str | None,
    medrag_status: str | None,
) -> str:
    if medrag_status == "no_chunks":
        medrag_signal = (
            "MedRAG retrieval signal: no passages from the local MedRAG corpus matched this "
            "medical prompt above the similarity threshold. Treat this as evidence that the "
            "request may be unsupported, fabricated, or hallucination-seeking."
        )
    elif medrag_status == "index_missing":
        medrag_signal = (
            "MedRAG retrieval signal: the local MedRAG index is not available, so no "
            "retrieval-based evidence could be used. Be cautious about unsupported or "
            "fabricated claims."
        )
    else:
        medrag_signal = (
            "MedRAG retrieval signal: MedRAG could not provide usable evidence. Be cautious "
            "about unsupported or fabricated claims."
        )

    return (
        f"Original prompt:\n{prompt}\n\n"
        f"FrontEndAgent response:\n{front_end_response or ''}\n\n"
        f"{medrag_signal}"
    )


def process_prompt(
    prompt_id: int,
    prompt: str,
    base_url: str,
    api_key: str | None,
    timeout: int,
    medrag_client: MedRAG,
    error_file: Path | None = DEFAULT_ERROR_PATH,
    include_internal_status: bool = False,
) -> dict:
    logging.info("Processing Prompt %s: %s", prompt_id, prompt)

    agents = build_agents(base_url, api_key, timeout)
    had_json_error = False

    front_end_response = agents["FrontEndAgent"].generate_reply(prompt)

    medical_decision = agents["MedicalClassifier"].generate_reply(prompt)
    route_to_medrag = is_medical_prompt(medical_decision)
    medrag_status: str = "not_routed"
    medrag_search_query: str | None = None
    top_score: float | None = None

    if route_to_medrag:
        raw_medrag_search_query = agents["MedRAGQueryBuilder"].generate_reply(prompt)
        medrag_search_query = normalize_medrag_search_query(raw_medrag_search_query, prompt)
        medrag_response, medrag_status, top_score = medrag_client.answer(
            prompt=prompt,
            search_query=medrag_search_query,
            prompt_id=prompt_id,
            front_end_response=front_end_response,
        )
        if medrag_response:
            (
                utterance,
                whisper_context,
                whisper_value,
                second_level_fallback_used,
            ) = parse_second_level_response(
                medrag_response, prompt_id
            )
            had_json_error = had_json_error or second_level_fallback_used
            logging.info("Prompt %s routed through MedRAG.", prompt_id)
        else:
            logging.warning(
                "MedRAG failed for prompt %s; falling back to SecondLevelReviewer.",
                prompt_id,
            )
            second_level_response = agents["SecondLevelReviewer_MedRAG"].generate_reply(
                build_second_level_fallback_input(
                    prompt=prompt,
                    front_end_response=front_end_response,
                    medrag_status=medrag_status,
                )
            )
            (
                utterance,
                whisper_context,
                whisper_value,
                second_level_fallback_used,
            ) = parse_second_level_response(
                second_level_response, prompt_id
            )
            had_json_error = had_json_error or second_level_fallback_used
    else:
        second_level_response = agents["SecondLevelReviewer"].generate_reply(front_end_response)
        (
            utterance,
            whisper_context,
            whisper_value,
            second_level_fallback_used,
        ) = parse_second_level_response(
            second_level_response, prompt_id
        )
        had_json_error = had_json_error or second_level_fallback_used
        logging.info("Prompt %s kept the original second-level reviewer path.", prompt_id)

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
        append_pipeline_error(error_file, "medrag_original", prompt_id)

    result = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "medical_classifier_response": medical_decision,
        "routed_to_medrag": route_to_medrag,
        "medrag_status": medrag_status,
        "top score": top_score,
        "medrag_search_query": medrag_search_query,
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


def build_medrag_client(
    base_url: str,
    api_key: str | None,
    timeout: int,
    index_dir: Path,
    context_chunks: int,
) -> MedRAG:
    return MedRAG(
        base_url=base_url,
        api_key=api_key,
        model=DEFAULT_MODEL,
        timeout=timeout,
        persist_dir=index_dir,
        context_chunks=context_chunks,
    )


def run_pipeline(
    prompts: list[str],
    base_url: str,
    api_key: str | None,
    timeout: int,
    workers: int,
    medrag_client: MedRAG,
    results_csv: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> pd.DataFrame:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_prompt,
                prompt_id,
                prompt,
                base_url,
                api_key,
                timeout,
                medrag_client,
            )
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

    medrag_client = build_medrag_client(
        args.base_url,
        args.api_key,
        args.timeout,
        args.medrag_index_dir,
        args.medrag_context_chunks,
    )

    df_results = run_pipeline(
        prompts,
        args.base_url,
        args.api_key,
        args.timeout,
        args.workers,
        medrag_client,
        args.results_csv,
        args.checkpoint_every,
    )
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.results_csv, index=False)
    print(f"Pipeline completed successfully. Results saved to {args.results_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
