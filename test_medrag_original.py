#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from medrag_original import (
    build_medrag_client,
    process_prompt,
    setup_logging,
)
from ovon_config import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MEDRAG_CONFIG,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_TIMEOUT,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT / "data" / "test_medrag_original_results.json"
DEFAULT_LOG_PATH = ROOT / "data" / "test_medrag_original.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run medrag_original on one medical and one non-medical prompt."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to the input prompt JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the test results JSON.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the test log file.",
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
        help="Number of MedRAG passages to retrieve per medical test prompt.",
    )
    return parser.parse_args()


def load_test_prompts(prompts_path: Path) -> list[dict]:
    if not prompts_path.exists():
        raise FileNotFoundError(f"Required prompt file not found: {prompts_path}")

    medical_record = None
    nonmedical_record = None

    with prompts_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            prompt = record.get("prompt")
            source_domain = record.get("source_domain")

            if not isinstance(prompt, str) or not prompt.strip():
                continue

            prompt_record = {
                "prompt_id": line_number,
                "source_domain": source_domain,
                "title": record.get("title"),
                "prompt": prompt,
            }

            if source_domain == "medical" and medical_record is None:
                medical_record = prompt_record
            elif source_domain != "medical" and nonmedical_record is None:
                nonmedical_record = prompt_record

            if medical_record and nonmedical_record:
                break

    if medical_record is None:
        raise ValueError("Could not find a medical prompt in the prompt file.")
    if nonmedical_record is None:
        raise ValueError("Could not find a non-medical prompt in the prompt file.")

    return [medical_record, nonmedical_record]


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    selected_prompts = load_test_prompts(args.prompts)
    medrag_client = build_medrag_client(
        args.base_url,
        args.api_key,
        args.timeout,
        args.medrag_index_dir,
        args.medrag_context_chunks,
    )

    results = []
    for record in selected_prompts:
        result = process_prompt(
            prompt_id=record["prompt_id"],
            prompt=record["prompt"],
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            medrag_client=medrag_client,
        )
        results.append(
            {
                "source_domain": record["source_domain"],
                "title": record["title"],
                "result": result,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2, ensure_ascii=False)

    print(f"Saved test results to {args.output}")
    for item in results:
        print(f"- {item['source_domain']}: {item['title']}")
        print(f"  classifier: {item['result'].get('medical_classifier_response')}")
        print(f"  routed_to_medrag: {item['result'].get('routed_to_medrag')}")
        print(f"  medrag_status: {item['result'].get('medrag_status')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
