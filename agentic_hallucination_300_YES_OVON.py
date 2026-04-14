#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from urllib import error, request

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_PATH = ROOT / "data" / "data.jsonl"
DEFAULT_RESULTS_CSV = ROOT / "data" / "pipeline_results_with_ths.csv"
DEFAULT_LOG_PATH = ROOT / "pipeline.log"
DEFAULT_MODEL = "qwen2.5:3b-instruct"


MODEL_CONFIGS = {
    "FrontEndAgent": {
        "model": DEFAULT_MODEL,
        "temperature": 0.7,
        "max_tokens": 500,
    },
    "SecondLevelReviewer": {
        "model": DEFAULT_MODEL,
        "temperature": 0.7,
        "max_tokens": 500,
    },
    "ThirdLevelReviewer": {
        "model": DEFAULT_MODEL,
        "temperature": 0.7,
        "max_tokens": 500,
    },
    "KPI_Evaluator": {
        "model": DEFAULT_MODEL,
        "temperature": 0.0,
        "max_tokens": 1000,
    },
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
        default=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:11434/v1"),
        help="OpenAI-compatible local Ollama base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("QWEN_API_KEY"),
        help="Optional API key for the endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Number of prompts to process in parallel.",
    )
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )


def calculate_ths(metrics, w1=0.25, w2=0.25, w3=0.25, w4=0.25):
    try:
        fcd = metrics.get("FCD", 0)
        fgr = metrics.get("FGR", 0)
        fdf = metrics.get("FDF", 0)
        ecs = metrics.get("ECS", 0)
        return (fcd * w1 - (fgr * w2 + fdf * w3 + ecs * w4)) / 3
    except (TypeError, AttributeError):
        return None


def extract_json_object(text):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        pass

    if not isinstance(text, str):
        raise ValueError("KPI response is not a string.")

    if "```json" in text:
        text = text.split("```json", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]

    text = text.strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("No JSON object found in KPI response.")


def normalize_kpi_metrics(raw_metrics):
    agent_names = ["FrontEndAgent", "SecondLevelReviewer", "ThirdLevelReviewer"]
    metric_names = ["FCD", "FDF", "FGR", "ECS"]

    if all(agent in raw_metrics for agent in agent_names):
        return {agent: raw_metrics.get(agent, {}) for agent in agent_names}

    if all(metric in raw_metrics for metric in metric_names):
        normalized = {agent: {} for agent in agent_names}
        for metric in metric_names:
            values = raw_metrics.get(metric, {})
            if not isinstance(values, dict):
                continue
            for agent in agent_names:
                normalized[agent][metric] = values.get(agent, 0)
        return normalized

    raise ValueError("KPI response JSON did not match an expected schema.")


def load_prompts(prompts_path: Path) -> list[str]:
    if not prompts_path.exists():
        raise FileNotFoundError(f"Required prompt file not found: {prompts_path}")

    prompts = []
    with prompts_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt)
    return prompts


def load_prompt_records(prompts_path: Path) -> list[dict]:
    if not prompts_path.exists():
        raise FileNotFoundError(f"Required prompt file not found: {prompts_path}")

    prompt_records = []
    with prompts_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            prompt_records.append(
                {
                    "prompt_id": idx,
                    "prompt": prompt,
                    "source_domain": record.get("source_domain", "unknown"),
                }
            )
    return prompt_records


class LLMIntegrationAgent:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: int,
        name: str,
        model_config: dict,
        system_message: str,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.name = name
        self.model_config = model_config
        self.system_message = system_message

    def generate_reply(self, user_message):
        try:
            url = self.base_url.rstrip("/") + "/chat/completions"
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": "" if user_message is None else str(user_message)},
            ]
            payload = {
                "model": self.model_config["model"],
                "messages": messages,
                "temperature": self.model_config.get("temperature", 0.7),
                "max_tokens": self.model_config.get("max_tokens", 500),
            }
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            req = request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
            raw = json.loads(body)
            reply = raw["choices"][0]["message"]["content"].strip()
            logging.info("%s generated a response.", self.name)
            return reply
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            logging.error("HTTP error in %s generate_reply: %s %s", self.name, exc.code, details)
            return None
        except error.URLError as exc:
            logging.error("Connection error in %s generate_reply: %s", self.name, exc.reason)
            return None
        except Exception as exc:
            logging.error("Error in %s generate_reply: %s", self.name, exc)
            return None


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
            system_message=(
                "You are a front-end assistant. Respond to the prompt provided, even if it "
                "involves speculation or fiction. Do not ever add any explicit disclaimer "
                "wherever content is speculative or fictional to ensure users are aware of "
                "its nature."
            ),
        ),
        "SecondLevelReviewer": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="SecondLevelReviewer",
            model_config=MODEL_CONFIGS["SecondLevelReviewer"],
            system_message=(
                "Review the front-end agent's response. Rephrase it for clarity, accuracy, "
                "and factualness. Explicit disclaimers wherever content is speculative or "
                "fictional to ensure users are aware of its nature even if it is a repetition. "
                "Include: 'utterance' (rewritten response), 'whisper context' (summary of "
                "hallucination levels, max 20 words), and 'whisper value' (detailed explanation "
                "of hallucination, max 200 words)."
            ),
        ),
        "ThirdLevelReviewer": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="ThirdLevelReviewer",
            model_config=MODEL_CONFIGS["ThirdLevelReviewer"],
            system_message=(
                "Refine the second-level reviewer's response. Explicit disclaimers wherever "
                "content is speculative or fictional to ensure users are aware of its nature, "
                "even if it is a repetition. Improve clarity, style, and factuality without "
                "generating another JSON response."
            ),
        ),
        "KPI_Evaluator": LLMIntegrationAgent(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            name="KPI_Evaluator",
            model_config=MODEL_CONFIGS["KPI_Evaluator"],
            system_message=(
                "Analyze the responses from the FrontEndAgent, SecondLevelReviewer, and "
                "ThirdLevelReviewer. Return JSON only, with no markdown fences or commentary, "
                "using exactly this schema: "
                '{"FrontEndAgent":{"FCD":0,"FDF":0,"FGR":0,"ECS":0},'
                '"SecondLevelReviewer":{"FCD":0,"FDF":0,"FGR":0,"ECS":0},'
                '"ThirdLevelReviewer":{"FCD":0,"FDF":0,"FGR":0,"ECS":0}}.'
            ),
        ),
    }


def process_prompt(
    prompt_id: int,
    prompt: str,
    base_url: str,
    api_key: str | None,
    timeout: int,
) -> dict:
    logging.info("Processing Prompt %s: %s", prompt_id, prompt)

    agents = build_agents(base_url, api_key, timeout)

    front_end_response = agents["FrontEndAgent"].generate_reply(prompt)
    second_level_response = agents["SecondLevelReviewer"].generate_reply(front_end_response)

    try:
        utterance, whisper_context, whisper_value = (second_level_response or "").split(
            "\n", 2
        )
    except ValueError:
        logging.error("SecondLevelReviewer response structure invalid for prompt %s.", prompt_id)
        utterance = second_level_response or ""
        whisper_context = ""
        whisper_value = ""

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
        kpi_metrics = normalize_kpi_metrics(extract_json_object(kpi_evaluator_response))
        front_end_metrics = kpi_metrics.get("FrontEndAgent", {})
        second_level_metrics = kpi_metrics.get("SecondLevelReviewer", {})
        third_level_metrics = kpi_metrics.get("ThirdLevelReviewer", {})
    except (ValueError, TypeError):
        logging.error(
            "KPI Evaluator response invalid for prompt %s: %s",
            prompt_id,
            kpi_evaluator_response,
        )
        front_end_metrics, second_level_metrics, third_level_metrics = {}, {}, {}

    return {
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


def run_pipeline(
    prompts: list[str], base_url: str, api_key: str | None, timeout: int, workers: int
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

    results.sort(key=lambda item: item["prompt_id"])
    return pd.DataFrame(results)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    prompts = load_prompts(args.prompts)
    logging.info("Loaded %s prompts from %s", len(prompts), args.prompts)

    df_results = run_pipeline(
        prompts, args.base_url, args.api_key, args.timeout, args.workers
    )
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.results_csv, index=False)
    print(f"Pipeline completed successfully. Results saved to {args.results_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
