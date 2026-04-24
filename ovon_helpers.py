#!/usr/bin/env python3
import json
import logging
import os
import threading
from pathlib import Path
from urllib import error, request

_ERROR_FILE_LOCK = threading.Lock()


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


def _coerce_score(value) -> float:
    score = float(value)
    if score < 1.0:
        return 1.0
    if score > 10.0:
        return 10.0
    return score


def calculate_ths(metrics, w1=0.25, w2=0.25, w3=0.25, w4=0.25):
    try:
        factuality = _coerce_score(metrics["factuality"])
        helpfulness = _coerce_score(metrics["helpfulness"])
        return (factuality + helpfulness) / 2.0
    except (TypeError, AttributeError, KeyError, ValueError):
        return None


def parse_kpi_metrics_response(
    kpi_evaluator_response: str | None,
) -> tuple[dict, dict, dict]:
    kpi_metrics = json.loads(kpi_evaluator_response or "")
    if not isinstance(kpi_metrics, dict):
        raise ValueError("KPI response is not a JSON object.")

    agent_names = ["FrontEndAgent", "SecondLevelReviewer", "ThirdLevelReviewer"]
    metric_names = ["factuality", "helpfulness"]

    normalized = {}
    for agent_name in agent_names:
        metrics = kpi_metrics.get(agent_name)
        if not isinstance(metrics, dict):
            raise ValueError(f"KPI response missing object for {agent_name}.")
        normalized_metrics = {}
        for metric in metric_names:
            if metric not in metrics:
                raise ValueError(f"KPI response missing '{metric}' for {agent_name}.")
            normalized_metrics[metric] = _coerce_score(metrics[metric])
        normalized[agent_name] = normalized_metrics

    front_end_metrics = normalized["FrontEndAgent"]
    second_level_metrics = normalized["SecondLevelReviewer"]
    third_level_metrics = normalized["ThirdLevelReviewer"]
    return front_end_metrics, second_level_metrics, third_level_metrics


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
    metric_names = ["factuality", "helpfulness"]

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
    with prompts_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt)
    return prompts


def append_pipeline_error(error_file: Path | None, pipeline_name: str, prompt_id: int) -> None:
    if error_file is None:
        return
    try:
        error_file.parent.mkdir(parents=True, exist_ok=True)
        line = f"{pipeline_name},{int(prompt_id)}\n"
        with _ERROR_FILE_LOCK:
            with error_file.open("a", encoding="utf-8") as file_obj:
                file_obj.write(line)
    except OSError as exc:
        logging.error(
            "Failed to append %s prompt %s to %s: %s",
            pipeline_name,
            prompt_id,
            error_file,
            exc,
        )


def load_pipeline_errors(error_file: Path) -> list[tuple[str, int]]:
    if not error_file.exists():
        return []
    errors: list[tuple[str, int]] = []
    with error_file.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not stripped:
                continue
            parts = [part.strip() for part in stripped.split(",", 1)]
            if len(parts) != 2:
                continue
            pipeline_name, prompt_id_text = parts
            try:
                prompt_id = int(prompt_id_text)
            except ValueError:
                continue
            errors.append((pipeline_name, prompt_id))
    return errors


def write_pipeline_errors(error_file: Path, errors: list[tuple[str, int]]) -> None:
    error_file.parent.mkdir(parents=True, exist_ok=True)
    unique_errors = sorted({(pipeline_name, int(prompt_id)) for pipeline_name, prompt_id in errors})
    temp_path = error_file.with_suffix(error_file.suffix + ".tmp")
    with _ERROR_FILE_LOCK:
        with temp_path.open("w", encoding="utf-8") as file_obj:
            for pipeline_name, prompt_id in unique_errors:
                file_obj.write(f"{pipeline_name},{prompt_id}\n")
        os.replace(temp_path, error_file)


def parse_second_level_response(
    second_level_response: str | None, prompt_id: int
) -> tuple[str, str, str, bool]:
    try:
        parsed = json.loads(second_level_response or "")
        if not isinstance(parsed, dict):
            raise ValueError("SecondLevelReviewer response is not a JSON object.")
        utterance = "" if parsed.get("utterance") is None else str(parsed.get("utterance", ""))
        whisper_context = (
            "" if parsed.get("whisper_context") is None else str(parsed.get("whisper_context", ""))
        )
        whisper_value = "" if parsed.get("whisper_value") is None else str(parsed.get("whisper_value", ""))
        return utterance, whisper_context, whisper_value, False
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    try:
        utterance, whisper_context, whisper_value = (second_level_response or "").split("\n", 2)
    except ValueError:
        logging.error("SecondLevelReviewer response structure invalid for prompt %s.", prompt_id)
        utterance = second_level_response or ""
        whisper_context = ""
        whisper_value = ""
    return utterance, whisper_context, whisper_value, True


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
            if "response_format" in self.model_config:
                payload["response_format"] = self.model_config["response_format"]
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
