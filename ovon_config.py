#!/usr/bin/env python3
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_PATH = ROOT / "data" / "data.jsonl"
DEFAULT_ERROR_PATH = ROOT / "data" / "error.txt"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_JSON_RESPONSE_FORMAT = {"type": "json_object"}
DEFAULT_BASE_URL = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:11434/v1")
DEFAULT_API_KEY = os.getenv("QWEN_API_KEY")
DEFAULT_TIMEOUT = 120
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4)))

DEFAULT_REVIEWER_CONFIG = {
    "model": DEFAULT_MODEL,
    "temperature": 0.7,
    "max_tokens": 500,
}
DEFAULT_JSON_REVIEWER_CONFIG = {
    "model": DEFAULT_MODEL,
    "temperature": 0.7,
    "max_tokens": 500,
    "response_format": DEFAULT_JSON_RESPONSE_FORMAT,
}
DEFAULT_CLASSIFIER_CONFIG = {
    "model": DEFAULT_MODEL,
    "temperature": 0.0,
    "max_tokens": 10,
}
DEFAULT_QUERY_BUILDER_CONFIG = {
    "model": DEFAULT_MODEL,
    "temperature": 0.0,
    "max_tokens": 40,
}
DEFAULT_EVALUATOR_CONFIG = {
    "model": DEFAULT_MODEL,
    "temperature": 0.0,
    "max_tokens": 1000,
    "response_format": DEFAULT_JSON_RESPONSE_FORMAT,
}
DEFAULT_MEDRAG_CONFIG = {
    "model": EMBED_MODEL,
    "index_dir": ROOT / "data" / "medrag_index",
    "embed_dim": 768,
    "context_chunks": 6,
    "min_score": 0.15,
}

FRONT_END_AGENT_SYSTEM_PROMPT = (
    "You are a front-end assistant. Respond to the prompt provided, even if it "
    "involves speculation or fiction. Do not ever add any explicit disclaimer "
    "wherever content is speculative or fictional to ensure users are aware of "
    "its nature."
)

MEDICAL_CLASSIFIER_SYSTEM_PROMPT = (
    "Decide whether the user's prompt is medical-related. Return exactly YES or NO. "
    "Classify by topic, not by factual accuracy, plausibility, or safety. "
    "Answer YES for any prompt about health, medicine, disease, diagnosis, symptoms, "
    "treatment, clinical trials, anatomy, physiology, drugs, surgery, patient care, "
    "public health, medical devices, or biomedical research, even if the prompt is "
    "fictional, speculative, contradictory, impossible, or asks about a made-up "
    "condition or cure. For example, invented diseases, fake therapies, impossible "
    "clinical trials, and hallucination-seeking medical questions are still medical. "
    "Answer NO only when the prompt is not primarily about a medical or health topic."
)

MEDRAG_QUERY_BUILDER_SYSTEM_PROMPT = (
    "Extract a short search query for medical retrieval. Return only 1 to 3 short medical "
    "keywords or key phrases separated by commas, and nothing else. Prefer broad concepts."
)

MEDRAG_GENERATION_SYSTEM_PROMPT = (
    "Review the front-end agent's response against the retrieved MedRAG "
    "textbook passages. Rephrase it for clarity, accuracy, and factualness "
    "using ONLY the provided sources, and cite them inline as [S<index>]. "
    "If the sources do not contain the answer, state explicitly in the rewritten "
    "response that MedRAG could not find a supporting source. "
    "Return exactly one JSON object with exactly this schema: "
    '{"utterance":"<rewritten response>","whisper_context":"<summary of hallucination levels, max 20 words>",'
    '"whisper_value":"<detailed explanation of hallucination, max 200 words>"}. '
    'Do not use any other keys.'
)

MEDRAG_GENERATION_USER_TEMPLATE = (
    "The following MedRAG passages were retrieved with search query "
    "'{search_query}':\n\n"
    "{context_block}\n\n"
    "Original prompt:\n{prompt}\n\n"
    "FrontEndAgent response:\n{front_end_response}"
)

SECOND_LEVEL_REVIEWER_SYSTEM_PROMPT = (
    "Review the front-end agent's response. Rephrase it for clarity, accuracy, "
    "and factualness. Explicit disclaimers wherever content is speculative or "
    "fictional to ensure users are aware of its nature even if it is a repetition. "
    "Return exactly one JSON object with exactly this schema: "
    '{"utterance":"<rewritten response>","whisper_context":"<summary of hallucination levels, max 20 words>",'
    '"whisper_value":"<detailed explanation of hallucination, max 200 words>"}. '
    'Do not use any other keys.'
)

SECOND_LEVEL_REVIEWER_MEDRAG_SYSTEM_PROMPT = (
    "Review the front-end agent's response. Rephrase it for clarity, accuracy, "
    "and factualness. Explicit disclaimers wherever content is speculative or "
    "fictional to ensure users are aware of its nature even if it is a repetition. "
    "If the input includes a MedRAG retrieval signal saying no trustworthy evidence "
    "was found, use that information. Return exactly one JSON object with exactly "
    'this schema: {"utterance":"<rewritten response>","whisper_context":"<summary of hallucination levels, max 20 words>",'
    '"whisper_value":"<detailed explanation of hallucination, max 200 words>"}. '
    'Do not use any other keys.'
)

THIRD_LEVEL_REVIEWER_SYSTEM_PROMPT = (
    "Refine the second-level reviewer's response. Explicit disclaimers wherever "
    "content is speculative or fictional to ensure users are aware of its nature, "
    "even if it is a repetition. Improve clarity, style, and factuality without "
    "generating another JSON response."
)

KPI_EVALUATOR_SYSTEM_PROMPT = (
    "Analyze the responses from the FrontEndAgent, SecondLevelReviewer, and "
    "ThirdLevelReviewer. For each agent, assign two integer scores from 1 to 10. "
    '"factuality" means how factually accurate, well-grounded, and non-hallucinatory '
    'the response is. "helpfulness" means how useful, clear, relevant, and actionable '
    "the response is for answering the user's prompt. Use the full 1-10 scale, where "
    "1 is extremely poor, 5 is mixed or mediocre, and 10 is excellent. Return exactly "
    "one JSON object with exactly this schema, replacing <integer 1-10> with your "
    'analysis: {"FrontEndAgent":{"factuality":<integer 1-10>,"helpfulness":<integer 1-10>},'
    '"SecondLevelReviewer":{"factuality":<integer 1-10>,"helpfulness":<integer 1-10>},'
    '"ThirdLevelReviewer":{"factuality":<integer 1-10>,"helpfulness":<integer 1-10>}} '
    'Do not use any other keys.'
)
