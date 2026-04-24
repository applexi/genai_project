#!/usr/bin/env python3
"""
Local MedRAG runtime client.

Loads the persisted FAISS index built by `build_medrag_index.py`, retrieves
top-k passages for a query via Ollama embeddings, and generates an answer
with the same local Ollama chat model used elsewhere in the pipeline.

Exposes `MedRAG.answer(*, prompt, search_query, prompt_id) -> (text, status, top_score)`,
matching the contract previously used by the GeoSI client so it plugs into
the OVON orchestrator.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.faiss import FaissVectorStore
from openai import OpenAI

from medrag_embed import NomicOllamaEmbedding
from ovon_config import (
    DEFAULT_JSON_RESPONSE_FORMAT,
    DEFAULT_MEDRAG_CONFIG,
    MEDRAG_GENERATION_SYSTEM_PROMPT,
    MEDRAG_GENERATION_USER_TEMPLATE,
)


class MedRAG:
    """Thread-safe local RAG client over the MedRAG FAISS index."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: int,
        persist_dir: Path,
        embed_model: str = DEFAULT_MEDRAG_CONFIG["model"],
        context_chunks: int = DEFAULT_MEDRAG_CONFIG["context_chunks"],
        min_score: float = DEFAULT_MEDRAG_CONFIG["min_score"],
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "ollama",
            timeout=timeout,
        )
        self.model = model
        self.persist_dir = Path(persist_dir)
        self.context_chunks = max(1, context_chunks)
        self.min_score = float(min_score)

        ollama_base = base_url.rstrip("/")
        if ollama_base.endswith("/v1"):
            ollama_base = ollama_base[: -len("/v1")]
        self.embed_model = NomicOllamaEmbedding(
            model_name=embed_model,
            base_url=ollama_base,
        )

        self._index_lock = threading.Lock()
        self._index = None
        self._retriever: VectorIndexRetriever | None = None
        self._index_missing = not (self.persist_dir / "index_store.json").exists()

        if self._index_missing:
            logging.warning(
                "MedRAG index not found at %s. Run `python3 build_medrag_index.py` first.",
                self.persist_dir,
            )

    def _ensure_retriever(self) -> VectorIndexRetriever | None:
        if self._index_missing:
            return None
        if self._retriever is not None:
            return self._retriever
        with self._index_lock:
            if self._retriever is not None:
                return self._retriever
            logging.info("MedRAG loading FAISS index from %s", self.persist_dir)
            vector_store = FaissVectorStore.from_persist_dir(str(self.persist_dir))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=str(self.persist_dir)
            )
            self._index = load_index_from_storage(
                storage_context, embed_model=self.embed_model
            )
            self._retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=self.context_chunks,
                embed_model=self.embed_model,
            )
            logging.info(
                "MedRAG index ready: %s nodes.",
                len(self._index.docstore.docs),
            )
            return self._retriever

    def retrieve(self, query_text: str) -> list[NodeWithScore]:
        retriever = self._ensure_retriever()
        if retriever is None:
            return []
        try:
            nodes = retriever.retrieve(query_text)
        except Exception as exc:
            logging.warning("MedRAG retrieval failed: %s", exc)
            return []
        kept = [n for n in nodes if (n.score or 0.0) >= self.min_score]
        return kept

    def generate_answer(
        self,
        prompt: str,
        search_query: str,
        nodes: list[NodeWithScore],
        front_end_response: str | None,
    ) -> str:
        context_block = "\n\n---\n\n".join(
            self._format_node(i, node) for i, node in enumerate(nodes, start=1)
        )
        user_prompt = MEDRAG_GENERATION_USER_TEMPLATE.format(
            search_query=search_query,
            context_block=context_block,
            prompt=prompt,
            front_end_response=front_end_response or "",
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MEDRAG_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=DEFAULT_JSON_RESPONSE_FORMAT,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _format_node(index: int, node: NodeWithScore) -> str:
        metadata = getattr(node.node, "metadata", {}) or {}
        title = metadata.get("title") or ""
        subset = metadata.get("subset") or ""
        header_bits = [f"S{index}"]
        if subset:
            header_bits.append(subset)
        if title:
            header_bits.append(title)
        header = " | ".join(header_bits)
        return f"[{header}]\n{node.node.get_content()}"

    def answer(
        self,
        *,
        prompt: str,
        search_query: str,
        prompt_id: int,
        front_end_response: str | None = None,
    ) -> tuple[str | None, str, float | None]:
        if self._index_missing:
            logging.warning(
                "MedRAG index missing for prompt %s; cannot retrieve.",
                prompt_id,
            )
            return None, "index_missing", None

        retrieval_query = f"{search_query}\n{prompt}".strip()
        nodes = self.retrieve(retrieval_query)
        if not nodes:
            logging.warning(
                "MedRAG retrieved no passages above min_score=%.2f for prompt %s.",
                self.min_score,
                prompt_id,
            )
            return None, "no_chunks", None

        response = self.generate_answer(prompt, search_query, nodes, front_end_response)
        top_score = nodes[0].score or 0.0
        logging.info(
            "MedRAG answered prompt %s with %s passages (top score=%.3f).",
            prompt_id,
            len(nodes),
            top_score,
        )
        return response, "success", top_score
