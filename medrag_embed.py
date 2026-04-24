#!/usr/bin/env python3
"""
Nomic-aware OllamaEmbedding wrapper.

The `nomic-embed-text` model expects task-specific prefixes:
  - "search_document: ..."  for corpus chunks being indexed
  - "search_query: ..."     for user queries at retrieval time

Without the prefix, retrieval quality measurably degrades. This wrapper
transparently adds the prefixes and L2-normalizes outputs so FAISS
`IndexFlatIP` gives cosine similarity directly.
"""
from __future__ import annotations

from typing import List

import numpy as np
from llama_index.embeddings.ollama import OllamaEmbedding


_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def _normalize(vector: List[float]) -> List[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


def _normalize_batch(vectors: List[List[float]]) -> List[List[float]]:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (arr / norms).tolist()


class NomicOllamaEmbedding(OllamaEmbedding):
    """OllamaEmbedding that adds Nomic prefixes and L2-normalizes outputs."""

    def _get_query_embedding(self, query: str) -> List[float]:
        return _normalize(super()._get_query_embedding(_QUERY_PREFIX + query))

    def _get_text_embedding(self, text: str) -> List[float]:
        return _normalize(super()._get_text_embedding(_DOC_PREFIX + text))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        prefixed = [_DOC_PREFIX + t for t in texts]
        return _normalize_batch(super()._get_text_embeddings(prefixed))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return _normalize(await super()._aget_query_embedding(_QUERY_PREFIX + query))

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return _normalize(await super()._aget_text_embedding(_DOC_PREFIX + text))

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        prefixed = [_DOC_PREFIX + t for t in texts]
        return _normalize_batch(await super()._aget_text_embeddings(prefixed))
