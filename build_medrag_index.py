#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import faiss
from datasets import load_dataset
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

from medrag_embed import NomicOllamaEmbedding
from ovon_config import (
    DEFAULT_BASE_URL,
    DEFAULT_MEDRAG_CONFIG,
)

AVAILABLE_SUBSETS = {
    "textbooks": "MedRAG/textbooks",
    "statpearls": "MedRAG/statpearls",
    "wikipedia": "MedRAG/wikipedia",
    "pubmed": "MedRAG/pubmed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index of MedRAG passages using Ollama embeddings."
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["textbooks"],
        choices=sorted(AVAILABLE_SUBSETS.keys()),
        help="Which MedRAG subsets to index. Default: textbooks.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_MEDRAG_CONFIG["index_dir"],
        help="Directory to persist the FAISS index and docstore.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_MEDRAG_CONFIG["model"],
        help="Ollama embedding model name.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Number of chunks to embed per Ollama call.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        help="Persist the index to disk every N newly-inserted nodes.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible Ollama base URL (default from ovon_config).",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="HuggingFace dataset split to load.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def build_embed_model(model_name: str, base_url: str, batch_size: int) -> NomicOllamaEmbedding:
    ollama_base = base_url.rstrip("/")
    if ollama_base.endswith("/v1"):
        ollama_base = ollama_base[: -len("/v1")]
    return NomicOllamaEmbedding(
        model_name=model_name,
        base_url=ollama_base,
        embed_batch_size=batch_size,
    )


def load_or_create_index(
    persist_dir: Path, embed_dim: int
) -> tuple[VectorStoreIndex, set[str]]:
    persist_dir.mkdir(parents=True, exist_ok=True)
    index_store_path = persist_dir / "index_store.json"

    if index_store_path.exists():
        logging.info("Loading existing index from %s", persist_dir)
        vector_store = FaissVectorStore.from_persist_dir(str(persist_dir))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=str(persist_dir)
        )
        index = load_index_from_storage(storage_context)
        seen_ids = set(index.docstore.docs.keys())
        logging.info("Loaded %s existing nodes.", len(seen_ids))
        return index, seen_ids

    logging.info("Creating new FAISS index (dim=%s) at %s", embed_dim, persist_dir)
    faiss_index = faiss.IndexFlatIP(embed_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context)
    return index, set()


def record_to_node(record: dict, subset_name: str) -> TextNode | None:
    text = record.get("contents") or record.get("content")
    if not text or not str(text).strip():
        return None
    node_id = record.get("id") or f"{subset_name}:{hash(text)}"
    metadata = {
        "subset": subset_name,
        "title": record.get("title") or "",
    }
    return TextNode(id_=str(node_id), text=str(text).strip(), metadata=metadata)


def flush_batch(
    index: VectorStoreIndex,
    batch: list[TextNode],
    persist_dir: Path,
    persist_now: bool,
) -> None:
    if not batch:
        return
    index.insert_nodes(batch)
    batch.clear()
    if persist_now:
        index.storage_context.persist(persist_dir=str(persist_dir))
        logging.info("Checkpoint persisted to %s", persist_dir)


def index_subset(
    index: VectorStoreIndex,
    seen_ids: set[str],
    subset_key: str,
    dataset_id: str,
    split: str,
    persist_dir: Path,
    batch_size: int,
    checkpoint_every: int,
) -> int:
    logging.info("Streaming %s (split=%s) ...", dataset_id, split)
    dataset = load_dataset(dataset_id, split=split, streaming=True)

    batch: list[TextNode] = []
    added = 0
    skipped = 0
    examined = 0
    since_checkpoint = 0
    start = time.time()

    for record in dataset:
        examined += 1
        node = record_to_node(record, subset_key)
        if node is None:
            continue
        if node.id_ in seen_ids:
            skipped += 1
            continue
        seen_ids.add(node.id_)
        batch.append(node)

        if len(batch) >= batch_size:
            flush_batch(index, batch, persist_dir, persist_now=False)
            added += batch_size
            since_checkpoint += batch_size
            if since_checkpoint >= checkpoint_every:
                index.storage_context.persist(persist_dir=str(persist_dir))
                since_checkpoint = 0
                elapsed = time.time() - start
                rate = added / elapsed if elapsed > 0 else 0.0
                logging.info(
                    "%s: +%s added (skipped=%s, examined=%s, rate=%.1f nodes/s)",
                    subset_key,
                    added,
                    skipped,
                    examined,
                    rate,
                )

    if batch:
        residual = len(batch)
        flush_batch(index, batch, persist_dir, persist_now=True)
        added += residual
    else:
        index.storage_context.persist(persist_dir=str(persist_dir))

    logging.info(
        "%s complete: added=%s skipped=%s examined=%s",
        subset_key,
        added,
        skipped,
        examined,
    )
    return added


def main() -> int:
    setup_logging()
    args = parse_args()

    embed_model = build_embed_model(
        args.embed_model, args.base_url, args.embed_batch_size
    )
    Settings.embed_model = embed_model

    index, seen_ids = load_or_create_index(args.persist_dir, DEFAULT_MEDRAG_CONFIG["embed_dim"])

    total_added = 0
    for subset_key in args.subsets:
        dataset_id = AVAILABLE_SUBSETS[subset_key]
        total_added += index_subset(
            index=index,
            seen_ids=seen_ids,
            subset_key=subset_key,
            dataset_id=dataset_id,
            split=args.dataset_split,
            persist_dir=args.persist_dir,
            batch_size=args.embed_batch_size,
            checkpoint_every=args.checkpoint_every,
        )

    logging.info(
        "Index build complete. Total added this run: %s. Persisted at %s",
        total_added,
        args.persist_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
