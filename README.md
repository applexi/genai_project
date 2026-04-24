# Multi-Agent Hallucination Mitigation Pipeline

This project uses local `Ollama` with `Qwen2.5-3B-Instruct` to:

1. generate general hallucination prompts
2. generate medical hallucination prompts
3. merge them into a balanced dataset
4. run a multi-agent OVON hallucination pipeline
5. analyze the resulting scores and plots

## Provenance

`original.py` is a script adaptation of Gosmar and Dahl's OVON-style hallucination mitigation work:

```bibtex
@misc{gosmar,
      title={Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks}, 
      author={Diego Gosmar and Deborah A. Dahl},
      year={2025},
      eprint={2501.13946},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.13946}, 
}
```

The medical-retrieval augmentation is provided by a local MedRAG pipeline built on the [MedRAG corpus](https://huggingface.co/MedRAG) introduced by Xiong et al.:

```bibtex
@article{xiong2024benchmarking,
      title={Benchmarking Retrieval-Augmented Generation for Medicine},
      author={Xiong, Guangzhi and Jin, Qiao and Lu, Zhiyong and Zhang, Aidong},
      year={2024},
      journal={Findings of the Association for Computational Linguistics: ACL 2024}
}
```

Relative to the original OVON / RAG papers, this repository makes several implementation-oriented changes:

- all OpenAI GPT model calls were rewritten to use a shared local base model, currently `qwen2.5:3b-instruct`
- parallelization was added to relevant scripts
- the common model/backend settings were centralized into shared config and helper modules
- medical retrieval is performed fully locally via LlamaIndex + FAISS over the MedRAG corpus, replacing any web-search or OpenAI Assistants/Vector-Stores dependency (the pipeline runs end-to-end against local Ollama with no external API keys)
- retrieval uses `nomic-embed-text` embeddings served by Ollama, with proper `search_document:` / `search_query:` prefixes and L2 normalization so FAISS `IndexFlatIP` yields exact cosine similarity
- the `SecondLevelReviewer` flow was extended so MedRAG-routed failures use a dedicated MedRAG-aware reviewer prompt and pass retrieval-failure evidence into the fallback review step, as some generated hallucinatory prompts do not have factual key terms for retrieval. Still, MedRAG not being able to find any relevant passages should still be treated as important evidence of hallucination.
- KPI scoring now uses a simpler 1-10 rubric for `factuality` and `helpfulness` for each agent, with `THS` defined as the average of those two scores rather than the earlier acronym-based formula.

## Setup

Install Ollama on macOS:

```bash
brew install ollama
```

Start the local Ollama server:

```bash
ollama serve
```

In a second terminal, pull the model:

```bash
ollama pull qwen2.5:3b-instruct
```

You can verify the model is available with:

```bash
ollama list
```

Run all other scripts in a separate terminal.

## MedRAG Setup

`medrag_original.py` is an adaptation of `original.py` that adds a `MedicalClassifier` before the second stage and routes medical prompts through a local MedRAG retriever implemented in `medrag.py`.

### 1. Install Python dependencies

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pull the embedding model

The local RAG pipeline retrieves passages using `nomic-embed-text` served by Ollama:

```bash
ollama pull nomic-embed-text
```

### 3. Build the MedRAG FAISS index

Download the MedRAG textbooks subset, embed it with Ollama, and persist a FAISS index to `data/medrag_index/`:

```bash
python3 build_medrag_index.py --subsets textbooks
```

Notes:

- The builder streams the dataset and is resumable: if you Ctrl-C mid-build, re-run the same command to continue from the last checkpoint.
- `textbooks` alone is ~126k passages and usually completes in about 20–40 minutes on an M-series Mac with Ollama. Vectors weigh in around 400 MB.
- Add `statpearls` for broader recall (`--subsets textbooks statpearls`, ~1 hr total). `wikipedia` and `pubmed` are orders of magnitude larger and not recommended for a laptop.
- Pass `--persist-dir /some/other/path` to put the index elsewhere; `medrag_original.py` accepts `--medrag-index-dir` to match.

### MedRAG Design Notes

Important implementation choices in the local MedRAG pipeline:

- **Nomic prefixes + L2 normalization.** `medrag_embed.NomicOllamaEmbedding` subclasses LlamaIndex's `OllamaEmbedding` to transparently prepend `search_document:` to corpus chunks and `search_query:` to user queries, as required by `nomic-embed-text` for best retrieval quality. Outputs are L2-normalized so FAISS `IndexFlatIP` yields exact cosine similarity in `[-1, 1]`, rather than using `IndexFlatL2` with distance-to-similarity conversion.
- **Cosine similarity threshold.** `MedRAG.min_score` defaults to `0.15`. Retrieved passages below this threshold are discarded, which surfaces genuinely off-topic medical prompts (a hallucination signal) as `medrag_status="no_chunks"`. Tune this per your evaluation set.
- **Resumable index build.** `build_medrag_index.py` streams the MedRAG dataset and deduplicates against the persisted docstore by stable record ID, so rerunning after a crash is safe and cheap. It also calls `storage_context.persist()` every `--checkpoint-every 2000` nodes (configurable), giving you a recovery point mid-build.
- **Embedding throughput.** Ollama serializes embed requests per model, so the `--embed-batch-size 64` default reduces per-request overhead but does not parallelize embeddings. Spawning multiple concurrent embed callers against the same Ollama instance does not help.
- **Thread-safe index sharing.** `medrag_original.py` constructs the `MedRAG` client once in `main()` and passes it into the `ThreadPoolExecutor` workers. `IndexFlatIP` reads, the embedder, and the `OpenAI` client are all thread-safe, so the FAISS index is loaded exactly once per process. The lazy-load path uses a lock to guard the first-access race.
- **Drop-in status contract.** `MedRAG.answer()` preserves the `(text, status)` tuple previously used by the GeoSI client, with new status strings: `"success"`, `"no_chunks"`, `"index_missing"`, `"not_routed"`. The MedRAG-aware `SecondLevelReviewer_MedRAG` fallback receives the retrieval-failure evidence so MedRAG's inability to find passages is itself treated as hallucination evidence.
- **Result CSV schema change.** Compared to the old GeoSI pipeline, the result columns were renamed: `routed_to_geosi` → `routed_to_medrag`, `geosi_status` → `medrag_status`, `geosi_search_query` → `medrag_search_query`. `analyze_hallucinations.py` is unaffected because it only reads `prompt_id`, `THS1`, `THS2`, `THS3`, `prompt`, and `source_domain`.

## Generate Prompt Datasets

In another terminal, run the general prompt generator from the project root:

```bash
python3 data/generate_general.py --workers 6
```

Default outputs:

- `data/general.jsonl`
- `data/general_summary.json`

Run the medical prompt generator:

```bash
python3 data/generate_medical.py --workers 6
```

Default outputs:

- `data/medical.jsonl`
- `data/medical_summary.json`

Notes:

- Both scripts use local Ollama by default at `http://127.0.0.1:11434/v1`
- Both scripts use `qwen2.5:3b-instruct` by default
- Records with malformed JSON responses are still saved with `raw_response` and `error_comment`

## Build The Final Balanced Dataset

After both datasets are generated, run:

```bash
python3 preprocess_data.py 500
```

This means:

- total output size is `500`
- `250` prompts come from `general.jsonl`
- `250` prompts come from `medical.jsonl`
- within each domain, each of the 5 techniques contributes the same number of prompts
- any row with `error_comment` or a missing `prompt` is skipped

Default outputs:

- `data/data.jsonl`
- `data/data_summary.json`

Important constraint:

- `x` must be divisible by `10`, because the pipeline requires `x/2` prompts per domain and equal counts across 5 techniques

## Example End-To-End Run

From the project root:

```bash
python3 data/generate_general.py --workers 6
python3 data/generate_medical.py --workers 6
python3 preprocess_data.py 500
```

## Run The OVON Pipeline

After creating `data/data.jsonl`, run the multi-agent pipeline:

```bash
python3 original.py --workers 6
```

To run the MedRAG-enabled variant instead (requires the FAISS index from the MedRAG Setup section):

```bash
python3 medrag_original.py --workers 6
```

Default outputs:

- `data/original_results.csv`
- `data/medrag_original_results.csv`
- `data/error.txt`
- logs can be found in the `data` folder

Notes:

- This script uses local Ollama by default at `http://127.0.0.1:11434/v1`
- Prompt processing is parallelized across prompts, while the three agent stages remain sequential within each prompt
- `SecondLevelReviewer` and `KPI_Evaluator` JSON failures do not stop the run; they are logged to `data/error.txt` as `original,<prompt_id>` or `medrag_original,<prompt_id>`
- Both pipeline scripts support `--checkpoint-every N` (default `10`) to atomically rewrite a partial results CSV every `N` completed prompts
- The MedRAG FAISS index is loaded once at process start and shared across worker threads (read-only), so per-prompt retrieval is a sub-millisecond FAISS lookup plus the usual Ollama embed/chat calls
- Medical prompts that return no MedRAG passages above the similarity threshold are treated as a hallucination signal and routed through the `SecondLevelReviewer_MedRAG` fallback
- The KPI evaluator now returns JSON with `factuality` and `helpfulness` (both 1-10) for each of `FrontEndAgent`, `SecondLevelReviewer`, and `ThirdLevelReviewer`; `THS1`, `THS2`, and `THS3` are the per-agent averages of those two scores

To rerun only prompts listed in `data/error.txt` and patch the existing CSV rows in place:

```bash
python3 rerun_json_errors.py
```

The repair script reruns only the failing `prompt_id` values, rewrites only the matching rows in `data/original_results.csv` or `data/medrag_original_results.csv`, and then rewrites `data/error.txt` with only the failures that remain.

To smoke-test the pipeline on one medical + one non-medical prompt:

```bash
python3 test_medrag_original.py
```

## Analyze OVON Results

After the pipeline finishes, generate the plots and stats:

```bash
python3 analyze_hallucinations.py --results-csv data/original_results.csv
python3 analyze_hallucinations.py --results-csv data/medrag_original_results.csv
```

Default outputs:

- `plots/original_results/all/...`
- `plots/original_results/medical/...`
- `plots/original_results/general/...`

Each subset folder contains:

- line and bar plots
- improvement plots
- `stats.txt`

The analysis script also verifies that `prompt_id` in the chosen results CSV matches the corresponding prompt order in `data/data.jsonl` before splitting results into `all`, `medical`, and `general`. The plots treat higher `THS` as better, because it is the average of the 1-10 `factuality` and `helpfulness` scores. The summary bar charts and `stats.txt` report average THS per prompt for each agent stage, not summed THS.
