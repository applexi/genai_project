# Local Qwen Pipeline

This project uses local `Ollama` with `Qwen2.5-3B-Instruct` to:

1. generate general hallucination prompts
2. generate medical hallucination prompts
3. merge them into a balanced dataset
4. run the multi-agent OVON hallucination pipeline
5. analyze the resulting scores and plots

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

## Generate Prompt Datasets

Run the general prompt generator from the project root:

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

- `data/balanced_500.jsonl`
- `data/balanced_500_summary.json`

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
python3 agentic_hallucination_300_YES_OVON.py --workers 6
```

Default outputs:

- `data/pipeline_results_with_ths.csv`
- `pipeline.log`

Notes:

- This script uses local Ollama by default at `http://127.0.0.1:11434/v1`
- It uses `qwen2.5:3b-instruct` for all agents
- Prompt processing is parallelized across prompts, while the three agent stages remain sequential within each prompt

## Analyze OVON Results

After the pipeline finishes, generate the plots and stats:

```bash
python3 analyze_agentic_hallucination_300_YES_OVON.py
```

Default outputs:

- `plots/all/...`
- `plots/medical/...`
- `plots/general/...`

Each subset folder contains:

- line and bar plots
- reduction plots
- `stats.txt`

The analysis script also verifies that `prompt_id` in `data/pipeline_results_with_ths.csv` matches the corresponding prompt order in `data/data.jsonl` before splitting results into `all`, `medical`, and `general`.

## Troubleshooting

If generation fails, check:

- Ollama is running: `ollama serve`
- the model is installed: `ollama list`
- the model name matches: `qwen2.5:3b-instruct`

If you want to point the generators at a different Ollama host, set:

```bash
export QWEN_BASE_URL="http://127.0.0.1:11434/v1"
```
