#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from ovon_config import DEFAULT_PROMPTS_PATH


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "handlabeled_data"
DEFAULT_OUTPUT_DIR = ROOT / "handlabeled_stats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stats.txt files for hand-labeled factuality/helpfulness results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing hand-labeled original_results.csv and medrag_original_results.csv.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to the input prompt JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated stats folders will be saved.",
    )
    return parser.parse_args()


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


def attach_prompt_metadata(df_results: pd.DataFrame, prompts_path: Path) -> pd.DataFrame:
    prompt_metadata = pd.DataFrame(load_prompt_records(prompts_path))
    merged = df_results.merge(
        prompt_metadata,
        on="prompt_id",
        how="left",
        suffixes=("", "_source"),
    )
    if merged["prompt_source"].isna().any():
        raise ValueError("Some prompt_id values were not found in the prompt file.")
    mismatched = merged["prompt"].astype(str) != merged["prompt_source"].astype(str)
    if mismatched.any():
        first_bad = merged.loc[mismatched, ["prompt_id", "prompt", "prompt_source"]].iloc[0]
        raise ValueError(
            f"Prompt alignment check failed at prompt_id={first_bad['prompt_id']}."
        )
    return merged.drop(columns=["prompt_source"])


def write_stats_file(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def build_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return [header_line, separator_line, *body_lines]


def short_subset_label(label: str) -> str:
    mapping = {
        "All": "All",
        "General": "Gen",
        "Medical": "Med",
    }
    return mapping.get(label, label)


def normalize_routed_to_medrag(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if pd.notna(value)
        else False
    )


def normalize_hallucination_label(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.isin([0.0, 1.0]))


def build_conditioned_subsets(
    df_results: pd.DataFrame, subset_label: str
) -> list[tuple[str, pd.DataFrame]]:
    short_label = short_subset_label(subset_label)
    conditioned_subsets: list[tuple[str, pd.DataFrame]] = [(short_label, df_results.copy())]
    if "routed_to_medrag" not in df_results.columns:
        return conditioned_subsets

    df_with_route = df_results.copy()
    df_with_route["routed_to_medrag_normalized"] = normalize_routed_to_medrag(
        df_with_route["routed_to_medrag"]
    )
    conditioned_subsets.extend(
        [
            ("RAG", df_with_route[df_with_route["routed_to_medrag_normalized"]].copy()),
            (
                "Orig",
                df_with_route[~df_with_route["routed_to_medrag_normalized"]].copy(),
            ),
        ]
    )

    if "source_domain" in df_with_route.columns:
        present_domains = [
            source_domain
            for source_domain in ("general", "medical")
            if (df_with_route["source_domain"] == source_domain).any()
        ]
        for source_domain in present_domains:
            short_domain = "Gen" if source_domain == "general" else "Med"
            for routed_value, routed_label in (
                (True, "RAG"),
                (False, "Orig"),
            ):
                conditioned_subsets.append(
                    (
                        f"{short_domain}+{routed_label}",
                        df_with_route[
                            (df_with_route["source_domain"] == source_domain)
                            & (df_with_route["routed_to_medrag_normalized"] == routed_value)
                        ].copy(),
                    )
                )
    return conditioned_subsets


def build_labeled_summary_row(df_results: pd.DataFrame, subset_label: str) -> list[str]:
    hallucination = normalize_hallucination_label(df_results["Hallucination or Not"])
    helpfulness = pd.to_numeric(df_results["Helpfulness (0-5)"], errors="coerce")
    labeled_mask = hallucination.notna() & helpfulness.notna()
    labeled_df = df_results.loc[labeled_mask].copy()
    labeled_hallucination = hallucination.loc[labeled_mask]
    labeled_helpfulness = helpfulness.loc[labeled_mask]

    if labeled_df.empty:
        avg_factuality = None
        avg_helpfulness = None
        avg_hallucination = None
    else:
        avg_factuality = (1.0 - labeled_hallucination).mean()
        avg_helpfulness = labeled_helpfulness.mean()
        avg_hallucination = labeled_hallucination.mean()

    return [
        subset_label,
        str(len(df_results)),
        str(len(labeled_df)),
        format_float(avg_factuality),
        format_float(avg_helpfulness),
        format_float(avg_hallucination),
    ]


def run_stats_for_subset(
    df_results: pd.DataFrame, output_dir: Path, subset_key: str, subset_label: str
) -> None:
    if df_results.empty:
        return
    conditioned_subsets = build_conditioned_subsets(df_results, subset_label)
    stats_lines: list[str] = [f"Subset: {subset_label}", ""]
    stats_lines.append(
        "Hallucination label uses `1 = hallucination` and `0 = not hallucination`, so factuality is reported as `1 - hallucination rate`."
    )
    stats_lines.append("")
    stats_lines.append("## Hand-Labeled Summary")
    stats_lines.extend(
        build_markdown_table(
            [
                "Group",
                "Rows",
                "Labeled Rows",
                "Avg Factuality",
                "Avg Helpfulness (0-5)",
                "Hallucination Rate",
            ],
            [build_labeled_summary_row(group_df, group_label) for group_label, group_df in conditioned_subsets],
        )
    )
    write_stats_file(output_dir / subset_key / "stats.txt", stats_lines)


def analyze_results_csv(results_csv: Path, prompts_path: Path, output_dir: Path) -> None:
    df_results = pd.read_csv(results_csv)
    df_results = attach_prompt_metadata(df_results, prompts_path)
    results_output_dir = output_dir / results_csv.stem

    run_stats_for_subset(df_results, results_output_dir, "all", "All")
    run_stats_for_subset(
        df_results[df_results["source_domain"] == "medical"].copy(),
        results_output_dir,
        "medical",
        "Medical",
    )
    run_stats_for_subset(
        df_results[df_results["source_domain"] == "general"].copy(),
        results_output_dir,
        "general",
        "General",
    )


def main() -> int:
    args = parse_args()
    for csv_name in ("original_results.csv", "medrag_original_results.csv"):
        analyze_results_csv(args.input_dir / csv_name, args.prompts, args.output_dir)
    print(f"Hand-labeled stats written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
