#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ovon_config import DEFAULT_PROMPTS_PATH


ROOT = Path(__file__).resolve().parent
DEFAULT_PLOTS_DIR = ROOT / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OVON analysis plots and stats from pipeline results."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        required=True,
        help="Path to the pipeline results CSV (example: original_results.csv or medrag_original_results.csv).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Path to the input prompt JSONL file.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory where generated plots will be saved.",
    )
    return parser.parse_args()


def load_results(results_csv: Path) -> pd.DataFrame:
    return pd.read_csv(results_csv)


def resolve_results_plots_dir(base_plots_dir: Path, results_csv: Path) -> Path:
    return base_plots_dir / results_csv.stem


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
    if prompt_metadata.empty:
        raise ValueError(f"No usable prompts found in {prompts_path}")

    if len(df_results) > len(prompt_metadata):
        raise ValueError(
            "Results CSV has more rows than prompt metadata; cannot align prompt_id safely."
        )

    merged = df_results.merge(
        prompt_metadata,
        on="prompt_id",
        how="left",
        suffixes=("", "_source"),
    )

    if merged["prompt_source"].isna().any():
        raise ValueError("Some prompt_id values in the results CSV were not found in data/data.jsonl.")

    mismatched = merged["prompt"].astype(str) != merged["prompt_source"].astype(str)
    if mismatched.any():
        first_bad = merged.loc[mismatched, ["prompt_id", "prompt", "prompt_source"]].iloc[0]
        raise ValueError(
            "Prompt alignment check failed between data/pipeline_results_with_ths.csv and data/data.jsonl "
            f"at prompt_id={first_bad['prompt_id']}."
        )

    return merged.drop(columns=["prompt_source"])


def save_plot(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_stats_file(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: float | None, digits: int = 2) -> str:
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


def calculate_improvements_from_totals(total_ths: dict[str, float]) -> dict[str, float | None]:
    ths1 = total_ths["FrontEndAgent (THS1)"]
    ths2 = total_ths["SecondLevelReviewer (THS2)"]
    ths3 = total_ths["ThirdLevelReviewer (THS3)"]

    def safe_pct(new_value: float, old_value: float) -> float | None:
        if pd.isna(new_value) or pd.isna(old_value) or old_value == 0:
            return None
        return ((new_value - old_value) / abs(old_value)) * 100

    return {
        "1st -> 2nd (%)": safe_pct(ths2, ths1),
        "2nd -> 3rd (%)": safe_pct(ths3, ths2),
        "1st -> 3rd (%)": safe_pct(ths3, ths1),
    }


def build_summary_table_rows(conditioned_subsets: list[tuple[str, pd.DataFrame]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for condition_label, conditioned_df in conditioned_subsets:
        total_ths = total_ths_variant_two(conditioned_df)
        if "top score" in conditioned_df.columns:
            top_scores = pd.to_numeric(conditioned_df["top score"], errors="coerce")
            top_score_count = int(top_scores.notna().sum())
            average_top_score = top_scores.dropna().mean()
        else:
            top_score_count = None
            average_top_score = None
        rows.append(
            [
                condition_label,
                str(len(conditioned_df)),
                format_float(total_ths["FrontEndAgent (THS1)"]),
                format_float(total_ths["SecondLevelReviewer (THS2)"]),
                format_float(total_ths["ThirdLevelReviewer (THS3)"]),
                "n/a" if top_score_count is None else str(top_score_count),
                format_float(average_top_score, 3),
            ]
        )
    return rows


def build_improvement_table_rows(conditioned_subsets: list[tuple[str, pd.DataFrame]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for condition_label, conditioned_df in conditioned_subsets:
        improvements = calculate_improvements_from_totals(total_ths_variant_two(conditioned_df))
        rows.append(
            [
                condition_label,
                format_float(improvements["1st -> 2nd (%)"]),
                format_float(improvements["2nd -> 3rd (%)"]),
                format_float(improvements["1st -> 3rd (%)"]),
            ]
        )
    return rows


def append_top_score_summary(stats_lines: list[str], df_results: pd.DataFrame) -> None:
    if "top score" not in df_results.columns:
        return

    top_scores = pd.to_numeric(df_results["top score"], errors="coerce")
    non_null_top_scores = top_scores.dropna()
    stats_lines.append("")
    stats_lines.append("MedRAG Top Score Summary:")
    stats_lines.append(f"Recorded top score count: {int(non_null_top_scores.notna().sum())}")
    if non_null_top_scores.empty:
        stats_lines.append("Average top score: n/a")
        return
    stats_lines.append(f"Average top score: {non_null_top_scores.mean():.3f}")


def build_melted_results(df_results: pd.DataFrame) -> pd.DataFrame:
    df_melted = pd.melt(
        df_results,
        id_vars=["prompt_id"],
        value_vars=["THS1", "THS2", "THS3"],
        var_name="Agent",
        value_name="Total Hallucination Score",
    )
    agent_mapping = {
        "THS1": "FrontEndAgent",
        "THS2": "SecondLevelReviewer",
        "THS3": "ThirdLevelReviewer",
    }
    df_melted["Agent"] = df_melted["Agent"].map(agent_mapping)
    return df_melted


def plot_line_chart(df_results: pd.DataFrame, plots_dir: Path, subset_label: str) -> None:
    df_melted = build_melted_results(df_results)
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_melted,
        x="prompt_id",
        y="Total Hallucination Score",
        hue="Agent",
        marker="o",
        linewidth=2.5,
    )
    plt.title(f"Total Hallucination Scores by Agent and Prompt ({subset_label})", fontsize=16)
    plt.xlabel("Prompt ID", fontsize=14)
    plt.ylabel("Total Hallucination Score", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Agent", title_fontsize=12, fontsize=12)
    plt.grid(True)
    save_plot(plots_dir / "01_line_plot.png")


def plot_grouped_bar_sorted_by_ths1(
    df_results: pd.DataFrame, plots_dir: Path, subset_label: str
) -> None:
    df_sorted = df_results.sort_values(by="THS1", ascending=False)
    df_melted = build_melted_results(df_sorted)
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df_melted,
        x="prompt_id",
        y="Total Hallucination Score",
        hue="Agent",
        palette="viridis",
    )
    plt.title(
        f"Total Hallucination Scores by Agent and Prompt (Sorted by THS1, {subset_label})",
        fontsize=16,
    )
    plt.xlabel("Prompt ID (Sorted by THS1)", fontsize=14)
    plt.ylabel("Total Hallucination Score", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Agent", title_fontsize=12, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_plot(plots_dir / "02_grouped_bar_sorted_by_ths1.png")


def plot_grouped_bar_sorted_within_prompt(
    df_results: pd.DataFrame, plots_dir: Path, subset_label: str
) -> None:
    df_melted = build_melted_results(df_results)
    df_melted = df_melted.sort_values(
        by=["prompt_id", "Total Hallucination Score"], ascending=[True, False]
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df_melted,
        x="prompt_id",
        y="Total Hallucination Score",
        hue="Agent",
        palette=["blue", "orange", "green"],
    )
    plt.title(
        f"Total Hallucination Scores by Agent and Prompt (Sorted within Each Prompt, {subset_label})",
        fontsize=16,
    )
    plt.xlabel("Prompt ID", fontsize=14)
    plt.ylabel("Total Hallucination Score", fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(title="Agent", title_fontsize=12, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    xticks = df_melted["prompt_id"].unique()
    plt.xticks(xticks[::10], fontsize=12, rotation=0)
    save_plot(plots_dir / "03_grouped_bar_sorted_within_prompt.png")


def total_ths_variant_one(df_results: pd.DataFrame) -> dict[str, float]:
    return {
        "1st_Agent (THS1)": df_results["THS1"].mean(),
        "2nd_Reviewer (THS2)": df_results["THS2"].mean(),
        "3rd_Reviewer (THS3)": df_results["THS3"].mean(),
    }


def total_ths_variant_two(df_results: pd.DataFrame) -> dict[str, float]:
    return {
        "FrontEndAgent (THS1)": df_results["THS1"].mean(),
        "SecondLevelReviewer (THS2)": df_results["THS2"].mean(),
        "ThirdLevelReviewer (THS3)": df_results["THS3"].mean(),
    }


def plot_total_ths_annotated(df_results: pd.DataFrame, plots_dir: Path, subset_label: str) -> None:
    total_ths = total_ths_variant_one(df_results)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(total_ths.keys(), total_ths.values(), color=["blue", "orange", "green"])
    for bar, value in zip(bars, total_ths.values()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom" if value > 0 else "top",
            fontsize=12,
            color="black",
        )
    plt.title(f"Average Total Hallucination Score by Agent ({subset_label})", fontsize=16)
    plt.xlabel("Agent", fontsize=14)
    plt.ylabel("Average Total Hallucination Score", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y")
    save_plot(plots_dir / "04_total_ths_annotated.png")


def plot_total_ths_simple(df_results: pd.DataFrame, plots_dir: Path, subset_label: str) -> None:
    total_ths = total_ths_variant_two(df_results)
    plt.figure(figsize=(8, 6))
    plt.bar(total_ths.keys(), total_ths.values(), color=["blue", "orange", "green"])
    plt.title(f"Average Total Hallucination Score by Agent ({subset_label})", fontsize=16)
    plt.xlabel("Agent", fontsize=14)
    plt.ylabel("Average Total Hallucination Score", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y")
    save_plot(plots_dir / "05_total_ths_simple.png")


def calculate_percentage_ths_improvement_all(total_ths: dict[str, float]) -> dict[str, float]:
    ths1 = total_ths["FrontEndAgent (THS1)"]
    ths2 = total_ths["SecondLevelReviewer (THS2)"]
    ths3 = total_ths["ThirdLevelReviewer (THS3)"]
    return {
        "Improvement (1st_agent -> 2nd_agent)": ((ths2 - ths1) / abs(ths1)) * 100,
        "Improvement (2nd_agent -> 3rd_agent)": ((ths3 - ths2) / abs(ths2)) * 100,
        "Improvement (1st_agent -> 3rd_agent)": ((ths3 - ths1) / abs(ths1)) * 100,
    }


def calculate_percentage_ths_improvement_frontend_paths(
    total_ths: dict[str, float],
) -> dict[str, float]:
    ths1 = total_ths["FrontEndAgent (THS1)"]
    ths2 = total_ths["SecondLevelReviewer (THS2)"]
    ths3 = total_ths["ThirdLevelReviewer (THS3)"]
    return {
        "(1st_agent -> 2nd_agent)": ((ths2 - ths1) / abs(ths1)) * 100,
        "(1st_agent -> 3rd_agent)": ((ths3 - ths1) / abs(ths1)) * 100,
    }


def plot_percentage_improvements(
    improvements: dict[str, float],
    plots_dir: Path,
    filename: str,
    colors: list[str],
    rotation: int,
    subset_label: str,
) -> None:
    labels = list(improvements.keys())
    values = list(improvements.values())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    for idx, value in enumerate(values):
        plt.text(
            idx,
            value + (5 if value > 0 else -5),
            f"{value:.2f}%",
            ha="center",
            fontsize=12,
            color="black",
        )
    plt.title(
        f"Percentage Improvement in Average Total Hallucination Scores ({subset_label})",
        fontsize=16,
    )
    plt.xlabel("Improvement Path", fontsize=14)
    plt.ylabel("Percentage Improvement (%)", fontsize=14)
    plt.xticks(fontsize=12, rotation=rotation)
    plt.yticks(fontsize=12)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_plot(plots_dir / filename)


def append_stats_summary(stats_lines: list[str], df_results: pd.DataFrame, subset_label: str) -> None:
    total_ths = total_ths_variant_two(df_results)
    stats_lines.append(f"Average Total Hallucination Scores ({subset_label}):")
    improvements_all = calculate_percentage_ths_improvement_all(total_ths)
    for label, value in total_ths.items():
        stats_lines.append(f"{label}: {value:.2f}")

    append_top_score_summary(stats_lines, df_results)

    stats_lines.append("")
    stats_lines.append(f"Percentage Improvements ({subset_label}):")
    for label, value in improvements_all.items():
        stats_lines.append(f"{label}: {value:.2f}%")

    improvements_frontend = calculate_percentage_ths_improvement_frontend_paths(total_ths)
    stats_lines.append("")
    stats_lines.append(f"Percentage Improvements ({subset_label}):")
    for label, value in improvements_frontend.items():
        stats_lines.append(f"{label}: {value:.2f}%")


def append_medrag_conditioned_stats(
    stats_lines: list[str], df_results: pd.DataFrame, subset_label: str
) -> None:
    if "routed_to_medrag" not in df_results.columns:
        return

    df_with_route = df_results.copy()
    df_with_route["routed_to_medrag_normalized"] = normalize_routed_to_medrag(
        df_with_route["routed_to_medrag"]
    )

    conditioned_subsets = [
        ("Routed to MedRAG", df_with_route[df_with_route["routed_to_medrag_normalized"]].copy()),
        (
            "Not Routed to MedRAG",
            df_with_route[~df_with_route["routed_to_medrag_normalized"]].copy(),
        ),
    ]

    if "source_domain" in df_with_route.columns:
        present_domains = [
            source_domain
            for source_domain in ("general", "medical")
            if (df_with_route["source_domain"] == source_domain).any()
        ]
        for source_domain in present_domains:
            for routed_value, routed_label in (
                (True, "Routed to MedRAG"),
                (False, "Not Routed to MedRAG"),
            ):
                conditioned_subsets.append(
                    (
                        f"{source_domain.title()} + {routed_label}",
                        df_with_route[
                            (df_with_route["source_domain"] == source_domain)
                            & (df_with_route["routed_to_medrag_normalized"] == routed_value)
                        ].copy(),
                    )
                )

    stats_lines.append("")
    stats_lines.append(f"MedRAG Route-Conditioned Statistics ({subset_label}):")
    for condition_label, conditioned_df in conditioned_subsets:
        stats_lines.append("")
        stats_lines.append(f"Condition: {condition_label}")
        stats_lines.append(f"Row count: {len(conditioned_df)}")
        if conditioned_df.empty:
            stats_lines.append("Skipped because the conditioned subset is empty.")
            continue
        append_stats_summary(stats_lines, conditioned_df, f"{subset_label} | {condition_label}")


def run_analysis_for_subset(
    df_results: pd.DataFrame, plots_dir: Path, subset_key: str, subset_label: str
) -> None:
    if df_results.empty:
        print(f"Skipping {subset_label} analysis because the subset is empty.")
        return

    subset_plots_dir = plots_dir / subset_key
    conditioned_subsets = build_conditioned_subsets(df_results, subset_label)
    stats_lines = [f"Subset: {subset_label}", ""]

    plot_line_chart(df_results, subset_plots_dir, subset_label)
    plot_grouped_bar_sorted_by_ths1(df_results, subset_plots_dir, subset_label)
    plot_grouped_bar_sorted_within_prompt(df_results, subset_plots_dir, subset_label)
    plot_total_ths_annotated(df_results, subset_plots_dir, subset_label)
    plot_total_ths_simple(df_results, subset_plots_dir, subset_label)

    stats_lines.append("")
    stats_lines.append("## Average THS Summary")
    stats_lines.extend(
        build_markdown_table(
            [
                "Group",
                "Rows",
                "THS1",
                "THS2",
                "THS3",
                "Top Score Count",
                "Avg Top Score",
            ],
            build_summary_table_rows(conditioned_subsets),
        )
    )

    plot_percentage_improvements(
        calculate_percentage_ths_improvement_all(total_ths_variant_two(df_results)),
        subset_plots_dir,
        "06_percentage_improvements_all.png",
        ["blue", "orange", "green"],
        15,
        subset_label,
    )

    stats_lines.append("")
    stats_lines.append("## Percentage Improvements")
    stats_lines.extend(
        build_markdown_table(
            ["Group", "1st -> 2nd (%)", "2nd -> 3rd (%)", "1st -> 3rd (%)"],
            build_improvement_table_rows(conditioned_subsets),
        )
    )

    plot_percentage_improvements(
        calculate_percentage_ths_improvement_frontend_paths(total_ths_variant_two(df_results)),
        subset_plots_dir,
        "07_percentage_improvements_frontend_paths.png",
        ["green", "orange", "blue"],
        0,
        subset_label,
    )
    write_stats_file(subset_plots_dir / "stats.txt", stats_lines)


def main() -> int:
    args = parse_args()
    df_results = load_results(args.results_csv)
    df_results = attach_prompt_metadata(df_results, args.prompts)
    results_plots_dir = resolve_results_plots_dir(args.plots_dir, args.results_csv)

    run_analysis_for_subset(df_results, results_plots_dir, "all", "All")
    run_analysis_for_subset(
        df_results[df_results["source_domain"] == "medical"].copy(),
        results_plots_dir,
        "medical",
        "Medical",
    )
    run_analysis_for_subset(
        df_results[df_results["source_domain"] == "general"].copy(),
        results_plots_dir,
        "general",
        "General",
    )
    print(f"Plots saved to {results_plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
