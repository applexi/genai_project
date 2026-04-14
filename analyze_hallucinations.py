#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_PATH = ROOT / "data" / "data.jsonl"
DEFAULT_RESULTS_CSV = ROOT / "data" / "pipeline_results_with_ths.csv"
DEFAULT_PLOTS_DIR = ROOT / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OVON analysis plots and stats from pipeline results."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Path to the pipeline results CSV.",
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
    df_sorted = df_results.sort_values(by="THS1", ascending=True)
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
        "1st_Agent (THS1)": df_results["THS1"].sum(),
        "2nd_Reviewer (THS2)": df_results["THS2"].sum(),
        "3rd_Reviewer (THS3)": df_results["THS3"].sum(),
    }


def total_ths_variant_two(df_results: pd.DataFrame) -> dict[str, float]:
    return {
        "FrontEndAgent (THS1)": df_results["THS1"].sum(),
        "SecondLevelReviewer (THS2)": df_results["THS2"].sum(),
        "ThirdLevelReviewer (THS3)": df_results["THS3"].sum(),
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
    plt.title(f"Total Hallucination Score by Agent ({subset_label})", fontsize=16)
    plt.xlabel("Agent", fontsize=14)
    plt.ylabel("Total Hallucination Score", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y")
    save_plot(plots_dir / "04_total_ths_annotated.png")


def plot_total_ths_simple(df_results: pd.DataFrame, plots_dir: Path, subset_label: str) -> None:
    total_ths = total_ths_variant_two(df_results)
    plt.figure(figsize=(8, 6))
    plt.bar(total_ths.keys(), total_ths.values(), color=["blue", "orange", "green"])
    plt.title(f"Total Hallucination Score by Agent ({subset_label})", fontsize=16)
    plt.xlabel("Agent", fontsize=14)
    plt.ylabel("Total Hallucination Score", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y")
    save_plot(plots_dir / "05_total_ths_simple.png")


def calculate_percentage_reduction_all(total_ths: dict[str, float]) -> dict[str, float]:
    ths1 = total_ths["FrontEndAgent (THS1)"]
    ths2 = total_ths["SecondLevelReviewer (THS2)"]
    ths3 = total_ths["ThirdLevelReviewer (THS3)"]
    return {
        "Reduction (1st_agent -> 2nd_agent)": ((ths1 - ths2) / abs(ths1)) * 100,
        "Reduction (2nd_agent -> 3rd_agent)": ((ths2 - ths3) / abs(ths2)) * 100,
        "Reduction (1st_agent -> 3rd_agent)": ((ths1 - ths3) / abs(ths1)) * 100,
    }


def calculate_percentage_reduction_frontend_paths(
    total_ths: dict[str, float],
) -> dict[str, float]:
    ths1 = total_ths["FrontEndAgent (THS1)"]
    ths2 = total_ths["SecondLevelReviewer (THS2)"]
    ths3 = total_ths["ThirdLevelReviewer (THS3)"]
    return {
        "(1st_agent -> 2nd_agent)": ((ths1 - ths2) / abs(ths1)) * 100,
        "(1st_agent -> 3rd_agent)": ((ths1 - ths3) / abs(ths1)) * 100,
    }


def plot_percentage_reductions(
    reductions: dict[str, float],
    plots_dir: Path,
    filename: str,
    colors: list[str],
    rotation: int,
    subset_label: str,
) -> None:
    labels = list(reductions.keys())
    values = list(reductions.values())
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
    plt.title(f"Percentage Reduction in Hallucination Scores ({subset_label})", fontsize=16)
    plt.xlabel("Reduction Path", fontsize=14)
    plt.ylabel("Percentage Reduction (%)", fontsize=14)
    plt.xticks(fontsize=12, rotation=rotation)
    plt.yticks(fontsize=12)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_plot(plots_dir / filename)


def run_analysis_for_subset(
    df_results: pd.DataFrame, plots_dir: Path, subset_key: str, subset_label: str
) -> None:
    if df_results.empty:
        print(f"Skipping {subset_label} analysis because the subset is empty.")
        return

    subset_plots_dir = plots_dir / subset_key
    stats_lines = [f"Subset: {subset_label}", f"Row count: {len(df_results)}", ""]

    plot_line_chart(df_results, subset_plots_dir, subset_label)
    plot_grouped_bar_sorted_by_ths1(df_results, subset_plots_dir, subset_label)
    plot_grouped_bar_sorted_within_prompt(df_results, subset_plots_dir, subset_label)
    plot_total_ths_annotated(df_results, subset_plots_dir, subset_label)
    plot_total_ths_simple(df_results, subset_plots_dir, subset_label)

    total_ths = total_ths_variant_two(df_results)
    print(f"Total Hallucination Scores ({subset_label}):")
    stats_lines.append(f"Total Hallucination Scores ({subset_label}):")
    reductions_all = calculate_percentage_reduction_all(total_ths)
    for label, value in total_ths.items():
        line = f"{label}: {value:.2f}"
        print(line)
        stats_lines.append(line)

    print(f"Percentage Reductions ({subset_label}):")
    stats_lines.append("")
    stats_lines.append(f"Percentage Reductions ({subset_label}):")
    for label, value in reductions_all.items():
        line = f"{label}: {value:.2f}%"
        print(line)
        stats_lines.append(line)
    plot_percentage_reductions(
        reductions_all,
        subset_plots_dir,
        "06_percentage_reductions_all.png",
        ["blue", "orange", "green"],
        15,
        subset_label,
    )

    reductions_frontend = calculate_percentage_reduction_frontend_paths(total_ths)
    print(f"Percentage Reductions ({subset_label}):")
    stats_lines.append("")
    stats_lines.append(f"Percentage Reductions ({subset_label}):")
    for label, value in reductions_frontend.items():
        line = f"{label}: {value:.2f}%"
        print(line)
        stats_lines.append(line)
    plot_percentage_reductions(
        reductions_frontend,
        subset_plots_dir,
        "07_percentage_reductions_frontend_paths.png",
        ["green", "orange", "blue"],
        0,
        subset_label,
    )
    write_stats_file(subset_plots_dir / "stats.txt", stats_lines)


def main() -> int:
    args = parse_args()
    df_results = load_results(args.results_csv)
    df_results = attach_prompt_metadata(df_results, args.prompts)

    run_analysis_for_subset(df_results, args.plots_dir, "all", "All")
    run_analysis_for_subset(
        df_results[df_results["source_domain"] == "medical"].copy(),
        args.plots_dir,
        "medical",
        "Medical",
    )
    run_analysis_for_subset(
        df_results[df_results["source_domain"] == "general"].copy(),
        args.plots_dir,
        "general",
        "General",
    )
    print(f"Plots saved to {args.plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
