#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GENERAL_JSONL = DATA_DIR / "general.jsonl"
MEDICAL_JSONL = DATA_DIR / "medical.jsonl"
GENERAL_SUMMARY = DATA_DIR / "general_summary.json"
MEDICAL_SUMMARY = DATA_DIR / "medical_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced dataset from general and medical prompts while "
            "skipping errored records."
        )
    )
    parser.add_argument(
        "x",
        type=int,
        help="Total number of prompts to include across both datasets.",
    )
    parser.add_argument(
        "--general-jsonl",
        type=Path,
        default=GENERAL_JSONL,
        help="Path to the general prompts JSONL file.",
    )
    parser.add_argument(
        "--medical-jsonl",
        type=Path,
        default=MEDICAL_JSONL,
        help="Path to the medical prompts JSONL file.",
    )
    parser.add_argument(
        "--general-summary",
        type=Path,
        default=GENERAL_SUMMARY,
        help="Path to the general summary JSON file.",
    )
    parser.add_argument(
        "--medical-summary",
        type=Path,
        default=MEDICAL_SUMMARY,
        help="Path to the medical summary JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL path. Defaults to data/balanced_<x>.jsonl.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional output summary JSON path. Defaults to data/balanced_<x>_summary.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def valid_record(record: Dict[str, object]) -> bool:
    if record.get("error_comment"):
        return False
    prompt = record.get("prompt")
    technique_key = record.get("technique_key")
    return isinstance(prompt, str) and bool(prompt.strip()) and isinstance(technique_key, str)


def group_valid_records(records: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        if not valid_record(record):
            continue
        grouped[str(record["technique_key"])].append(record)

    for technique_records in grouped.values():
        technique_records.sort(key=lambda item: int(item.get("global_index", 0)))
    return dict(grouped)


def infer_techniques(
    general_summary: Dict[str, object], medical_summary: Dict[str, object]
) -> List[str]:
    general_allocations = general_summary.get("allocations", {})
    medical_allocations = medical_summary.get("allocations", {})
    if not isinstance(general_allocations, dict) or not isinstance(medical_allocations, dict):
        raise ValueError("Both summary files must contain an 'allocations' object.")

    general_keys = list(general_allocations.keys())
    medical_keys = list(medical_allocations.keys())
    if set(general_keys) != set(medical_keys):
        raise ValueError("General and medical summaries do not define the same techniques.")

    return general_keys


def build_balanced_selection(
    *,
    x: int,
    techniques: List[str],
    general_records: Dict[str, List[Dict[str, object]]],
    medical_records: Dict[str, List[Dict[str, object]]],
) -> List[Dict[str, object]]:
    if x <= 0:
        raise ValueError("x must be positive.")
    if x % 2 != 0:
        raise ValueError("x must be even so half can be general and half medical.")

    per_domain = x // 2
    technique_count = len(techniques)
    if technique_count == 0:
        raise ValueError("No techniques were found in the summary files.")
    if per_domain % technique_count != 0:
        raise ValueError(
            f"x must make x/2 divisible by {technique_count} techniques. "
            f"Received x={x}."
        )

    per_technique = per_domain // technique_count

    for technique in techniques:
        general_available = len(general_records.get(technique, []))
        medical_available = len(medical_records.get(technique, []))
        if general_available < per_technique or medical_available < per_technique:
            max_per_technique = min(general_available, medical_available)
            max_total = max_per_technique * technique_count * 2
            raise ValueError(
                "Balanced selection is not possible. "
                f"Technique '{technique}' has {general_available} valid general prompts and "
                f"{medical_available} valid medical prompts; need {per_technique} from each. "
                f"At this bottleneck, the largest feasible x is {max_total}."
            )

    selected: List[Dict[str, object]] = []
    for source_name, grouped_records in (
        ("general", general_records),
        ("medical", medical_records),
    ):
        for technique in techniques:
            for record in grouped_records[technique][:per_technique]:
                selected.append(
                    {
                        "source_domain": source_name,
                        **record,
                    }
                )
    return selected


def main() -> int:
    args = parse_args()
    output_path = args.output or (DATA_DIR / f"data.jsonl")
    summary_output_path = args.summary_output or (
        DATA_DIR / f"data_summary.json"
    )

    general_summary = load_json(args.general_summary)
    medical_summary = load_json(args.medical_summary)
    techniques = infer_techniques(general_summary, medical_summary)

    general_records = group_valid_records(load_jsonl(args.general_jsonl))
    medical_records = group_valid_records(load_jsonl(args.medical_jsonl))

    selected = build_balanced_selection(
        x=args.x,
        techniques=techniques,
        general_records=general_records,
        medical_records=medical_records,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    per_domain = args.x // 2
    per_technique = per_domain // len(techniques)
    summary = {
        "requested_total": args.x,
        "written_total": len(selected),
        "general_count": per_domain,
        "medical_count": per_domain,
        "technique_count": len(techniques),
        "per_technique_per_domain": per_technique,
        "output_file": str(output_path),
        "techniques": techniques,
    }
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote balanced dataset to {output_path}", file=sys.stderr)
    print(f"Wrote summary to {summary_output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
