#!/usr/bin/env python3
"""Calculate line-level Extract Method precision, recall, and F1."""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VALID_FILE = ROOT_DIR / "dataset" / "em_valid.json"
DEFAULT_TEST_FILE = ROOT_DIR / "dataset" / "em_test.json"

LINE_RANGE_PATTERN = re.compile(
    r"\bLines?\s*:\s*\[?\s*(\d+)\s*(?:-|–|—|,|\bto\b)\s*(\d+)\s*\]?",
    flags=re.IGNORECASE,
)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def metrics(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }


def safe_name(value: str, fallback: str = "sample") -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return name or fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predicted Extract Method line numbers with the lines "
            "covered by merged_method_lines ground truth."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Prediction directory containing valid/ and test/ subdirectories.",
    )
    parser.add_argument("--valid-file", default=str(DEFAULT_VALID_FILE))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))
    parser.add_argument(
        "--split",
        choices=["all", "valid", "test"],
        default="all",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Default: <results-dir>/metrics_summary.csv",
    )
    parser.add_argument(
        "--details-file",
        default=None,
        help="Default: <results-dir>/metrics_samples.csv",
    )
    return parser.parse_args()


def load_records(json_file: str) -> list[dict[str, Any]]:
    path = Path(json_file)
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return records


def ground_truth_ranges(record: dict[str, Any], sample_id: str) -> set[tuple[int, int]]:
    opportunities = record.get("opportunities")
    if not isinstance(opportunities, list):
        raise ValueError(f"Sample {sample_id}: opportunities must be an array.")

    ranges: set[tuple[int, int]] = set()
    for opportunity in opportunities:
        if not isinstance(opportunity, dict):
            raise ValueError(f"Sample {sample_id}: opportunity must be an object.")
        lines = opportunity.get("merged_method_lines")
        if (
            not isinstance(lines, list)
            or len(lines) != 2
            or not all(isinstance(value, int) for value in lines)
        ):
            raise ValueError(
                f"Sample {sample_id}: merged_method_lines must be [start, end]."
            )
        start, end = lines
        ranges.add((min(start, end), max(start, end)))
    return ranges


def predicted_ranges(prediction: str) -> set[tuple[int, int]]:
    cleaned = prediction.replace("*", "").replace("`", "")
    return {
        (min(int(start), int(end)), max(int(start), int(end)))
        for start, end in LINE_RANGE_PATTERN.findall(cleaned)
    }


def prediction_file(
    results_dir: Path,
    split: str,
    index: int,
    sample_id: str,
) -> Path:
    expected = results_dir / split / f"{index:06d}_{safe_name(sample_id)}.txt"
    if expected.exists():
        return expected

    # Accept older result names that included the split in the filename.
    legacy = results_dir / f"{split}_{index:06d}_{safe_name(sample_id)}.txt"
    return legacy if legacy.exists() else expected


def covered_lines(ranges: set[tuple[int, int]]) -> set[int]:
    lines: set[int] = set()
    for start, end in ranges:
        lines.update(range(start, end + 1))
    return lines


def line_counts(
    predicted_ranges: set[tuple[int, int]],
    ground_truth_ranges: set[tuple[int, int]],
) -> Counts:
    predicted = covered_lines(predicted_ranges)
    ground_truth = covered_lines(ground_truth_ranges)
    matches = predicted & ground_truth
    return Counts(
        tp=len(matches),
        fp=len(predicted - matches),
        fn=len(ground_truth - matches),
    )


def ranges_text(ranges: set[tuple[int, int]]) -> str:
    return ";".join(f"{start}-{end}" for start, end in sorted(ranges))


def lines_text(lines: set[int]) -> str:
    return ";".join(str(line) for line in sorted(lines))


def evaluate_split(
    split: str,
    json_file: str,
    results_dir: Path,
) -> tuple[Counts, list[dict[str, Any]], int, int]:
    records = load_records(json_file)
    summary = Counts()
    rows: list[dict[str, Any]] = []
    missing_files = 0
    unparsed_files = 0

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{json_file}: record {index - 1} must be an object.")

        sample_id = str(record.get("id", index - 1))
        gt_ranges = ground_truth_ranges(record, sample_id)
        result_file = prediction_file(results_dir, split, index, sample_id)

        file_exists = result_file.exists()
        if file_exists:
            prediction = result_file.read_text(encoding="utf-8")
            pred_ranges = predicted_ranges(prediction)
            if not pred_ranges:
                unparsed_files += 1
        else:
            prediction = ""
            pred_ranges = set()
            missing_files += 1

        pred_lines = covered_lines(pred_ranges)
        gt_lines = covered_lines(gt_ranges)
        counts = line_counts(pred_ranges, gt_ranges)
        summary.add(counts)
        sample_metrics = counts.metrics()
        rows.append(
            {
                "split": split,
                "index": index,
                "sample_id": sample_id,
                "prediction_file": str(result_file),
                "file_exists": file_exists,
                "predicted_ranges": ranges_text(pred_ranges),
                "ground_truth_ranges": ranges_text(gt_ranges),
                "predicted_lines": lines_text(pred_lines),
                "ground_truth_lines": lines_text(gt_lines),
                **sample_metrics,
            }
        )

    return summary, rows, missing_files, unparsed_files


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    selected_splits = (
        [args.split] if args.split != "all" else ["valid", "test"]
    )
    dataset_files = {"valid": args.valid_file, "test": args.test_file}

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    overall = Counts()
    overall_samples = 0
    overall_missing = 0
    overall_unparsed = 0

    for split in selected_splits:
        counts, rows, missing, unparsed = evaluate_split(
            split,
            dataset_files[split],
            results_dir,
        )
        detail_rows.extend(rows)
        overall.add(counts)
        overall_samples += len(rows)
        overall_missing += missing
        overall_unparsed += unparsed
        summary_rows.append(
            {
                "split": split,
                "samples": len(rows),
                "missing_prediction_files": missing,
                "unparsed_prediction_files": unparsed,
                **counts.metrics(),
            }
        )

    summary_rows.append(
        {
            "split": "overall",
            "samples": overall_samples,
            "missing_prediction_files": overall_missing,
            "unparsed_prediction_files": overall_unparsed,
            **overall.metrics(),
        }
    )

    summary_file = Path(args.summary_file) if args.summary_file else (
        results_dir / "metrics_summary.csv"
    )
    details_file = Path(args.details_file) if args.details_file else (
        results_dir / "metrics_samples.csv"
    )
    write_csv(summary_file, summary_rows)
    write_csv(details_file, detail_rows)

    for row in summary_rows:
        print(
            f"{row['split']}: samples={row['samples']} "
            f"TP={row['TP']} FP={row['FP']} FN={row['FN']} "
            f"precision={row['Precision']:.6f} "
            f"recall={row['Recall']:.6f} F1={row['F1']:.6f} "
            f"missing={row['missing_prediction_files']} "
            f"unparsed={row['unparsed_prediction_files']}"
        )
    print(f"Saved summary to {summary_file}")
    print(f"Saved sample details to {details_file}")


if __name__ == "__main__":
    main()
