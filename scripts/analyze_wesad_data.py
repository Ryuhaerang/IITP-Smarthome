#!/usr/bin/env python3
"""Inspect label distributions across WESAD dataset splits."""

import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from wesad.config import ExperimentConfig, parse_config
from wesad.data import DataPreparationResult, prepare_wesad_data


def compute_distribution(labels: Iterable[int], label_order: Sequence[str]) -> Tuple[List[Tuple[str, int, float]], int]:
    """Return counts and percentages for each label given integer-encoded labels."""
    if hasattr(labels, "__array__"):
        label_ids = np.asarray(labels, dtype=np.int64)
    else:
        label_ids = np.fromiter(labels, dtype=np.int64)
    if label_ids.size == 0:
        return [(label, 0, 0.0) for label in label_order], 0
    counts = np.bincount(label_ids, minlength=len(label_order))
    total = int(counts.sum())
    breakdown = []
    for idx, label in enumerate(label_order):
        count = int(counts[idx])
        percentage = (count / total * 100.0) if total else 0.0
        breakdown.append((label, count, percentage))
    return breakdown, total


def print_distribution(summary: List[Tuple[str, int, float]], total: int, header: str) -> None:
    """Pretty-print the computed label distribution."""
    print(header)
    print(f"Total samples: {total}")
    for label, count, percentage in summary:
        print(f"  {label:>10}: {count:5d} ({percentage:6.2f}%)")
    print()


def analyze(prepared: DataPreparationResult, config: ExperimentConfig) -> None:
    """Compute per-split and overall label distributions."""
    label_order = config.label_order

    train_summary, train_total = compute_distribution(prepared.train_labels, label_order)
    val_summary, val_total = compute_distribution(prepared.val_labels, label_order)
    test_summary, test_total = compute_distribution(prepared.test_labels, label_order)

    combined_labels = np.concatenate(
        [prepared.train_labels, prepared.val_labels, prepared.test_labels]
    ).astype(np.int64)
    overall_summary, overall_total = compute_distribution(combined_labels, label_order)

    print_distribution(train_summary, train_total, "Train split")
    print_distribution(val_summary, val_total, "Validation split")
    print_distribution(test_summary, test_total, "Test split")
    print_distribution(overall_summary, overall_total, "Overall dataset")

    print("Subject splits:")
    for split_name, users in prepared.subject_splits.items():
        print(f"  {split_name}: {', '.join(users)}")


def main() -> None:
    config = parse_config()
    prepared = prepare_wesad_data(config)
    analyze(prepared, config)


if __name__ == "__main__":
    main()
