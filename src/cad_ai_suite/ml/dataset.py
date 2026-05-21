from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cad_ai_suite.cad.feature_extraction import (
    FEATURE_KEYS,
    extract_step_features,
    normalize_rows,
    vectorize_features,
)
from cad_ai_suite.data.label_store import DEFAULT_DB_PATH, LabelRecord, list_labels


@dataclass(frozen=True)
class TrainingDataset:
    features: list[list[float]]
    labels: list[int]
    label_names: list[str]
    paths: list[str]
    means: list[float]
    stds: list[float]


def load_training_dataset(db_path: str | Path = DEFAULT_DB_PATH) -> TrainingDataset:
    records = [record for record in list_labels(db_path) if Path(record.path).exists()]
    if not records:
        raise ValueError("No labeled STEP files found. Label files before training.")

    label_names = sorted({record.label for record in records})
    label_to_index = {label: index for index, label in enumerate(label_names)}

    raw_rows = []
    label_indices = []
    paths = []
    for record in records:
        raw_rows.append(vectorize_features(extract_step_features(record.path)))
        label_indices.append(label_to_index[record.label])
        paths.append(record.path)

    normalized_rows, means, stds = normalize_rows(raw_rows)
    return TrainingDataset(
        features=normalized_rows,
        labels=label_indices,
        label_names=label_names,
        paths=paths,
        means=means,
        stds=stds,
    )


def describe_records(records: list[LabelRecord]) -> str:
    lines = []
    for record in records:
        tags = ", ".join(record.tags) if record.tags else "no tags"
        lines.append(f"{record.label}: {record.path} ({tags})")
    return "\n".join(lines)


def normalize_single_feature_row(row: list[float], means: list[float], stds: list[float]) -> list[float]:
    if len(row) != len(FEATURE_KEYS):
        raise ValueError(f"Expected {len(FEATURE_KEYS)} features, got {len(row)}")
    return [
        (value - mean) / (std or 1.0)
        for value, mean, std in zip(row, means, stds)
    ]
