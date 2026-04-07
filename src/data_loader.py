"""Data loading and validation functions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

REQUIRED_COLUMNS = ("text", "label")
SPLITS = ("train", "dev", "test")


class DatasetValidationError(Exception):
    """Raised when dataset structure or content is invalid."""


def _validate_split(df: pd.DataFrame, split_name: str, dataset_name: str) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise DatasetValidationError(
            f"Dataset '{dataset_name}' split '{split_name}' is missing required columns: {missing_columns}."
        )

    cleaned = df.loc[:, REQUIRED_COLUMNS].copy()
    cleaned["text"] = cleaned["text"].astype(str)
    cleaned["label"] = cleaned["label"].astype(str)

    before_rows = len(cleaned)
    cleaned = cleaned.replace({"text": {"": pd.NA}, "label": {"": pd.NA}})
    cleaned = cleaned.dropna(subset=["text", "label"]).reset_index(drop=True)

    if cleaned.empty:
        raise DatasetValidationError(
            f"Dataset '{dataset_name}' split '{split_name}' has no valid rows after dropping missing text/label."
        )

    dropped = before_rows - len(cleaned)
    if dropped > 0:
        print(
            f"[WARN] Dropped {dropped} empty/missing rows from {dataset_name}/{split_name}."
        )

    return cleaned


def load_tsv(file_path: Path, split_name: str, dataset_name: str) -> pd.DataFrame:
    """Load and validate one TSV split file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file for dataset '{dataset_name}', split '{split_name}': {file_path}"
        )

    df = pd.read_csv(file_path, sep="\t")
    return _validate_split(df=df, split_name=split_name, dataset_name=dataset_name)


def resolve_dataset_dir(data_root: Path, dataset_name: str) -> Path:
    """Resolve dataset path with a compatibility fallback to legacy root layout."""
    standard_path = data_root / dataset_name
    legacy_path = Path(dataset_name)

    if standard_path.exists():
        return standard_path
    if legacy_path.exists():
        print(
            f"[WARN] Using legacy dataset location '{legacy_path}'. "
            f"Prefer '{standard_path}' for the documented structure."
        )
        return legacy_path

    raise FileNotFoundError(
        f"Could not find dataset directory for '{dataset_name}'. "
        f"Expected '{standard_path}' (or legacy '{legacy_path}')."
    )


def load_dataset_splits(data_root: Path, dataset_name: str) -> Dict[str, pd.DataFrame]:
    """Load and validate train/dev/test splits for a dataset."""
    dataset_dir = resolve_dataset_dir(data_root=data_root, dataset_name=dataset_name)

    splits: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        split_path = dataset_dir / f"{split}.tsv"
        splits[split] = load_tsv(file_path=split_path, split_name=split, dataset_name=dataset_name)

    return splits


def build_train_dev_text_label(
    splits: Dict[str, pd.DataFrame],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return text and label series for train/dev/test splits."""
    train_df = splits["train"]
    dev_df = splits["dev"]
    test_df = splits["test"]
    return (
        train_df["text"],
        train_df["label"],
        dev_df["text"],
        dev_df["label"],
        test_df["text"],
        test_df["label"],
    )
