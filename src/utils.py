"""General utility helpers for the experiment pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create directory recursively when missing."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    """Save object as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def save_text(text: str, path: Path) -> None:
    """Save plain text to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe to CSV without index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def log(message: str) -> None:
    """Simple logger for consistent console output."""
    print(f"[INFO] {message}")
