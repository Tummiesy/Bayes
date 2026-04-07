"""Text preprocessing utilities for short-text classification."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable, List, Set

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration flags for text preprocessing."""

    lowercase: bool = True
    strip: bool = True
    collapse_whitespace: bool = True
    remove_punctuation: bool = False
    remove_digits: bool = False
    remove_stopwords: bool = False


_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
_DIGIT_PATTERN = re.compile(r"\d+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(
    text: str,
    config: PreprocessConfig,
    stopwords: Set[str] | None = None,
) -> str:
    """Preprocess one text string with configurable light-normalization steps."""
    processed = "" if text is None else str(text)

    if config.lowercase:
        processed = processed.lower()
    if config.strip:
        processed = processed.strip()
    if config.remove_punctuation:
        processed = processed.translate(_PUNCT_TRANSLATION)
    if config.remove_digits:
        processed = _DIGIT_PATTERN.sub("", processed)
    if config.collapse_whitespace:
        processed = _WHITESPACE_PATTERN.sub(" ", processed).strip()
    if config.remove_stopwords:
        stopwords = stopwords or set(ENGLISH_STOP_WORDS)
        processed = " ".join(token for token in processed.split() if token not in stopwords)

    return processed


def preprocess_corpus(texts: Iterable[str], config: PreprocessConfig) -> List[str]:
    """Preprocess an iterable of text strings and return a list."""
    stopwords = set(ENGLISH_STOP_WORDS) if config.remove_stopwords else None
    return [preprocess_text(text, config=config, stopwords=stopwords) for text in texts]
