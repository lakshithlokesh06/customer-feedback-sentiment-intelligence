"""Review validation and preparation without dropping or overwriting input data."""

from dataclasses import dataclass
import re

import pandas as pd

from src.config import MIN_REVIEW_LENGTH
from src.data.loader import DataValidationError


@dataclass
class PreparedReviews:
    """Prepared rows, quality counts, and collision-safe metadata column names."""

    data: pd.DataFrame
    quality: dict[str, int]
    text_column: str
    status_column: str


def review_columns(data: pd.DataFrame) -> list[str]:
    """Order text-like columns first without selecting one for the user."""
    return sorted(data.columns, key=lambda name: not (
        pd.api.types.is_object_dtype(data[name]) or pd.api.types.is_string_dtype(data[name])
    ))


def dataset_summary(data: pd.DataFrame) -> dict[str, int | str]:
    """Profile the original dataframe; whitespace-only cells count as missing."""
    return {
        "Rows": len(data), "Columns": len(data.columns),
        "Missing values": int(data.replace(r"^\s*$", pd.NA, regex=True).isna().sum().sum()),
        "Duplicate rows": int(data.duplicated().sum()),
        "Memory": f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    }


def _status(value: object) -> tuple[object, str]:
    if not pd.api.types.is_scalar(value):
        return pd.NA, "non_text"
    if pd.isna(value):
        return pd.NA, "missing"
    if not isinstance(value, str):
        return pd.NA, "non_text"
    text = value.strip()
    if not text:
        return pd.NA, "empty"
    # CSV mixed-type columns may represent numeric cells as strings.
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return pd.NA, "non_text"
    if len(text) < MIN_REVIEW_LENGTH:
        return pd.NA, "too_short"
    return text, "valid"


def prepare_reviews(data: pd.DataFrame, column: str) -> PreparedReviews:
    """Copy all original records and append normalized text and quality status."""
    if column not in data.columns or not data.columns.is_unique:
        raise DataValidationError("Select a unique column from the loaded dataset.")
    prepared = data.copy(deep=True)
    names = []
    for base in ("prepared_review", "review_status"):
        name = base
        while name in prepared.columns or name in names:
            name = "_" + name
        names.append(name)
    checks = [_status(value) for value in data[column]]
    prepared[names[0]] = pd.array([text for text, _ in checks], dtype="string")
    statuses = [status for _, status in checks]
    prepared[names[1]] = statuses
    quality = {
        "Total Rows": len(data), "Valid Reviews": statuses.count("valid"),
        "Missing Reviews": statuses.count("missing"), "Empty Reviews": statuses.count("empty"),
        "Invalid Reviews": statuses.count("non_text") + statuses.count("too_short"),
    }
    return PreparedReviews(prepared, quality, *names)
