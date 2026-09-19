"""Small, UI-independent CSV reader with user-safe validation errors."""

import csv
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.config import MAX_UPLOAD_BYTES, MAX_DATASET_ROWS, MAX_DATASET_COLUMNS

CSVSource = str | Path | BinaryIO


class DataValidationError(ValueError):
    """An input cannot be loaded as a usable CSV dataset."""


def validate_csv_file(source: CSVSource, *, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Validate extension, readability and size; return bounded file contents.

    Binary streams are rewound before reading and their position is restored.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if not isinstance(source, (str, Path)) and not all(
        callable(getattr(source, method, None)) for method in ("read", "seek", "tell")
    ):
        raise DataValidationError("Provide a CSV path or a readable binary file.")
    name = str(source) if isinstance(source, (str, Path)) else str(getattr(source, "name", "dataset.csv"))
    if Path(name).suffix.lower() != ".csv":
        raise DataValidationError("Choose a file with a .csv extension.")
    try:
        if isinstance(source, (str, Path)):
            with Path(source).open("rb") as handle:
                content = handle.read(max_bytes + 1)
        else:
            position = source.tell()
            try:
                source.seek(0)
                content = source.read(max_bytes + 1)
            finally:
                source.seek(position)
    except FileNotFoundError:
        raise DataValidationError("The CSV file is missing. Restore it and try again.") from None
    except (OSError, ValueError, AttributeError, TypeError):
        raise DataValidationError("The CSV file could not be read. Check file access and try again.") from None
    if not isinstance(content, bytes):
        raise DataValidationError("Provide a CSV file opened in binary mode.")
    if len(content) > max_bytes:
        raise DataValidationError(f"The CSV exceeds the {max_bytes / (1024 * 1024):g} MB size limit.")
    if not content.strip():
        raise DataValidationError("The CSV file is empty. Add a header and at least one data row.")
    return content


def is_dataset_empty(data: pd.DataFrame) -> bool:
    """Return whether a dataset contains no nonblank values."""
    return data.empty or data.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all").empty


def read_csv_safely(source: CSVSource, *, max_bytes: int = MAX_UPLOAD_BYTES,
                    max_rows: int = MAX_DATASET_ROWS) -> pd.DataFrame:
    """Read a UTF-8, comma-delimited CSV without inferring a review column."""
    content = validate_csv_file(source, max_bytes=max_bytes)
    try:
        decoded = content.decode("utf-8-sig")
        if "\x00" in decoded:
            raise DataValidationError("The CSV contains invalid text. Export it as UTF-8 CSV.")
        rows = csv.reader(StringIO(decoded), strict=True)
        header = next(rows)
        if len(header) > MAX_DATASET_COLUMNS:
            raise DataValidationError(f"CSV files may contain at most {MAX_DATASET_COLUMNS} columns.")
        names = [name.strip() for name in header]
        if not names or any(not name for name in names):
            raise DataValidationError("Each CSV column must have a nonblank header.")
        if len(set(names)) != len(names):
            raise DataValidationError("CSV column names must be unique.")
        count = 0
        for row in rows:
            count += 1
            if count > max_rows:
                raise DataValidationError(f"The dataset exceeds the {max_rows:,} row limit. Split it into smaller files.")
            if row and len(row) != len(header):
                raise DataValidationError("CSV rows have inconsistent column counts. Check commas and quoting.")
        data = pd.read_csv(BytesIO(content), encoding="utf-8-sig", skip_blank_lines=False,
                           keep_default_na=False, na_values=[""])
    except UnicodeDecodeError:
        raise DataValidationError("The CSV encoding is unsupported. Save the file as UTF-8 CSV.") from None
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration):
        raise DataValidationError("The CSV could not be parsed. Check its header, commas, and quoting.") from None
    if is_dataset_empty(data):
        raise DataValidationError("The dataset has no usable rows. Add at least one nonblank record.")
    return data
