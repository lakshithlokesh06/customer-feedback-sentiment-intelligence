"""Dataset lifecycle helpers shared by the Streamlit controls."""

from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from src.data.preprocessor import dataset_summary


def clear_dataset(state: MutableMapping[str, Any]) -> None:
    """Discard dataset-derived state while leaving navigation intact."""
    state["dataset_revision"] = state.get("dataset_revision", 0) + 1
    for key in ("dataset", "dataset_name", "dataset_id", "profile", "review_column", "prepared", "analysis", "analytics_context", "text_context"):
        state.pop(key, None)


def set_dataset(state: MutableMapping[str, Any], data: pd.DataFrame, name: str, identity: str) -> None:
    """Replace a dataset and reset any incompatible review selection."""
    profile = dataset_summary(data)
    clear_dataset(state)
    state.update(dataset=data, dataset_name=name, dataset_id=identity, profile=profile)


def reset_review_selection(state: MutableMapping[str, Any], column: str | None) -> None:
    """Invalidate prepared and analyzed data only when the selection changes."""
    if state.get("review_column") != column:
        state["review_column"] = column
        state.pop("prepared", None)
        state.pop("analysis", None)
        state.pop("analytics_context", None)
        state.pop("text_context", None)
