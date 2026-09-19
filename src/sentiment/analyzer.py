"""Offline VADER service; user data is never cached across calls or sessions."""

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD, SENTIMENT_COLUMNS
from src.data.preprocessor import PreparedReviews


class SentimentError(ValueError):
    """A safe, actionable sentiment service failure."""


@dataclass
class AnalysisResult:
    """Analyzed data and logical-to-physical output names for collision safety."""

    data: pd.DataFrame
    columns: dict[str, str]


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Reuse the bundled, offline VADER lexicon and emoji resources."""
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        raise SentimentError(
            "VADER could not initialize. Reinstall the project requirements to restore its bundled resources, then retry."
        ) from None


def sentiment_label(compound: float) -> str:
    """Classify a VADER compound score using the central inclusive thresholds."""
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_review(text: str) -> dict[str, float | str]:
    """Score one nonblank review without changing case, punctuation, or emojis."""
    if not isinstance(text, str) or not text.strip():
        raise SentimentError("Select a nonblank text review before analysis.")
    try:
        scores = get_analyzer().polarity_scores(text)
        return {
            "sentiment_negative": scores["neg"], "sentiment_neutral": scores["neu"],
            "sentiment_positive": scores["pos"], "sentiment_compound": scores["compound"],
            "sentiment_label": sentiment_label(scores["compound"]),
        }
    except SentimentError:
        raise
    except Exception:
        raise SentimentError("This review could not be analyzed. Check the review text and retry.") from None


def analyze_dataframe(prepared: PreparedReviews) -> AnalysisResult:
    """Score only valid rows; preserve input, index, row order, and every column.

    Duplicate text is scored once per call. Existing sentiment-named columns
    remain untouched; new output names receive leading underscores if needed.
    """
    data = prepared.data
    if prepared.text_column not in data or prepared.status_column not in data:
        raise SentimentError("Prepare a review column before running sentiment analysis.")
    columns: dict[str, str] = {}
    for logical in SENTIMENT_COLUMNS:
        name = logical
        while name in data.columns or name in columns.values():
            name = "_" + name
        columns[logical] = name
    outputs = {name: [] for name in SENTIMENT_COLUMNS}
    cache: dict[str, dict[str, float | str]] = {}
    for text, status in zip(data[prepared.text_column], data[prepared.status_column]):
        if status == "valid":
            if not isinstance(text, str) or not text.strip():
                raise SentimentError("Prepared reviews are inconsistent. Select the review column again.")
            if text not in cache:
                cache[text] = analyze_review(text)
            scores = cache[text]
        else:
            scores = {name: "Not analyzed" if name == "sentiment_label" else None for name in SENTIMENT_COLUMNS}
        for name in SENTIMENT_COLUMNS:
            outputs[name].append(scores[name])
    result = data.copy(deep=True)
    for logical, physical in columns.items():
        result[physical] = pd.array(outputs[logical], dtype="string" if logical == "sentiment_label" else "Float64")
    return AnalysisResult(result, columns)


def sentiment_summary(result: AnalysisResult) -> dict[str, int]:
    """Count scored reviews; unusable records are excluded from percentages."""
    labels = result.data[result.columns["sentiment_label"]]
    return {
        "Analyzed Reviews": int(labels.isin(["Positive", "Neutral", "Negative"]).sum()),
        **{f"{label} Reviews": int(labels.eq(label).sum()) for label in ("Positive", "Neutral", "Negative")},
    }
