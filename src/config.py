"""Shared product identity, paths, and application settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
APP_TITLE = "Customer Feedback Sentiment Intelligence"
TAGLINE = "Transform customer feedback into actionable sentiment insights."
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_customer_feedback.csv"
NAVIGATION = ("Overview", "Sentiment Analytics", "Text Insights", "Review Explorer", "About")
SAMPLE_COLUMNS = ("review_id", "review", "rating", "date", "product")
DEFAULT_MAX_UPLOAD_MB = 10


def _upload_limit() -> int:
    """Use a safe default when a local setting is invalid."""
    try:
        value = int(os.getenv("FEEDBACK_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB)))
        return value if 1 <= value <= 100 else DEFAULT_MAX_UPLOAD_MB
    except ValueError:
        return DEFAULT_MAX_UPLOAD_MB


MAX_UPLOAD_BYTES = _upload_limit() * 1024 * 1024
KPI_LABELS = ("Total reviews", "Positive sentiment", "Neutral sentiment", "Negative sentiment")
