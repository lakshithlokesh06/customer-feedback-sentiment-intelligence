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

MAX_DATASET_ROWS = 100_000
MAX_DATASET_COLUMNS = 200
PREVIEW_ROWS = 100
MIN_REVIEW_LENGTH = 3
SETUP_MESSAGE = "Load a dataset and select a review column to begin analysis."

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
SENTIMENT_COLUMNS = (
    "sentiment_negative", "sentiment_neutral", "sentiment_positive",
    "sentiment_compound", "sentiment_label",
)
SENTIMENT_COLORS = {
    "Positive": "color: #17634b; background-color: #e7f3ec",
    "Neutral": "color: #44546a; background-color: #edf1f6",
    "Negative": "color: #993b3b; background-color: #faeeee",
    "Not analyzed": "color: #666666; background-color: #f3f3f3",
}

SENTIMENT_CHART_COLORS = {'Positive': '#247A59', 'Neutral': '#7B8794', 'Negative': '#B65353'}
MAX_CHART_CATEGORIES = 15
MAX_CATEGORY_CARDINALITY = 50
MAX_RATING_CARDINALITY = 20

TEXT_KEYWORD_DEFAULT = 15
TEXT_PHRASE_DEFAULT = 10
TEXT_DISPLAY_MIN = 5
TEXT_DISPLAY_MAX = 20
TEXT_MIN_TOKEN_LENGTH = 2
TEXT_LARGE_CORPUS_SIZE = 30
TEXT_LARGE_MIN_REVIEWS = 2
TEXT_REPRESENTATIVE_LIMIT = 3

EXPLORER_PAGE_SIZES = (10, 25, 50)
EXPLORER_METADATA_LIMIT = 2
