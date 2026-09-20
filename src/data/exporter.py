"""In-memory UTF-8 CSV exports without global caches or filesystem writes."""
import pandas as pd

FULL_EXPORT_NAME = 'customer_feedback_sentiment_analysis.csv'
FILTERED_EXPORT_NAME = 'customer_feedback_filtered_reviews.csv'


def export_csv(data: pd.DataFrame) -> bytes:
    """Preserve physical columns and current row order; serialize nulls as blanks.

    Source values are not modified or spreadsheet-formula escaped: consumers
    should import untrusted CSV text as text rather than evaluate its contents.
    """
    return data.to_csv(index=False, na_rep='', lineterminator='\n').encode('utf-8')
