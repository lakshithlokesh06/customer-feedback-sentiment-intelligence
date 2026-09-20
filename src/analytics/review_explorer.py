"""Vectorized review filtering, stable sorting, and bounded pagination."""
from dataclasses import dataclass, field
from math import ceil
import pandas as pd
from src.analytics.sentiment_metrics import parse_dates
from src.sentiment.analyzer import AnalysisResult

LABELS = ('Positive', 'Neutral', 'Negative', 'Not analyzed')


@dataclass
class ReviewFilters:
    search: str = ''
    labels: tuple[str, ...] = LABELS
    compound: tuple[float, float] | None = None
    categories: dict[str, tuple[list[str], bool]] = field(default_factory=dict)
    rating_column: str | None = None
    rating_range: tuple[float, float] | None = None
    date_column: str | None = None
    date_range: tuple[object, object] | None = None


def filter_reviews(result: AnalysisResult, review_column: str, filters: ReviewFilters) -> pd.DataFrame:
    """Select matching rows without modifying source values or physical column names.

    Null compound values on explicitly selected Not analyzed rows bypass the
    compound range only; search and all active metadata filters still apply.
    """
    data = result.data
    labels = data[result.columns['sentiment_label']]
    mask = labels.isin(filters.labels)
    query = filters.search.strip()
    if query:
        mask &= data[review_column].astype('string').str.contains(query, case=False, regex=False, na=False)
    if filters.compound is not None:
        score = data[result.columns['sentiment_compound']]
        mask &= score.between(*filters.compound).fillna(False) | labels.eq('Not analyzed')
    for column, (values, missing) in filters.categories.items():
        series = data[column].astype('string')
        mask &= series.isin(values) | (series.isna() & missing)
    if filters.rating_column and filters.rating_range is not None:
        ratings = pd.to_numeric(data[filters.rating_column], errors='coerce')
        mask &= ratings.between(*filters.rating_range).fillna(False)
    if filters.date_column and filters.date_range is not None:
        dates = parse_dates(data[filters.date_column]).dt.normalize()
        start, end = (pd.Timestamp(value, tz='UTC') for value in filters.date_range)
        mask &= dates.between(start, end).fillna(False)
    return data.loc[mask].copy()


def sort_reviews(data: pd.DataFrame, result: AnalysisResult, sort: str,
                 date_column: str | None = None, rating_column: str | None = None) -> pd.DataFrame:
    """Sort stably with null scores/dates/ratings last; ties retain source order."""
    if sort == 'Original order':
        return data.copy()
    if sort in ('Most Positive', 'Most Negative', 'Most Neutral'):
        values = data[result.columns['sentiment_compound']]
        if sort == 'Most Neutral':
            values = values.abs()
        ascending = sort != 'Most Positive'
    elif sort in ('Newest', 'Oldest') and date_column:
        values = parse_dates(data[date_column])
        ascending = sort == 'Oldest'
    elif sort in ('Rating High → Low', 'Rating Low → High') and rating_column:
        values = pd.to_numeric(data[rating_column], errors='coerce')
        ascending = sort == 'Rating Low → High'
    else:
        return data.copy()
    # Positional permutation also supports duplicate source index labels.
    positions = values.reset_index(drop=True).sort_values(ascending=ascending, kind='stable', na_position='last').index
    return data.iloc[positions].copy()


def paginate(data: pd.DataFrame, page: int, page_size: int) -> tuple[pd.DataFrame, int, int]:
    """Clamp a one-based page and return a bounded slice, page number and count."""
    if page_size <= 0:
        raise ValueError('Page size must be positive')
    pages = max(1, ceil(len(data) / page_size))
    page = max(1, min(page, pages))
    return data.iloc[(page - 1) * page_size:page * page_size].copy(), page, pages
