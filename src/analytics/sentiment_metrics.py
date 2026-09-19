"""Pure analytics over analyzed rows; original results remain untouched."""
import pandas as pd
from src.sentiment.analyzer import AnalysisResult
from src.config import MAX_CHART_CATEGORIES, MAX_CATEGORY_CARDINALITY, MAX_RATING_CARDINALITY

LABELS = ('Positive', 'Neutral', 'Negative')


def analyzed_rows(result: AnalysisResult) -> pd.DataFrame:
    """Select scored records and use canonical names in an independent frame."""
    rows = result.data.loc[result.data[result.columns['sentiment_label']].isin(LABELS)].copy()
    # Keep metadata separately when an uploaded column conflicts with output names.
    return rows.rename(columns={physical: logical for logical, physical in result.columns.items()
                                if physical != logical}).loc[:, lambda frame: ~frame.columns.duplicated(keep='last')]


def filter_sentiment(data: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    return data.loc[data.sentiment_label.isin(labels)].copy()


def overall(data: pd.DataFrame) -> pd.DataFrame:
    counts = data.sentiment_label.value_counts().reindex(LABELS, fill_value=0)
    return pd.DataFrame({'Sentiment': LABELS, 'Count': counts.values,
                         'Percent': counts.values / len(data) * 100 if len(data) else [0., 0., 0.]})


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse textual/datetime dates in UTC; never interpret numbers as epochs."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns, UTC]')
    return pd.to_datetime(series, errors='coerce', format='mixed', utc=True)


def metadata_columns(data: pd.DataFrame, excluded: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Suggest usable dates, low-cardinality groups, and named numeric ratings."""
    dates, categories, ratings = [], [], []
    for name in data.columns:
        if name in excluded or name.startswith('sentiment_'):
            continue
        series = data[name]
        if not pd.api.types.is_numeric_dtype(series) and parse_dates(series).notna().mean() >= .5:
            dates.append(name)
        unique = series.nunique()
        if name not in dates and 1 <= unique <= MAX_CATEGORY_CARDINALITY and (unique < len(data) or unique <= 5) and not any(token in name.lower() for token in ('id', 'review', 'text')):
            if not pd.api.types.is_numeric_dtype(series):
                categories.append(name)
        if unique <= MAX_RATING_CARDINALITY and any(token in name.lower() for token in ('rating', 'stars', 'score')) and pd.to_numeric(series, errors='coerce').notna().any():
            ratings.append(name)
    return dates, categories, ratings


def grouped(data: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Aggregate counts, class shares, and compound mean for aligned groups."""
    work = data.assign(_group=groups)
    counts = pd.crosstab(work._group, work.sentiment_label).reindex(columns=LABELS, fill_value=0)
    result = counts.div(counts.sum(axis=1), axis=0).mul(100)
    result['Review count'] = counts.sum(axis=1)
    result['Average compound'] = work.groupby('_group').sentiment_compound.mean()
    return result.rename_axis('Group').reset_index()


def by_category(data: pd.DataFrame, column: str, limit: int = MAX_CHART_CATEGORIES) -> pd.DataFrame:
    result = grouped(data, data[column].astype('string').fillna('(Missing)'))
    return result.sort_values(['Review count', 'Group'], ascending=[False, True]).head(limit)


def by_time(data: pd.DataFrame, column: str, frequency: str) -> pd.DataFrame:
    dates = parse_dates(data[column]).dt.tz_localize(None)
    groups = dates.dt.to_period(frequency).dt.start_time
    return grouped(data, groups).sort_values('Group')


def by_rating(data: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int | None]:
    """Compare numeric ratings; mismatch only for integral values on 1–5."""
    ratings = pd.to_numeric(data[column], errors='coerce')
    summary = grouped(data, ratings)
    supported = five_star_scale(ratings)
    mismatch = ((ratings.ge(4) & data.sentiment_label.eq('Negative')) |
                (ratings.le(2) & data.sentiment_label.eq('Positive')))
    return summary, int(mismatch.sum()) if supported else None


def average_compound(data: pd.DataFrame) -> float | None:
    """Return mean score for a scored subset, or None when empty."""
    value = data.sentiment_compound.mean()
    return None if pd.isna(value) else float(value)


def five_star_scale(series: pd.Series) -> bool:
    """Check numeric metadata across the source, regardless of review usability."""
    values = pd.to_numeric(series, errors='coerce').dropna()
    return bool(len(values) and values.between(1, 5).all() and values.mod(1).eq(0).all())
