"""Sentiment Analytics page; calculations and figures live in separate modules."""
import streamlit as st
from src.config import SETUP_MESSAGE, COMPOUND_HELP
from src.analytics.sentiment_metrics import (
    LABELS, analyzed_rows, filter_sentiment, overall, metadata_columns,
    parse_dates, by_category, by_time, by_rating, average_compound, five_star_scale,
)
from src.visualization import charts


def render_analytics_page() -> None:
    st.subheader('Sentiment Analytics')
    st.write('Compare sentiment, follow trends, and explore differences across categories and ratings.')
    state = st.session_state
    if state.get('dataset') is None:
        st.info(SETUP_MESSAGE)
        return
    if state.get('review_column') is None:
        st.info(SETUP_MESSAGE)
        return
    if state.get('analysis') is None:
        st.info('Run sentiment analysis from Overview to unlock analytics.')
        return
    result = state['analysis']
    context = state.get('analytics_context')
    if context is None or context[0] is not result:
        data = analyzed_rows(result)
        prepared = state['prepared']
        excluded = [state['review_column'], prepared.text_column, prepared.status_column, *result.columns.values()]
        context = (result, data, metadata_columns(data, excluded))
        state['analytics_context'] = context
    _, data, (dates, categories, ratings) = context
    revision = state.get('dataset_revision', 0)
    key = f'analytics_labels_{revision}_{state["review_column"]}'
    if key not in state:
        state[key] = list(LABELS)
    if st.button('Reset filters'):
        state[key] = list(LABELS)
    labels = st.multiselect('Sentiment labels', LABELS, key=key)
    selected = filter_sentiment(data, labels)
    summary = overall(selected)
    metrics = {'Total Reviews': len(result.data), 'Analyzed Reviews': len(selected),
               **{f'{label} Reviews': int(summary.loc[summary.Sentiment.eq(label), 'Count'].iloc[0]) for label in LABELS},
               'Average Compound Score': f'{average_compound(selected):.3f}' if len(selected) else '—'}
    items = list(metrics.items())
    for start in (0, 3):
        for column, (label, value) in zip(st.columns(3), items[start:start + 3]):
            with column, st.container(border=True):
                st.metric(label, value, help=COMPOUND_HELP if label == 'Average Compound Score' else None)
    st.caption(f'{len(result.data) - len(data):,} reviews excluded from analysis. {len(data) - len(selected):,} analyzed reviews hidden by filters. Total Reviews is the unfiltered source count.')
    if selected.empty:
        st.info('No analyzed reviews match these filters. Reset filters to view all available results.')
        return
    st.plotly_chart(charts.distribution(summary), width='stretch', config={'displayModeBar': False})
    st.plotly_chart(charts.compound(selected), width='stretch', config={'displayModeBar': False})
    st.caption('−1: more negative · 0: more neutral · +1: more positive. Dotted lines mark −0.05 and +0.05.')
    st.plotly_chart(charts.components(selected), width='stretch', config={'displayModeBar': False})
    st.subheader('Trends over time')
    if dates:
        date = st.selectbox('Date column', dates)
        parsed = parse_dates(data[date])
        span = (parsed.max() - parsed.min()).days
        options = {'Day': 'D'}
        if span >= 7:
            options['Week'] = 'W'
        if span >= 28:
            options['Month'] = 'M'
        grouping = st.selectbox('Time grouping', list(options))
        invalid = int(parse_dates(selected[date]).isna().sum())
        st.caption(f'{invalid} rows with invalid/missing dates excluded from this chart. Dates normalized to UTC; weeks start Monday.')
        times = by_time(selected, date, options[grouping])
        if times.empty:
            st.info('No valid dates in the filtered reviews.')
        else:
            st.plotly_chart(charts.trend(times), width='stretch', config={'displayModeBar': False})
    else:
        st.info('No usable date column found.')
    st.subheader('Category comparison')
    if categories:
        category = st.selectbox('Group by', categories)
        groups = by_category(selected, category)
        st.caption('Showing up to 15 groups by analyzed review volume; ties use category name. Missing values form a separate group.')
        st.plotly_chart(charts.comparison(groups, f'Sentiment by {category}'), width='stretch', config={'displayModeBar': False})
        with st.expander('Category detail table'):
            st.dataframe(groups.rename(columns={label: f'{label} %' for label in LABELS}), hide_index=True, width='stretch')
    else:
        st.info('No suitable low-cardinality category column found.')
    st.subheader('Rating vs sentiment')
    if ratings:
        rating = st.selectbox('Rating column', ratings)
        mismatch_supported = five_star_scale(result.data[rating])
        rating_summary, mismatch = by_rating(selected, rating)
        if not rating_summary.empty:
            st.plotly_chart(charts.comparison(rating_summary, f'Text sentiment by {rating}'), width='stretch', config={'displayModeBar': False})
        if mismatch_supported:
            st.metric('Rating–sentiment mismatch', mismatch or 0, help='On a 1–5 scale: ratings 4–5 with Negative text, or 1–2 with Positive text. This highlights disagreement; it does not prove an error.')
            st.caption('Assuming a 1–5 star scale: 4–5 with Negative text, or 1–2 with Positive text. Rating 3 is excluded. This is descriptive, not a claim of fraud or error.')
        else:
            st.info('Mismatch analytics skipped: ratings do not consistently resemble an integral 1–5 scale.')
        st.caption('Missing/non-numeric ratings are excluded. Ratings never alter text sentiment.')
    else:
        st.info('No suitable numeric rating column found.')
