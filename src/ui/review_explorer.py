"""Review investigation page with session-local filters and on-demand exports."""
import math
import pandas as pd
import streamlit as st

from src.analytics.review_explorer import LABELS, ReviewFilters, filter_reviews, sort_reviews, paginate
from src.analytics.sentiment_metrics import metadata_columns, parse_dates
from src.config import EXPLORER_PAGE_SIZES, EXPLORER_METADATA_LIMIT, SENTIMENT_COLORS
from src.data.exporter import export_csv, FULL_EXPORT_NAME, FILTERED_EXPORT_NAME


def reset_explorer() -> None:
    """Reset only explorer controls and prepared downloads."""
    for key in list(st.session_state):
        if key.startswith('explorer_') and key != 'explorer_result':
            del st.session_state[key]


def controls(result, review_column: str):
    """Keep the control panel compact and metadata-dependent."""
    state = st.session_state
    if st.button('Reset filters', key='explorer_reset'):
        reset_explorer()
        st.rerun()
    prepared = state['prepared']
    excluded = [review_column, prepared.text_column, prepared.status_column, *result.columns.values()]
    if 'explorer_metadata' not in state:
        state['explorer_metadata'] = metadata_columns(result.data, excluded)
    dates, categories, ratings = state['explorer_metadata']
    st.markdown('**Search & Sentiment**')
    query = st.text_input('Search original review text', key='explorer_query')
    labels = st.multiselect('Sentiment labels', LABELS, default=list(LABELS), key='explorer_labels')
    filters = ReviewFilters(search=query, labels=tuple(labels))
    descriptions = []
    if query.strip():
        descriptions.append(f'Search: {query.strip()[:60]}')
    if tuple(labels) != LABELS:
        descriptions.append('Sentiment: ' + (', '.join(labels) or 'none'))
    selected_date = selected_rating = None
    selected_categories = []
    with st.expander('Additional filters'):
        if st.checkbox('Filter compound score', key='explorer_compound_on'):
            filters.compound = st.slider('Compound range', -1., 1., (-1., 1.), .01, key='explorer_compound')
            descriptions.append(f'Compound: {filters.compound[0]:.2f} to {filters.compound[1]:.2f}')
            st.caption('Selected Not analyzed rows bypass this range; other filters still apply.')
        if categories:
            selected_categories = st.multiselect('Metadata columns', categories, max_selections=EXPLORER_METADATA_LIMIT, key='explorer_categories')
            for column in selected_categories:
                values = sorted(result.data[column].dropna().astype(str).unique())
                chosen = st.multiselect(f'{column} values', values, default=values, key=f'explorer_cat_{column}')
                missing = st.checkbox(f'Include missing {column}', value=True, key=f'explorer_missing_{column}')
                filters.categories[column] = (chosen, missing)
                if chosen != values or not missing:
                    descriptions.append(f'{column}: {len(chosen)} values' + (' + missing' if missing else ''))
        if ratings:
            selected_rating = st.selectbox('Rating column', ratings, key='explorer_rating_column')
            values = pd.to_numeric(result.data[selected_rating], errors='coerce').dropna()
            values = values[values.map(math.isfinite)]
            if not values.empty and st.checkbox('Filter rating', key='explorer_rating_on'):
                lower, upper = float(values.min()), float(values.max())
                if lower == upper:
                    st.caption(f'Only rating {lower:g} is present; missing ratings will be excluded.')
                    filters.rating_range = (lower, upper)
                else:
                    filters.rating_range = st.slider('Rating range', lower, upper, (lower, upper), key=f'explorer_rating_range_{selected_rating}')
                filters.rating_column = selected_rating
                descriptions.append(f'Rating: {filters.rating_range[0]:g} to {filters.rating_range[1]:g}')
        if dates:
            selected_date = st.selectbox('Date column', dates, key='explorer_date_column')
            values = parse_dates(result.data[selected_date]).dropna()
            if not values.empty and st.checkbox('Filter dates', key='explorer_date_on'):
                bounds = st.date_input('Date range', (values.min().date(), values.max().date()),
                                       min_value=values.min().date(), max_value=values.max().date(), key=f'explorer_date_range_{selected_date}')
                if len(bounds) == 2:
                    filters.date_column, filters.date_range = selected_date, bounds
                    descriptions.append(f'Dates: {bounds[0]} to {bounds[1]}')
                else:
                    st.info('Choose an end date to apply the date filter.')
                st.caption('Active date/rating filters exclude missing or invalid metadata. Dates use UTC.')
    sorts = ['Original order', 'Most Positive', 'Most Negative', 'Most Neutral']
    if selected_date:
        sorts += ['Newest', 'Oldest']
    if selected_rating:
        sorts += ['Rating High → Low', 'Rating Low → High']
    st.markdown('**Sort & Display**')
    sort = st.selectbox('Sort results', sorts, key='explorer_sort')
    size = st.selectbox('Reviews per page', EXPLORER_PAGE_SIZES, key='explorer_size')
    return filters, sort, size, selected_date, selected_rating, selected_categories, descriptions


def render_exports(result, matches, signature) -> None:
    """Convert to CSV only after explicit preparation; keep bytes in this session."""
    with st.expander('CSV downloads'):
        st.caption('Full export preserves every analyzed-result row. Filtered export follows current filters and sorting, across all pages.')
        if st.button('Prepare full analyzed CSV'):
            with st.spinner('Preparing CSV…'):
                st.session_state['explorer_full_csv'] = export_csv(result.data)
        if 'explorer_full_csv' in st.session_state:
            st.download_button('Download full analyzed CSV', st.session_state['explorer_full_csv'], FULL_EXPORT_NAME, 'text/csv', on_click='ignore')
        if st.session_state.get('explorer_export_signature') != signature:
            st.session_state.pop('explorer_filtered_csv', None)
        if st.button('Prepare filtered CSV', disabled=matches.empty):
            with st.spinner('Preparing filtered CSV…'):
                st.session_state['explorer_filtered_csv'] = export_csv(matches)
                st.session_state['explorer_export_signature'] = signature
        if 'explorer_filtered_csv' in st.session_state:
            st.download_button('Download filtered CSV', st.session_state['explorer_filtered_csv'], FILTERED_EXPORT_NAME, 'text/csv', on_click='ignore')
        elif matches.empty:
            st.caption('Filtered download is unavailable because no reviews match.')


def render_cards(page, result, review_column, metadata) -> None:
    """Render bounded original review text, quality status, scores, and metadata."""
    for index, row in page.iterrows():
        with st.container(border=True):
            label = row[result.columns['sentiment_label']]
            # Styling includes only our fixed labels and configured CSS, never user HTML.
            st.markdown(f'<span style="{SENTIMENT_COLORS[label]}; padding:4px 8px; border-radius:4px">{label}</span>', unsafe_allow_html=True)
            st.caption(f'Source row index: {index}')
            text = row[review_column]
            st.text('Missing review value' if pd.isna(text) else str(text))
            status_column = st.session_state['prepared'].status_column
            st.caption(f'Preparation status: {row[status_column]}')
            if label != 'Not analyzed':
                scores = ' · '.join(f'{name.capitalize()}: {row[result.columns[f"sentiment_{name}"]]:.3f}'
                                    for name in ('compound', 'positive', 'neutral', 'negative'))
                st.caption(scores)
            else:
                st.caption('No sentiment scores were generated for this record.')
            for name in dict.fromkeys(metadata):
                st.text(f'{name}: {row[name]}')


def render_review_explorer_page() -> None:
    st.subheader('Review Explorer')
    state = st.session_state
    if state.get('dataset') is None:
        st.info('Load a dataset to explore customer reviews.')
        return
    if state.get('review_column') is None:
        st.info('Select the review text column to continue.')
        return
    result = state.get('analysis')
    if result is None:
        st.info('Run sentiment analysis from Overview to unlock the Review Explorer.')
        return
    if state.get('explorer_result') is not result:
        reset_explorer()
        state['explorer_result'] = result
    filters, sort, size, date, rating, categories, descriptions = controls(result, state['review_column'])
    signature = repr((filters, sort, date, rating))
    page_signature = (signature, size)
    if page_signature != state.get('explorer_signature'):
        state['explorer_page'] = 1
        state['explorer_signature'] = page_signature
    matches = sort_reviews(filter_reviews(result, state['review_column'], filters), result, sort, date, rating)
    st.caption(f'Showing {len(matches):,} of {len(result.data):,} reviews · ' + (' · '.join(descriptions) if descriptions else 'Showing all analyzed records.'))
    if not matches.empty:
        labels = matches[result.columns['sentiment_label']]
        mean = matches.loc[labels.ne('Not analyzed'), result.columns['sentiment_compound']].mean()
        metrics = [('Matching Reviews', len(matches)), *[(label, int(labels.eq(label).sum())) for label in LABELS[:3]],
                   ('Average Compound Score', '—' if pd.isna(mean) else f'{mean:.3f}')]
        for start in (0, 3):
            for col, (label, value) in zip(st.columns(3 if start == 0 else 2), metrics[start:start + 3]):
                with col, st.container(border=True):
                    st.metric(label, value)
    render_exports(result, matches, signature)
    if matches.empty:
        st.info('No reviews match the current filters.')
        return
    _, current, pages = paginate(matches, int(state.get('explorer_page', 1)), size)
    state['explorer_page'] = current
    page_number = st.number_input('Page', min_value=1, max_value=pages, step=1, key='explorer_page')
    page, current, pages = paginate(matches, int(page_number), size)
    st.caption(f'Page {current} of {pages} · {len(page)} reviews on this page. Source row indices are UI references, not added export columns.')
    metadata = [*categories, *([rating] if rating else []), *([date] if date else [])]
    if not categories:
        metadata = [*state['explorer_metadata'][1][:1], *metadata]
    render_cards(page, result, state['review_column'], metadata)
