"""Text Insights page with session-local corpus reuse and explicit empty states."""
import logging
import streamlit as st
from src.config import SETUP_MESSAGE

from src.analytics.text_insights import (
    ALL_SCOPE, build_corpus, corpus_statistics, minimum_reviews, top_terms, representative_reviews,
)
from src.analytics.sentiment_metrics import LABELS
from src.config import TEXT_KEYWORD_DEFAULT, TEXT_PHRASE_DEFAULT, TEXT_DISPLAY_MIN, TEXT_DISPLAY_MAX
from src.visualization.text_charts import term_chart

logger = logging.getLogger(__name__)


def render_terms(scope, limit: int, title: str, *, phrases: bool = False, sentiment: str = 'Neutral') -> None:
    """Render real counts or an honest empty state for the chosen scope."""
    terms = top_terms(scope, limit, phrases=phrases)
    if terms.empty:
        st.info('No meaningful phrases meet the review-frequency threshold.' if phrases else
                'No meaningful keywords meet the review-frequency threshold.')
        return
    st.plotly_chart(term_chart(terms, title, sentiment), width='stretch', config={'displayModeBar': False})
    st.caption(f'{scope.review_count:,} reviews in this scope · Minimum {minimum_reviews(scope)} review mentions. Hover for occurrence counts and review coverage.')


def render_representatives(result, scope: str) -> None:
    """Show a bounded, class-specific ranking with optional original metadata."""
    st.subheader('Representative reviews')
    options = list(LABELS) if scope == ALL_SCOPE else [scope]
    label = st.selectbox('Representative sentiment', options)
    st.markdown(f"**{'Most Neutral Signals' if label == 'Neutral' else f'Strongest {label} Signals'}**")
    reviews = representative_reviews(result, label)
    if reviews.empty:
        st.info(f'No {label} reviews are available in this scope.')
    for _, row in reviews.iterrows():
        with st.container(border=True):
            st.text(str(row[st.session_state['review_column']]))
            st.caption(f"{label} · Compound Score {row[result.columns['sentiment_compound']]:.3f}")
            metadata = [f'{name}: {row[name]}' for name in ('product', 'rating', 'date') if name in reviews.columns]
            if metadata:
                st.text(' · '.join(metadata))
    st.caption('Signals are ranked by compound score within each class; they are examples, not a statistically representative sample.')


def render_text_insights_page() -> None:
    st.subheader('Text Insights')
    st.write('Discover frequently mentioned words and phrases, then read the reviews behind them.')
    state = st.session_state
    if state.get('dataset') is None:
        st.info(SETUP_MESSAGE)
        return
    if state.get('review_column') is None:
        st.info(SETUP_MESSAGE)
        return
    result = state.get('analysis')
    if result is None:
        st.info('Run sentiment analysis from Overview to unlock text insights.')
        return
    context = state.get('text_context')
    if context is None or context[0] is not result:
        try:
            with st.spinner('Preparing text insights…'):
                corpus = build_corpus(result, state['review_column'])
            state['text_context'] = (result, corpus)
        except Exception:
            logger.exception('Text insights preparation failed')
            st.error('Text insights could not be prepared. Please retry or use a smaller dataset.')
            return
    _, corpus = state['text_context']
    scope_name = st.selectbox('Sentiment scope', [ALL_SCOPE, *LABELS])
    with st.expander('How frequencies are counted'):
        st.write('Review frequency counts reviews mentioning a word or phrase at least once. Word frequency counts every occurrence, including repetitions within one review. Charts rank by review frequency; hover to compare both counts.')
    with st.expander('Display options'):
        keywords = st.slider('Keywords to display', TEXT_DISPLAY_MIN, TEXT_DISPLAY_MAX, TEXT_KEYWORD_DEFAULT)
        phrases = st.slider('Phrases to display', TEXT_DISPLAY_MIN, TEXT_DISPLAY_MAX, TEXT_PHRASE_DEFAULT)
    scope = corpus[scope_name]
    st.caption(f'{scope.review_count:,} analyzed reviews in scope. Unusable rows are excluded. English-focused lexical counts; no semantic topic or root-cause analysis.')
    if not scope.review_count:
        st.info('No analyzed reviews are available for this sentiment scope.')
        return
    metrics = list(corpus_statistics(scope).items())
    for start in (0, 2):
        for column, (label, value) in zip(st.columns(2), metrics[start:start + 2]):
            with column, st.container(border=True):
                st.metric(label, value)
    st.caption('Words / review counts words before stopword filtering. Keyword counts use meaningful tokens. Bars rank by reviews mentioning a term, then raw occurrences; duplicate review rows each count separately.')
    render_terms(scope, keywords, 'Most mentioned keywords')
    render_terms(scope, phrases, 'Common two-word phrases', phrases=True)
    if scope_name == ALL_SCOPE:
        for label, title in [('Positive', 'What customers appreciate'), ('Negative', 'Common customer concerns')]:
            st.subheader(title)
            subset = corpus[label]
            if not subset.review_count:
                st.info(f'No {label} reviews are available.')
                continue
            render_terms(subset, keywords, f'{label} keywords', sentiment=label)
            render_terms(subset, phrases, f'{label} phrases', phrases=True, sentiment=label)
        st.caption('These are frequent terms in sentiment-labeled reviews, not confirmed praise, defects, or causes. English stopwords and simple word matching limit multilingual interpretation.')
    render_representatives(result, scope_name)
