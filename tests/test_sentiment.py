"""VADER behavior, preservation, resources, and session UI regression tests."""
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT, SENTIMENT_COLUMNS
from src.data.preprocessor import prepare_reviews
from src.data.session import reset_review_selection, set_dataset, clear_dataset
from src.sentiment.analyzer import (
    SentimentError, get_analyzer, sentiment_label, analyze_review,
    analyze_dataframe, sentiment_summary,
)


@pytest.mark.parametrize('score,label', [(0.05, 'Positive'), (0.04999, 'Neutral'),
    (-0.05, 'Negative'), (-0.04999, 'Neutral'), (0, 'Neutral'), (1, 'Positive'), (-1, 'Negative')])
def test_thresholds(score, label):
    assert sentiment_label(score) == label
    fake = Mock()
    fake.polarity_scores.return_value = dict(neg=0, neu=1, pos=0, compound=score)
    with patch('src.sentiment.analyzer.get_analyzer', return_value=fake):
        assert analyze_review('A review')['sentiment_label'] == label


@pytest.mark.parametrize('text,label', [
    ('I love this excellent product! Absolutely wonderful!', 'Positive'),
    ('I hate this terrible product. Awful and broken.', 'Negative'),
    ('The package contains a cable.', 'Neutral'),
])
def test_actual_vader(text, label):
    result = analyze_review(text)
    assert result['sentiment_label'] == label
    assert -1 <= result['sentiment_compound'] <= 1
    assert abs(sum(result[name] for name in SENTIMENT_COLUMNS[:3]) - 1) < 0.002


def test_resources_and_reuse():
    get_analyzer.cache_clear()
    first = get_analyzer()
    assert first is get_analyzer()
    assert first.lexicon
    assert first.emojis
    get_analyzer.cache_clear()
    with patch('src.sentiment.analyzer.SentimentIntensityAnalyzer', side_effect=OSError('internal path')):
        with pytest.raises(SentimentError, match='Reinstall') as caught:
            get_analyzer()
        assert 'internal path' not in str(caught.value)
    assert get_analyzer().lexicon  # failed initialization is not cached


def test_preservation_duplicates_and_unusable():
    data = pd.DataFrame({'review': ['Good service!', None, ' ', 42, 'ok', 'Good service!'],
                         'rating': [1, 5, 4, 3, 2, 5]}, index=[5, 2, 2, 9, 4, 1])
    original = data.copy(deep=True)
    prepared = prepare_reviews(data, 'review')
    before = prepared.data.copy(deep=True)
    with patch('src.sentiment.analyzer.analyze_review', wraps=analyze_review) as scoring:
        result = analyze_dataframe(prepared)
    assert scoring.call_count == 1
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(prepared.data, before)
    pd.testing.assert_frame_equal(result.data[before.columns], before)
    assert result.data.index.tolist() == [5, 2, 2, 9, 4, 1]
    assert len(result.data) == 6
    assert set(SENTIMENT_COLUMNS).issubset(result.data.columns)
    assert result.data['sentiment_label'].tolist() == ['Positive', *(['Not analyzed'] * 4), 'Positive']
    assert result.data[list(SENTIMENT_COLUMNS[:4])].iloc[1:5].isna().all().all()
    assert sentiment_summary(result)['Analyzed Reviews'] == 2
    assert result.data['sentiment_compound'].iloc[0] == result.data['sentiment_compound'].iloc[-1]


def test_collision_preserves_existing_sentiment_columns():
    data = pd.DataFrame({'text': ['Excellent service'], **{name: ['original'] for name in SENTIMENT_COLUMNS}})
    prepared = prepare_reviews(data, 'text')
    result = analyze_dataframe(prepared)
    pd.testing.assert_frame_equal(result.data[data.columns], data)
    assert all(logical != physical for logical, physical in result.columns.items())
    assert sentiment_summary(result)['Positive Reviews'] == 1


@pytest.mark.parametrize('text', ['😀 😍 ❤️', '!!! ??? ...', 'Great GREAT great!!!',
    'https://example.com I love this', 'El producto funciona bien', '产品很好',
    'good ' * 5000, '12345'])
def test_edge_text_does_not_crash(text):
    result = analyze_dataframe(prepare_reviews(pd.DataFrame({'text': [text]}), 'text'))
    assert len(result.data) == 1
    if text == '12345':
        assert result.data['sentiment_label'].iloc[0] == 'Not analyzed'
    else:
        assert -1 <= result.data['sentiment_compound'].iloc[0] <= 1


def test_emphasis_and_emoji():
    assert analyze_review('This is GREAT!!!')['sentiment_compound'] > analyze_review('This is great')['sentiment_compound']
    assert analyze_review('😍 😍 😍')['sentiment_label'] == 'Positive'


def test_all_unusable_and_empty():
    for data in (pd.DataFrame({'text': [None, '', 42]}), pd.DataFrame({'text': []})):
        with patch('src.sentiment.analyzer.get_analyzer', side_effect=AssertionError('Must not initialize')):
            result = analyze_dataframe(prepare_reviews(data, 'text'))
        assert len(result.data) == len(data)
        assert sentiment_summary(result)['Analyzed Reviews'] == 0


def test_session_reset():
    state = {'review_column': 'a', 'prepared': 'old', 'analysis': 'old'}
    reset_review_selection(state, 'a')
    assert state['analysis'] == 'old'
    reset_review_selection(state, 'b')
    assert 'analysis' not in state and 'prepared' not in state
    state['analysis'] = 'old'
    set_dataset(state, pd.DataFrame({'text': ['Great']}), 'new', 'new')
    assert 'analysis' not in state
    state['analysis'] = 'old'
    clear_dataset(state)
    assert 'analysis' not in state


def _ready_app():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.button[0].click().run()
    app.selectbox[0].select('review').run()
    return app


def test_app_explicit_analysis_persistence_and_reset():
    app = _ready_app()
    assert 'analysis' not in app.session_state
    next(b for b in app.button if b.label == 'Analyze Sentiment').click().run()
    assert not app.exception
    assert len(app.session_state['analysis'].data) == 20
    assert app.metric[0].label == 'Analyzed Reviews'
    for page in ['Sentiment Analytics', 'Review Explorer', 'Text Insights', 'Overview']:
        with patch('src.sentiment.analyzer.analyze_dataframe', side_effect=AssertionError('Unexpected scoring')):
            app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        assert len(app.session_state['analysis'].data) == 20
    app.selectbox[0].select('product').run()
    assert 'analysis' not in app.session_state
    next(b for b in app.button if b.label == 'Analyze Sentiment').click().run()
    assert 'analysis' in app.session_state
    app.sidebar.button[0].click().run()
    assert 'analysis' not in app.session_state
    assert app.selectbox[0].value is None


def test_app_error_safe_and_no_action_for_invalid_column():
    app = _ready_app()
    with patch('src.sentiment.analyzer.analyze_dataframe', side_effect=SentimentError('VADER resources unavailable. Reinstall requirements.')):
        next(b for b in app.button if b.label == 'Analyze Sentiment').click().run()
    assert not app.exception
    assert any('resources unavailable' in error.value for error in app.error)
    assert 'analysis' not in app.session_state
    app.selectbox[0].select('rating').run()
    assert not any(b.label == 'Analyze Sentiment' for b in app.button)


def test_unexpected_service_failure_is_safe():
    fake = Mock()
    fake.polarity_scores.side_effect = RuntimeError('secret internal details')
    with patch('src.sentiment.analyzer.get_analyzer', return_value=fake):
        with pytest.raises(SentimentError) as caught:
            analyze_review('Good service')
    assert 'secret' not in str(caught.value)
    app = _ready_app()
    with patch('src.sentiment.analyzer.analyze_dataframe', side_effect=RuntimeError('secret internal details')):
        next(b for b in app.button if b.label == 'Analyze Sentiment').click().run()
    assert not app.exception
    assert any('could not complete' in error.value for error in app.error)
    assert all('secret' not in error.value for error in app.error)
    assert 'analysis' not in app.session_state
