"""Lexical counts, document coverage, representative rankings and UI checks."""
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.analytics.text_insights import (
    ALL_SCOPE, build_corpus, tokenize, extract_text, top_terms, corpus_statistics,
    representative_reviews, minimum_reviews,
)
from src.sentiment.analyzer import AnalysisResult
from src.config import PROJECT_ROOT
from src.data.session import clear_dataset, reset_review_selection


def result_for(texts, labels=None, scores=None):
    labels = labels if labels is not None else ['Positive'] * len(texts)
    scores = scores if scores is not None else [.5] * len(texts)
    return AnalysisResult(pd.DataFrame({'review': texts, 'sentiment_label': labels,
                                      'sentiment_compound': scores}),
                          {'sentiment_label': 'sentiment_label', 'sentiment_compound': 'sentiment_compound'})


def test_cleaning_and_negation():
    assert tokenize('The BATTERY, life!!! https://example.com/a is GREAT.') == ['battery', 'life', 'great']
    assert tokenize('not no never') == ['not', 'no', 'never']
    assert tokenize("I don't like it. It can't work and won't last.") == ['not', 'like', 'not', 'work', 'not']
    assert tokenize('a & $ 123 _ x') == []
    assert tokenize('www.example.com battery') == ['battery']
    assert tokenize('just really useful battery') == ['useful', 'battery']


@pytest.mark.parametrize('text', ['', None, 'the and or is it', '😀 😍 ❤️', '!!! ???'])
def test_empty_and_uninformative(text):
    assert tokenize(text) == []


def test_keyword_document_frequency_and_order():
    result = result_for(['battery battery battery', 'service battery', 'service'])
    corpus = build_corpus(result, 'review')
    terms = top_terms(corpus[ALL_SCOPE], 10).set_index('Term')
    assert terms.index.tolist() == ['battery', 'service']
    assert terms.loc['battery', 'Occurrences'] == 4
    assert terms.loc['battery', 'Reviews'] == 2
    assert terms.loc['battery', 'Coverage (%)'] == pytest.approx(200 / 3)
    assert corpus_statistics(corpus[ALL_SCOPE])['Meaningful words'] == 6


def test_document_frequency_beats_repetition():
    corpus = build_corpus(result_for(['battery ' * 30, 'service', 'service']), 'review')
    assert top_terms(corpus[ALL_SCOPE], 1).Term.tolist() == ['service']


def test_scopes_and_unanalyzed_exclusion():
    corpus = build_corpus(result_for(['great battery', 'broken cable', 'ignore phrase'],
                                    ['Positive', 'Negative', 'Not analyzed']), 'review')
    assert set(top_terms(corpus['Positive'], 10).Term) == {'great', 'battery'}
    assert set(top_terms(corpus['Negative'], 10).Term) == {'broken', 'cable'}
    assert corpus[ALL_SCOPE].review_count == 2
    assert top_terms(corpus['Neutral'], 10).empty
    for label in ('Positive', 'Negative'):
        other = 'Negative' if label == 'Positive' else 'Positive'
        single = build_corpus(result_for(['battery life'], [label]), 'review')
        assert top_terms(single[other], 10).empty


def test_bigrams_preserve_adjacency_and_boundaries():
    tokens, phrases, _ = extract_text('Customer service is great. Battery life works; easy setup.')
    assert 'customer service' in phrases and 'battery life' in phrases and 'easy setup' in phrases
    assert 'service great' not in phrases and 'great battery' not in phrases
    assert extract_text('good https://example.com battery')[1] == []
    corpus = build_corpus(result_for(['battery life battery life', 'battery life']), 'review')
    phrases = top_terms(corpus[ALL_SCOPE], 10, phrases=True).set_index('Term')
    assert phrases.loc['battery life', 'Occurrences'] == 3
    assert phrases.loc['battery life', 'Reviews'] == 2


def test_minimum_frequency_small_and_large():
    tiny = build_corpus(result_for(['unique keyword']), 'review')[ALL_SCOPE]
    assert minimum_reviews(tiny) == 1
    assert not top_terms(tiny, 10).empty
    large = build_corpus(result_for(['common battery'] * 29 + ['rare unicorn']), 'review')[ALL_SCOPE]
    assert minimum_reviews(large) == 2
    assert 'unicorn' not in top_terms(large, 20).Term.tolist()


def test_representatives_and_preservation():
    result = result_for(['p1', 'p2', 'n1', 'n2', 'z1', 'z2', 'p3'],
                        ['Positive', 'Positive', 'Negative', 'Negative', 'Neutral', 'Neutral', 'Positive'],
                        [.5, .9, -.4, -.8, -.01, .04, .9])
    before = result.data.copy(deep=True)
    assert representative_reviews(result, 'Positive').review.tolist() == ['p2', 'p3', 'p1']
    assert representative_reviews(result, 'Negative').review.tolist() == ['n2', 'n1']
    assert representative_reviews(result, 'Neutral').review.tolist() == ['z1', 'z2']
    build_corpus(result, 'review')
    pd.testing.assert_frame_equal(before, result.data)


def test_unicode_duplicates_and_empty_dataframe():
    for text in ['这个产品很好', 'Très bon produit', '😀', '!!!', 'the and it']:
        corpus = build_corpus(result_for([text]), 'review')
        assert corpus[ALL_SCOPE].review_count == 1
        assert isinstance(top_terms(corpus[ALL_SCOPE], 10), pd.DataFrame)
    with patch('src.analytics.text_insights.extract_text', wraps=extract_text) as extraction:
        corpus = build_corpus(result_for(['battery life'] * 3), 'review')
    assert extraction.call_count == 1
    assert top_terms(corpus[ALL_SCOPE], 10).Reviews.tolist() == [3, 3]
    assert top_terms(build_corpus(result_for([]), 'review')[ALL_SCOPE], 10).empty


def test_collision_mapping_and_cache_reset():
    result = result_for(['Good service'])
    result.data['_sentiment_label'] = result.data.sentiment_label
    result.data['sentiment_label'] = 'original metadata'
    result.columns['sentiment_label'] = '_sentiment_label'
    assert build_corpus(result, 'review')[ALL_SCOPE].review_count == 1
    assert len(representative_reviews(result, 'Positive')) == 1
    state = {'review_column': 'review', 'text_context': 'old'}
    reset_review_selection(state, 'other')
    assert 'text_context' not in state
    state['text_context'] = 'old'
    clear_dataset(state)
    assert 'text_context' not in state


def test_text_insights_ui_workflow():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Text Insights').run()
    assert app.info[0].value == 'Load a dataset to explore text insights.'
    app.sidebar.button[0].click().run()
    assert app.info[0].value == 'Select the review text column to continue.'
    app.sidebar.radio[0].set_value('Overview').run()
    app.selectbox[0].select('review').run()
    app.sidebar.radio[0].set_value('Text Insights').run()
    assert 'Run sentiment analysis' in app.info[0].value
    app.sidebar.radio[0].set_value('Overview').run()
    next(b for b in app.button if b.label == 'Analyze Sentiment').click().run()
    app.sidebar.radio[0].set_value('Text Insights').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 6
    assert any('Strongest Positive Signals' in item.value for item in app.markdown)
    with patch('src.ui.text_insights.build_corpus', side_effect=AssertionError('Unnecessary tokenization')):
        app.selectbox[0].select('Negative').run()
        assert not app.exception
        assert len(app.get('plotly_chart')) == 2
        assert any('Strongest Negative Signals' in item.value for item in app.markdown)
        app.slider[0].set_value(5).run()
        assert not app.exception
        app.sidebar.radio[0].set_value('Overview').run()
        app.sidebar.radio[0].set_value('Text Insights').run()
        assert not app.exception
    assert 'analysis' in app.session_state


def test_ui_empty_scopes_and_uninformative_text():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.button[0].click().run()
    app.selectbox[0].select('review').run()
    app.session_state['analysis'] = result_for(['the and it'], ['Neutral'], [0.])
    app.sidebar.radio[0].set_value('Text Insights').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 0
    assert any('No Positive' in message.value for message in app.info)
    assert any('No Negative' in message.value for message in app.info)
    app.selectbox[0].select('Negative').run()
    assert any('No analyzed reviews' in message.value for message in app.info)
    assert not app.exception


def test_ui_preparation_failure_is_friendly():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.button[0].click().run()
    app.selectbox[0].select('review').run()
    app.session_state['analysis'] = result_for(['Good service'])
    with patch('src.ui.text_insights.build_corpus', side_effect=RuntimeError('internal details')):
        app.sidebar.radio[0].set_value('Text Insights').run()
    assert not app.exception
    assert 'could not be prepared' in app.error[0].value
    assert 'internal details' not in app.error[0].value
