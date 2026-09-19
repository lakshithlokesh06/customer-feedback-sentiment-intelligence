"""Deterministic aggregation and Streamlit dashboard regression coverage."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.config import PROJECT_ROOT
from src.sentiment.analyzer import AnalysisResult
from src.analytics.sentiment_metrics import (
    analyzed_rows, filter_sentiment, overall, parse_dates, metadata_columns,
    by_category, by_time, by_rating,
)
from src.visualization import charts


def fixture_data():
    return pd.DataFrame({'sentiment_label': ['Positive', 'Negative', 'Neutral', 'Not analyzed'],
        'sentiment_compound': [0.8, -0.6, 0., None],
        'sentiment_positive': [.8, .1, 0., None], 'sentiment_neutral': [.2, .2, 1., None],
        'sentiment_negative': [0., .7, 0., None], 'product': ['A', 'A', None, 'B'],
        'date': ['2026-01-01', '2026-01-02', 'bad', None], 'rating': [1, 5, 3, 4]})


def scored():
    data = fixture_data()
    return analyzed_rows(AnalysisResult(data, {name: name for name in data if name.startswith('sentiment_')}))


def test_counts_percentages_exclusion_and_mean():
    data = scored()
    assert len(data) == 3
    summary = overall(data)
    assert summary.Count.tolist() == [1, 1, 1]
    assert summary.Percent.sum() == pytest.approx(100)
    assert data.sentiment_compound.mean() == pytest.approx(.2 / 3)
    assert len(fixture_data()) == 4


@pytest.mark.parametrize('label', ['Positive', 'Neutral', 'Negative'])
def test_single_class_and_empty(label):
    data = filter_sentiment(scored(), [label])
    assert overall(data).Count.sum() == 1
    assert overall(data).Percent.max() == 100
    assert overall(filter_sentiment(data, [])).Count.sum() == 0
    for fig in (charts.distribution(overall(data)), charts.compound(data), charts.components(data)):
        assert fig.data


def test_categories_missing_and_limit():
    result = by_category(scored(), 'product')
    assert result.Group.tolist() == ['A', '(Missing)']
    assert result['Review count'].tolist() == [2, 1]
    assert result.iloc[0].Positive == 50
    assert result.iloc[0].Negative == 50
    assert len(by_category(scored(), 'product', limit=1)) == 1


def test_dates_and_time():
    data = scored()
    before = data.copy(deep=True)
    assert parse_dates(data.date).notna().sum() == 2
    assert parse_dates(pd.Series([1, 2, 3])).isna().all()
    assert len(by_time(data, 'date', 'D')) == 2
    weekly = by_time(data, 'date', 'W')
    assert len(weekly) == 1
    assert weekly['Review count'].iloc[0] == 2
    assert weekly['Average compound'].iloc[0] == pytest.approx(.1)
    pd.testing.assert_frame_equal(data, before)
    data['date'] = 'invalid'
    assert by_time(data, 'date', 'D').empty


def test_rating_mismatches_and_unsupported():
    data = scored()
    summary, mismatch = by_rating(data, 'rating')
    assert mismatch == 2
    assert summary['Review count'].sum() == 3
    data['rating'] = [3, 3, 3]
    assert by_rating(data, 'rating')[1] == 0
    for values in ([1, 10, 3], [0, 1, 2], [1.5, 5, 3]):
        data['rating'] = values
        assert by_rating(data, 'rating')[1] is None


def test_metadata_and_filter():
    data = scored()
    dates, categories, ratings = metadata_columns(data, ['sentiment_label'])
    assert dates == ['date']
    assert 'product' in categories
    assert ratings == ['rating']
    original = data.copy(deep=True)
    assert filter_sentiment(data, ['Positive', 'Negative']).shape[0] == 2
    pd.testing.assert_frame_equal(data, original)


def test_empty_aggregations():
    data = scored().iloc[:0]
    assert overall(data).Percent.sum() == 0
    assert by_category(data, 'product').empty
    assert by_time(data, 'date', 'D').empty
    assert by_rating(data, 'rating')[1] is None


def test_analytics_workflow():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Sentiment Analytics').run()
    assert app.info[0].value == 'Load a dataset to begin sentiment analysis.'
    app.sidebar.button[0].click().run()
    assert 'Select the review text column' in app.info[0].value
    app.sidebar.radio[0].set_value('Overview').run()
    app.selectbox[0].select('review').run()
    app.sidebar.radio[0].set_value('Sentiment Analytics').run()
    assert 'Run sentiment analysis' in app.info[0].value
    app.sidebar.radio[0].set_value('Overview').run()
    next(b for b in app.button if b.label == 'Analyze Sentiment').click().run()
    app.sidebar.radio[0].set_value('Sentiment Analytics').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 6
    assert [s.value for s in app.selectbox] == ['date', 'Day', 'product', 'rating']
    app.multiselect[0].set_value(['Positive']).run()
    assert not app.exception
    assert int(app.metric[1].value) < 20
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 0
    next(b for b in app.button if b.label == 'Reset filters').click().run()
    assert app.metric[1].value == '20'
    assert not app.warning
    app.sidebar.radio[0].set_value('Overview').run()
    assert 'analysis' in app.session_state


def test_collision_mapping_and_numeric_metadata_safety():
    original = fixture_data()
    original['_sentiment_label'] = original['sentiment_label']
    original['sentiment_label'] = 'uploaded metadata'
    mapping = {name: name for name in original if name.startswith('sentiment_')}
    mapping['sentiment_label'] = '_sentiment_label'
    result = analyzed_rows(AnalysisResult(original, mapping))
    assert len(result) == 3
    assert overall(result).Count.tolist() == [1, 1, 1]
    assert original.sentiment_label.eq('uploaded metadata').all()


def test_month_timezone_missing_rating_and_average():
    from src.analytics.sentiment_metrics import average_compound
    data = scored()
    data['date'] = ['2026-01-01T01:00:00+02:00', '2026-01-12', '2026-02-01']
    assert len(by_time(data, 'date', 'M')) == 3  # first timestamp is December in UTC
    data['rating'] = ['1', 'missing', None]
    summary, mismatch = by_rating(data, 'rating')
    assert summary['Review count'].sum() == 1
    assert mismatch == 1
    assert average_compound(data.iloc[:0]) is None


def test_metadata_excludes_identifiers_and_text():
    data = pd.concat([scored()] * 4, ignore_index=True)
    data['review_id'] = [f'r{i}' for i in range(len(data))]
    data['feedback'] = [f'Long review {i}' for i in range(len(data))]
    _, categories, _ = metadata_columns(data, ['feedback'])
    assert 'review_id' not in categories and 'feedback' not in categories


def test_rating_scale_checks_source_values():
    from src.analytics.sentiment_metrics import five_star_scale
    assert five_star_scale(pd.Series([1, 5, None]))
    assert not five_star_scale(pd.Series([1, 5, 10]))
    assert not five_star_scale(pd.Series([None]))
