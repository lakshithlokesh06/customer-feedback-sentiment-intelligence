"""Review filtering, stable pagination, UTF-8 export, and explorer UI checks."""
from io import BytesIO
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.analytics.review_explorer import ReviewFilters, filter_reviews, sort_reviews, paginate
from src.config import PROJECT_ROOT
from src.data.exporter import export_csv
from src.data.preprocessor import prepare_reviews
from src.data.session import clear_dataset, reset_review_selection
from src.sentiment.analyzer import analyze_dataframe


def sample_result():
    original = pd.DataFrame({'review': ['LOVE delivery [a+b]', 'Terrible delivery!', 'The package contains a cable.', None, 'Très good service'],
                             'product': ['A', 'B', None, None, 'A'], 'rating': [10, 2, None, 7, 8],
                             'date': ['2026-01-03', '2026-01-01', 'bad', None, '2026-01-02']})
    prepared = prepare_reviews(original, 'review')
    result = analyze_dataframe(prepared)
    # Deterministic expected labels/scores independent of lexicon changes.
    result.data[result.columns['sentiment_label']] = ['Positive', 'Negative', 'Neutral', 'Not analyzed', 'Positive']
    result.data[result.columns['sentiment_compound']] = [.9, -.8, .02, pd.NA, .5]
    return result


@pytest.mark.parametrize('query,expected', [('DELIV', [0, 1]), (' [a+b] ', [0]), ('', [0,1,2,3,4]), ('  ', [0,1,2,3,4]), ('.*', []), ('Très', [4])])
def test_literal_search(query, expected):
    assert filter_reviews(sample_result(), 'review', ReviewFilters(search=query)).index.tolist() == expected


@pytest.mark.parametrize('labels,expected', [(('Positive',), [0,4]), (('Negative',), [1]), (('Neutral',), [2]), (('Not analyzed',), [3]), (('Positive','Negative'), [0,1,4]), ((), [])])
def test_sentiment(labels, expected):
    assert filter_reviews(sample_result(), 'review', ReviewFilters(labels=labels)).index.tolist() == expected


def test_compound_null_bypass_and_metadata():
    result = sample_result()
    assert filter_reviews(result, 'review', ReviewFilters(compound=(.7,1))).index.tolist() == [0,3]
    assert filter_reviews(result, 'review', ReviewFilters(labels=('Positive',), compound=(.7,1))).index.tolist() == [0]
    assert filter_reviews(result, 'review', ReviewFilters(categories={'product': ([],True)})).index.tolist() == [2,3]
    assert filter_reviews(result, 'review', ReviewFilters(categories={'product': (['A'],False)})).index.tolist() == [0,4]
    assert filter_reviews(result, 'review', ReviewFilters(rating_column='rating', rating_range=(7,10))).index.tolist() == [0,3,4]
    assert filter_reviews(result, 'review', ReviewFilters(date_column='date', date_range=('2026-01-02','2026-01-03'))).index.tolist() == [0,4]


@pytest.mark.parametrize('sort,expected', [('Original order',[0,1,2,3,4]), ('Most Positive',[0,4,2,1,3]), ('Most Negative',[1,2,4,0,3]), ('Most Neutral',[2,4,1,0,3]), ('Newest',[0,4,1,2,3]), ('Oldest',[1,4,0,2,3]), ('Rating High → Low',[0,4,3,1,2]), ('Rating Low → High',[1,3,4,0,2])])
def test_sort(sort, expected):
    result = sample_result()
    before = result.data.copy(deep=True)
    assert sort_reviews(result.data, result, sort, 'date','rating').index.tolist() == expected
    pd.testing.assert_frame_equal(result.data, before)


def test_stability_duplicate_index_and_pagination():
    result = sample_result()
    data = result.data.copy()
    data.index = [0,0,2,3,4]
    data[result.columns['sentiment_compound']] = [.5,.5,0,pd.NA,.5]
    assert sort_reviews(data,result,'Most Positive').review.iloc[:3].tolist() == [data.review.iloc[0],data.review.iloc[1],data.review.iloc[4]]
    page, current, pages = paginate(data, 100, 2)
    assert (len(page), current, pages) == (1,3,3)
    assert paginate(data, -1, 2)[1] == 1
    assert paginate(data.iloc[:0], 5, 10)[1:] == (1,1)
    with pytest.raises(ValueError):
        paginate(data,1,0)


def test_csv_full_filtered_unicode_nulls_collisions():
    result = sample_result()
    result.data['original_extra'] = ['hello, "world"\nnext', 'b', 'c','d','e']
    before = result.data.copy(deep=True)
    encoded = export_csv(result.data)
    parsed = pd.read_csv(BytesIO(encoded))
    assert len(parsed) == 5
    assert list(parsed.columns) == list(result.data.columns)
    assert parsed.review.iloc[-1] == 'Très good service'
    assert parsed['original_extra'].iloc[0] == 'hello, "world"\nnext'
    assert parsed['sentiment_label'].iloc[3] == 'Not analyzed'
    assert pd.isna(parsed['sentiment_compound'].iloc[3])
    selected = filter_reviews(result,'review',ReviewFilters(labels=('Positive',)))
    assert len(pd.read_csv(BytesIO(export_csv(selected)))) == 2
    pd.testing.assert_frame_equal(result.data,before)
    collision = analyze_dataframe(prepare_reviews(pd.DataFrame({'review':['Great service'], 'sentiment_label':['original']}),'review'))
    exported = pd.read_csv(BytesIO(export_csv(collision.data)))
    assert exported.sentiment_label.iloc[0] == 'original'
    assert collision.columns['sentiment_label'] in exported


def ready_app():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.button[0].click().run()
    app.selectbox[0].select('review').run()
    next(b for b in app.button if b.label == 'Analyze Sentiment').click().run()
    app.sidebar.radio[0].set_value('Review Explorer').run()
    return app


def test_explorer_states_and_workflow():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Review Explorer').run()
    assert app.info[0].value == 'Load a dataset and select a review column to begin analysis.'
    app.sidebar.button[0].click().run()
    assert app.info[0].value == 'Load a dataset and select a review column to begin analysis.'
    app.sidebar.radio[0].set_value('Overview').run()
    app.selectbox[0].select('review').run()
    app.sidebar.radio[0].set_value('Review Explorer').run()
    assert 'Run sentiment analysis' in app.info[0].value
    app = ready_app()
    assert not app.exception
    assert app.metric[0].value == '20'
    assert len([item for item in app.caption if item.value.startswith('Source row index:')]) == 10
    app.number_input[0].set_value(2).run()
    assert not app.exception
    app.text_input[0].set_value('coffee').run()
    assert app.number_input[0].value == 1
    assert int(app.metric[0].value) < 20
    for label in ('Positive','Neutral','Negative'):
        app.multiselect[0].set_value([label]).run()
        assert not app.exception
    next(b for b in app.button if b.label=='Reset filters').click().run()
    assert app.metric[0].value == '20'
    assert app.text_input[0].value == ''
    assert not app.exception and not app.warning
    next(b for b in app.button if b.label=='Prepare full analyzed CSV').click().run()
    assert len(pd.read_csv(BytesIO(app.session_state['explorer_full_csv']))) == 20
    app.text_input[0].set_value('coffee').run()
    next(b for b in app.button if b.label=='Prepare filtered CSV').click().run()
    assert len(pd.read_csv(BytesIO(app.session_state['explorer_filtered_csv']))) == int(app.metric[0].value)
    app.text_input[0].set_value('no-match-xyz').run()
    assert 'explorer_filtered_csv' not in app.session_state
    assert any('No reviews match' in item.value for item in app.info)
    assert next(b for b in app.button if b.label=='Prepare filtered CSV').disabled
    app.sidebar.radio[0].set_value('Overview').run()
    assert 'analysis' in app.session_state


def test_not_analyzed_ui_and_state_reset():
    app = ready_app()
    result = sample_result()
    prepared = prepare_reviews(result.data[['review','product','rating','date']], 'review')
    app.session_state['analysis'] = result
    app.session_state['prepared'] = prepared
    app.run()
    app.multiselect[0].set_value(['Not analyzed']).run()
    assert not app.exception
    assert app.metric[0].value == '1'
    assert any('No sentiment scores' in item.value for item in app.caption)
    state = {'explorer_query':'old','analysis':result,'review_column':'review'}
    reset_review_selection(state,'product')
    assert 'explorer_query' not in state
    state['explorer_query']='old'
    clear_dataset(state)
    assert 'explorer_query' not in state


def test_metadata_controls_sorting_and_reset():
    app = ready_app()
    next(m for m in app.multiselect if m.label=='Metadata columns').set_value(['product']).run()
    next(m for m in app.multiselect if m.label=='product values').set_value(['Coffee Maker']).run()
    assert app.metric[0].value == '5'
    next(c for c in app.checkbox if c.label=='Filter rating').check().run()
    next(s for s in app.slider if s.label=='Rating range').set_value((4.,5.)).run()
    assert app.metric[0].value == '2'
    next(c for c in app.checkbox if c.label=='Filter dates').check().run()
    next(s for s in app.selectbox if s.label=='Sort results').select('Newest').run()
    assert not app.exception
    assert app.date_input
    next(b for b in app.button if b.label=='Reset filters').click().run()
    assert app.metric[0].value == '20'
    assert not any(c.value for c in app.checkbox)
    assert next(s for s in app.selectbox if s.label=='Sort results').value == 'Original order'
    assert not app.warning


def test_filter_preserves_source_and_filtered_export_sort():
    result = sample_result()
    before = result.data.copy(deep=True)
    filtered = filter_reviews(result, 'review', ReviewFilters(search='delivery'))
    sorted_rows = sort_reviews(filtered,result,'Most Negative')
    exported = pd.read_csv(BytesIO(export_csv(sorted_rows)))
    assert exported.review.tolist() == ['Terrible delivery!', 'LOVE delivery [a+b]']
    pd.testing.assert_frame_equal(result.data,before)
