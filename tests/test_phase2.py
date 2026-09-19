"""Phase 2 validation, preparation, and state regression checks."""
from io import BytesIO, StringIO
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
from src.data.loader import DataValidationError, read_csv_safely, is_dataset_empty
from src.data.preprocessor import prepare_reviews, review_columns, dataset_summary
from src.data.session import set_dataset, clear_dataset


@pytest.mark.parametrize('source', [None, 5, [], {}, b'a\nb', StringIO('a\nb')])
def test_unsupported_inputs(source):
    with pytest.raises(DataValidationError):
        read_csv_safely(source)


def test_unreadable():
    with patch('pathlib.Path.open', side_effect=PermissionError):
        with pytest.raises(DataValidationError, match='could not be read'):
            read_csv_safely('blocked.csv')


@pytest.mark.parametrize('content', [b'', b'a,b\n', b'\n', b'a,b\n,\n', b'a,a\nx,y', b'a,b\nx,y,z', b'a\n\xff'])
def test_invalid_csv(content):
    with pytest.raises(DataValidationError):
        read_csv_safely(BytesIO(content))


def test_limits_and_blank_row_preservation():
    with pytest.raises(DataValidationError, match='row limit'):
        read_csv_safely(BytesIO(b'a\nx\ny'), max_rows=1)
    with pytest.raises(DataValidationError, match='columns'):
        read_csv_safely(BytesIO(','.join(f'c{i}' for i in range(201)).encode()))
    data = read_csv_safely(BytesIO(b'a,b\nhello,1\n\nNA,2\n'))
    assert len(data) == 3
    assert data.iloc[1].isna().all()
    assert data.iloc[2, 0] == 'NA'
    assert is_dataset_empty(pd.DataFrame(columns=['a']))
    assert is_dataset_empty(pd.DataFrame(index=[0]))


def test_review_quality_and_preservation():
    data = pd.DataFrame({'text': ['  Good service  ', None, '', '  ', 12, '12', 'ok', True, 'Fine'],
                         'prepared_review': list(range(9)), 'review_status': ['original'] * 9})
    before = data.copy(deep=True)
    result = prepare_reviews(data, 'text')
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(result.data[data.columns], before)
    assert result.text_column == '_prepared_review'
    assert result.status_column == '_review_status'
    assert result.quality == {'Total Rows': 9, 'Valid Reviews': 2, 'Missing Reviews': 1,
                              'Empty Reviews': 2, 'Invalid Reviews': 4}
    assert result.data[result.text_column].iloc[0] == 'Good service'
    assert result.data[result.status_column].tolist() == ['valid', 'missing', 'empty', 'empty', 'non_text', 'non_text', 'too_short', 'non_text', 'valid']
    assert len(result.data) == len(data)
    with pytest.raises(DataValidationError):
        prepare_reviews(data, 'unknown')


def test_profile_and_order():
    data = pd.DataFrame({'rating': [1, 1, 2], 'text': ['good', 'good', None]})
    assert review_columns(data) == ['text', 'rating']
    profile = dataset_summary(data)
    assert profile['Missing values'] == 1
    assert profile['Duplicate rows'] == 1
    assert prepare_reviews(data.iloc[:0], 'text').quality['Total Rows'] == 0


def test_state_reset():
    state = {'review_column': 'old', 'prepared': 'old'}
    data = pd.DataFrame({'feedback': ['Good service']})
    set_dataset(state, data, 'new.csv', 'new')
    assert 'review_column' not in state and 'prepared' not in state
    assert state['dataset_id'] == 'new'
    clear_dataset(state)
    assert 'dataset' not in state


def test_selection_persistence_and_reset():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.button[0].click().run()
    assert app.selectbox[0].value is None
    app.selectbox[0].select('review').run()
    assert app.session_state['prepared'].quality['Valid Reviews'] == 20
    for page in ['Text Insights', 'Review Explorer', 'Sentiment Analytics', 'Overview']:
        with patch('src.data.loader.read_csv_safely', side_effect=AssertionError('Unexpected reload')):
            app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    assert app.selectbox[0].value == 'review'
    app.sidebar.button[0].click().run()
    assert app.selectbox[0].value is None
    assert 'prepared' not in app.session_state


def test_uploaded_csv_flow():
    upload = BytesIO(b'feedback,rating\nGreat service,5\n  ,2\n,3\n123,1\n')
    upload.name = 'uploaded.csv'
    with patch('streamlit.file_uploader', return_value=upload):
        app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
        next(button for button in app.button if button.label == 'Load uploaded CSV').click().run()
        assert not app.exception
        assert app.session_state['dataset_name'] == 'uploaded.csv'
        app.selectbox[0].select('feedback').run()
        assert app.session_state['prepared'].quality['Valid Reviews'] == 1
        assert len(app.session_state['prepared'].data) == 4
        next(button for button in app.button if button.label == 'Load uploaded CSV').click().run()
        assert app.selectbox[0].value == 'feedback'
        app.sidebar.button[0].click().run()
        assert app.session_state['dataset_name'] == 'Fictional sample'
        assert app.selectbox[0].value is None
        next(button for button in app.button if button.label == 'Load uploaded CSV').click().run()
        assert app.session_state['dataset_name'] == 'uploaded.csv'
        assert app.selectbox[0].value is None
        assert not app.exception


def test_preview_limit_and_failed_upload_preserves_data():
    upload = BytesIO(b'feedback\n' + b'Good service\n' * 120)
    upload.name = 'large.csv'
    with patch('streamlit.file_uploader', return_value=upload):
        app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
        next(b for b in app.button if b.label == 'Load uploaded CSV').click().run()
        assert len(app.session_state['dataset']) == 120
        assert len(app.dataframe[0].value) == 100
    bad = BytesIO(b'a,b\nwrong,width,extra')
    bad.name = 'bad.csv'
    with patch('streamlit.file_uploader', return_value=bad):
        app.run()
        next(b for b in app.button if b.label == 'Load uploaded CSV').click().run()
        assert not app.exception
        assert app.error
        assert app.session_state['dataset_name'] == 'large.csv'
        assert len(app.session_state['dataset']) == 120


def test_numeric_review_column_does_not_unlock_analytics():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.button[0].click().run()
    app.selectbox[0].select('rating').run()
    assert app.session_state['prepared'].quality['Valid Reviews'] == 0
    assert app.error
    app.sidebar.radio[0].set_value('Sentiment Analytics').run()
    assert 'Run sentiment analysis from Overview' in app.info[0].value
    assert len(app.get('plotly_chart')) == 0
    assert not app.exception
