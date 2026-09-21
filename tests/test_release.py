"""Release regressions across uploaded data, navigation, and export failures."""
from io import BytesIO
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest
from src.config import PROJECT_ROOT, NAVIGATION


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_uploaded_analysis_navigation_export_and_replacement():
    upload = BytesIO(b'feedback,alternate,rating,date,product\nExcellent service,Useful product,5,2026-01-01,Widget\nTerrible service,Broken product,1,2026-01-02,Widget\n   ,Fine product,3,2026-01-03,Widget\n,Good product,3,2026-01-04,Widget\n123,Great product,2,2026-01-05,Widget\n')
    upload.name = 'release.csv'
    with patch('streamlit.file_uploader', return_value=upload):
        app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
        button(app, 'Load uploaded CSV').click().run()
        app.selectbox[0].select('feedback').run()
        assert app.session_state['prepared'].quality['Valid Reviews'] == 2
        button(app, 'Analyze Sentiment').click().run()
        original = app.session_state['analysis'].data.copy(deep=True)
        for page in NAVIGATION:
            app.sidebar.radio[0].set_value(page).run()
            assert not app.exception, page
            pd.testing.assert_frame_equal(app.session_state['analysis'].data, original)
        app.sidebar.radio[0].set_value('Review Explorer').run()
        button(app, 'Prepare full analyzed CSV').click().run()
        exported = pd.read_csv(BytesIO(app.session_state['explorer_full_csv']))
        assert len(exported) == 5
        assert exported.sentiment_label.eq('Not analyzed').sum() == 3
        assert exported.loc[exported.sentiment_label.eq('Not analyzed'), 'sentiment_compound'].isna().all()
        app.multiselect[0].set_value(['Not analyzed']).run()
        button(app, 'Prepare filtered CSV').click().run()
        assert len(pd.read_csv(BytesIO(app.session_state['explorer_filtered_csv']))) == 3
        app.sidebar.radio[0].set_value('Overview').run()
        app.selectbox[0].select('alternate').run()
        assert 'analysis' not in app.session_state
        assert 'explorer_full_csv' not in app.session_state
        button(app, 'Analyze Sentiment').click().run()
        app.sidebar.button[0].click().run()
        assert app.session_state['dataset_name'] == 'Fictional sample'
        assert 'analysis' not in app.session_state
        assert app.selectbox[0].value is None
        assert not app.exception


def test_export_failure_has_actionable_message():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.button[0].click().run()
    app.selectbox[0].select('review').run()
    button(app, 'Analyze Sentiment').click().run()
    app.sidebar.radio[0].set_value('Review Explorer').run()
    with patch('src.ui.review_explorer.export_csv', side_effect=MemoryError('private implementation detail')):
        button(app, 'Prepare full analyzed CSV').click().run()
    assert not app.exception
    assert 'CSV download could not be prepared' in app.error[0].value
    assert 'private implementation' not in app.error[0].value
    assert 'explorer_full_csv' not in app.session_state
