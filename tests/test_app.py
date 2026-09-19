"""Smoke checks for the Streamlit shell and dataset lifecycle."""

import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.config import NAVIGATION, PROJECT_ROOT
from src.data.loader import DataValidationError


class AppTests(unittest.TestCase):
    def test_navigation_and_dataset_lifecycle(self):
        app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '—')
        app.sidebar.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '20')
        self.assertEqual(len(app.dataframe[0].value), 20)
        for page in NAVIGATION:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, page)
        app.sidebar.button[1].click().run()
        app.sidebar.radio[0].set_value('Overview').run()
        self.assertEqual(app.metric[0].value, '—')
        self.assertEqual(len(app.dataframe), 0)

    def test_missing_sample_is_friendly(self):
        app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
        with patch('src.data.loader.read_csv_safely', side_effect=DataValidationError('The CSV file is missing. Restore it and try again.')):
            app.sidebar.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn('missing', app.error[0].value)


if __name__ == '__main__':
    unittest.main()
