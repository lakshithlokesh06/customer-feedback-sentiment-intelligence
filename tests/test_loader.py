"""Validation boundaries for local files and upload-compatible binary streams."""

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.config import SAMPLE_COLUMNS, SAMPLE_DATA_PATH
from src.data.loader import DataValidationError, is_dataset_empty, read_csv_safely, validate_csv_file


class LoaderTests(unittest.TestCase):
    def test_sample(self):
        data = read_csv_safely(SAMPLE_DATA_PATH)
        self.assertEqual(len(data), 20)
        self.assertEqual(tuple(data.columns), SAMPLE_COLUMNS)
        self.assertTrue(data.review_id.is_unique)
        self.assertTrue(data.rating.between(1, 5).all())
        pd.to_datetime(data.date, errors="raise")

    def test_stream_position_preserved(self):
        stream = BytesIO(b'review,rating\n"Great, thanks",5\n')
        stream.seek(4)
        self.assertEqual(read_csv_safely(stream).iloc[0]['review'], 'Great, thanks')
        self.assertEqual(stream.tell(), 4)

    def test_invalid_content(self):
        cases = [b'', b' \n', b'review,rating\n', b'review,rating\n,\n',
                 b'review,rating\n  ,  \n', b'review,rating\na,1,extra\n',
                 b'review,rating\na\n', b'review,review\na,b\n',
                 b',rating\na,5\n', b'review\n\xff\n', b'review\n\x00\n',
                 b'review\n"unterminated\n']
        for content in cases:
            with self.subTest(content=content), self.assertRaises(DataValidationError):
                read_csv_safely(BytesIO(content))

    def test_missing_file(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DataValidationError, 'missing'):
                read_csv_safely(Path(directory) / 'missing.csv')

    def test_extension_and_limit(self):
        stream = BytesIO(b'review\nhello\n')
        stream.name = 'reviews.txt'
        with self.assertRaisesRegex(DataValidationError, 'extension'):
            validate_csv_file(stream)
        stream.name = 'reviews.CSV'
        with self.assertRaisesRegex(DataValidationError, 'size limit'):
            validate_csv_file(stream, max_bytes=4)

    def test_bom_and_multiline(self):
        data = read_csv_safely(BytesIO(b'\xef\xbb\xbfreview,rating\n"Good\nproduct",4\n'))
        self.assertEqual(data.iloc[0]['review'], 'Good\nproduct')

    def test_empty_dataframe(self):
        self.assertTrue(is_dataset_empty(pd.DataFrame()))
        self.assertTrue(is_dataset_empty(pd.DataFrame({'review': [' ', None]})))
        self.assertFalse(is_dataset_empty(pd.DataFrame({'rating': [0]})))


if __name__ == '__main__':
    unittest.main()
