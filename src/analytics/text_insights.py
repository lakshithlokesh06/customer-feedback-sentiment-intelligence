"""Offline lexical insights; no semantic inference or shared user-data cache."""
from collections import Counter
from dataclasses import dataclass, field
import re
from statistics import mean, median

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.analytics.sentiment_metrics import LABELS
from src.config import (
    TEXT_MIN_TOKEN_LENGTH, TEXT_LARGE_CORPUS_SIZE, TEXT_LARGE_MIN_REVIEWS,
    TEXT_DISPLAY_MAX, TEXT_REPRESENTATIVE_LIMIT,
)
from src.sentiment.analyzer import AnalysisResult

ALL_SCOPE = 'All analyzed reviews'
# Generic emphasis adds little topical information; keep customer/domain words.
STOPWORDS = (frozenset(ENGLISH_STOP_WORDS) | {'just', 'really'}) - {'not', 'no', 'never'}
URL = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)
PARTS = re.compile(r"[^\W\d_]+|[\d_]+|[^\w\s]", re.UNICODE)


@dataclass
class TextScope:
    """Review counts, token counts and document frequencies for a sentiment scope."""
    lengths: list[int] = field(default_factory=list)
    words: Counter = field(default_factory=Counter)
    word_reviews: Counter = field(default_factory=Counter)
    phrases: Counter = field(default_factory=Counter)
    phrase_reviews: Counter = field(default_factory=Counter)

    @property
    def review_count(self) -> int:
        return len(self.lengths)


def extract_text(text: object) -> tuple[list[str], list[str], int]:
    """Return meaningful tokens, original-adjacent bigrams, and pre-stopword length.

    Punctuation, symbols, numbers, URLs and removed words break phrase adjacency.
    Apostrophe negations are expanded before tokenization.
    """
    if not isinstance(text, str):
        return [], [], 0
    cleaned = URL.sub(' . ', text.lower().replace('’', "'"))
    cleaned = re.sub(r"\bcan['’]t\b", 'can not', cleaned)
    cleaned = re.sub(r"\bwon['’]t\b", 'will not', cleaned)
    cleaned = re.sub(r"n['’]t\b", ' not', cleaned)
    tokens, phrases = [], []
    previous = None
    length = 0
    for part in PARTS.findall(cleaned):
        if part.isalpha():
            length += 1
        useful = part.isalpha() and len(part) >= TEXT_MIN_TOKEN_LENGTH and part not in STOPWORDS
        if useful:
            tokens.append(part)
            if previous is not None:
                phrases.append(f'{previous} {part}')
            previous = part
        else:
            previous = None
    return tokens, phrases, length


def tokenize(text: object) -> list[str]:
    """Return lowercase meaningful words while preserving useful negation."""
    return extract_text(text)[0]


def build_corpus(result: AnalysisResult, text_column: str) -> dict[str, TextScope]:
    """Tokenize each unique analyzed review once and aggregate per-review coverage."""
    scopes = {scope: TextScope() for scope in (ALL_SCOPE, *LABELS)}
    cache = {}
    for text, label in zip(result.data[text_column], result.data[result.columns['sentiment_label']]):
        if label not in LABELS:
            continue
        if text not in cache:
            cache[text] = extract_text(text)
        tokens, phrases, length = cache[text]
        word_counts, phrase_counts = Counter(tokens), Counter(phrases)
        for scope in (scopes[ALL_SCOPE], scopes[label]):
            scope.lengths.append(length)
            scope.words.update(word_counts)
            scope.word_reviews.update(word_counts.keys())
            scope.phrases.update(phrase_counts)
            scope.phrase_reviews.update(phrase_counts.keys())
    return scopes


def minimum_reviews(scope: TextScope) -> int:
    """Allow single mentions for small scopes; require repetition for larger ones."""
    return TEXT_LARGE_MIN_REVIEWS if scope.review_count >= TEXT_LARGE_CORPUS_SIZE else 1


def top_terms(scope: TextScope, limit: int, *, phrases: bool = False) -> pd.DataFrame:
    """Rank by review frequency, then occurrences, then alphabetical order."""
    occurrences = scope.phrases if phrases else scope.words
    coverage = scope.phrase_reviews if phrases else scope.word_reviews
    threshold = minimum_reviews(scope)
    rows = [(term, count, coverage[term], 100 * coverage[term] / scope.review_count)
            for term, count in occurrences.items() if coverage[term] >= threshold]
    frame = pd.DataFrame(rows, columns=['Term', 'Occurrences', 'Reviews', 'Coverage (%)'])
    return frame.sort_values(['Reviews', 'Occurrences', 'Term'], ascending=[False, False, True]).head(
        max(1, min(limit, TEXT_DISPLAY_MAX))).reset_index(drop=True)


def corpus_statistics(scope: TextScope) -> dict[str, float | int]:
    """Summarize meaningful tokens and review lengths before stopword removal."""
    return {'Meaningful words': sum(scope.words.values()), 'Unique keywords': len(scope.words),
            'Average words / review': round(mean(scope.lengths), 1) if scope.lengths else 0,
            'Median words / review': median(scope.lengths) if scope.lengths else 0}


def representative_reviews(result: AnalysisResult, label: str, limit: int = TEXT_REPRESENTATIVE_LIMIT) -> pd.DataFrame:
    """Rank within the requested class, keeping source order for equal scores."""
    data = result.data.loc[result.data[result.columns['sentiment_label']].eq(label)]
    score = result.columns['sentiment_compound']
    return data.sort_values(score, ascending=label != 'Positive',
                           key=(lambda values: values.abs()) if label == 'Neutral' else None,
                           kind='stable').head(max(1, min(limit, TEXT_REPRESENTATIVE_LIMIT))).copy()
