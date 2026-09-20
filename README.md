# Customer Feedback Sentiment Intelligence

Transform customer feedback into actionable sentiment insights.

A Python and Streamlit portfolio project for exploring customer review data. **Phase 5 adds offline keyword, phrase, and representative-review insights to the existing upload, sentiment, and analytics workflow.** Sentiment comes from review text only; no machine-learning training or LLM is used.

## Available in Phase 5

- Wide dashboard with sidebar navigation: Overview, Sentiment Analytics, Text Insights, Review Explorer, and About.
- Helpful empty states, actual sentiment KPIs after analysis, and real Plotly analytics in the Sentiment Analytics page.
- Load and clear a bundled fictional sample dataset; preview original records and their count.
- Reusable CSV validation for file extensions, size, encoding, headers, row consistency, and empty datasets, with readable errors.
- CSV upload and switching between uploaded and sample datasets without restarting.
- Dataset profiling: row/column counts, missing cells, duplicate rows, and estimated dataframe memory.
- Explicit review-column selection with text-like columns listed first; no silent selection.
- Review quality summaries and prepared review text with per-row status, preserving all original data.
- Explicit VADER analysis with Positive, Neutral, and Negative labels, component scores, compound scores, and a bounded colored preview.
- Overall and sentiment-specific keywords, frequent adjacent phrases, corpus statistics, and strongest/most neutral review signals.
- Central configuration, persistent session results, and automated loader, preparation, sentiment, and UI tests.

## Planned features

Individual review filtering and analyzed-data exports remain planned. No paid API, LLM, authentication, or database is used.

## Tech stack

Python 3.12+; Streamlit; Pandas; NumPy; Plotly; NLTK; VADER Sentiment (`vaderSentiment`); scikit-learn; python-dotenv.

VADER uses the already-installed `vaderSentiment` package. NLTK and other analytics libraries remain available for later work; no NLTK download is needed. Requirements specify compatible version ranges rather than a fully locked environment.

## Project structure

```text
customer-feedback-sentiment-intelligence/
├── app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── session.py
│   ├── sentiment/
│   │   ├── __init__.py
│   │   └── analyzer.py
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── sentiment_metrics.py
│   │   └── text_insights.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── charts.py
│   │   └── text_charts.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── analytics.py
│   │   └── text_insights.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── data/
│   └── sample_customer_feedback.csv
├── assets/
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── test_app.py
│   ├── test_loader.py
│   ├── test_phase2.py
│   ├── test_sentiment.py
│   ├── test_analytics.py
│   └── test_text_insights.py
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

The data layer has no Streamlit dependency. The sentiment service owns offline scoring and output preservation; analytics aggregation, Plotly visualization, and Streamlit page rendering are separated into their own modules. `app.py` owns the interface and session state; `src/config.py` owns shared application constants. Native Streamlit theming lives in `.streamlit/config.toml`.

## Local installation

Use Python 3.12 or newer. Python 3.12 is the verified baseline. From the repository directory on macOS/Linux:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `python3` is older than 3.12, use `python3.12 -m venv .venv` instead. Open the local URL printed by Streamlit (normally `http://localhost:8501`). Stop it with Ctrl+C.

No environment file is required. Optionally copy `.env.example` to `.env` and set `FEEDBACK_MAX_UPLOAD_MB` (default 10; accepted range 1–100). This controls both the CSV uploader and reusable loader limit. The uploader overrides the fallback server limit in `.streamlit/config.toml`. Invalid environment values fall back to 10 MB.

## Sample data

`data/sample_customer_feedback.csv` contains 20 fictional reviews across four products, dated August 1–20, 2026. Columns are `review_id` (unique identifier), `review` (original text), `rating` (1–5), `date` (ISO date), and `product`. The text includes positive, negative, neutral, and mixed experiences. It has no sentiment labels or scores; ratings are not converted into sentiment.

Use **Load sample dataset** in the sidebar to preview it. **Clear dataset** resets the session preview. A missing or invalid sample produces a friendly error. The generic loader accepts UTF-8 comma-delimited CSV files and binary file-like objects; it does not enforce the sample schema or select review columns. The app checks the sample's expected columns separately.

## Upload and prepare a dataset

1. Open **Overview** and choose **Upload CSV** or **Use sample dataset**.
2. Select a CSV and press **Load uploaded CSV**, or press **Use sample dataset**. The sidebar sample shortcut remains available.
3. Inspect profiling cards and the first 100 records.
4. Explicitly choose **Select review text column**. Text-like columns appear first; none is chosen automatically.
5. Review quality counts and expand the prepared preview to inspect per-row status.
6. Press **Analyze Sentiment** when at least one valid review exists. A spinner displays while real processing runs.
7. Inspect label counts, percentages (of analyzed rows only), and the first 100 sentiment results on Overview or Review Explorer.

Choosing a source or file alone does not replace the active dataset: its load button does. The current source is displayed above its profile. Failed uploads preserve the previous valid dataset. Navigation preserves data and the selected column; loading a different dataset or clearing resets preparation. Re-loading the identical uploaded bytes keeps the current selection. Data stays in the current Streamlit session and is not saved to disk.

### Validation and limits

Only UTF-8 (including BOM), comma-delimited `.csv` files are supported. Validation checks input types, readable binary streams, file size, header names, duplicate headers (including surrounding-whitespace differences), consistent record widths, encoding, and usable data. Zero-row, zero-column, and wholly blank datasets are rejected with friendly messages. Pandas parses column types; mixed numeric/text columns may contain numeric strings, which review validation also detects.

Limits are defined in `src/config.py`: **10 MB by default**, **100,000 data rows**, **200 columns**, and **100 preview rows**. Over-limit files are rejected with guidance; previews are capped, not the underlying accepted dataset. Fully blank records within a usable file are retained. CSV parser field-length restrictions also apply to unusually long individual cells. Missing-cell counts include whitespace-only cells; duplicate rows are reported without removal. Memory is an estimate for the original dataframe, not total process memory.

### Review quality rules

Prepared data is a separate copy containing every original row and column. Two additional columns hold trimmed review text and status (`prepared_review` and `review_status`; underscores are prefixed if those names already exist). Original review values are never overwritten.

- `valid`: text containing at least three characters after trimming, excluding numeric-only strings.
- `missing`: null value or an empty CSV cell.
- `empty`: a blank or whitespace-only string.
- `non_text`: numeric, boolean, other non-string values, or numeric-only strings.
- `too_short`: nonblank text shorter than three characters.

**Invalid Reviews** combines `non_text` and `too_short`; Missing and Empty are separate, mutually exclusive counts. Unusable records remain in the prepared dataset with a null prepared text and their reason. Common literal strings such as `NA` are preserved rather than treated as null automatically. No records are silently removed. A selected column with zero valid reviews displays an error and does not unlock the analytics placeholders. These checks are basic data-quality rules, not semantic or language analysis.

## Sentiment engine and output

`src/sentiment/analyzer.py` uses `vaderSentiment.SentimentIntensityAnalyzer` instead of the NLTK wrapper because the existing package bundles both its lexicon and emoji mappings. This provides offline initialization and emoji handling without downloading NLTK resources or committing downloaded artifacts. The analyzer is initialized lazily once per process and reused. If its resources cannot be read, the UI asks the user to reinstall project requirements; failed initialization can be retried and is not cached.

Classification uses central constants in `src/config.py`:

- Compound **≥ 0.05**: **Positive**.
- Compound **≤ −0.05**: **Negative**.
- Otherwise: **Neutral**.

The new analyzed dataframe retains every original column, prepared column, row, index, and row order. It adds `sentiment_negative`, `sentiment_neutral`, `sentiment_positive`, `sentiment_compound`, and `sentiment_label`. Component scores describe negative/neutral/positive proportions; compound ranges from −1 to 1. Existing columns with those names are never overwritten: generated output names receive leading underscores, and the UI follows that mapping.

Only rows marked `valid` are scored. Missing, blank, numeric/non-text, and too-short rows stay in place, with null numeric sentiment scores and **Not analyzed** labels. A valid review with no recognized sentiment can receive Neutral; that differs from an unusable review. Rating values are never used for classification.

Scoring runs only on an explicit button click. Results live in the current session and survive navigation. Changing the selected review column, loading a new dataset, or clearing data invalidates results. Identical uploaded bytes keep the existing selection and analysis. Duplicate reviews are scored once within each analysis call, while every duplicate row is retained. Only the analyzer resource is shared across sessions; uploaded data, duplicate-text caches, and results are not globally cached. Failed scoring publishes no partial result; if a prior completed result exists, it remains available.

### Sentiment limitations

VADER is a rule/lexicon-based model primarily designed for English. Non-English text is accepted without a reliability claim. Sarcasm, context, domain-specific language, and mixed sentiment can be misclassified. Scores describe text sentiment, not factual correctness, human intent, or a calibrated probability. Case, punctuation, URLs, and emoji are passed through after Phase 2 whitespace trimming. Numeric-only text remains unusable under Phase 2 rules. Very short feedback (including a single emoji) can be excluded by the existing minimum length. Long reviews are supported within existing CSV limits, but large amounts of unique text take longer to process. There is no topic modeling, aspect analysis, export, model training, or LLM functionality in this phase.

## Sentiment Analytics dashboard

After loading a dataset, selecting its review column, and clicking **Analyze Sentiment**, open **Sentiment Analytics**. The page never triggers scoring automatically. It provides:

- Six KPIs: total source reviews, filtered analyzed reviews, three sentiment counts, and mean compound score.
- Sentiment counts and percentages, a compound histogram with the unchanged ±0.05 boundaries, and mean VADER component scores.
- Optional date trends, category comparisons, and rating comparisons when compatible metadata exists.
- A sentiment-label multiselect and **Reset filters**. Every chart follows the selected labels; original data is unchanged. Empty selections show a helpful message.

`Not analyzed` rows are excluded from counts, percentages, and averages; their exclusion count remains visible. Total Reviews is always the unfiltered source row count. Analytics frames and metadata suggestions are reused within the current session until analysis changes; no uploaded data is cached globally.

### Optional metadata handling

**Dates:** Non-numeric columns with at least 50% parseable date values are offered in a selector, without requiring a column named `date`. Dates normalize to UTC. Invalid/missing dates are omitted only from the trend chart, with a visible count. Day grouping is always available; Week appears for spans of at least seven days, Month for at least 28 days. Weekly buckets begin Monday. Ambiguous date strings follow Pandas parsing conventions; ISO dates are recommended. Numeric timestamps are not auto-interpreted.

**Categories:** Low-cardinality non-numeric columns (at most 50 distinct values) are suggested, excluding selected review text, preparation/output fields, date columns, and obvious ID/text names. Unique-valued columns are omitted unless they have five or fewer categories. The chart and compact table show count, sentiment shares, and mean compound. Up to 15 groups appear, selected by highest analyzed volume and alphabetical ties; missing values have a separate group. These heuristics can omit unusual but meaningful metadata.

**Ratings:** Numeric or numeric-like columns whose names contain `rating`, `stars`, or `score`, with at most 20 distinct values, are offered. A stacked chart compares text sentiment across rating values. Missing/non-numeric ratings are omitted; ratings never change sentiment labels. Mismatch analytics assumes a 1–5 scale only when all numeric ratings across the entire source dataset are integral and within 1–5. A mismatch is rating 4–5 with Negative text, or rating 1–2 with Positive text. Rating 3 and Neutral text do not trigger a mismatch. Other scales skip the mismatch metric with an explanation. A subset of an unknown scale can resemble 1–5, so the UI explicitly labels this assumption. Mismatches are descriptive, not claims of fraud or error.

Charts use consistent semantic colors and responsive full-width sizing. KPIs use two rows of three cards rather than a six-card-wide layout. Semantic topic extraction remains unimplemented; lightweight lexical insights are available in Text Insights.

## Text Insights

Run sentiment analysis in Overview, then open **Text Insights**. This page does not trigger VADER or change stored text, scores, or labels. Choose **All analyzed reviews**, **Positive**, **Neutral**, or **Negative**. Display options control the number of keywords (default 15) and phrases (default 10), each within 5–20.

### Preprocessing and metrics

The pipeline lowercases temporary text, removes obvious HTTP(S)/www URLs, tokenizes Unicode alphabetic words, and excludes numbers, punctuation, isolated symbols, and tokens shorter than two characters. Common apostrophe contractions are expanded to preserve negation. It uses scikit-learn's bundled English stopwords with **not**, **no**, and **never** retained. No network resource download, new dependency, stemming, or lemmatization is required. The small custom supplement removes **just** and **really** as generic filler/emphasis; customer and product vocabulary is not removed.

Keyword charts rank terms by **review frequency**, then raw occurrence count, then alphabetical order. Repeating a word ten times in one review contributes ten occurrences but only one review mention. Hover details include both counts and the percentage of reviews in the current sentiment scope containing the term. Duplicate dataset rows remain separate review documents, consistently with previous phases.

Minimum document frequency is one for scopes with fewer than 30 reviews, and two for scopes of 30 or more. These thresholds and display limits live in `src/config.py`; each sentiment subset uses its own scope size. Empty subsets, stopword-only text, or phrases below the threshold display helpful empty states rather than fabricated results.

Bigrams are adjacent meaningful words from the source sequence, such as **battery life**. Removed stopwords, punctuation, URLs, numbers, and symbols break adjacency: the pipeline does not join distant words or cross sentence boundaries. Inflected forms remain separate words. This is lexical phrase counting, not semantic topic modeling.

### Praise, concerns, and examples

The All scope shows **What customers appreciate** from Positive reviews and **Common customer concerns** from Negative reviews, with keywords and phrases for each. Choosing one sentiment focuses charts and representative examples on that class. Frequent words in these subsets are not confirmed praise, defects, business failures, or root causes; VADER labels and simple word counts can miss context and negation scope.

Representative examples are limited to three original reviews. **Strongest Positive Signals** ranks Positive reviews by descending compound, **Strongest Negative Signals** ranks Negative reviews by ascending compound, and **Most Neutral Signals** ranks Neutral reviews by absolute compound closest to zero. Equal scores retain source order. Original text, label, compound, and available product/rating/date metadata are displayed. These examples are selected extremes or neutral signals, not a statistically representative sample.

The compact corpus summary shows meaningful-word occurrences, unique keywords, and average/median words per review. Review length counts alphabetic words before stopword filtering, after URL removal and contraction expansion. Corpus statistics describe the selected scope and do not imply quality. Only analyzed rows contribute; unusable records remain in the source.

Tokenization is reused for duplicate text during corpus construction. Aggregated results are held only in the current session and reused when changing controls or navigation; changing the dataset, selected review column, or analysis invalidates them. No uploaded text is cached globally.

### Text limitations

Text insights are primarily designed for English. Non-English and emoji-heavy text can be processed without a language-support claim; emoji and symbols do not become keywords. Very short or entirely uninformative reviews may yield no terms. Stopword filtering can remove words useful in some domains. There is no semantic topic modeling, aspect-based sentiment, transformer NLP, LLM summarization, automated root-cause detection, word cloud, or export workflow. Phase 6 has not been implemented.

## Validation

Install the development dependency (pytest) and run:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall .
python -m pytest
python -m unittest discover -s tests -v
python -m streamlit run app.py --server.headless true
```

`compileall .` also traverses the ignored virtual environment; for a faster project-only check use `python -m compileall -q app.py src tests`. Tests retain the Phase 1 checks and cover input errors, row and column limits, quality categories, original-data preservation, source switching, explicit selection, and navigation persistence. Sentiment tests cover actual positive/neutral/negative scoring, exact threshold boundaries, unusable rows, duplicate text, emphasis, emoji, multilingual/long text, column collisions, resource failures, and session invalidation. Text-insight tests cover cleaning, negation, URLs, counts/coverage, phrase boundaries, dynamic thresholds, sentiment scopes, representative ordering, Unicode, preservation, and session reuse. User-facing tracebacks are disabled; unexpected loading errors are logged to the server console.
