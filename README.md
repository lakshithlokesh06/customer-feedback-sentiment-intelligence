# Customer Feedback Sentiment Intelligence

Transform customer feedback into actionable sentiment insights.

A Python and Streamlit portfolio project for exploring customer review data. **Phase 3 adds offline VADER sentiment classification to the existing CSV upload and preparation workflow.** Sentiment comes from review text only; no machine-learning training or LLM is used.

## Available in Phase 3

- Wide dashboard with sidebar navigation: Overview, Sentiment Analytics, Text Insights, Review Explorer, and About.
- Helpful empty states, actual sentiment KPIs after analysis, and clearly labeled planned chart areas.
- Load and clear a bundled fictional sample dataset; preview original records and their count.
- Reusable CSV validation for file extensions, size, encoding, headers, row consistency, and empty datasets, with readable errors.
- CSV upload and switching between uploaded and sample datasets without restarting.
- Dataset profiling: row/column counts, missing cells, duplicate rows, and estimated dataframe memory.
- Explicit review-column selection with text-like columns listed first; no silent selection.
- Review quality summaries and prepared review text with per-row status, preserving all original data.
- Explicit VADER analysis with Positive, Neutral, and Negative labels, component scores, compound scores, and a bounded colored preview.
- Central configuration, persistent session results, and automated loader, preparation, sentiment, and UI tests.

## Planned features

Sentiment trend/distribution charts, text insights, review filtering, and analyzed-data exports remain planned. No paid API, LLM, authentication, or database is used.

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
│   │   └── __init__.py
│   ├── visualization/
│   │   └── __init__.py
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
│   └── test_sentiment.py
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

The data layer has no Streamlit dependency. The sentiment service owns offline scoring and output preservation; analytics and visualization packages reserve simple boundaries for later phases. `app.py` owns the interface and session state; `src/config.py` owns shared application constants. Native Streamlit theming lives in `.streamlit/config.toml`.

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

VADER is a rule/lexicon-based model primarily designed for English. Non-English text is accepted without a reliability claim. Sarcasm, context, domain-specific language, and mixed sentiment can be misclassified. Scores describe text sentiment, not factual correctness, human intent, or a calibrated probability. Case, punctuation, URLs, and emoji are passed through after Phase 2 whitespace trimming. Numeric-only text remains unusable under Phase 2 rules. Very short feedback (including a single emoji) can be excluded by the existing minimum length. Long reviews are supported within existing CSV limits, but large amounts of unique text take longer to process. There are no sentiment charts, topic modeling, aspect analysis, exports, model training, or LLM features in this phase.

## Validation

Install the development dependency (pytest) and run:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall .
python -m pytest
python -m unittest discover -s tests -v
python -m streamlit run app.py --server.headless true
```

`compileall .` also traverses the ignored virtual environment; for a faster project-only check use `python -m compileall -q app.py src tests`. Tests retain the Phase 1 checks and cover input errors, row and column limits, quality categories, original-data preservation, source switching, explicit selection, and navigation persistence. Sentiment tests cover actual positive/neutral/negative scoring, exact threshold boundaries, unusable rows, duplicate text, emphasis, emoji, multilingual/long text, column collisions, resource failures, and session invalidation. User-facing tracebacks are disabled; unexpected loading errors are logged to the server console.
