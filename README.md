# Customer Feedback Sentiment Intelligence

Transform customer feedback into actionable sentiment insights.

A Python and Streamlit portfolio project for exploring customer review data. **Phase 2 adds CSV upload, dataset profiling, and review preparation to the existing dashboard.** Sentiment classification, VADER scoring, and machine-learning training are not implemented.

## Available in Phase 2

- Wide dashboard with sidebar navigation: Overview, Sentiment Analytics, Text Insights, Review Explorer, and About.
- Helpful empty states, placeholder sentiment KPIs, and clearly labeled planned analytics areas.
- Load and clear a bundled fictional sample dataset; preview original records and their count.
- Reusable CSV validation for file extensions, size, encoding, headers, row consistency, and empty datasets, with readable errors.
- CSV upload and switching between uploaded and sample datasets without restarting.
- Dataset profiling: row/column counts, missing cells, duplicate rows, and estimated dataframe memory.
- Explicit review-column selection with text-like columns listed first; no silent selection.
- Review quality summaries and prepared review text with per-row status, preserving all original data.
- Central configuration, persistent session state, and automated loader, preparation, and UI tests.

## Planned features

Local sentiment scoring, sentiment trends, text insights, review filtering, and analyzed-data exports. No sentiment scoring is performed in this phase. No paid API, LLM, authentication, or database is used.

## Tech stack

Python 3.12+; Streamlit; Pandas; NumPy; Plotly; NLTK; VADER Sentiment (`vaderSentiment`); scikit-learn; python-dotenv.

Analytics libraries are installed for future phases; Phase 2 does not initialize models or download NLTK resources. Requirements specify compatible version ranges rather than a fully locked environment.

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
│   │   └── __init__.py
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
│   └── test_phase2.py
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

The data layer has no Streamlit dependency. The sentiment, analytics, and visualization packages reserve simple module boundaries for later phases. `app.py` owns the interface and session state; `src/config.py` owns shared application constants. Native Streamlit theming lives in `.streamlit/config.toml`.

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

## Validation

Install the development dependency (pytest) and run:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall .
python -m pytest
python -m unittest discover -s tests -v
python -m streamlit run app.py --server.headless true
```

`compileall .` also traverses the ignored virtual environment; for a faster project-only check use `python -m compileall -q app.py src tests`. Tests retain the Phase 1 checks and cover input errors, row and column limits, quality categories, original-data preservation, source switching, explicit selection, and navigation persistence. User-facing tracebacks are disabled; unexpected loading errors are logged to the server console.
