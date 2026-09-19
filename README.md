# Customer Feedback Sentiment Intelligence

Transform customer feedback into actionable sentiment insights.

A Python and Streamlit portfolio project for exploring customer review data. **Phase 1 delivers the project foundation and dashboard shell.** Sentiment classification, VADER scoring, and machine-learning training are not implemented.

## Available in Phase 1

- Wide dashboard with sidebar navigation: Overview, Sentiment Analytics, Text Insights, Review Explorer, and About.
- Helpful empty states, placeholder sentiment KPIs, and clearly labeled planned analytics areas.
- Load and clear a bundled fictional sample dataset; preview original records and their count.
- Reusable CSV validation for file extensions, size, encoding, headers, row consistency, and empty datasets, with readable errors.
- Central configuration, optional environment settings, and automated loader and UI smoke tests.

## Planned features

CSV uploads, review-column selection, local sentiment scoring, sentiment trends, text insights, review filtering, and analyzed-data exports. There is no upload control or automatic review-column detection in this phase. No paid API, LLM, authentication, or database is used.

## Tech stack

Python 3.12+; Streamlit; Pandas; NumPy; Plotly; NLTK; VADER Sentiment (`vaderSentiment`); scikit-learn; python-dotenv.

Analytics libraries are installed for future phases; Phase 1 does not initialize models or download NLTK resources. Requirements specify compatible version ranges rather than a fully locked environment.

## Project structure

```text
customer-feedback-sentiment-intelligence/
├── app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
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
│   └── test_loader.py
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── requirements.txt
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

No environment file is required. Optionally copy `.env.example` to `.env` and set `FEEDBACK_MAX_UPLOAD_MB` (default 10; accepted range 1–100). This controls the reusable loader limit; Streamlit's separate server upload limit is configured in `.streamlit/config.toml` for future use. Invalid environment values fall back to 10 MB.

## Sample data

`data/sample_customer_feedback.csv` contains 20 fictional reviews across four products, dated August 1–20, 2026. Columns are `review_id` (unique identifier), `review` (original text), `rating` (1–5), `date` (ISO date), and `product`. The text includes positive, negative, neutral, and mixed experiences. It has no sentiment labels or scores; ratings are not converted into sentiment.

Use **Load sample dataset** in the sidebar to preview it. **Clear dataset** resets the session preview. A missing or invalid sample produces a friendly error. The generic loader accepts UTF-8 comma-delimited CSV files and binary file-like objects; it does not enforce the sample schema or select review columns. The app checks the sample's expected columns separately.

## Validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py src tests
python -m streamlit run app.py --server.headless true
```

Tests cover CSV input boundaries, sample schema, navigation, loading/clearing, and friendly missing-resource errors. Compilation is scoped to project code to avoid traversing the virtual environment. User-facing tracebacks are disabled in Streamlit configuration; unexpected sample-loading errors are logged to the local server console.
