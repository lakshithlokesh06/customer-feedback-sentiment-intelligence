# Customer Feedback Sentiment Intelligence

Transform customer feedback into actionable sentiment insights.

## Overview

A Streamlit workspace for turning customer-review CSV files into understandable sentiment summaries. Validate your data, choose the review column, run offline VADER analysis, and investigate the individual reviews behind trends. A fictional 20-review sample demonstrates the complete workflow without an upload.

## Key features

- **CSV upload and validation:** friendly errors, bounded file sizes, dataset profiling, and previews.
- **Review quality checks:** identify missing, blank, non-text, numeric-only, and very short reviews without deleting records.
- **VADER sentiment analysis:** Positive, Neutral, and Negative labels with compound and component scores.
- **Sentiment analytics:** distributions, time-based trends, category comparisons, and rating–sentiment comparisons.
- **Text insights:** keyword and two-word phrase frequencies, Positive/Negative mentions, and example reviews.
- **Review Explorer:** literal text search, sentiment and metadata filters, sorting, and pagination.
- **CSV export:** full results or the current filtered and sorted selection, generated in memory.

## How it works

```text
Upload CSV (or load sample)
   ↓
Select Review Column
   ↓
Validate & Prepare Reviews
   ↓
Run VADER Sentiment Analysis
   ↓
Explore Analytics & Text Insights
   ↓
Filter Reviews
   ↓
Export Results
```

Overview owns dataset setup and analysis. The other pages reuse session results. Changing the dataset or review column clears incompatible analysis and downloads. Navigation preserves the dataset, selection, and analysis; page-specific filter widgets may reset when navigating away.

## Tech stack

| Tool | Purpose |
| --- | --- |
| Python 3.12 | Tested runtime |
| Streamlit | Application and session state |
| Pandas / NumPy | Data preparation, summaries, and numeric checks |
| Plotly | Interactive charts |
| vaderSentiment | Bundled sentiment lexicon and scoring rules |
| scikit-learn | Bundled English stopword list |
| python-dotenv | Optional local settings |
| pytest / unittest / Streamlit AppTest | Automated validation |

No API keys, model downloads, database, or paid service are needed. NLTK is not required.

## Project structure

```text
app.py                         # Entry point and Overview workflow
src/
  config.py                    # Identity, limits, thresholds, presentation
  data/
    loader.py                  # Strict CSV validation
    preprocessor.py            # Review preparation and quality checks
    session.py                 # Dataset and analysis lifecycle
    exporter.py                # In-memory CSV output
  sentiment/analyzer.py        # Offline VADER analysis
  analytics/
    sentiment_metrics.py       # Aggregations and metadata detection
    text_insights.py            # Keywords, phrases, example reviews
    review_explorer.py         # Filtering, sorting, pagination
  visualization/
    charts.py                  # Sentiment figures
    text_charts.py             # Frequency figures
  ui/
    analytics.py
    text_insights.py
    review_explorer.py
    about.py
  utils/helpers.py             # Display formatting
data/sample_customer_feedback.csv
 tests/                        # Unit and application regression tests
.streamlit/config.toml         # Theme, error display, upload ceiling
requirements.txt
requirements-dev.txt
.env.example                   # Optional upload-limit setting
```

## Data validation and limits

Only comma-separated UTF-8 CSV files (including a UTF-8 BOM) are supported. Validation rejects unreadable or malformed files, duplicate/blank headers, inconsistent row widths, empty data, and datasets containing only empty columns. Error messages do not expose Python exceptions.

Default limits are **10 MB**, **100,000 rows**, and **200 columns**. Previews show at most **100 rows**; Review Explorer pages contain 10, 25, or 50 reviews. The optional `FEEDBACK_MAX_UPLOAD_MB` setting accepts 1–100 MB; keep `.streamlit/config.toml`'s server upload ceiling consistent if increasing it. The CSV parser also has a per-field safety limit; unusually long single-cell text may be rejected. Limits reduce risk but do not guarantee every accepted dataset fits the host's memory.

Choose a review column explicitly; text columns appear first. Preparation copies all source rows and columns, trims valid strings, and adds `prepared_review` and `review_status`. Reviews need at least three trimmed characters; numeric-only text is excluded. Original values remain unchanged. Generated names receive leading underscores when needed to preserve existing columns.

Unusable reviews retain the label **Not analyzed** and empty scores. They are excluded from sentiment and text aggregates, remain searchable/exportable in Review Explorer, and are never silently removed. Duplicate reviews are retained and contribute separately to counts.

## Sentiment methodology

VADER is a lexicon and rule-based model. It scores the selected text only; ratings do not change sentiment.

| Output | Meaning |
| --- | --- |
| Compound Score | Normalized overall sentiment from −1 to +1 |
| Positive | Compound Score ≥ 0.05 |
| Negative | Compound Score ≤ −0.05 |
| Neutral | −0.05 < Compound Score < 0.05 |
| Component scores | Positive, Neutral, and Negative proportions; not confidence probabilities |

The package includes its lexicon, so scoring requires no runtime network access. Analysis runs only after the user presses **Analyze Sentiment**.

## Analytics

Sentiment shares use analyzed reviews in the selected scope. Compound and component distributions show the score balance. Date trends use parseable dates normalized to UTC, with day/week/month grouping as available; weeks start Monday. Missing dates are excluded from trend charts. Category charts show up to 15 groups by review volume, with missing values grouped separately.

Rating–sentiment mismatch highlights 4–5 star ratings with Negative text and 1–2 star ratings with Positive text, only when the source resembles an integral 1–5 scale. It does not imply fraud or an error. Metadata detection is heuristic and may not recognize every dataset schema.

Text Insights uses Unicode tokenization and English stopwords while retaining basic negation. Keyword and bigram charts rank by reviews mentioning a term, then occurrences. Repeated mentions in one review increase word frequency but count once toward review frequency. Scopes with fewer than 30 reviews require one mention; larger scopes require two reviews. Example reviews are ranked by sentiment score, not statistically sampled.

Review Explorer combines literal, case-insensitive search with sentiment, compound, category, rating, and date filters. Selected Not analyzed reviews bypass the compound filter; active date/rating filters exclude invalid metadata. Exports include all matching pages and omit the dataframe index. CSV preserves original text, including spreadsheet formula-like strings; interpret external spreadsheet imports accordingly.

## Run locally

```bash
git clone git@github.com:lakshithlokesh06/customer-feedback-sentiment-intelligence.git
cd customer-feedback-sentiment-intelligence
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown by Streamlit. No `.env` file is required. To try the app, load the sample from the sidebar, select `review` in Overview, and click **Analyze Sentiment**.

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m compileall .
python -m pytest
python -m unittest discover -s tests
python -m pip check
git diff --check
```

Tests cover CSV and review validation, source preservation, sentiment boundaries and failures, analytics, keywords, filtering, exports, and session lifecycle. AppTest exercises the main workflow without a browser.

## Deployment readiness

Use `app.py` as the entry point and `requirements.txt` for dependencies. Sample resources resolve relative to the repository, not a developer's machine. Core functionality requires no secrets, external NLP downloads, or writable export directory.

For Streamlit Community Cloud, select this repository, `main`, and `app.py`; select the tested **Python 3.12** runtime in Advanced settings. Community Cloud configures Python through that UI; no speculative runtime configuration file is included. See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). This repository does not claim an active deployment or CI service.

Uploaded files are processed within the running application session. Exports are generated in memory. The application does not intentionally persist uploaded datasets to a database. Session lifetime and infrastructure behavior depend on the hosting environment. Clearing data resets app state; it is not a guarantee of secure deletion from host memory.

## Limitations

- VADER is primarily English-focused. Sarcasm, specialized language, and context can be misinterpreted.
- Lexicon sentiment is not human understanding or proof of factual correctness.
- Keywords identify mentions, not confirmed praise, defects, or root causes.
- Metadata detection uses heuristics; verify suggested fields and date interpretation.
- Large or unusually verbose datasets may require more memory and processing time than a small hosting instance provides.
- Session state is temporary; export results you need to keep. There are no user accounts or durable storage.

## Future possibilities

Aspect-based sentiment, multilingual models, transformer comparisons, topic clustering, and PDF reports are potential future directions. They are not implemented.
