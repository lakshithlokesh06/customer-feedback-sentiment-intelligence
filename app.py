"""Streamlit interface with explicit offline VADER analysis."""

import logging
import hashlib

import pandas as pd
import streamlit as st

from src.config import APP_TITLE, KPI_LABELS, NAVIGATION, SAMPLE_COLUMNS, SAMPLE_DATA_PATH, TAGLINE
from src.config import MAX_UPLOAD_BYTES, MAX_DATASET_ROWS, PREVIEW_ROWS, MIN_REVIEW_LENGTH, SETUP_MESSAGE
from src.data.preprocessor import prepare_reviews, review_columns
from src.data.session import clear_dataset, set_dataset, reset_review_selection
from src.config import POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD, SENTIMENT_COLORS
from src.sentiment.analyzer import analyze_dataframe, sentiment_summary, SentimentError
from src.data.loader import DataValidationError, read_csv_safely
from src.utils.helpers import format_count
from src.ui.about import render_about_page
from src.ui.analytics import render_analytics_page
from src.ui.text_insights import render_text_insights_page
from src.ui.review_explorer import render_review_explorer_page

logger = logging.getLogger(__name__)
st.set_page_config(page_title=APP_TITLE, page_icon="◈", layout="wide")


def load_sample() -> bool:
    """Load the bundled dataset while presenting safe, actionable failures."""
    try:
        with st.spinner("Loading sample dataset…"):
            data = read_csv_safely(SAMPLE_DATA_PATH)
        if not set(SAMPLE_COLUMNS).issubset(data.columns):
            raise DataValidationError("The sample dataset is incomplete. Restore the bundled CSV file.")
        set_dataset(st.session_state, data, "Fictional sample", "sample")
        return True
    except DataValidationError as exc:
        st.error(str(exc))
    except Exception:
        logger.exception("Unexpected sample resource failure")
        st.error("The sample dataset is temporarily unavailable. Check the local installation and try again.")

    return False


def render_sidebar() -> str:
    """Render navigation and explicit sample-data controls."""
    with st.sidebar:
        st.markdown("### FEEDBACK / INTELLIGENCE")
        st.caption("Customer experience workspace")
        st.divider()
        page = st.radio("Workspace", NAVIGATION, label_visibility="collapsed")
        st.divider()
        st.markdown("**Dataset**")
        if st.button("Load sample dataset", width="stretch"):
            load_sample()
        if st.session_state.get("dataset") is not None:
            st.caption(f"{st.session_state.get('dataset_name', 'Dataset')} · {format_count(len(st.session_state['dataset']))} rows")
            if st.button("Clear dataset", width="stretch"):
                clear_dataset(st.session_state)
                st.rerun()
        else:
            st.caption("No dataset loaded")
        st.divider()
        st.caption("Explore the story behind each review.")
        st.caption("Upload, prepare, and analyze review text from Overview.")
    return page


def render_kpis(data: pd.DataFrame | None) -> None:
    """Show real record counts and clearly uncomputed sentiment metrics."""
    if st.session_state.get("analysis") is not None:
        render_sentiment_summary()
        return
    for index, (column, label) in enumerate(zip(st.columns(4), KPI_LABELS)):
        with column:
            with st.container(border=True):
                value = format_count(len(data)) if index == 0 and data is not None else "—"
                st.metric(label, value)
                st.caption("Dataset records" if index == 0 else "Awaiting sentiment analysis")


def render_reviews(data: pd.DataFrame) -> None:
    """Show a bounded preview of the original dataset."""
    st.subheader("Dataset preview")
    st.dataframe(data.head(PREVIEW_ROWS), hide_index=True, width="stretch")
    st.caption(f"{format_count(len(data))} reviews · Showing up to {PREVIEW_ROWS} rows. Original values are preserved.")


def metric_cards(values: dict) -> None:
    """Render compact profiling and review-quality metrics."""
    items = list(values.items())
    for start in range(0, len(items), 3):
        group = items[start:start + 3]
        for column, (label, value) in zip(st.columns(len(group)), group):
            with column, st.container(border=True):
                st.metric(label, format_count(value) if isinstance(value, int) else value)


def render_loading() -> None:
    """Offer explicit source controls without rereading on navigation."""
    st.subheader("Dataset setup")
    source = st.radio("Dataset source", ("Upload CSV", "Use sample dataset"), horizontal=True)
    if source == "Use sample dataset":
        if st.button("Use sample dataset") and load_sample():
            st.rerun()
    else:
        uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"], max_upload_size=MAX_UPLOAD_BYTES // 1024**2,
                                    help=f"UTF-8 CSV; up to {MAX_UPLOAD_BYTES // 1024**2} MB and {MAX_DATASET_ROWS:,} rows.")
        if uploaded is not None and st.button("Load uploaded CSV"):
            identity = hashlib.sha256(uploaded.getbuffer()).hexdigest()
            if st.session_state.get("dataset_id") == identity:
                st.info("This dataset is already loaded; your review selection has been retained.")
            else:
                try:
                    with st.spinner("Loading and validating CSV…"):
                        data = read_csv_safely(uploaded)
                        set_dataset(st.session_state, data, uploaded.name, identity)
                    st.rerun()
                except DataValidationError as exc:
                    st.error(str(exc))
                    st.caption("The previous dataset, if any, is still active.")
                except Exception:
                    logger.exception("Unable to load uploaded dataset")
                    st.error("Unable to load this dataset. Check the CSV and try a smaller file.")
    st.caption("Choose a source and press its load button to replace the active dataset.")


def render_setup(data: pd.DataFrame) -> None:
    """Profile data and persist an explicit review-column selection."""
    st.caption(f"Active dataset: {st.session_state.get('dataset_name', 'Dataset')}")
    metric_cards(st.session_state["profile"])
    st.caption("Missing values include whitespace-only cells. Duplicate rows are retained.")
    render_reviews(data)
    options = review_columns(data)
    selected = st.session_state.get("review_column")
    column = st.selectbox("Select review text column", options,
                          index=options.index(selected) if selected in options else None,
                          placeholder="Choose a column containing customer feedback",
                          help="Choose the column containing written reviews. Text columns appear first. Changing this selection clears previous analysis.",
                          key=f"review_selector_{st.session_state.get('dataset_revision', 0)}")
    reset_review_selection(st.session_state, column)
    if column is None:
        st.info(SETUP_MESSAGE)
        return
    if "prepared" not in st.session_state:
        st.session_state["prepared"] = prepare_reviews(data, column)
    prepared = st.session_state["prepared"]
    st.subheader("Review quality")
    metric_cards(prepared.quality)
    unusable = len(data) - prepared.quality["Valid Reviews"]
    st.caption(f"Reviews require at least {MIN_REVIEW_LENGTH} trimmed characters. Numeric-only values are not review text.")
    if unusable:
        st.warning(f"{unusable:,} reviews will be excluded from analysis. All rows remain available with a review-quality status.")
    if prepared.quality["Valid Reviews"] and st.session_state.get("analysis") is None:
        st.success("Review data is prepared. Use Analyze Sentiment below to score valid reviews.")
    elif not prepared.quality["Valid Reviews"]:
        st.error("This column has no usable reviews. Select another column or load another dataset.")
    with st.expander("Inspect prepared reviews and row status"):
        st.dataframe(prepared.data.head(PREVIEW_ROWS), hide_index=True, width="stretch")
        st.caption(f"Added columns: {prepared.text_column}, {prepared.status_column}. Status: valid, missing, empty, non_text, or too_short.")


def render_sentiment_summary() -> None:
    """Show actual label counts and shares of analyzed reviews."""
    summary = sentiment_summary(st.session_state["analysis"])
    metric_cards(summary)
    total = summary["Analyzed Reviews"]
    if total:
        st.caption("Share of analyzed reviews: " + " · ".join(
            f"{label}: {summary[f'{label} Reviews'] / total:.1%}"
            for label in ("Positive", "Neutral", "Negative")
        ))


def render_sentiment_preview() -> None:
    """Display bounded, adaptive results with restrained sentiment colors."""
    result = st.session_state["analysis"]
    st.subheader("Sentiment results")
    review = st.session_state["review_column"]
    preferred = [review, *result.columns.values(), st.session_state["prepared"].status_column]
    preferred += [name for name in ("rating", "date", "product") if name in result.data]
    names = list(dict.fromkeys(preferred))
    preview = result.data[names].head(PREVIEW_ROWS)
    styled = preview.style.map(lambda value: SENTIMENT_COLORS.get(value, ""),
                               subset=[result.columns["sentiment_label"]])
    st.dataframe(styled, hide_index=True, width="stretch")
    st.caption(f"First {PREVIEW_ROWS} rows at most. Unusable rows are retained as Not analyzed with null scores.")
    if any(logical != physical for logical, physical in result.columns.items()):
        st.caption("Existing sentiment columns were preserved; new output names have leading underscores.")


def render_analysis_action() -> None:
    """Run scoring only on a button click; publish results atomically."""
    prepared = st.session_state.get("prepared")
    if prepared is None or not prepared.quality["Valid Reviews"]:
        return
    if st.button("Analyze Sentiment", type="primary"):
        try:
            with st.spinner("Analyzing valid reviews with VADER…"):
                result = analyze_dataframe(prepared)
            st.session_state["analysis"] = result
            st.rerun()
        except SentimentError as exc:
            st.error(str(exc))
        except Exception:
            logger.exception("Sentiment analysis failed")
            st.error("Sentiment analysis could not complete. Please retry or use a smaller dataset.")
    if st.session_state.get("analysis") is not None:
        st.success("Sentiment analysis complete. Results are retained for this dataset and review column.")
        render_sentiment_preview()
    st.caption(
        f"VADER is a rule/lexicon-based model. Compound ranges from −1 to 1; "
        f"Positive ≥ {POSITIVE_THRESHOLD}, Negative ≤ {NEGATIVE_THRESHOLD}, otherwise Neutral. "
        "Scores reflect text sentiment, not factual correctness or human intent. "
        "VADER is primarily designed for English; multilingual results may be unreliable."
    )


def main() -> None:
    """Route the existing dashboard and persist dataset preparation across pages."""
    page = render_sidebar()
    st.caption("CUSTOMER EXPERIENCE / FEEDBACK INTELLIGENCE")
    st.title(APP_TITLE)
    st.write(TAGLINE)
    st.divider()
    if page == "Overview":
        st.subheader("Overview")
        st.write("Load a dataset, select your review column, check review quality, then analyze sentiment.")
        render_loading()
        data = st.session_state.get("dataset")
        kpi_area = st.container()
        if data is None:
            st.info(SETUP_MESSAGE)
        else:
            render_setup(data)
            render_analysis_action()
        with kpi_area:
            render_kpis(data)
        if st.session_state.get("analysis") is not None:
            st.info("Explore results using Sentiment Analytics, Text Insights, or Review Explorer in the sidebar.")
    elif page == "Sentiment Analytics":
        render_analytics_page()
    elif page == "Text Insights":
        render_text_insights_page()
    elif page == "About":
        render_about_page()
    elif page == "Review Explorer":
        render_review_explorer_page()

    st.divider()
    st.caption("Customer Feedback Sentiment Intelligence · Methodology and data handling in About")


if __name__ == "__main__":
    main()
