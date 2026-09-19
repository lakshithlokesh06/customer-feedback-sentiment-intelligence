"""Phase 1 Streamlit interface; analysis is intentionally not implemented."""

import logging

import pandas as pd
import streamlit as st

from src.config import APP_TITLE, KPI_LABELS, NAVIGATION, SAMPLE_COLUMNS, SAMPLE_DATA_PATH, TAGLINE
from src.data.loader import DataValidationError, read_csv_safely
from src.utils.helpers import format_count

logger = logging.getLogger(__name__)
st.set_page_config(page_title=APP_TITLE, page_icon="◈", layout="wide")


def load_sample() -> None:
    """Load the bundled dataset while presenting safe, actionable failures."""
    try:
        data = read_csv_safely(SAMPLE_DATA_PATH)
        if not set(SAMPLE_COLUMNS).issubset(data.columns):
            raise DataValidationError("The sample dataset is incomplete. Restore the bundled CSV file.")
        st.session_state["dataset"] = data
    except DataValidationError as exc:
        st.error(str(exc))
    except Exception:
        logger.exception("Unexpected sample resource failure")
        st.error("The sample dataset is temporarily unavailable. Check the local installation and try again.")


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
            st.caption(f"Fictional sample · {format_count(len(st.session_state['dataset']))} reviews")
            if st.button("Clear dataset", width="stretch"):
                st.session_state.pop("dataset", None)
                st.rerun()
        else:
            st.caption("No dataset loaded")
        st.divider()
        st.caption("PHASE 1 · FOUNDATION")
        st.caption("Sample preview is available. Uploads and sentiment analysis are planned for later phases.")
    return page


def render_kpis(data: pd.DataFrame | None) -> None:
    """Show real record counts and clearly uncomputed sentiment metrics."""
    for index, (column, label) in enumerate(zip(st.columns(4), KPI_LABELS)):
        with column:
            with st.container(border=True):
                value = format_count(len(data)) if index == 0 and data is not None else "—"
                st.metric(label, value)
                st.caption("Dataset records" if index == 0 else "Awaiting sentiment analysis")


def render_analytics() -> None:
    """Reserve clearly labeled areas for future visual analytics."""
    st.subheader("Sentiment at a glance")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Sentiment distribution**")
        st.caption("PLANNED")
        st.info("Positive, neutral, and negative review shares will appear here after analysis is available.")
    with right, st.container(border=True):
        st.markdown("**Sentiment over time**")
        st.caption("PLANNED")
        st.info("Explore changes in customer sentiment across review dates in a future phase.")


def render_reviews(data: pd.DataFrame | None) -> None:
    """Preview raw sample rows without implying that scoring has happened."""
    st.subheader("Review explorer")
    st.caption("Inspect original review records. Sentiment labels, filtering, and export are planned.")
    if data is None:
        with st.container(border=True):
            st.markdown("**Your customer stories will appear here**")
            st.write("Load the fictional sample dataset from the sidebar to preview the review table.")
    else:
        st.dataframe(data, hide_index=True, width="stretch")
        st.caption(f"{format_count(len(data))} original records · No sentiment labels or scores have been generated.")


def main() -> None:
    """Route the dashboard shell to the selected workspace section."""
    page = render_sidebar()
    data = st.session_state.get("dataset")
    st.caption("CUSTOMER EXPERIENCE / FEEDBACK INTELLIGENCE")
    st.title(APP_TITLE)
    st.write(TAGLINE)
    st.divider()
    if page == "Overview":
        st.subheader("Understand the voice of your customers")
        st.write("A focused workspace for turning customer reviews into a clearer view of product experience.")
        if data is None:
            st.info("Start with the sample dataset. Load 20 fictional reviews using the sidebar to explore the foundation of your feedback workspace.")
        else:
            st.success("Sample dataset loaded. You can preview the original records; sentiment analysis is not available in Phase 1.")
        render_kpis(data)
        st.write("")
        render_analytics()
        st.write("")
        render_reviews(data)
    elif page == "Sentiment Analytics":
        st.subheader("Sentiment analytics")
        st.write("A dedicated view for understanding sentiment patterns across your feedback.")
        render_kpis(data)
        render_analytics()
    elif page == "Text Insights":
        st.subheader("Text insights")
        with st.container(border=True):
            st.markdown("**Discover what customers are talking about**")
            st.info("Keywords, recurring themes, and text exploration are planned for a future phase. No text analysis has been performed.")
    elif page == "Review Explorer":
        render_reviews(data)
    else:
        st.subheader("About this project")
        st.write("Customer Feedback Sentiment Intelligence is a portfolio project for exploring customer experience through review data.")
        st.markdown("**Available now:** dashboard navigation, safe CSV-loading utilities, and a fictional sample dataset preview.")
        st.markdown("**Planned:** CSV uploads, review-column selection, local sentiment scoring, interactive insights, review filters, and exports.")
        st.caption("Built with Python and Streamlit. No external paid API, LLM, authentication, or database is used.")
    st.divider()
    st.caption("Phase 1 · Project foundation · Sample data is fictional")


if __name__ == "__main__":
    main()
