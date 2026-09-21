"""Product, methodology, and data-handling information."""
import streamlit as st
from src.config import POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD


def render_about_page() -> None:
    st.subheader('About')
    st.write('Understand the methods and limitations behind your customer feedback insights.')
    st.markdown('### From customer reviews to useful evidence')
    st.write('Upload a customer-feedback CSV, validate and prepare reviews, and classify their text sentiment. Explore trends, categories, ratings, common praise and concerns; search individual reviews and export your results.')
    st.markdown('### Sentiment methodology')
    st.write('VADER Sentiment uses a lexicon and language rules to score review text. Ratings and other metadata do not influence the classification. Its bundled lexicon works without a runtime download.')
    st.markdown(f'**Compound Score** ranges from −1 to +1. **Positive:** ≥ {POSITIVE_THRESHOLD}; **Negative:** ≤ {NEGATIVE_THRESHOLD}; **Neutral:** between these thresholds. Positive, Neutral, and Negative component scores describe the balance of sentiment in the text; they are not confidence probabilities.')
    st.write('Not analyzed reviews remain in results with empty scores when their text is missing, blank, non-text, numeric-only, or too short.')
    st.markdown('### Technology')
    st.write('Python · Streamlit · Pandas · NumPy · Plotly · VADER Sentiment · scikit-learn (English stopwords) · python-dotenv (optional configuration) · pytest (testing)')
    st.markdown('### Interpretation and limitations')
    st.markdown('- Primarily optimized for English; sarcasm and context may be misinterpreted.\n- Lexicon sentiment is not human understanding and does not establish factual correctness.\n- Frequent keywords and phrases describe mentions, not proven root causes.\n- Date, category, and rating suggestions use heuristics; verify selected metadata before interpreting comparisons.')
    st.markdown('### Privacy and data handling')
    st.write('Uploaded files are processed within the running application session. CSV exports are generated in memory. The application does not intentionally persist uploaded datasets to a database. Session lifetime and infrastructure behavior depend on the hosting environment; this is not a guarantee of permanent storage or automatic secure deletion.')
