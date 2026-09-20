"""Readable horizontal keyword/phrase charts using the existing theme."""
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from src.config import SENTIMENT_CHART_COLORS
from src.visualization.charts import finish


def term_chart(data: pd.DataFrame, title: str, sentiment: str = 'Neutral') -> Figure:
    """Display review frequency, with occurrences and coverage in hover details."""
    figure = px.bar(data, x='Reviews', y='Term', orientation='h', title=title,
                    color_discrete_sequence=[SENTIMENT_CHART_COLORS[sentiment]],
                    hover_data={'Occurrences': True, 'Coverage (%)': ':.1f'},
                    labels={'Reviews': 'Reviews mentioning term'})
    figure.update_yaxes(autorange='reversed', title=None)
    largest_count = int(data.Reviews.max()) if not data.empty else 1
    figure.update_xaxes(dtick=max(1, (largest_count + 4) // 5), tickformat=',d')
    figure.update_layout(height=max(350, len(data) * 28 + 140), showlegend=False)
    return finish(figure)
