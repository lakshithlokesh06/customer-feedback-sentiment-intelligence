"""Consistent Plotly figures built from real sentiment aggregations."""
import plotly.express as px
from src.analytics.sentiment_metrics import LABELS
from src.config import POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD, SENTIMENT_CHART_COLORS as COLORS



def finish(figure):
    figure.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(family='sans-serif'), margin=dict(l=15, r=15, t=100, b=25),
                         legend=dict(orientation='h', y=1.25, x=0, title_text=''))
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor='#e8edf2')
    return figure


def distribution(summary):
    fig = px.bar(summary, x='Sentiment', y='Count', color='Sentiment',
                 color_discrete_map=COLORS, hover_data={'Percent': ':.1f'}, title='Sentiment distribution')
    fig.update_layout(showlegend=False)
    return finish(fig)


def compound(data):
    fig = px.histogram(data, x='sentiment_compound', nbins=40, range_x=[-1, 1],
                       title='Compound score distribution', labels={'sentiment_compound': 'Compound score'},
                       color_discrete_sequence=['#52758D'])
    for threshold in (NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD):
        fig.add_vline(x=threshold, line_dash='dot', line_color='#7B8794')
    return finish(fig)


def components(data):
    values = [data[f'sentiment_{label.lower()}'].mean() for label in LABELS]
    fig = px.bar(x=list(LABELS), y=values, color=list(LABELS), color_discrete_map=COLORS,
                 labels={'x': 'Component', 'y': 'Average score'}, title='Average VADER components')
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 1])
    return finish(fig)


def comparison(summary, title):
    summary = summary.assign(Group=summary.Group.astype(str))
    fig = px.bar(summary, y='Group', x=list(LABELS), orientation='h', barmode='stack', color_discrete_map=COLORS,
                 hover_data={'Review count': True, 'Average compound': ':.3f'}, title=title,
                 labels={'value': 'Share (%)', 'variable': 'Sentiment'})
    fig.update_layout(height=max(400, len(summary) * 35 + 150))
    fig.update_yaxes(autorange='reversed')
    return finish(fig)


def trend(summary):
    fig = px.line(summary, x='Group', y='Average compound', markers=True,
                  hover_data=['Review count'], title='Sentiment over time', labels={'Group': 'Date'})
    fig.update_yaxes(range=[-1, 1])
    return finish(fig)
