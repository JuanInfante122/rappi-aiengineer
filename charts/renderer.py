"""Plotly chart renderer with Rappi brand styling.

Dispatches to the correct chart builder (line, bar, box, scatter) based
on the viz_hint dict produced by agent.intent.parse_viz_hint(). Falls
back to bar chart for unknown types and to first/second DataFrame columns
when viz_hint column names do not match the actual data.
"""

import pandas as pd
import plotly.graph_objects as go

RAPPI_ORANGE = "#FF441F"
RAPPI_TEAL = "#00B5AD"
RAPPI_YELLOW = "#F2C94C"
GRID_COLOR = "#EAEAEA"

COLOR_SEQUENCE = [RAPPI_ORANGE, RAPPI_TEAL, RAPPI_YELLOW]


def build_chart(df: pd.DataFrame, viz_hint: dict) -> go.Figure:
    """Build a Plotly Figure based on viz_hint type and column mapping.

    Dispatches to the appropriate chart builder. Falls back to bar chart
    for unknown types. Gracefully handles column name mismatches by
    using the first and second DataFrame columns as fallback.

    Args:
        df: DataFrame with query results (at least 2 columns expected).
        viz_hint: Dict with 'type', 'x_col', 'y_col' from parse_viz_hint.

    Returns:
        Configured Plotly Figure ready for st.plotly_chart().
    """
    viz_type = viz_hint.get("type", "bar")
    x_col = viz_hint.get("x_col", "")
    y_col = viz_hint.get("y_col", "")

    # Graceful column fallback when viz_hint columns are absent from the result
    if x_col not in df.columns and len(df.columns) > 0:
        x_col = df.columns[0]
    if y_col not in df.columns and len(df.columns) > 1:
        y_col = df.columns[1]

    dispatchers = {
        "line": _build_line,
        "bar": _build_bar,
        "box": _build_box,
        "scatter": _build_scatter,
    }
    builder = dispatchers.get(viz_type, _build_bar)
    return builder(df, x_col, y_col)


def _base_layout() -> dict:
    """Return base Plotly layout dict with Rappi brand styling.

    White background, subtle grid lines, sans-serif font, compact margins.
    No data labels — tooltips only.
    """
    return dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="sans-serif", size=13, color="#1A1A1A"),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
    )


def _build_line(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Build line chart for temporal trends (VIZ-02)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="lines+markers",
        line=dict(color=RAPPI_ORANGE, width=2),
        marker=dict(color=RAPPI_ORANGE, size=6),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(**_base_layout())
    return fig


def _build_bar(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Build bar chart for zone/country comparisons (VIZ-03)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[y_col],
        marker=dict(color=RAPPI_ORANGE),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(**_base_layout())
    return fig


def _build_box(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Build box plot for metric distribution across zones (VIZ-04)."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df[x_col],
        y=df[y_col],
        marker=dict(color=RAPPI_ORANGE),
        line=dict(color=RAPPI_ORANGE),
    ))
    fig.update_layout(**_base_layout())
    return fig


def _build_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Build scatter plot for cross-metric correlations (VIZ-05)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="markers",
        marker=dict(color=RAPPI_ORANGE, size=8),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(**_base_layout())
    return fig
