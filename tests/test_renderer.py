"""Unit tests for the Plotly chart renderer dispatcher.

Tests verify that build_chart returns valid Plotly Figures for each
supported chart type with correct trace types and Rappi brand colors.
"""

import pandas as pd
import plotly.graph_objects as go

from charts.renderer import build_chart, RAPPI_ORANGE


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "week": [0, 1, 2, 3],
        "value": [10.5, 12.3, 11.0, 14.2],
    })


def _sample_df_zones() -> pd.DataFrame:
    return pd.DataFrame({
        "zone": ["A", "B", "C"],
        "metric": [0.85, 0.72, 0.91],
    })


def test_line_chart_returns_figure():
    df = _sample_df()
    fig = build_chart(df, {"type": "line", "x_col": "week", "y_col": "value"})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.data[0].mode == "lines+markers"


def test_bar_chart_returns_figure():
    df = _sample_df_zones()
    fig = build_chart(df, {"type": "bar", "x_col": "zone", "y_col": "metric"})
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Bar)


def test_box_chart_returns_figure():
    df = _sample_df_zones()
    fig = build_chart(df, {"type": "box", "x_col": "zone", "y_col": "metric"})
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Box)


def test_scatter_chart_returns_figure():
    df = _sample_df()
    fig = build_chart(df, {"type": "scatter", "x_col": "week", "y_col": "value"})
    assert isinstance(fig, go.Figure)
    assert fig.data[0].mode == "markers"


def test_unknown_type_defaults_to_bar():
    df = _sample_df_zones()
    fig = build_chart(df, {"type": "unknown", "x_col": "zone", "y_col": "metric"})
    assert isinstance(fig.data[0], go.Bar)


def test_column_fallback_x():
    df = _sample_df()
    fig = build_chart(df, {"type": "bar", "x_col": "nonexistent", "y_col": "value"})
    assert isinstance(fig, go.Figure)
    # Should use first column (week) as x
    assert list(fig.data[0].x) == [0, 1, 2, 3]


def test_column_fallback_y():
    df = _sample_df()
    fig = build_chart(df, {"type": "bar", "x_col": "week", "y_col": "nonexistent"})
    assert isinstance(fig, go.Figure)
    # Should use second column (value) as y
    assert list(fig.data[0].y) == [10.5, 12.3, 11.0, 14.2]


def test_line_chart_uses_rappi_orange():
    df = _sample_df()
    fig = build_chart(df, {"type": "line", "x_col": "week", "y_col": "value"})
    assert fig.data[0].line.color == RAPPI_ORANGE


def test_bar_chart_uses_rappi_orange():
    df = _sample_df_zones()
    fig = build_chart(df, {"type": "bar", "x_col": "zone", "y_col": "metric"})
    assert fig.data[0].marker.color == RAPPI_ORANGE


def test_layout_white_background():
    df = _sample_df()
    fig = build_chart(df, {"type": "line", "x_col": "week", "y_col": "value"})
    assert fig.layout.plot_bgcolor == "white"
    assert fig.layout.paper_bgcolor == "white"
