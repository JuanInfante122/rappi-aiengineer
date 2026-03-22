"""Report-specific Plotly chart builders for the HTML insights report.

Maps each detector type to a tailored chart that best communicates the
statistical finding. Tracks a module-level flag to ensure Plotly CDN
is included exactly once per generated report.
"""

from __future__ import annotations

import plotly.graph_objects as go

from charts.renderer import RAPPI_ORANGE, RAPPI_TEAL
from insights.scorer import METRIC_CONFIG

# Module-level flag: True after the first chart has been rendered in the
# current report generation pass. Reset with reset_chart_counter() at the
# start of each new report so each HTML file loads Plotly CDN exactly once.
_first_chart_rendered: bool = False

_DETECTOR_MAP: dict[str, object] = {}  # populated after function definitions

_BASE_LAYOUT = dict(
    template="plotly_white",
    height=280,
    margin=dict(l=40, r=20, t=40, b=30),
    font=dict(size=11),
)


def reset_chart_counter() -> None:
    """Reset the CDN-tracking flag at the start of each report generation.

    Must be called once before building any charts for a new report so that
    the first chart receives ``include_plotlyjs="cdn"`` and all subsequent
    charts receive ``include_plotlyjs=False``.
    """
    global _first_chart_rendered
    _first_chart_rendered = False


def build_report_chart(insight: dict) -> str:
    """Build an HTML fragment for a Plotly chart matching the insight detector type.

    Dispatches to the appropriate builder based on ``insight["detector_type"]``.
    Tracks whether this is the first chart in the report to include the Plotly
    CDN script exactly once. Returns an empty string on any error so the card
    renders without a chart rather than crashing the report.

    Args:
        insight: Insight evidence dict from run_insights_engine. Must contain
            "detector_type" and the fields relevant to the chart type.

    Returns:
        HTML string fragment (without full HTML boilerplate). Empty string on error.
    """
    global _first_chart_rendered

    try:
        detector = insight.get("detector_type", "wow")
        builder = _DETECTOR_MAP.get(detector, _build_wow_chart)
        fig = builder(insight)

        include_js = "cdn" if not _first_chart_rendered else False
        _first_chart_rendered = True

        return fig.to_html(full_html=False, include_plotlyjs=include_js)
    except Exception:
        return ""


def _build_wow_chart(insight: dict) -> go.Figure:
    """Build a point-and-reference chart for week-over-week anomaly insights.

    Shows the anomalous current value as a red diamond marker at L0W
    and an optional dashed peer-average reference line.

    Args:
        insight: Insight dict with at least "current_value", "metric_name",
            "zone_id", and optionally "peer_avg".

    Returns:
        Configured Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=["L0W"],
        y=[insight["current_value"]],
        mode="markers",
        marker=dict(color="red", size=14, symbol="diamond"),
        name="Valor actual",
        hovertemplate="L0W: %{y}<extra></extra>",
    ))

    if insight.get("peer_avg") is not None:
        fig.add_hline(
            y=insight["peer_avg"],
            line_dash="dash",
            line_color=RAPPI_TEAL,
            annotation_text="Promedio peers",
        )

    fig.update_layout(
        title=f"{insight['metric_name']} - {insight['zone_id']}",
        **_BASE_LAYOUT,
    )
    return fig


def _build_trend_chart(insight: dict) -> go.Figure:
    """Build a point chart highlighting a sustained trend for a zone-metric.

    Shows the current value at L0W with an annotation indicating the trend
    direction and duration. Adds a peer-average reference line when available.

    Args:
        insight: Insight dict with "current_value", "metric_name", "zone_id",
            "trend_direction", "trend_weeks", and optionally "peer_avg".

    Returns:
        Configured Plotly Figure.
    """
    fig = go.Figure()

    trend_label = (
        f"{insight.get('trend_direction', '')} "
        f"({insight.get('trend_weeks', 0)} semanas)"
    )

    fig.add_trace(go.Scatter(
        x=["L0W"],
        y=[insight["current_value"]],
        mode="markers+text",
        marker=dict(color=RAPPI_ORANGE, size=14),
        text=[trend_label],
        textposition="top center",
        name="Tendencia",
        hovertemplate="L0W: %{y}<extra></extra>",
    ))

    if insight.get("peer_avg") is not None:
        fig.add_hline(
            y=insight["peer_avg"],
            line_dash="dash",
            line_color=RAPPI_TEAL,
            annotation_text="Promedio peers",
        )

    fig.update_layout(
        title=f"{insight['metric_name']} tendencia - {insight['zone_id']}",
        **_BASE_LAYOUT,
    )
    return fig


def _build_peer_chart(insight: dict) -> go.Figure:
    """Build a horizontal bar chart comparing a zone to its peers.

    Bar color reflects performance direction: orange (bad) when the zone is
    on the wrong side of the peer average, teal (good) when it is on the
    right side. The bad direction is looked up from METRIC_CONFIG so that
    "Restaurants Markdowns / GMV" is handled correctly (higher = bad).

    Args:
        insight: Insight dict with "current_value", "metric_name", "zone_id",
            and "peer_avg".

    Returns:
        Configured Plotly Figure.
    """
    bad_direction = METRIC_CONFIG.get(insight["metric_name"], {}).get("bad_direction", "decrease")
    current = insight["current_value"]
    peer_avg = insight.get("peer_avg", 0.0) or 0.0

    if bad_direction == "decrease":
        # Below average is bad for most metrics
        bar_color = RAPPI_ORANGE if current < peer_avg else RAPPI_TEAL
    else:
        # Above average is bad (e.g. Restaurants Markdowns / GMV)
        bar_color = RAPPI_ORANGE if current > peer_avg else RAPPI_TEAL

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[current],
        y=[insight["zone_id"]],
        orientation="h",
        marker_color=bar_color,
        name=insight["zone_id"],
        hovertemplate="%{x}<extra></extra>",
    ))

    if insight.get("peer_avg") is not None:
        fig.add_vline(
            x=insight["peer_avg"],
            line_dash="dash",
            line_color="#666",
            annotation_text="Promedio peers",
        )

    fig.update_layout(
        title=f"{insight['metric_name']} vs Peers",
        **_BASE_LAYOUT,
    )
    return fig


def _build_correlation_chart(insight: dict) -> go.Figure:
    """Build a placeholder scatter for cross-zone correlation insights.

    Correlation insights span all zones (zone_id="CROSS_ZONE") so no
    per-zone time series is available. A minimal placeholder marker with
    an annotation communicates the finding instead.

    Args:
        insight: Insight dict with "metric_name".

    Returns:
        Configured Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker=dict(size=1, color=RAPPI_ORANGE),
        showlegend=False,
    ))

    fig.add_annotation(
        x=0,
        y=0,
        text=(
            f"Correlacion significativa detectada entre zonas "
            f"para {insight['metric_name']}"
        ),
        showarrow=False,
        font=dict(size=12),
        bgcolor="#FFF9E6",
        bordercolor="#D69E2E",
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        title=f"Correlacion: {insight['metric_name']}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        **_BASE_LAYOUT,
    )
    return fig


def _build_opportunity_chart(insight: dict) -> go.Figure:
    """Build a point chart annotating a sustained improvement opportunity.

    Shows the current value at L0W and annotates the number of weeks the
    gap to peers has been sustained. Adds a peer-average reference when
    available.

    Args:
        insight: Insight dict with "current_value", "metric_name", "zone_id",
            "trend_weeks", and optionally "peer_avg".

    Returns:
        Configured Plotly Figure.
    """
    fig = go.Figure()

    gap_label = f"Brecha sostenida {insight.get('trend_weeks', 0)} semanas"

    fig.add_trace(go.Scatter(
        x=["L0W"],
        y=[insight["current_value"]],
        mode="markers+text",
        marker=dict(color=RAPPI_TEAL, size=14),
        text=[gap_label],
        textposition="top center",
        name="Oportunidad",
        hovertemplate="L0W: %{y}<extra></extra>",
    ))

    if insight.get("peer_avg") is not None:
        fig.add_hline(
            y=insight["peer_avg"],
            line_dash="dash",
            line_color=RAPPI_ORANGE,
            annotation_text="Promedio peers",
        )

    fig.update_layout(
        title=f"Oportunidad: {insight['metric_name']} - {insight['zone_id']}",
        **_BASE_LAYOUT,
    )
    return fig


# Populate dispatcher map after function definitions
_DETECTOR_MAP = {
    "wow": _build_wow_chart,
    "trend": _build_trend_chart,
    "peer": _build_peer_chart,
    "correlation": _build_correlation_chart,
    "opportunity": _build_opportunity_chart,
}
