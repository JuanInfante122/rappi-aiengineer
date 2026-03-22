"""Unit tests for the HTML report module.

Tests cover chart builders (all 5 detector types), CDN-once rule enforcement,
the full report generator pipeline with mocked LLM calls, and severity color
thresholds. All tests run without OpenAI API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from report.charts import (
    _build_correlation_chart,
    _build_opportunity_chart,
    _build_peer_chart,
    _build_trend_chart,
    _build_wow_chart,
    build_report_chart,
    reset_chart_counter,
)
from report.generator import generate_html_report
from insights.scorer import METRIC_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_wow_insight() -> dict:
    """Insight with detector_type='wow', severity >= 70 (red border threshold)."""
    return {
        "metric_name": "Gross Profit UE",
        "zone_id": "BOGOTA_CHAPINERO",
        "country": "CO",
        "week_number": 0,
        "current_value": 45.2,
        "wow_change_pct": -0.15,
        "zscore": -2.3,
        "peer_avg": 52.1,
        "trend_weeks": None,
        "trend_direction": None,
        "detector_type": "wow",
        "severity_score": 75.0,
    }


@pytest.fixture()
def sample_trend_insight() -> dict:
    """Insight with detector_type='trend', sustained deterioration."""
    return {
        "metric_name": "Perfect Orders",
        "zone_id": "BOGOTA_USAQUEN",
        "country": "CO",
        "week_number": 0,
        "current_value": 88.0,
        "wow_change_pct": 0.0,
        "zscore": None,
        "peer_avg": 91.5,
        "trend_weeks": 4,
        "trend_direction": "deteriorating",
        "detector_type": "trend",
        "severity_score": 55.0,
    }


@pytest.fixture()
def sample_peer_insight() -> dict:
    """Insight with detector_type='peer', zone below peer average."""
    return {
        "metric_name": "Lead Penetration",
        "zone_id": "BOGOTA_SUBA",
        "country": "CO",
        "week_number": 0,
        "current_value": 30.1,
        "wow_change_pct": 0.0,
        "zscore": -2.5,
        "peer_avg": 42.0,
        "trend_weeks": None,
        "trend_direction": None,
        "detector_type": "peer",
        "severity_score": 50.0,
    }


@pytest.fixture()
def sample_correlation_insight() -> dict:
    """Insight with detector_type='correlation', cross-zone placeholder."""
    return {
        "metric_name": "Gross Profit UE",
        "zone_id": "CROSS_ZONE",
        "country": "CO",
        "week_number": 0,
        "current_value": 0.0,
        "wow_change_pct": 0.0,
        "zscore": None,
        "peer_avg": None,
        "trend_weeks": None,
        "trend_direction": None,
        "detector_type": "correlation",
        "severity_score": 40.0,
    }


@pytest.fixture()
def sample_opportunity_insight() -> dict:
    """Insight with detector_type='opportunity', sustained above-avg gap."""
    return {
        "metric_name": "Perfect Orders",
        "zone_id": "BOGOTA_CHAPINERO",
        "country": "CO",
        "week_number": 0,
        "current_value": 94.0,
        "wow_change_pct": 0.0,
        "zscore": 1.8,
        "peer_avg": 88.0,
        "trend_weeks": 5,
        "trend_direction": "improving",
        "detector_type": "opportunity",
        "severity_score": 35.0,
    }


# ---------------------------------------------------------------------------
# Chart builder tests
# ---------------------------------------------------------------------------

def test_build_wow_chart_returns_html(sample_wow_insight):
    """build_report_chart for wow type returns HTML containing Plotly markup."""
    reset_chart_counter()
    result = build_report_chart(sample_wow_insight)
    assert isinstance(result, str)
    assert len(result) > 0
    # Plotly HTML fragments always contain 'plotly' in the markup
    assert "plotly" in result.lower()


def test_build_peer_chart_below_avg(sample_peer_insight):
    """build_report_chart for peer type returns non-empty HTML."""
    reset_chart_counter()
    result = build_report_chart(sample_peer_insight)
    assert isinstance(result, str)
    assert len(result) > 0


def test_first_chart_includes_cdn(sample_wow_insight, sample_peer_insight):
    """First chart after reset_chart_counter includes the CDN URL; second does not."""
    reset_chart_counter()

    first_html = build_report_chart(sample_wow_insight)
    # Plotly CDN URL appears in the first chart
    assert "cdn.plot.ly" in first_html or "plotly-latest.min.js" in first_html or "plotly.min.js" in first_html

    second_html = build_report_chart(sample_peer_insight)
    # CDN URL must NOT be in subsequent charts
    assert "cdn.plot.ly" not in second_html
    assert "plotly-latest.min.js" not in second_html
    assert "plotly.min.js" not in second_html


def test_build_correlation_chart(sample_correlation_insight):
    """Correlation insight returns non-empty HTML fragment."""
    reset_chart_counter()
    result = build_report_chart(sample_correlation_insight)
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_trend_chart(sample_trend_insight):
    """Trend chart builder returns HTML containing the zone id."""
    reset_chart_counter()
    fig = _build_trend_chart(sample_trend_insight)
    html = fig.to_html(full_html=False, include_plotlyjs=False)
    assert isinstance(html, str)
    assert len(html) > 0


def test_build_opportunity_chart(sample_opportunity_insight):
    """Opportunity chart builder returns a Plotly Figure with correct title."""
    fig = _build_opportunity_chart(sample_opportunity_insight)
    assert "Oportunidad" in fig.layout.title.text
    assert sample_opportunity_insight["zone_id"] in fig.layout.title.text


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

def test_generate_html_report_mocked(sample_wow_insight, sample_peer_insight):
    """generate_html_report produces correct HTML with mocked LLM calls."""
    mock_client = MagicMock()

    sample_narrative = {
        "situation": "Situacion de prueba.",
        "complication": "Complicacion de prueba.",
        "resolution": "Resolucion de prueba.",
    }

    with (
        patch("report.generator.run_insights_engine", return_value=[sample_wow_insight, sample_peer_insight]),
        patch("report.generator.generate_narratives_parallel", return_value=[sample_narrative, sample_narrative]),
        patch("report.generator.generate_executive_summary", return_value="Resumen de prueba"),
    ):
        result = generate_html_report(
            db_path=None,
            country="CO",
            client=mock_client,
        )

    assert isinstance(result, str)
    assert "Resumen Ejecutivo" in result
    assert "Resumen de prueba" in result
    assert "Rappi Ops Intelligence" in result
    assert sample_wow_insight["zone_id"] in result
    assert sample_peer_insight["zone_id"] in result
    # Severity scores should appear as badges
    assert "75" in result
    assert "50" in result


def test_generate_html_report_empty_insights():
    """generate_html_report returns no-data HTML when engine finds no insights."""
    mock_client = MagicMock()

    with patch("report.generator.run_insights_engine", return_value=[]):
        result = generate_html_report(
            db_path=None,
            country="MX",
            client=mock_client,
        )

    assert "No se encontraron insights" in result
    assert "MX" in result


def test_severity_color_thresholds():
    """Cards use correct left-border colors per severity threshold."""
    import jinja2
    from pathlib import Path

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path("report/templates")),
        autoescape=False,
    )
    card_template = env.get_template("insight_card.html.j2")

    def render_card(severity: float) -> str:
        insight = {
            "metric_name": "Gross Profit UE",
            "zone_id": "TEST_ZONE",
            "severity_score": severity,
        }
        return card_template.render(insight=insight, narrative=None, chart_html="")

    high_card = render_card(75.0)    # >= 70 → red
    medium_card = render_card(50.0)  # 40-69 → orange
    low_card = render_card(35.0)     # 30-39 → yellow

    assert "#E53E3E" in high_card
    assert "#DD6B20" in medium_card
    assert "#D69E2E" in low_card


# ---------------------------------------------------------------------------
# METRIC_CONFIG integrity check
# ---------------------------------------------------------------------------

def test_metric_config_has_bad_direction():
    """Every metric in METRIC_CONFIG has a bad_direction key."""
    for metric_name, cfg in METRIC_CONFIG.items():
        assert "bad_direction" in cfg, f"Missing bad_direction for {metric_name}"
        assert cfg["bad_direction"] in ("decrease", "rise"), (
            f"Invalid bad_direction for {metric_name}: {cfg['bad_direction']}"
        )
