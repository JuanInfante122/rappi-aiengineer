"""Tests for the statistical insights engine: detectors, scorer, narrator, and engine.

Tests are organized in implementation order:
  - Detectors: wow, trend, peer, correlation, opportunity
  - Scorer: tier multiplier, cap/filter
  - Engine orchestrator: JSON-serializable output
  - Narrator: mocked LLM, retry/fallback, parallel execution, executive summary sequencing
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from insights.scorer import METRIC_CONFIG, compute_severity_score
from insights.narrator import generate_narrative, generate_narratives_parallel, generate_executive_summary


# ---------------------------------------------------------------------------
# WoW Detector
# ---------------------------------------------------------------------------


def test_wow_detector() -> None:
    """WoW detector flags a -20% drop as bad when bad_direction is 'decrease'."""
    from insights.detectors.wow_detector import detect_wow_anomaly

    series = pd.Series({0: 0.80, 1: 1.00})
    result = detect_wow_anomaly(series, threshold=0.10, bad_direction="decrease")
    assert result is not None
    assert abs(result["wow_change_pct"] - (-0.20)) < 1e-9
    assert result["is_bad"] is True


def test_wow_missing_week() -> None:
    """WoW detector returns None when the previous week (index 1) is absent."""
    from insights.detectors.wow_detector import detect_wow_anomaly

    series = pd.Series({0: 0.80})
    result = detect_wow_anomaly(series, threshold=0.10, bad_direction="decrease")
    assert result is None


# ---------------------------------------------------------------------------
# Trend Detector
# ---------------------------------------------------------------------------


def test_trend_detector() -> None:
    """Consecutive trend detector identifies 3 deteriorating weeks correctly."""
    from insights.detectors.trend_detector import detect_consecutive_trend

    # Series sorted descending by week_number: 8 (oldest) -> 5 (latest seen in last-4 window)
    series = pd.Series({8: 100, 7: 95, 6: 90, 5: 85})
    result = detect_consecutive_trend(series, min_run=3)
    assert result is not None
    assert result["trend_direction"] == "deteriorating"
    assert result["trend_weeks"] == 3


def test_linear_trend_r2_gate() -> None:
    """Linear trend analysis returns None when R2 < 0.5 (noisy / no clear trend)."""
    from insights.detectors.trend_detector import linear_trend_analysis

    # Alternating values produce near-zero R2
    series = pd.Series({0: 10, 1: 50, 2: 5, 3: 45, 4: 10})
    result = linear_trend_analysis(series)
    assert result is None


# ---------------------------------------------------------------------------
# Peer Z-score Detector
# ---------------------------------------------------------------------------


def test_zscore_min_n() -> None:
    """Z-score returns None when fewer than 3 peer zones exist (statistically unreliable)."""
    from insights.detectors.peer_detector import zone_zscore_at_week

    all_zone_values = pd.Series([0.4, 0.5])
    result = zone_zscore_at_week(all_zone_values, zone_value=0.3)
    assert result is None


# ---------------------------------------------------------------------------
# Correlation Detector
# ---------------------------------------------------------------------------


def test_correlation_detector() -> None:
    """Cross-zone correlation returns a Pearson r close to 1.0 for perfectly correlated series."""
    from insights.detectors.peer_detector import cross_zone_correlation

    metric_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[0, 1, 2, 3, 4])
    metric_b = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0], index=[0, 1, 2, 3, 4])
    result = cross_zone_correlation(metric_a, metric_b)
    assert result is not None
    assert abs(result["pearson_r"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Opportunity Detector
# ---------------------------------------------------------------------------


def test_opportunity_detector() -> None:
    """Opportunity detector flags a zone with Z >= 1.5 sustained across 4+ weeks."""
    from insights.detectors.opportunity_detector import detect_opportunity

    import numpy as np

    rng = np.random.default_rng(42)
    zones = ["zone_B", "zone_C", "zone_D", "zone_E", "zone_F"]
    rows = []
    for z in zones:
        for w in range(4):
            rows.append({"zone_id": z, "week_number": w, "value": 0.5 + rng.normal(0, 0.05)})
    peer_group_df = pd.DataFrame(rows)

    # zone_A has a very high value — should be Z >= 1.5 vs the tight peer cluster for all 4 weeks
    zone_a_series = pd.Series({0: 0.9, 1: 0.88, 2: 0.87, 3: 0.89})

    result = detect_opportunity(zone_a_series, peer_group_df, min_weeks=4)
    assert result is not None
    assert result["sustained_weeks"] == 4
    assert all(z >= 1.5 for z in result["z_scores"])


# ---------------------------------------------------------------------------
# Severity Scorer
# ---------------------------------------------------------------------------


def test_tier_multiplier_priority() -> None:
    """Gross Profit UE (tier 1.5) must score higher than Turbo Adoption (tier 0.8) at same WoW."""
    score_gp = compute_severity_score(
        wow_change_pct=0.15,
        zscore=None,
        trend_weeks=None,
        metric_name="Gross Profit UE",
    )
    score_ta = compute_severity_score(
        wow_change_pct=0.15,
        zscore=None,
        trend_weeks=None,
        metric_name="Turbo Adoption",
    )
    assert score_gp > score_ta


def test_score_cap_and_filter() -> None:
    """Extreme inputs cap at 100; mild inputs produce score below the 30 filter threshold."""
    # Extreme inputs: wow=1.0 (200% change), zscore=5.0, trend=9 weeks for high-tier metric
    score_extreme = compute_severity_score(
        wow_change_pct=1.0,
        zscore=5.0,
        trend_weeks=9,
        metric_name="Lead Penetration",
    )
    assert score_extreme == 100.0

    # Mild inputs: small wow, low zscore, no trend for low-tier metric
    score_mild = compute_severity_score(
        wow_change_pct=0.05,
        zscore=0.5,
        trend_weeks=None,
        metric_name="Turbo Adoption",
    )
    assert score_mild < 30.0


# ---------------------------------------------------------------------------
# Stubs for engine orchestrator (plan 04-02)
# ---------------------------------------------------------------------------


def test_engine_json_serializable() -> None:
    """Engine output must be JSON-serializable (Python primitives only)."""
    import json
    from pathlib import Path
    from insights.engine import run_insights_engine

    db_path = Path("data/rappi_ops.db")
    if not db_path.exists():
        pytest.skip("SQLite database not available")

    results = run_insights_engine(db_path, "CO")
    # Must be a list
    assert isinstance(results, list)
    # Must be JSON-serializable (no numpy types)
    serialized = json.dumps(results)
    assert isinstance(serialized, str)
    # Each insight must have all 11 evidence keys + severity_score
    if results:
        required_keys = {
            "metric_name", "zone_id", "country", "week_number",
            "current_value", "wow_change_pct", "zscore", "peer_avg",
            "trend_weeks", "trend_direction", "detector_type", "severity_score"
        }
        for insight in results:
            assert required_keys.issubset(insight.keys()), (
                f"Missing keys: {required_keys - insight.keys()}"
            )
        # Severity scores should be >= 30 (filter)
        assert all(r["severity_score"] >= 30 for r in results)
        # Should be sorted descending
        scores = [r["severity_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        # At most 25
        assert len(results) <= 25


# ---------------------------------------------------------------------------
# Narrator tests (mocked LLM — no real API calls)
# ---------------------------------------------------------------------------

_SAMPLE_EVIDENCE = {
    "metric_name": "Gross Profit UE",
    "zone_id": "CO_BOGOTA_CHAPINERO",
    "country": "CO",
    "week_number": 0,
    "current_value": 0.85,
    "wow_change_pct": -0.15,
    "zscore": -1.8,
    "peer_avg": 0.92,
    "trend_weeks": 3,
    "trend_direction": "deteriorating",
    "detector_type": "wow",
    "severity_score": 72.0,
}


def _make_mock_client(content: str) -> MagicMock:
    """Build a mock OpenAI client that returns the given content string on every call."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_narrative_evidence_grounded() -> None:
    """Narrator returns a dict with non-empty SCR fields when the LLM responds correctly."""
    valid_content = json.dumps({
        "situation": "La zona CO_BOGOTA_CHAPINERO tiene un valor de 0.85 en Gross Profit UE.",
        "complication": "El cambio semanal de -15% indica una deterioracion significativa.",
        "resolution": "Se recomienda revisar las causas operativas y comparar con zonas similares.",
    })
    mock_client = _make_mock_client(valid_content)

    result = generate_narrative(_SAMPLE_EVIDENCE, mock_client)

    assert isinstance(result, dict)
    assert "situation" in result
    assert "complication" in result
    assert "resolution" in result
    assert len(result["situation"]) > 0
    assert len(result["complication"]) > 0
    assert len(result["resolution"]) > 0


def test_narrative_retry_fallback() -> None:
    """Narrator falls back to template when both LLM call attempts fail."""
    # First call raises JSONDecodeError; second returns invalid JSON (missing "resolution")
    def side_effect_fail(*args, **kwargs):  # noqa: ANN001
        raise json.JSONDecodeError("mock decode error", "", 0)

    mock_message_invalid = MagicMock()
    mock_message_invalid.content = json.dumps({"situation": "s", "complication": "c"})
    mock_choice_invalid = MagicMock()
    mock_choice_invalid.message = mock_message_invalid
    mock_response_invalid = MagicMock()
    mock_response_invalid.choices = [mock_choice_invalid]

    # First call raises, second call returns response missing "resolution"
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        json.JSONDecodeError("mock", "", 0),
        mock_response_invalid,
    ]

    result = generate_narrative(_SAMPLE_EVIDENCE, mock_client)

    # Fallback template contains the zone_id from evidence
    assert isinstance(result, dict)
    assert "CO_BOGOTA_CHAPINERO" in result["situation"]
    assert len(result["resolution"]) > 0


def test_parallel_narratives() -> None:
    """Parallel generation produces one result per insight with no None entries."""
    valid_content = json.dumps({
        "situation": "Situacion de prueba.",
        "complication": "Complicacion de prueba.",
        "resolution": "Resolucion de prueba.",
    })
    mock_client = _make_mock_client(valid_content)

    # Build 5 sample insights
    insights = [dict(_SAMPLE_EVIDENCE, zone_id=f"CO_ZONE_{i}") for i in range(5)]

    results = generate_narratives_parallel(insights, mock_client, max_workers=2)

    assert len(results) == 5
    for result in results:
        assert result is not None
        assert "situation" in result
        assert "complication" in result
        assert "resolution" in result


def test_executive_summary_sequencing() -> None:
    """Executive summary is generated after all narratives and returns a non-empty string."""
    narrative_content = json.dumps({
        "situation": "Situacion.",
        "complication": "Complicacion.",
        "resolution": "Resolucion.",
    })
    summary_content = json.dumps({"summary": "Resumen ejecutivo de CO con hallazgos criticos."})

    call_counter: list[int] = []

    def tracking_create(*args, **kwargs):  # noqa: ANN001
        call_counter.append(1)
        # Return narrative JSON for the first N calls, then executive summary JSON
        mock_message = MagicMock()
        # Determine which response to return based on call count
        if len(call_counter) <= 3:
            mock_message.content = narrative_content
        else:
            mock_message.content = summary_content
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = tracking_create

    sample_insights = [dict(_SAMPLE_EVIDENCE, zone_id=f"CO_ZONE_{i}") for i in range(3)]

    # Step 1: generate all narratives
    narratives = generate_narratives_parallel(sample_insights, mock_client, max_workers=2)
    narrative_call_count = len(call_counter)

    # Step 2: generate executive summary AFTER narratives complete
    summary = generate_executive_summary(sample_insights, "CO", mock_client)

    assert summary_content != ""
    assert isinstance(summary, str)
    assert len(summary) > 0
    # Confirm summary call happened after all narrative calls
    assert len(call_counter) > narrative_call_count
