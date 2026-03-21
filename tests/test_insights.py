"""Tests for the statistical insights engine: detectors, scorer, and stubs for narrator/engine.

Tests are organized in order of implementation:
  - Task 1 (this file): Scorer tests + test scaffold
  - Task 2: Detector implementations that make RED tests go GREEN
  - Plans 02/03: Stubs below are completed when engine orchestrator and narrator are built
"""

from __future__ import annotations

import pandas as pd
import pytest

from insights.scorer import METRIC_CONFIG, compute_severity_score


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
# Stubs for LLM narrator (plan 04-03)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="implemented in plan 04-03")
def test_narrative_evidence_grounded() -> None:
    """Narrator must only cite numbers present in the evidence dict — no hallucinated values."""
    pass


@pytest.mark.skip(reason="implemented in plan 04-03")
def test_narrative_retry_fallback() -> None:
    """Narrator retries once on Pydantic validation failure then returns template fallback."""
    pass


@pytest.mark.skip(reason="implemented in plan 04-03")
def test_parallel_narratives() -> None:
    """ThreadPoolExecutor generates all insight narratives in parallel (max 5 workers)."""
    pass


@pytest.mark.skip(reason="implemented in plan 04-03")
def test_executive_summary_sequencing() -> None:
    """Executive summary is generated only after all individual narratives complete."""
    pass
