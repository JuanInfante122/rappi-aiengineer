"""Severity scorer for the insights engine.

Converts statistical detector outputs (WoW change, Z-score, trend weeks)
into a single severity score (0-100) weighted by metric business tier.
"""

from __future__ import annotations

# Metric configuration: tier multiplier and bad_direction for each of the 13 metrics.
# Tier reflects business priority (higher = more important to Operations).
# bad_direction: "decrease" means a drop is bad (most metrics); "rise" means an
# increase is bad (only Restaurants Markdowns / GMV — higher markdown = margin erosion).
METRIC_CONFIG: dict[str, dict] = {
    "Gross Profit UE": {
        "tier": 1.5,
        "bad_direction": "decrease",
    },
    "Perfect Orders": {
        "tier": 1.5,
        "bad_direction": "decrease",
    },
    "Lead Penetration": {
        "tier": 1.5,
        "bad_direction": "decrease",
    },
    "% PRO Users Who Breakeven": {
        "tier": 1.0,
        "bad_direction": "decrease",
    },
    "% Restaurants Sessions With Optimal Assortment": {
        "tier": 1.0,
        "bad_direction": "decrease",
    },
    "Non-Pro PTC > OP": {
        "tier": 1.0,
        "bad_direction": "decrease",
    },
    "Pro Adoption (Last Week Status)": {
        "tier": 1.0,
        "bad_direction": "decrease",
    },
    "Restaurants Markdowns / GMV": {
        "tier": 1.0,
        "bad_direction": "rise",
    },
    "Restaurants SS > ATC CVR": {
        "tier": 1.0,
        "bad_direction": "decrease",
    },
    "Restaurants SST > SS CVR": {
        "tier": 1.0,
        "bad_direction": "decrease",
    },
    "Turbo Adoption": {
        "tier": 0.8,
        "bad_direction": "decrease",
    },
    "Retail SST > SS CVR": {
        "tier": 0.8,
        "bad_direction": "decrease",
    },
    "MLTV Top Verticals Adoption": {
        "tier": 0.8,
        "bad_direction": "decrease",
    },
}


def compute_severity_score(
    wow_change_pct: float | None,
    zscore: float | None,
    trend_weeks: int | None,
    metric_name: str,
) -> float:
    """Compute a weighted severity score (0-100) for a single insight.

    Combines three statistical signals into a base score, then applies a
    metric-tier multiplier. The tier reflects business importance so that
    a Gross Profit UE drop ranks higher than a Turbo Adoption drop of
    identical magnitude.

    Args:
        wow_change_pct: Week-over-week fractional change (e.g. -0.15 = -15%).
            Pass None if WoW data is unavailable.
        zscore: Cross-zone Z-score at current week. Pass None if fewer than
            3 peers exist (zone_zscore_at_week returns None in that case).
        trend_weeks: Number of consecutive deteriorating/improving weeks from
            detect_consecutive_trend. Pass None if no sustained trend found.
        metric_name: Exact metric name from METRIC_CONFIG keys.

    Returns:
        Severity score in [0.0, 100.0]. Higher = more urgent.

    Raises:
        KeyError: If metric_name is not in METRIC_CONFIG.
    """
    wow_component = min(abs(wow_change_pct) / 0.50, 1.0) if wow_change_pct is not None else 0.0
    zscore_component = min(abs(zscore) / 3.0, 1.0) if zscore is not None else 0.0
    trend_component = min(trend_weeks / 9.0, 1.0) if trend_weeks is not None else 0.0

    base = (wow_component * 0.40 + zscore_component * 0.40 + trend_component * 0.20) * 100
    tier = METRIC_CONFIG[metric_name]["tier"]
    return min(base * tier, 100.0)
