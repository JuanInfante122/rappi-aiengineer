"""Opportunity detector: finds zones that consistently outperform their peers.

An "opportunity" is a zone that maintains a Z-score >= 1.5 standard deviations
above its same-country, same-zone_type peers for at least 4 consecutive weeks.
These zones are candidates for best-practice sharing with underperforming peers.
"""

from __future__ import annotations

import pandas as pd

from insights.detectors.peer_detector import zone_zscore_at_week


def detect_opportunity(
    zone_series: pd.Series,
    peer_group_df: pd.DataFrame,
    min_weeks: int = 4,
) -> dict | None:
    """Detect a zone that sustainably outperforms its peers across multiple weeks.

    Checks whether the target zone's Z-score vs its peer group is >= 1.5 for
    each of the most recent min_weeks weeks. All weeks must hold — a single
    week below threshold breaks the streak.

    Args:
        zone_series: Target zone's metric values indexed by week_number
            (0 = latest). Values for weeks 0 through min_weeks-1 are used.
        peer_group_df: DataFrame with columns [zone_id, week_number, value]
            containing all peers in the same country+zone_type group.
        min_weeks: Number of consecutive recent weeks that must all exceed
            the Z >= 1.5 threshold (default 4).

    Returns:
        Dict with "z_scores" (list of floats, one per week) and
        "sustained_weeks" (int equal to min_weeks), or None if the
        divergence is not sustained across all required weeks.
    """
    z_scores: list[float] = []

    for week in range(min_weeks):
        if week not in zone_series.index:
            return None

        zone_val = zone_series[week]
        week_peers = peer_group_df[peer_group_df["week_number"] == week]["value"]

        z = zone_zscore_at_week(week_peers.reset_index(drop=True), zone_value=float(zone_val))
        if z is None or z < 1.5:
            return None

        z_scores.append(z)

    return {"z_scores": z_scores, "sustained_weeks": min_weeks}
