"""Week-over-week anomaly detector for operational metrics.

Compares the latest week (week_number=0) against the previous week
(week_number=1) to detect significant percentage changes.
"""

from __future__ import annotations

import math

import pandas as pd


def detect_wow_anomaly(
    series: pd.Series,
    threshold: float = 0.10,
    bad_direction: str = "decrease",
) -> dict | None:
    """Detect a statistically significant week-over-week change.

    Compares index 0 (latest week) against index 1 (previous week).
    Returns None when data is insufficient or the change is within threshold.

    Args:
        series: Metric values indexed by week_number (0 = latest, 1 = previous).
        threshold: Minimum absolute WoW fractional change to report (default 10%).
        bad_direction: "decrease" means a drop is bad; "rise" means an increase
            is bad (used for Restaurants Markdowns / GMV).

    Returns:
        Dict with keys "wow_change_pct" (signed float) and "is_bad" (bool),
        or None if comparison is not possible or change is below threshold.
    """
    if 0 not in series.index or 1 not in series.index:
        return None

    current = series[0]
    previous = series[1]

    if math.isnan(current) or math.isnan(previous):
        return None

    if previous == 0:
        return None

    wow = (current - previous) / abs(previous)

    if abs(wow) <= threshold:
        return None

    is_bad = (wow < 0 and bad_direction == "decrease") or (wow > 0 and bad_direction == "rise")
    return {"wow_change_pct": float(wow), "is_bad": is_bad}
