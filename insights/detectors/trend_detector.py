"""Consecutive-trend and linear-trend detectors for operational metrics.

Trend analysis operates on a time series indexed by week_number where
0 is the latest week and higher values are older (e.g. 8 = L8W).
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


def detect_consecutive_trend(
    series: pd.Series,
    min_run: int = 3,
) -> dict | None:
    """Detect a sustained consecutive trend in the most recent weeks.

    Checks whether the last min_run values all move in the same direction.
    Index must be week_number (0 = latest, 8 = oldest); the series is sorted
    descending so that the highest index (oldest week) comes first, making
    diffs reflect chronological change from past to present.

    Args:
        series: Metric values indexed by week_number (integers).
        min_run: Minimum number of consecutive weeks required to report.

    Returns:
        Dict with "trend_direction" ("deteriorating" or "improving") and
        "trend_weeks" (int), or None if no sustained trend is found or
        there are fewer than min_run non-NaN data points.
    """
    # Sort descending (highest week_number = oldest first) for chronological order.
    # week_number=0 is the latest, week_number=8 is the oldest in this dataset.
    clean = series.dropna().sort_index(ascending=False)
    if len(clean) < min_run:
        return None

    # Take the first min_run + 1 values (oldest weeks) for the trend window,
    # then check the last min_run diffs for the most recent movement.
    window = clean.iloc[: min_run + 1]
    diffs = window.diff().dropna()

    if len(diffs) < min_run:
        return None

    last_diffs = diffs.iloc[-min_run:]

    if (last_diffs < 0).all():
        return {"trend_direction": "deteriorating", "trend_weeks": min_run}
    if (last_diffs > 0).all():
        return {"trend_direction": "improving", "trend_weeks": min_run}

    return None


def linear_trend_analysis(series: pd.Series) -> dict | None:
    """Fit a linear regression to the series and return slope/R2 if trend is strong.

    Uses scipy linregress on the numeric index values. Only reports a result
    when R2 >= 0.5 — below that threshold the noise dominates the trend.

    Args:
        series: Metric values indexed by week_number (integers, 0 = latest).

    Returns:
        Dict with "slope" (float) and "r2" (float) when R2 >= 0.5,
        or None if fewer than 3 non-NaN points or R2 < 0.5.
    """
    clean = series.dropna()
    if len(clean) < 3:
        return None

    x = clean.index.astype(float).values
    y = clean.values.astype(float)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = r_value ** 2

    if r2 < 0.5:
        return None

    return {"slope": float(slope), "r2": float(r2)}
