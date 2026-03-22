"""Cross-zone peer benchmarking: Z-score and Pearson correlation detectors.

Both functions operate on pre-aggregated zone-level values rather than
raw database rows, keeping the detectors stateless and easily testable.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


def zone_zscore_at_week(
    all_zone_values: pd.Series,
    zone_value: float,
) -> float | None:
    """Compute the Z-score of a single zone relative to all peers at one week.

    Requires at least 3 peer values to produce a meaningful Z-score; fewer
    peers make the mean/std estimates too noisy to trust.

    Args:
        all_zone_values: Values for all zones in the same country+zone_type
            group at a specific week_number (NaN entries are dropped).
        zone_value: The target zone's value to score against the group.

    Returns:
        Signed Z-score (float) where positive means above average,
        or None if fewer than 3 non-NaN peers or the std deviation is 0.
    """
    clean = all_zone_values.dropna()
    if len(clean) < 3:
        return None

    mean = float(clean.mean())
    std = float(clean.std())

    if std == 0:
        return None

    return float((zone_value - mean) / std)


def cross_zone_correlation(
    metric_a_vals: pd.Series,
    metric_b_vals: pd.Series,
) -> dict | None:
    """Compute Pearson correlation between two metric series aligned by shared index.

    Designed for cross-zone comparisons at a single week_number: each index
    entry is a zone_id, and values are the metric readings for those zones.
    Requires at least 3 matched pairs for a reliable correlation estimate.

    Args:
        metric_a_vals: Values for metric A indexed by zone_id.
        metric_b_vals: Values for metric B indexed by zone_id.

    Returns:
        Dict with "pearson_r" (float) and "p_value" (float),
        or None if fewer than 3 aligned non-NaN pairs exist.
    """
    aligned = pd.concat([metric_a_vals, metric_b_vals], axis=1).dropna()
    if len(aligned) < 3:
        return None

    a_vals = aligned.iloc[:, 0].values.astype(float)
    b_vals = aligned.iloc[:, 1].values.astype(float)

    r, p = stats.pearsonr(a_vals, b_vals)
    return {"pearson_r": float(r), "p_value": float(p)}
