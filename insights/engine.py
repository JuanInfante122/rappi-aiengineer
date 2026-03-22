"""Insights engine orchestrator: composes all detectors and scorer into a pipeline.

Loads metric data from SQLite, runs 5 detectors per zone-metric combination,
builds evidence dicts with all required keys, scores and ranks insights, then
returns the top 25 as a JSON-serializable list of Python dicts.

This is the public API consumed by Phase 5 for report generation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from insights.scorer import METRIC_CONFIG, compute_severity_score
from insights.detectors.wow_detector import detect_wow_anomaly
from insights.detectors.trend_detector import detect_consecutive_trend, linear_trend_analysis  # noqa: F401
from insights.detectors.peer_detector import zone_zscore_at_week, cross_zone_correlation
from insights.detectors.opportunity_detector import detect_opportunity

# Metric pairs to test for cross-zone correlation.
# Chosen based on operational hypotheses: GP driven by quality, volume driven by penetration,
# and speed adoption linked to lead conversion.
_CORRELATION_PAIRS = [
    ("Gross Profit UE", "Perfect Orders"),
    ("Lead Penetration", "Orders"),
    ("Turbo Adoption", "Lead Penetration"),
]

# Required keys in every evidence dict — validated in _build_evidence to catch regressions.
_EVIDENCE_KEYS = {
    "metric_name", "zone_id", "country", "week_number",
    "current_value", "wow_change_pct", "zscore", "peer_avg",
    "trend_weeks", "trend_direction", "detector_type",
}


def _load_metric_data(db_path: Path, country: str, metric_name: str) -> pd.DataFrame:
    """Load all rows for a given country and metric from SQLite.

    Args:
        db_path: Path to the SQLite database file.
        country: Country code (e.g., 'CO', 'BR').
        metric_name: Exact metric name matching a key in METRIC_CONFIG.

    Returns:
        DataFrame with columns [zone_id, zone_type, week_number, value].
        Returns an empty DataFrame if no data is found.
    """
    query = (
        "SELECT zone_id, zone_type, week_number, value "
        "FROM raw_input_metrics "
        "WHERE country = ? AND metric_name = ? "
        "ORDER BY zone_id, week_number ASC"
    )
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=(country, metric_name))
    finally:
        conn.close()
    return df


def _build_evidence(
    metric_name: str,
    zone_id: str,
    country: str,
    current_value: float,
    wow_change_pct: float,
    zscore: float | None,
    peer_avg: float | None,
    trend_weeks: int | None,
    trend_direction: str | None,
    detector_type: str,
) -> dict:
    """Build a fully-populated evidence dict for one insight.

    All 11 keys from the evidence contract are required. Numeric values are
    explicitly cast to Python primitives to prevent numpy type leakage that
    would break JSON serialization downstream.

    Args:
        metric_name: Exact metric name string.
        zone_id: Normalized zone identifier string.
        country: Country code string.
        current_value: Latest week metric value (cast to float).
        wow_change_pct: Signed WoW fractional change (0.0 for correlation insights).
        zscore: Cross-zone Z-score, or None if fewer than 3 peers.
        peer_avg: Mean value across same-country same-zone_type peers, or None.
        trend_weeks: Consecutive weeks in same direction, or None.
        trend_direction: "improving" or "deteriorating", or None.
        detector_type: One of "wow", "trend", "peer", "correlation", "opportunity".

    Returns:
        Evidence dict with all 11 required keys present.

    Raises:
        AssertionError: If the built dict is missing any of the 11 required keys.
    """
    evidence = {
        "metric_name": str(metric_name),
        "zone_id": str(zone_id),
        "country": str(country),
        "week_number": int(0),
        "current_value": float(current_value),
        "wow_change_pct": float(wow_change_pct),
        "zscore": float(zscore) if zscore is not None else None,
        "peer_avg": float(peer_avg) if peer_avg is not None else None,
        "trend_weeks": int(trend_weeks) if trend_weeks is not None else None,
        "trend_direction": str(trend_direction) if trend_direction is not None else None,
        "detector_type": str(detector_type),
    }
    assert _EVIDENCE_KEYS.issubset(evidence.keys()), (
        f"Missing keys in evidence dict: {_EVIDENCE_KEYS - evidence.keys()}"
    )
    return evidence


def _run_detectors_for_zone(
    zone_id: str,
    zone_type: str,
    zone_series: pd.Series,
    all_zones_at_week0: pd.Series,
    peer_group_df: pd.DataFrame,
    metric_name: str,
    country: str,
    bad_direction: str,
) -> list[dict]:
    """Run all 5 detectors on a single zone-metric combination.

    Each detector that fires contributes one evidence dict to the output.
    The peer Z-score detector is gated at abs(zscore) > 2.0 — zones within
    2 standard deviations of their peers do not produce peer-type insights.

    Args:
        zone_id: Normalized zone identifier.
        zone_type: Zone type string (e.g., 'Wealthy', 'Non Wealthy').
        zone_series: This zone's metric values indexed by week_number.
        all_zones_at_week0: All zone values at week_number=0 indexed by zone_id.
        peer_group_df: DataFrame [zone_id, week_number, value] for all zones
            in the same country+zone_type group.
        metric_name: Metric name string.
        country: Country code string.
        bad_direction: "decrease" or "rise" from METRIC_CONFIG.

    Returns:
        List of evidence dicts (one per triggered detector, can be empty).
    """
    results: list[dict] = []

    # Skip zones that have no week 0 value.
    if 0 not in zone_series.index:
        return results

    current_value = zone_series[0]
    peer_avg = float(all_zones_at_week0.mean()) if len(all_zones_at_week0) > 0 else None

    # --- WoW anomaly detector ---
    wow_result = detect_wow_anomaly(zone_series, threshold=0.10, bad_direction=bad_direction)
    if wow_result is not None:
        results.append(_build_evidence(
            metric_name=metric_name,
            zone_id=zone_id,
            country=country,
            current_value=current_value,
            wow_change_pct=wow_result["wow_change_pct"],
            zscore=None,
            peer_avg=peer_avg,
            trend_weeks=None,
            trend_direction=None,
            detector_type="wow",
        ))

    # --- Consecutive trend detector ---
    trend_result = detect_consecutive_trend(zone_series, min_run=3)
    if trend_result is not None:
        results.append(_build_evidence(
            metric_name=metric_name,
            zone_id=zone_id,
            country=country,
            current_value=current_value,
            wow_change_pct=0.0,
            zscore=None,
            peer_avg=peer_avg,
            trend_weeks=trend_result["trend_weeks"],
            trend_direction=trend_result["trend_direction"],
            detector_type="trend",
        ))

    # --- Peer Z-score detector (gated at abs(zscore) > 2.0) ---
    zscore = zone_zscore_at_week(all_zones_at_week0, float(current_value))
    if zscore is not None and abs(zscore) > 2.0:
        results.append(_build_evidence(
            metric_name=metric_name,
            zone_id=zone_id,
            country=country,
            current_value=current_value,
            wow_change_pct=0.0,
            zscore=zscore,
            peer_avg=peer_avg,
            trend_weeks=None,
            trend_direction=None,
            detector_type="peer",
        ))

    # --- Opportunity detector ---
    opportunity_result = detect_opportunity(zone_series, peer_group_df, min_weeks=4)
    if opportunity_result is not None:
        results.append(_build_evidence(
            metric_name=metric_name,
            zone_id=zone_id,
            country=country,
            current_value=current_value,
            wow_change_pct=0.0,
            zscore=zscore,
            peer_avg=peer_avg,
            trend_weeks=opportunity_result["sustained_weeks"],
            trend_direction="improving",
            detector_type="opportunity",
        ))

    return results


def _run_correlations(db_path: Path, country: str) -> list[dict]:
    """Run cross-zone Pearson correlation for pre-defined metric pairs.

    Correlations are computed at week_number=0 across all zones. For the
    "Orders" metric, the raw_orders table is queried instead of
    raw_input_metrics because orders are stored separately.

    Only pairs with p_value < 0.05 and abs(pearson_r) > 0.5 generate insights.
    Correlation evidence uses placeholder values (current_value=0.0,
    wow_change_pct=0.0) because correlations are cross-zone, not zone-specific.

    Args:
        db_path: Path to the SQLite database file.
        country: Country code string.

    Returns:
        List of correlation evidence dicts.
    """
    results: list[dict] = []

    for metric_a, metric_b in _CORRELATION_PAIRS:
        # Load metric A at week 0 — aggregate by zone_id to ensure a unique index.
        df_a = _load_metric_data(db_path, country, metric_a)
        df_a_w0 = df_a[df_a["week_number"] == 0].groupby("zone_id")["value"].mean()

        # Load metric B — orders come from raw_orders table.
        if metric_b == "Orders":
            conn = sqlite3.connect(db_path)
            try:
                query = (
                    "SELECT zone_id, value FROM raw_orders "
                    "WHERE country = ? AND week_number = 0"
                )
                df_b_raw = pd.read_sql_query(query, conn, params=(country,))
            finally:
                conn.close()
            df_b_w0 = df_b_raw.groupby("zone_id")["value"].mean()
        else:
            df_b = _load_metric_data(db_path, country, metric_b)
            df_b_w0 = df_b[df_b["week_number"] == 0].groupby("zone_id")["value"].mean()

        corr_result = cross_zone_correlation(df_a_w0, df_b_w0)
        if corr_result is None:
            continue

        pearson_r = corr_result["pearson_r"]
        p_value = corr_result["p_value"]

        if p_value < 0.05 and abs(pearson_r) > 0.5:
            results.append(_build_evidence(
                metric_name=metric_a,
                zone_id="CROSS_ZONE",
                country=country,
                current_value=0.0,
                wow_change_pct=0.0,
                zscore=None,
                peer_avg=None,
                trend_weeks=None,
                trend_direction=None,
                detector_type="correlation",
            ))

    return results


def _serialize_primitives(value: object) -> object:
    """Cast numpy scalar types to Python primitives for JSON serialization.

    pandas/numpy operations produce numpy.float64 and numpy.int64 values that
    are not accepted by json.dumps. This function recursively sanitizes a value
    so that dicts and lists of dicts coming out of the engine are safe to
    serialize with the standard library json module.

    Args:
        value: Any Python value, potentially a numpy scalar, dict, list, or None.

    Returns:
        The value converted to a plain Python primitive (int, float, str, bool,
        None, list, or dict).
    """
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _serialize_primitives(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_primitives(v) for v in value]
    return value


def run_insights_engine(db_path: Path, country: str) -> list[dict]:
    """Run the full insights pipeline for one country and return ranked results.

    Iterates all 13 metrics, applies all 5 detectors to each zone-metric pair,
    runs cross-zone correlations, scores each candidate insight, and returns the
    top 25 with severity_score >= 30 sorted descending.

    Args:
        db_path: Path to the SQLite database file (must exist and contain
            raw_input_metrics and raw_orders tables).
        country: Country code to analyze (e.g., 'CO', 'BR', 'AR').

    Returns:
        List of up to 25 insight dicts, each containing all 11 evidence keys
        plus "severity_score". Sorted by severity_score descending.
        All values are JSON-serializable Python primitives.
    """
    all_evidence: list[dict] = []

    for metric_name, metric_cfg in METRIC_CONFIG.items():
        bad_direction = metric_cfg["bad_direction"]

        df = _load_metric_data(db_path, country, metric_name)
        if df.empty:
            continue

        # Cross-zone snapshot at week 0 for peer benchmarking.
        # Aggregate per zone_id to ensure a unique index for Z-score computation.
        week0_mask = df["week_number"] == 0
        all_zones_at_week0 = df[week0_mask].groupby("zone_id")["value"].mean()

        for zone_id in df["zone_id"].unique():
            zone_mask = df["zone_id"] == zone_id
            zone_df = df[zone_mask]
            # Aggregate duplicate week entries (same zone can appear multiple times
            # in the raw data); use mean so the series index is unique.
            zone_series = zone_df.groupby("week_number")["value"].mean()

            # Build peer group for opportunity detector: same zone_type
            if not zone_df.empty:
                zone_type = zone_df["zone_type"].iloc[0]
                peer_group_df = df[df["zone_type"] == zone_type][
                    ["zone_id", "week_number", "value"]
                ]
            else:
                zone_type = ""
                peer_group_df = pd.DataFrame(columns=["zone_id", "week_number", "value"])

            zone_evidence = _run_detectors_for_zone(
                zone_id=zone_id,
                zone_type=zone_type,
                zone_series=zone_series,
                all_zones_at_week0=all_zones_at_week0,
                peer_group_df=peer_group_df,
                metric_name=metric_name,
                country=country,
                bad_direction=bad_direction,
            )
            all_evidence.extend(zone_evidence)

    # Correlation insights across all zones
    correlation_evidence = _run_correlations(db_path, country)
    all_evidence.extend(correlation_evidence)

    # Score each insight
    for ev in all_evidence:
        ev["severity_score"] = compute_severity_score(
            wow_change_pct=ev["wow_change_pct"],
            zscore=ev["zscore"],
            trend_weeks=ev["trend_weeks"],
            metric_name=ev["metric_name"],
        )

    # Filter at severity >= 30, sort descending, keep top 25
    filtered = [ev for ev in all_evidence if ev["severity_score"] >= 30]
    filtered.sort(key=lambda x: x["severity_score"], reverse=True)
    top25 = filtered[:25]

    # Final serialization pass: eliminate any residual numpy scalars
    return [_serialize_primitives(ev) for ev in top25]
