"""Automated verification of ETL correctness.

Tests verify that the Excel-to-SQLite pipeline produces the expected
long-format schema with correct row counts, metric names, zone_id
formatting, week_number ranges, and performance indexes.

Expected counts come from direct Excel inspection (2026-03-20):
- RAW_INPUT_METRICS: 12,573 wide rows x 9 week columns = 113,157 long rows
- RAW_ORDERS: 1,242 wide rows x 9 week columns = 11,178 long rows
"""

import re

from tests.conftest import db_conn  # noqa: F401 — imported for pytest fixture


EXPECTED_METRIC_NAMES = sorted([
    '% PRO Users Who Breakeven',
    '% Restaurants Sessions With Optimal Assortment',
    'Gross Profit UE',
    'Lead Penetration',
    'MLTV Top Verticals Adoption',
    'Non-Pro PTC > OP',
    'Perfect Orders',
    'Pro Adoption (Last Week Status)',
    'Restaurants Markdowns / GMV',
    'Restaurants SS > ATC CVR',
    'Restaurants SST > SS CVR',
    'Retail SST > SS CVR',
    'Turbo Adoption',
])


def test_metrics_row_count(db_conn) -> None:
    """ETL produces exactly 113,157 rows in raw_input_metrics."""
    cursor = db_conn.execute("SELECT COUNT(*) FROM raw_input_metrics")
    count = cursor.fetchone()[0]
    assert count == 113157, f"Expected 113157 rows, got {count}"


def test_orders_row_count(db_conn) -> None:
    """ETL produces exactly 11,178 rows in raw_orders."""
    cursor = db_conn.execute("SELECT COUNT(*) FROM raw_orders")
    count = cursor.fetchone()[0]
    assert count == 11178, f"Expected 11178 rows, got {count}"


def test_metric_names_count(db_conn) -> None:
    """raw_input_metrics contains exactly 13 distinct metric names."""
    cursor = db_conn.execute(
        "SELECT COUNT(DISTINCT metric_name) FROM raw_input_metrics"
    )
    count = cursor.fetchone()[0]
    assert count == 13, f"Expected 13 distinct metrics, got {count}"


def test_metric_names_exact(db_conn) -> None:
    """All 13 metric names match the verified list from direct Excel inspection."""
    cursor = db_conn.execute(
        "SELECT DISTINCT metric_name FROM raw_input_metrics ORDER BY metric_name"
    )
    actual = sorted([row[0] for row in cursor.fetchall()])
    assert actual == EXPECTED_METRIC_NAMES, (
        f"Metric name mismatch.\nExpected: {EXPECTED_METRIC_NAMES}\nActual: {actual}"
    )


def test_countries_count(db_conn) -> None:
    """raw_input_metrics spans exactly 9 countries."""
    cursor = db_conn.execute(
        "SELECT COUNT(DISTINCT country) FROM raw_input_metrics"
    )
    count = cursor.fetchone()[0]
    assert count == 9, f"Expected 9 countries, got {count}"


def test_zone_id_format(db_conn) -> None:
    """All zone_id values are uppercase with underscores only — no spaces or hyphens.

    Validates that build_zone_id() applied .upper().replace(' ', '_').replace('-', '_')
    consistently to all rows, enabling reliable LIKE '%TERM%' queries.
    """
    zone_id_pattern = re.compile(r'^[A-Z0-9_]+$')
    cursor = db_conn.execute("SELECT DISTINCT zone_id FROM raw_input_metrics")
    zone_ids = [row[0] for row in cursor.fetchall()]
    assert len(zone_ids) > 0, "No zone_ids found in raw_input_metrics"
    invalid = [z for z in zone_ids if not zone_id_pattern.match(z)]
    assert not invalid, (
        f"Found {len(invalid)} zone_ids with invalid format: {invalid[:10]}"
    )


def test_week_number_range(db_conn) -> None:
    """week_number values are integers 0 through 8 inclusive, no nulls.

    0 = latest week (L0W_ROLL), 8 = oldest week (L8W_ROLL).
    This ordering is critical for ORDER BY week_number DESC in SQL queries.
    """
    cursor = db_conn.execute(
        "SELECT DISTINCT week_number FROM raw_input_metrics ORDER BY week_number"
    )
    week_numbers = [row[0] for row in cursor.fetchall()]
    assert week_numbers == list(range(9)), (
        f"Expected week_numbers [0..8], got {week_numbers}"
    )
    # Verify no nulls
    cursor = db_conn.execute(
        "SELECT COUNT(*) FROM raw_input_metrics WHERE week_number IS NULL"
    )
    null_count = cursor.fetchone()[0]
    assert null_count == 0, f"Found {null_count} rows with NULL week_number"


def test_indexes_exist(db_conn) -> None:
    """Required indexes exist on raw_input_metrics for SQL agent query performance."""
    cursor = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='raw_input_metrics'"
    )
    index_names = {row[0] for row in cursor.fetchall()}
    assert 'idx_metrics_country_metric_week' in index_names, (
        f"Missing index idx_metrics_country_metric_week. Found: {index_names}"
    )
    assert 'idx_metrics_zone_metric' in index_names, (
        f"Missing index idx_metrics_zone_metric. Found: {index_names}"
    )
