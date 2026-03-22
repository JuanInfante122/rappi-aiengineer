"""CSV-to-SQLite ETL pipeline for Rappi operations data.

Reads RAW_INPUT_METRICS and RAW_ORDERS from pre-exported CSV files,
reshapes them from wide format (one column per week) to long format
(one row per zone/metric/week), and writes to SQLite with indexes.

CSV files were exported from data/rappi_data.xlsx (kept for reference)
and committed to the repository for faster startup — no openpyxl overhead.

Always rebuilds from scratch — drop + recreate guarantees freshness
and eliminates the risk of stale data during demos.

Column name source: direct Excel inspection on 2026-03-20.
Expected output: 113,157 rows in raw_input_metrics, 11,178 in raw_orders.
"""

import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd


METRICS_CSV_PATH = Path("data/raw_input_metrics.csv")
ORDERS_CSV_PATH = Path("data/raw_orders.csv")
DB_PATH = Path("data/rappi_ops.db")

# Exact column names verified from Excel inspection (2026-03-20)
METRICS_ID_COLS = [
    'COUNTRY', 'CITY', 'ZONE', 'ZONE_TYPE', 'ZONE_PRIORITIZATION', 'METRIC',
]
METRICS_WEEK_COLS = [
    'L8W_ROLL', 'L7W_ROLL', 'L6W_ROLL', 'L5W_ROLL',
    'L4W_ROLL', 'L3W_ROLL', 'L2W_ROLL', 'L1W_ROLL', 'L0W_ROLL',
]
# L8W_ROLL -> 8 (oldest week), L0W_ROLL -> 0 (latest/newest week)
METRICS_WEEK_MAP = {col: 8 - i for i, col in enumerate(METRICS_WEEK_COLS)}

ORDERS_ID_COLS = ['COUNTRY', 'CITY', 'ZONE', 'METRIC']
ORDERS_WEEK_COLS = ['L8W', 'L7W', 'L6W', 'L5W', 'L4W', 'L3W', 'L2W', 'L1W', 'L0W']
ORDERS_WEEK_MAP = {col: 8 - i for i, col in enumerate(ORDERS_WEEK_COLS)}


def build_zone_id(country: str, city: str, zone: str) -> str:
    """Build a normalized zone identifier suitable for SQL LIKE queries.

    Concatenates country, city, and zone with underscores, then:
    1. Strips Unicode accents via NFKD decomposition + ASCII encoding so
       accented chars (á, é, ã, ñ) become their base ASCII equivalents.
    2. Uppercases the result.
    3. Replaces all non-alphanumeric characters (spaces, hyphens, slashes,
       parentheses, etc.) with underscores.
    4. Collapses consecutive underscores to a single one and strips leading/
       trailing underscores.

    This ensures SQL queries like `WHERE zone_id LIKE '%BOGOTA%'` work
    reliably regardless of accented, mixed-case, or punctuated source data.

    Args:
        country: Country code from source data (e.g., 'CO').
        city: City name from source data (e.g., 'Bogotá').
        zone: Zone name from source data (e.g., 'San Martín de Porras').

    Returns:
        Normalized zone identifier (e.g., 'CO_BOGOTA_SAN_MARTIN_DE_PORRAS').
    """
    raw = f"{country}_{city}_{zone}"
    # Strip accents: decompose to NFD and drop combining diacritical marks
    normalized = unicodedata.normalize('NFKD', raw).encode('ascii', 'ignore').decode('ascii')
    uppercased = normalized.upper()
    # Replace any non-alphanumeric character (except underscore) with underscore
    underscored = re.sub(r'[^A-Z0-9_]', '_', uppercased)
    # Collapse multiple consecutive underscores and strip leading/trailing ones
    return re.sub(r'_+', '_', underscored).strip('_')


def run_etl() -> None:
    """Load Excel data into SQLite, always rebuilding both tables from scratch.

    Reads RAW_INPUT_METRICS and RAW_ORDERS sheets, melts wide weekly columns
    to long format, adds week_number (integer 0-8) and normalized zone_id,
    then writes to SQLite with four performance indexes.

    Prints progress to stdout so both local and container runs are debuggable.

    Raises:
        FileNotFoundError: If METRICS_CSV_PATH or ORDERS_CSV_PATH does not exist.
        sqlite3.Error: If the database write fails.
    """
    print(f"Loading {METRICS_CSV_PATH} and {ORDERS_CSV_PATH}...")

    df_metrics_wide = pd.read_csv(METRICS_CSV_PATH)
    df_orders_wide = pd.read_csv(ORDERS_CSV_PATH)

    print(f"  RAW_INPUT_METRICS: {len(df_metrics_wide):,} rows (wide format)")
    print(f"  RAW_ORDERS: {len(df_orders_wide):,} rows (wide format)")

    # Reshape metrics: wide -> long (one row per zone x metric x week)
    df_metrics = df_metrics_wide.melt(
        id_vars=METRICS_ID_COLS,
        value_vars=METRICS_WEEK_COLS,
        var_name='week_label',
        value_name='value',
    )
    df_metrics['week_number'] = df_metrics['week_label'].map(METRICS_WEEK_MAP)
    df_metrics['zone_id'] = df_metrics.apply(
        lambda r: build_zone_id(r['COUNTRY'], r['CITY'], r['ZONE']), axis=1
    )
    df_metrics = df_metrics.rename(columns={
        'COUNTRY': 'country',
        'CITY': 'city',
        'ZONE': 'zone',
        'ZONE_TYPE': 'zone_type',
        'ZONE_PRIORITIZATION': 'zone_prioritization',
        'METRIC': 'metric_name',
    })

    # Reshape orders: wide -> long (one row per zone x week)
    df_orders = df_orders_wide.melt(
        id_vars=ORDERS_ID_COLS,
        value_vars=ORDERS_WEEK_COLS,
        var_name='week_label',
        value_name='value',
    )
    df_orders['week_number'] = df_orders['week_label'].map(ORDERS_WEEK_MAP)
    df_orders['zone_id'] = df_orders.apply(
        lambda r: build_zone_id(r['COUNTRY'], r['CITY'], r['ZONE']), axis=1
    )
    df_orders = df_orders.rename(columns={
        'COUNTRY': 'country',
        'CITY': 'city',
        'ZONE': 'zone',
        'METRIC': 'metric_name',
    })

    # Write to SQLite — always rebuild to guarantee data freshness
    conn = sqlite3.connect(DB_PATH)
    try:
        # Drop existing tables before recreating to prevent duplicate rows on re-run
        conn.executescript("""
            DROP TABLE IF EXISTS raw_input_metrics;
            DROP TABLE IF EXISTS raw_orders;
        """)

        df_metrics.to_sql('raw_input_metrics', conn, if_exists='replace', index=False)
        df_orders.to_sql('raw_orders', conn, if_exists='replace', index=False)

        # Indexes for SQL agent query patterns: country+metric+week and zone+metric
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_metrics_country_metric_week
                ON raw_input_metrics (country, metric_name, week_number);
            CREATE INDEX IF NOT EXISTS idx_metrics_zone_metric
                ON raw_input_metrics (zone_id, metric_name);
            CREATE INDEX IF NOT EXISTS idx_orders_country_week
                ON raw_orders (country, week_number);
            CREATE INDEX IF NOT EXISTS idx_orders_zone
                ON raw_orders (zone_id);
        """)
        conn.commit()
    finally:
        conn.close()

    print(f"Loaded {len(df_metrics):,} rows into raw_input_metrics")
    print(f"Loaded {len(df_orders):,} rows into raw_orders")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    run_etl()
