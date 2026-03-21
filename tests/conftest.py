"""Shared pytest fixtures for database access.

Runs the ETL pipeline once per test session and provides a SQLite
connection for all tests that need to query the generated tables.
"""

import sqlite3
from pathlib import Path

import pytest

from data.load_data import run_etl


DB_PATH = Path("data/rappi_ops.db")


@pytest.fixture(scope="session")
def db_conn():
    """Run ETL once and yield a SQLite connection for the entire test session.

    The ETL always rebuilds from scratch, so the session-scoped fixture
    guarantees a fresh, consistent state across all tests without paying
    the cost of multiple ETL runs.

    Yields:
        sqlite3.Connection: Open connection to data/rappi_ops.db.
    """
    run_etl()
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()
