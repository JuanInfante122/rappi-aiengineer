"""Unit tests for agent/intent.py — VIZ_HINT parsing, SQL extraction, and DataFrame output.

Tests cover the four exported functions:
- parse_viz_hint: extract visualization metadata from agent response text
- strip_viz_hint: remove VIZ_HINT line before displaying to user
- extract_last_sql: retrieve last SQL from AgentExecutor intermediate_steps
- run_sql_to_df: execute SQL against SQLite and return a DataFrame
"""

import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# parse_viz_hint tests
# ---------------------------------------------------------------------------


def test_parse_viz_hint_line():
    """VIZ_HINT with 'line' type must return correct type/x_col/y_col dict."""
    from agent.intent import parse_viz_hint

    text = "Some answer\nVIZ_HINT: line | week_number | value"
    result = parse_viz_hint(text)
    assert result is not None
    assert result["type"] == "line"
    assert result["x_col"] == "week_number"
    assert result["y_col"] == "value"


def test_parse_viz_hint_bar_uppercase():
    """VIZ_HINT with uppercase type 'BAR' must be lowercased in returned dict."""
    from agent.intent import parse_viz_hint

    text = "VIZ_HINT: BAR | zone_id | avg_value"
    result = parse_viz_hint(text)
    assert result is not None
    assert result["type"] == "bar"
    assert result["x_col"] == "zone_id"
    assert result["y_col"] == "avg_value"


def test_parse_viz_hint_missing():
    """Text without VIZ_HINT must return None — no chart hint present."""
    from agent.intent import parse_viz_hint

    result = parse_viz_hint("No hint here")
    assert result is None


def test_parse_viz_hint_table_type():
    """VIZ_HINT with 'table' type must be parsed correctly."""
    from agent.intent import parse_viz_hint

    text = "Here is your data.\nVIZ_HINT: table | zone_id | value"
    result = parse_viz_hint(text)
    assert result is not None
    assert result["type"] == "table"


def test_parse_viz_hint_scatter_type():
    """VIZ_HINT with 'scatter' type must be parsed correctly."""
    from agent.intent import parse_viz_hint

    text = "VIZ_HINT: scatter | x_metric | y_metric"
    result = parse_viz_hint(text)
    assert result is not None
    assert result["type"] == "scatter"
    assert result["x_col"] == "x_metric"
    assert result["y_col"] == "y_metric"


# ---------------------------------------------------------------------------
# strip_viz_hint tests
# ---------------------------------------------------------------------------


def test_strip_viz_hint_removes_line():
    """VIZ_HINT line must be stripped from text before user display."""
    from agent.intent import strip_viz_hint

    text = "Answer text\nVIZ_HINT: line | x | y"
    result = strip_viz_hint(text)
    assert result == "Answer text"


def test_strip_viz_hint_no_hint():
    """Text without VIZ_HINT must be returned unchanged."""
    from agent.intent import strip_viz_hint

    result = strip_viz_hint("No hint")
    assert result == "No hint"


def test_strip_viz_hint_trailing_whitespace():
    """Stripped text must have trailing whitespace removed."""
    from agent.intent import strip_viz_hint

    text = "Answer text  \nVIZ_HINT: bar | a | b  "
    result = strip_viz_hint(text)
    assert not result.endswith(" ")
    assert "VIZ_HINT" not in result


# ---------------------------------------------------------------------------
# extract_last_sql tests
# ---------------------------------------------------------------------------


def test_extract_last_sql_with_steps():
    """intermediate_steps with sql_db_query tool must return the last SQL string."""
    from agent.intent import extract_last_sql

    mock_action = SimpleNamespace(tool="sql_db_query", tool_input={"query": "SELECT zone_id FROM raw_input_metrics LIMIT 5"})
    result = {"intermediate_steps": [(mock_action, "result_output")]}
    sql = extract_last_sql(result)
    assert sql == "SELECT zone_id FROM raw_input_metrics LIMIT 5"


def test_extract_last_sql_empty():
    """Empty intermediate_steps must return None — no SQL was executed."""
    from agent.intent import extract_last_sql

    result = {"intermediate_steps": []}
    assert extract_last_sql(result) is None


def test_extract_last_sql_string_input():
    """tool_input as plain string (not dict) must be handled gracefully."""
    from agent.intent import extract_last_sql

    mock_action = SimpleNamespace(tool="sql_db_query", tool_input="SELECT * FROM t")
    result = {"intermediate_steps": [(mock_action, "obs")]}
    sql = extract_last_sql(result)
    assert sql == "SELECT * FROM t"


def test_extract_last_sql_returns_last():
    """When multiple SQL steps exist, the last one must be returned."""
    from agent.intent import extract_last_sql

    action1 = SimpleNamespace(tool="sql_db_query", tool_input={"query": "SELECT 1"})
    action2 = SimpleNamespace(tool="sql_db_query", tool_input={"query": "SELECT 2 FROM t"})
    result = {"intermediate_steps": [(action1, "obs1"), (action2, "obs2")]}
    sql = extract_last_sql(result)
    assert sql == "SELECT 2 FROM t"


def test_extract_last_sql_missing_key():
    """Result dict without 'intermediate_steps' key must return None gracefully."""
    from agent.intent import extract_last_sql

    result = {"output": "some answer"}
    assert extract_last_sql(result) is None


def test_extract_last_sql_non_sql_tool():
    """Steps with non-SQL tools must be ignored."""
    from agent.intent import extract_last_sql

    schema_action = SimpleNamespace(tool="sql_db_schema", tool_input={"tables": "raw_input_metrics"})
    result = {"intermediate_steps": [(schema_action, "schema_output")]}
    assert extract_last_sql(result) is None


# ---------------------------------------------------------------------------
# run_sql_to_df tests
# ---------------------------------------------------------------------------


def test_run_sql_to_df_valid():
    """Valid SQL against a temp SQLite DB must return a non-empty DataFrame."""
    from agent.intent import run_sql_to_df

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE metrics (zone TEXT, value REAL)")
        conn.execute("INSERT INTO metrics VALUES ('BOGOTA', 42.5)")
        conn.execute("INSERT INTO metrics VALUES ('MEDELLIN', 38.1)")
        conn.commit()
        conn.close()

        df = run_sql_to_df("SELECT zone, value FROM metrics ORDER BY value DESC", db_path)
        assert not df.empty
        assert len(df) == 2
        assert "zone" in df.columns
        assert "value" in df.columns


def test_run_sql_to_df_invalid():
    """Invalid SQL must return an empty DataFrame without raising an exception."""
    from agent.intent import run_sql_to_df

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        df = run_sql_to_df("SELECT * FROM nonexistent_table_xyz", db_path)
        assert df.empty
