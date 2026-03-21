"""Intent classification and VIZ_HINT parsing for the Rappi chat UI.

Extracts visualization metadata from agent responses, retrieves executed SQL
from intermediate_steps, and re-executes SQL to produce DataFrames for charts.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

VIZ_HINT_PATTERN = re.compile(
    r'VIZ_HINT:\s*(\w+)\s*\|\s*(\S+)\s*\|\s*(\S+)',
    re.IGNORECASE,
)


def parse_viz_hint(text: str) -> dict | None:
    """Extract VIZ_HINT metadata from agent response text.

    Parses the VIZ_HINT line appended by the system prompt to determine
    chart type and axis columns for Plotly rendering.

    Args:
        text: Full agent response text that may contain a VIZ_HINT line.

    Returns:
        Dict with 'type', 'x_col', 'y_col' keys, or None if no hint found.
    """
    match = VIZ_HINT_PATTERN.search(text)
    if not match:
        return None
    return {
        "type": match.group(1).lower(),
        "x_col": match.group(2),
        "y_col": match.group(3),
    }


def strip_viz_hint(text: str) -> str:
    """Remove VIZ_HINT line from text before displaying to user.

    Args:
        text: Agent response text potentially containing VIZ_HINT.

    Returns:
        Text with VIZ_HINT line removed and trailing whitespace stripped.
    """
    return VIZ_HINT_PATTERN.sub('', text).strip()


def extract_last_sql(result: dict) -> str | None:
    """Extract the last SQL query from AgentExecutor intermediate_steps.

    Walks the intermediate_steps list looking for sql_db_query tool calls
    and returns the last SQL string found. Handles both dict and string
    tool_input formats from different LangChain agent types.

    Args:
        result: Dict from agent.invoke() with 'intermediate_steps' key.

    Returns:
        Last SQL query string, or None if no SQL found.
    """
    steps = result.get("intermediate_steps", [])
    sql_queries: list[str] = []
    for action, _observation in steps:
        if hasattr(action, "tool") and "sql_db_query" in action.tool:
            tool_input = action.tool_input
            if isinstance(tool_input, dict):
                sql_queries.append(tool_input.get("query", ""))
            elif isinstance(tool_input, str):
                sql_queries.append(tool_input)
    return sql_queries[-1] if sql_queries else None


def run_sql_to_df(sql: str, db_path: Path) -> pd.DataFrame:
    """Execute SQL query and return result as DataFrame for chart rendering.

    Re-runs the agent's last SQL against the database to get structured
    data for Plotly. Returns empty DataFrame on any error to allow
    graceful degradation to table fallback.

    Args:
        sql: SQL query string from extract_last_sql.
        db_path: Path to the SQLite database file.

    Returns:
        DataFrame with query results, or empty DataFrame on error.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()
