"""Unit tests for LangChain SQL agent and system prompt correctness.

Tests verify that:
- SYSTEM_PREFIX contains all required template variables and metric names
- Critical SQL rules are embedded in the system prompt
- AgentType imports from the correct package (langchain_classic)
- create_agent() returns a properly configured AgentExecutor
"""

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Wave 0: SYSTEM_PREFIX content tests
# ---------------------------------------------------------------------------


def test_system_prefix_has_dialect_placeholder():
    """SYSTEM_PREFIX must contain {dialect} or create_sql_agent raises KeyError."""
    from agent.prompts import SYSTEM_PREFIX

    assert "{dialect}" in SYSTEM_PREFIX, "SYSTEM_PREFIX missing {dialect} template placeholder"


def test_system_prefix_has_top_k_placeholder():
    """SYSTEM_PREFIX must contain {top_k} or create_sql_agent raises KeyError."""
    from agent.prompts import SYSTEM_PREFIX

    assert "{top_k}" in SYSTEM_PREFIX, "SYSTEM_PREFIX missing {top_k} template placeholder"


def test_system_prefix_has_all_13_metrics():
    """All 13 exact metric name strings must appear in SYSTEM_PREFIX for correct SQL generation."""
    from agent.prompts import SYSTEM_PREFIX

    expected_metrics = [
        "% PRO Users Who Breakeven",
        "% Restaurants Sessions With Optimal Assortment",
        "Gross Profit UE",
        "Lead Penetration",
        "MLTV Top Verticals Adoption",
        "Non-Pro PTC > OP",
        "Perfect Orders",
        "Pro Adoption (Last Week Status)",
        "Restaurants Markdowns / GMV",
        "Restaurants SS > ATC CVR",
        "Restaurants SST > SS CVR",
        "Retail SST > SS CVR",
        "Turbo Adoption",
    ]
    for metric in expected_metrics:
        assert metric in SYSTEM_PREFIX, f"SYSTEM_PREFIX missing metric: '{metric}'"


def test_system_prefix_has_metric_name_filter_rule():
    """The mandatory WHERE metric_name rule must be in SYSTEM_PREFIX to prevent cross-metric mixing."""
    from agent.prompts import SYSTEM_PREFIX

    assert "WHERE metric_name" in SYSTEM_PREFIX, (
        "SYSTEM_PREFIX missing critical SQL rule: 'WHERE metric_name'"
    )


def test_system_prefix_has_viz_hint():
    """VIZ_HINT instruction must appear so the Streamlit UI can parse chart type from responses."""
    from agent.prompts import SYSTEM_PREFIX

    assert "VIZ_HINT" in SYSTEM_PREFIX, "SYSTEM_PREFIX missing VIZ_HINT instruction"


def test_system_prefix_has_proactive_suggestions():
    """Proactive suggestions instruction must be in SYSTEM_PREFIX."""
    from agent.prompts import SYSTEM_PREFIX

    assert "3 related" in SYSTEM_PREFIX, (
        "SYSTEM_PREFIX missing proactive suggestions instruction ('3 related')"
    )


def test_agent_type_import():
    """AgentType must import from langchain_classic, not langchain (which dropped it in v1.x)."""
    from langchain_classic.agents import AgentType

    assert AgentType.OPENAI_FUNCTIONS, "AgentType.OPENAI_FUNCTIONS not available"


# ---------------------------------------------------------------------------
# Wave 1: SYSTEM_PREFIX structural completeness
# ---------------------------------------------------------------------------


def test_system_prefix_has_few_shot_examples():
    """Few-shot SQL examples must be in SYSTEM_PREFIX for better query generation."""
    from agent.prompts import SYSTEM_PREFIX

    assert "Lead Penetration" in SYSTEM_PREFIX and "SELECT" in SYSTEM_PREFIX, (
        "SYSTEM_PREFIX should contain few-shot SQL examples"
    )


def test_system_prefix_has_zone_uppercase_rule():
    """Zone LIKE queries must use uppercase terms — rule must be in SYSTEM_PREFIX."""
    from agent.prompts import SYSTEM_PREFIX

    assert "LIKE '%TERM%'" in SYSTEM_PREFIX or "UPPERCASE" in SYSTEM_PREFIX, (
        "SYSTEM_PREFIX missing zone LIKE uppercase rule"
    )


# ---------------------------------------------------------------------------
# Wave 2: Agent creation test (requires real DB)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("data/rappi_ops.db").exists(),
    reason="data/rappi_ops.db not built — run ETL first",
)
def test_create_agent_returns_executor(db_conn):
    """create_agent() must return an AgentExecutor with an invoke method."""
    from agent.sql_agent import create_agent

    agent = create_agent()
    assert hasattr(agent, "invoke"), "create_agent() did not return an AgentExecutor-like object"
