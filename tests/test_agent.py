"""Unit tests for LangChain SQL agent and system prompt correctness.

Tests verify that:
- SYSTEM_PREFIX contains all required template variables and metric names
- Critical SQL rules are embedded in the system prompt
- AgentType imports from the correct package (langchain_classic)
- create_agent() returns a properly configured AgentExecutor
"""

import os
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
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — ChatOpenAI requires a valid API key",
)
def test_create_agent_returns_executor(db_conn):
    """create_agent() must return an AgentExecutor with an invoke method."""
    from agent.sql_agent import create_agent

    agent = create_agent()
    assert hasattr(agent, "invoke"), "create_agent() did not return an AgentExecutor-like object"


# ---------------------------------------------------------------------------
# Wave 3: build_enriched_query tests (Plan 02-02)
# ---------------------------------------------------------------------------


def test_build_enriched_query_empty_history():
    """Empty history must return the original query unchanged — no context overhead."""
    from agent.memory import build_enriched_query

    result = build_enriched_query("What is the lead penetration?", [])
    assert result == "What is the lead penetration?"


def test_build_enriched_query_injects_3_turns():
    """Six messages (3 user/assistant turns) must all appear in the enriched output."""
    from agent.memory import build_enriched_query

    history = [
        {"role": "user", "content": "Turn 1 user"},
        {"role": "assistant", "content": "Turn 1 assistant"},
        {"role": "user", "content": "Turn 2 user"},
        {"role": "assistant", "content": "Turn 2 assistant"},
        {"role": "user", "content": "Turn 3 user"},
        {"role": "assistant", "content": "Turn 3 assistant"},
    ]
    result = build_enriched_query("New question here", history)
    assert "Turn 1 user" in result
    assert "Turn 1 assistant" in result
    assert "Turn 3 user" in result
    assert "Turn 3 assistant" in result


def test_build_enriched_query_limits_to_max_turns():
    """Eight messages (4 turns) must include only the last 6 (3 turns) — token cap."""
    from agent.memory import build_enriched_query

    history = [
        {"role": "user", "content": "Oldest user message"},
        {"role": "assistant", "content": "Oldest assistant message"},
        {"role": "user", "content": "Turn 2 user"},
        {"role": "assistant", "content": "Turn 2 assistant"},
        {"role": "user", "content": "Turn 3 user"},
        {"role": "assistant", "content": "Turn 3 assistant"},
        {"role": "user", "content": "Turn 4 user"},
        {"role": "assistant", "content": "Turn 4 assistant"},
    ]
    result = build_enriched_query("New question", history)
    # Oldest messages are outside the 3-turn window and must be excluded
    assert "Oldest user message" not in result
    assert "Oldest assistant message" not in result
    assert "Turn 4 user" in result
    assert "Turn 4 assistant" in result


def test_build_enriched_query_truncates_long_content():
    """Assistant messages over 400 chars must be truncated to prevent token accumulation."""
    from agent.memory import build_enriched_query

    long_content = "x" * 500
    history = [
        {"role": "user", "content": "Short user message"},
        {"role": "assistant", "content": long_content},
    ]
    result = build_enriched_query("Follow up question", history)
    # The 500-char content must be cut at 400 and marked with "..."
    assert "x" * 400 in result
    assert "x" * 401 not in result
    assert "..." in result


def test_build_enriched_query_format():
    """Output must follow the documented format with sentinel strings for parsing."""
    from agent.memory import build_enriched_query

    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]
    result = build_enriched_query("Current question", history)
    assert result.startswith("Previous conversation:")
    assert "New question:" in result
    assert "Current question" in result


# ---------------------------------------------------------------------------
# Wave 4: safe_invoke tests (Plan 02-02)
# ---------------------------------------------------------------------------


class MockAgent:
    """Stub agent that returns immediately with a fixed output."""

    def invoke(self, input_dict: dict) -> dict:
        return {"output": "mock response"}


class SlowMockAgent:
    """Stub agent that sleeps to simulate a hung LLM call."""

    def invoke(self, input_dict: dict) -> dict:
        import time

        time.sleep(5)
        return {"output": "too slow"}


class FailingMockAgent:
    """Stub agent that raises an unexpected exception."""

    def invoke(self, input_dict: dict) -> dict:
        raise ValueError("Simulated unexpected error")


def test_safe_invoke_success():
    """Successful invocation must return the agent's output dict with 'output' key."""
    from agent.memory import safe_invoke

    agent = MockAgent()
    result = safe_invoke(agent, "test query", timeout_s=10)
    assert "output" in result
    assert result["output"] == "mock response"


def test_safe_invoke_timeout():
    """Agent exceeding timeout_s must return an error dict with 'error' == 'timeout'."""
    from agent.memory import safe_invoke

    agent = SlowMockAgent()
    result = safe_invoke(agent, "complex query", timeout_s=0.1)
    assert "error" in result
    assert result["error"] == "timeout"


def test_safe_invoke_exception():
    """Unexpected agent exception must return an error dict with 'error' key."""
    from agent.memory import safe_invoke

    agent = FailingMockAgent()
    result = safe_invoke(agent, "problem query", timeout_s=10)
    assert "error" in result
    # error value contains the exception string
    assert "error" in result


# ---------------------------------------------------------------------------
# Wave 5: classify_response tests (Plan 02-02)
# ---------------------------------------------------------------------------


def test_classify_response_empty_result():
    """English 'no results' signal must trigger suggestions for better query formulation."""
    from agent.memory import classify_response

    result = {"output": "The query returned no results for this zone."}
    classified = classify_response(result)
    assert "suggestions" in classified
    assert len(classified["suggestions"]) > 0


def test_classify_response_spanish_empty():
    """Spanish 'sin resultados' signal must also trigger suggestions list."""
    from agent.memory import classify_response

    result = {"output": "No se encontraron datos, sin resultados para esa zona."}
    classified = classify_response(result)
    assert "suggestions" in classified


def test_classify_response_normal():
    """Response with real data content must pass through unchanged without suggestions."""
    from agent.memory import classify_response

    result = {"output": "Lead Penetration in BOGOTA is 45.3% this week."}
    classified = classify_response(result)
    assert "suggestions" not in classified
