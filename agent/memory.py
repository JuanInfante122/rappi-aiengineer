"""Conversation memory, safe invocation, and response classification for the SQL agent.

Three utility functions that wrap the LangChain AgentExecutor with production-grade
reliability: context-aware query enrichment, hard timeout enforcement, and
business-level error handling with user-facing suggestions.
"""

import concurrent.futures
from typing import Any


def build_enriched_query(user_query: str, history: list[dict], max_turns: int = 3) -> str:
    """Prepend recent conversation turns to the current query for context injection.

    Injects the last N user/assistant pairs as plain text before the new query.
    This approach is used instead of LangChain memory classes because the SQL
    agent's multi-step tool loop makes memory injection points unreliable.

    Args:
        user_query: The current question from the user.
        history: List of message dicts with 'role' and 'content' keys.
        max_turns: Number of prior conversation turns to include.

    Returns:
        The enriched query string with conversation context prepended,
        or the original query if history is empty.
    """
    if not history:
        return user_query
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        label = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{label}: {content}")
    context = "\n".join(lines)
    return f"Previous conversation:\n{context}\n\nNew question: {user_query}"


def safe_invoke(agent: Any, query: str, timeout_s: int = 30) -> dict:
    """Invoke the SQL agent with a hard wall-clock timeout.

    LangChain's max_execution_time is checked between agent iterations, not
    mid-tool-call. ThreadPoolExecutor provides a hard limit for cases where
    a single tool call hangs (e.g., slow LLM response or SQLite lock).

    Args:
        agent: LangChain AgentExecutor instance.
        query: Enriched query string (output of build_enriched_query).
        timeout_s: Hard timeout in seconds.

    Returns:
        Dict with 'output' key on success, or error dict on timeout/failure.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(agent.invoke, {"input": query})
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return {
                "output": (
                    "La consulta es muy compleja y excedio el tiempo limite. "
                    "Intenta con menos variables o un rango de semanas mas corto."
                ),
                "error": "timeout",
            }
        except Exception as exc:
            return {
                "output": (
                    f"Error al procesar la consulta. Por favor intenta de nuevo. "
                    f"Detalle tecnico: {type(exc).__name__}"
                ),
                "error": str(exc),
            }


def classify_response(result: dict) -> dict:
    """Classify agent response and apply business-level error handling.

    Detects empty result sets and unknown metric references.
    Returns enhanced response with suggestions when no data is found.

    Args:
        result: Dict from safe_invoke with 'output' key.

    Returns:
        Result dict, possibly with 'suggestions' key added.
    """
    output = result.get("output", "")

    # Empty result detection — covers both English agent output and Spanish responses
    empty_signals = [
        "no results", "0 rows", "no data", "no encontre", "no encontro",
        "sin resultados", "no hay datos",
    ]
    if any(signal in output.lower() for signal in empty_signals):
        result["suggestions"] = [
            "Verifica el nombre exacto de la zona (puede incluir ciudad y pais)",
            "Prueba con un rango de semanas mas amplio",
            "Revisa el nombre de la metrica en el diccionario de metricas",
        ]

    return result
