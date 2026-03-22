"""LLM-powered narrative generation: converts statistical insights into Spanish SCR narratives.

Provides structured narrative generation using GPT-4o-mini for individual insights and
GPT-4o for the executive summary. The SCR (Situation-Complication-Resolution) format
provides context, problem framing, and action guidance for each statistical anomaly.

Parallel execution via ThreadPoolExecutor allows all 25 top insights to be narrated
concurrently, reducing total latency from ~25x to ~5x serial time.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from openai import OpenAI


NARRATOR_PROMPT = """Eres un analista de operaciones de Rappi. Genera una narrativa en formato SCR
(Situacion-Complicacion-Resolucion) en espanol para el siguiente hallazgo.

DATOS DEL HALLAZGO (solo puedes citar numeros que aparezcan aqui -- no inventes cifras):
{evidence_json}

INSTRUCCIONES:
- Situacion (3-4 oraciones): describe el estado actual usando los datos del hallazgo.
- Complicacion (3-4 oraciones): explica por que esto es preocupante para operaciones.
- Resolucion (3-4 oraciones): sugiere accion o investigacion. Nunca dejes Resolucion vacia.
- Si no hay una accion clara, recomienda investigar las causas operativas y comparar con zonas de mejor desempeno.
- Usa solo los numeros del hallazgo. Si no tienes suficientes datos, indica incertidumbre.

Responde UNICAMENTE con un JSON con las claves: "situation", "complication", "resolution".
"""

EXECUTIVE_SUMMARY_PROMPT = """Eres el Director de Operaciones de Rappi. Genera un resumen ejecutivo
en espanol de los hallazgos mas criticos para {country}.

TOP HALLAZGOS (ordenados por severidad):
{insights_summary}

INSTRUCCIONES:
- Escribe 3 a 5 parrafos cortos, uno por hallazgo critico.
- Cada parrafo: zona, metrica, que paso, y que hacer.
- Usa solo los datos proporcionados, no inventes cifras.
- Tono: ejecutivo, directo, orientado a accion.

Responde UNICAMENTE con un JSON con la clave: "summary" (string con el resumen completo).
"""


class SCRNarrative(BaseModel):
    """Structured narrative: Situation-Complication-Resolution.

    Used for Pydantic validation of LLM JSON output. All three fields
    are required — an empty resolution is a prompt-level violation.
    """

    situation: str
    complication: str
    resolution: str


def _fallback_narrative(evidence: dict) -> dict:
    """Return a template-based narrative when the LLM fails twice.

    Used as a last resort after one retry so the output list never
    contains None entries and the HTML report always has renderable content.

    Args:
        evidence: Insight evidence dict from the engine.

    Returns:
        Dict with keys "situation", "complication", "resolution".
    """
    zone = evidence.get("zone_id", "zona desconocida")
    metric = evidence.get("metric_name", "metrica desconocida")
    wow = evidence.get("wow_change_pct", 0)
    return {
        "situation": f"La zona {zone} presenta un cambio de {wow:.1%} en {metric} esta semana.",
        "complication": f"Este cambio en {metric} requiere atencion del equipo de operaciones.",
        "resolution": (
            f"Se recomienda investigar las causas operativas en {zone} y "
            f"comparar con las zonas de mejor desempeno del mismo tipo."
        ),
    }


def _call_narrator_api(evidence: dict, client: "OpenAI") -> dict:
    """Make a single LLM call and return the parsed SCR dict.

    Args:
        evidence: Insight evidence dict from the engine.
        client: OpenAI client instance.

    Returns:
        Dict with "situation", "complication", "resolution" keys.

    Raises:
        json.JSONDecodeError: If response cannot be decoded.
        ValidationError: If decoded dict fails Pydantic validation.
    """
    prompt = NARRATOR_PROMPT.format(
        evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2)
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    raw_dict = json.loads(response.choices[0].message.content)
    narrative = SCRNarrative.model_validate(raw_dict)
    return narrative.model_dump()


def generate_narrative(evidence: dict, client: "OpenAI") -> dict:
    """Generate a single SCR narrative for one insight evidence dict.

    Calls GPT-4o-mini with the evidence JSON, validates the response as
    a SCRNarrative, and retries once on any parse or validation failure.
    Falls back to a template string if both attempts fail.

    Args:
        evidence: Insight evidence dict from run_insights_engine. Must contain
            at minimum "metric_name", "zone_id", "wow_change_pct".
        client: OpenAI client instance (reads OPENAI_API_KEY from env).

    Returns:
        Dict with keys "situation", "complication", "resolution". Never None.
    """
    try:
        return _call_narrator_api(evidence, client)
    except (json.JSONDecodeError, ValidationError, Exception):
        # Retry once before falling back to template
        try:
            return _call_narrator_api(evidence, client)
        except Exception:
            return _fallback_narrative(evidence)


def generate_narratives_parallel(
    insights: list[dict],
    client: "OpenAI",
    max_workers: int = 5,
) -> list[dict]:
    """Generate SCR narratives for all insights in parallel via a thread pool.

    Preserves insertion order — result index i corresponds to insights index i.
    On any exception in a worker thread, assigns the template fallback so the
    output list never contains None entries.

    Args:
        insights: List of insight evidence dicts from run_insights_engine.
        client: OpenAI client instance.
        max_workers: Maximum concurrent LLM calls.

    Returns:
        List of narrative dicts in the same order as input. Length == len(insights).
    """
    results: list[dict | None] = [None] * len(insights)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_narrative, insight, client): idx
            for idx, insight in enumerate(insights)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = _fallback_narrative(insights[idx])

    # Final safety pass — replace any remaining None with fallback
    for idx, result in enumerate(results):
        if result is None:
            results[idx] = _fallback_narrative(insights[idx])

    return results  # type: ignore[return-value]


def generate_executive_summary(
    insights: list[dict],
    country: str,
    client: "OpenAI",
) -> str:
    """Generate an executive summary for the top 5 critical insights of a country.

    Uses GPT-4o (not gpt-4o-mini) because executive summaries are read by
    Directors and require higher reasoning quality. Called sequentially after
    all individual narratives are generated.

    Args:
        insights: List of insight dicts, already sorted by severity_score descending.
            May include "narrative" key if narratives were pre-attached.
        country: Country code (e.g., "CO", "BR") for the prompt header.
        client: OpenAI client instance.

    Returns:
        Executive summary string in Spanish. Returns a template fallback
        on any API or parse error so callers never receive an empty string.
    """
    top5 = insights[:5]

    summary_lines = []
    for i, insight in enumerate(top5, start=1):
        zone = insight.get("zone_id", "N/A")
        metric = insight.get("metric_name", "N/A")
        wow = insight.get("wow_change_pct", 0.0)
        severity = insight.get("severity_score", 0.0)
        # Include situation from pre-attached narrative if available
        situation = ""
        if "narrative" in insight and isinstance(insight["narrative"], dict):
            situation = insight["narrative"].get("situation", "")

        line = (
            f"{i}. Zona: {zone} | Metrica: {metric} | "
            f"Cambio WoW: {wow:.1%} | Severidad: {severity:.1f}"
        )
        if situation:
            line += f"\n   Situacion: {situation}"
        summary_lines.append(line)

    insights_summary = "\n".join(summary_lines)

    try:
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            country=country,
            insights_summary=insights_summary,
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = json.loads(response.choices[0].message.content)
        return raw["summary"]
    except Exception:
        return (
            f"Resumen ejecutivo no disponible. "
            f"Se encontraron {len(insights)} hallazgos para {country}."
        )
