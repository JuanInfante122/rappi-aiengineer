"""HTML executive report generator.

Wires the insights engine, LLM narrator, Plotly chart builders, and Jinja2
templates into a single pipeline that produces a self-contained HTML file
for a given country. The output can be opened in any browser without any
external dependencies beyond the Plotly CDN.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from insights.engine import run_insights_engine
from insights.narrator import generate_narratives_parallel, generate_executive_summary
from report.charts import build_report_chart, reset_chart_counter

if TYPE_CHECKING:
    from openai import OpenAI


def generate_html_report(
    db_path: Path,
    country: str,
    client: "OpenAI",
    output_path: Path | None = None,
) -> str:
    """Generate a self-contained HTML executive report for one country.

    Runs the full pipeline: statistical detection via run_insights_engine,
    parallel SCR narrative generation via generate_narratives_parallel,
    executive summary via generate_executive_summary, and Plotly chart
    building with CDN-once rule enforced by reset_chart_counter.

    Each insight is rendered as a card in a 2-column grid sorted by
    severity descending. The executive summary appears above the cards.

    Args:
        db_path: Path to the SQLite database file (must exist and contain
            raw_input_metrics and raw_orders tables).
        country: Country code to analyze and report on (e.g., 'CO', 'BR').
        client: OpenAI client instance used for narrative and summary generation.
        output_path: Optional path to write the HTML file. Parent directories
            are created automatically. If None, only the string is returned.

    Returns:
        Self-contained HTML string. Can be written to a .html file and opened
        in any browser. Returns minimal HTML with a no-data message if the
        insights engine finds no insights for the country.
    """
    # Step 1: Run insights engine
    insights = run_insights_engine(db_path, country)

    if not insights:
        html = (
            "<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'>"
            f"<title>Reporte {country}</title></head><body>"
            f"<p>No se encontraron insights para {country}.</p>"
            "</body></html>"
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding="utf-8")
        return html

    # Step 2: Generate SCR narratives in parallel
    narratives = generate_narratives_parallel(insights, client)

    # Step 3: Attach narratives to insights for executive summary context
    for i, insight in enumerate(insights):
        insight["narrative"] = narratives[i]

    # Step 4: Generate executive summary (uses pre-attached narratives)
    executive_summary = generate_executive_summary(insights, country, client)

    # Step 5: Reset chart CDN flag for this report generation pass
    reset_chart_counter()

    # Step 6: Build Jinja2 environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=False,
    )
    card_template = env.get_template("insight_card.html.j2")
    base_template = env.get_template("base.html.j2")

    # Step 7: Render each insight card (sorted by severity descending — engine already sorts)
    cards: list[str] = []
    for insight in insights:
        chart_html = build_report_chart(insight)
        card_html = card_template.render(
            insight=insight,
            narrative=insight.get("narrative"),
            chart_html=chart_html,
        )
        cards.append(card_html)

    # Step 8: Render full page
    html = base_template.render(
        country=country,
        executive_summary=executive_summary,
        cards=cards,
    )

    # Step 9: Write to disk if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html
