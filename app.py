"""Streamlit chat application for Rappi Operations Intelligence.

Chat UI that wires together the SQL agent and intent/renderer modules into
a production-quality interface with sidebar, custom CSS, compound messages
(text + chart + buttons), and the prefill-without-submit pattern for
example questions.
"""

import base64
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agent.intent import (
    extract_last_sql,
    parse_viz_hint,
    run_sql_to_df,
    strip_viz_hint,
)
from agent.memory import build_enriched_query, classify_response, safe_invoke
from agent.sql_agent import create_agent
from charts.renderer import build_chart
from data.load_data import run_etl
from report.emailer import send_report_email
from report.generator import generate_html_report

load_dotenv()

DB_PATH = Path("data/rappi_ops.db")
LOGO_PATH = Path("assets/rappi_logo.svg")

# Example questions: Spanish, ascending complexity
EXAMPLE_QUESTIONS = [
    "Cuales son las 5 zonas con mayor Lead Penetration esta semana?",
    "Muestrame la evolucion de Gross Profit UE en Chapinero las ultimas 8 semanas",
    "Compara el Perfect Order entre zonas Wealthy y Non Wealthy en Mexico",
    "Que zonas tienen alto Lead Penetration pero bajo Perfect Order?",
    "Cuales son las zonas que mas crecen en ordenes y que podria explicar el crecimiento?",
]

# Available metrics list — matches the 13 metrics in SYSTEM_PREFIX
AVAILABLE_METRICS = [
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

# Custom CSS for Rappi-branded chat bubble styling
CHAT_CSS = """
<style>
/* User message bubble */
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #F4F4F4;
    border-radius: 15px 15px 5px 15px;
}

/* Bot message bubble */
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #FFFFFF;
    border-left: 2px solid #FF441F;
    border-radius: 5px 15px 15px 5px;
}

/* Narrative blockquote accent */
blockquote {
    border-left: 4px solid #FF441F !important;
    padding-left: 12px;
    margin: 8px 0;
    color: #1A1A1A;
}

/* Chart container shadow */
.stPlotlyChart {
    background: #FFFFFF;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 8px;
}
</style>
"""


def get_openai_key() -> str:
    """Read OPENAI_API_KEY from st.secrets (cloud) or env var (local).

    Tries st.secrets first for Streamlit Cloud deployments, then falls
    back to the environment variable set via .env or shell for local dev.
    Stops the app if no key is found.

    Returns:
        The OpenAI API key string.
    """
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            st.error(
                "OPENAI_API_KEY no configurada. "
                "Agrega tu clave en .streamlit/secrets.toml o en .env"
            )
            st.stop()
        return key


@st.cache_resource
def get_database() -> Path:
    """Build SQLite DB from CSV on first run and return its path.

    Uses @st.cache_resource so ETL runs only once per server process,
    not on every browser refresh. Safe to share across users because
    the database is read-only after build.

    Returns:
        Path to the SQLite database file.
    """
    if not DB_PATH.exists():
        run_etl()
    return DB_PATH


@st.cache_data
def get_available_countries(db_path: Path) -> list[str]:
    """Query distinct country codes from raw_input_metrics.

    Cached per server process since country list is static within a dataset.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Sorted list of country code strings (e.g., ['AR', 'BR', 'CO']).
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT country FROM raw_input_metrics ORDER BY country"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def get_agent():
    """Get or create per-session SQL agent instance.

    Uses st.session_state instead of @st.cache_resource because
    ChatOpenAI is stateful — sharing across users is unsafe. Validates
    the OpenAI key exists before creating the agent to give a clear
    error message rather than a cryptic OpenAI API error.

    Returns:
        Configured AgentExecutor instance.
    """
    if "agent" not in st.session_state:
        get_openai_key()  # validates key exists before creating agent
        db_path = get_database()
        st.session_state["agent"] = create_agent(db_path)
    return st.session_state["agent"]


def render_chart_compound(msg: dict, index: int) -> None:
    """Render chart + CSV download + SQL toggle for an assistant message.

    Handles empty DataFrames, missing or malformed viz_hint, and narrative-
    only responses. Uses session_state toggle for SQL display instead of
    st.expander — st.expander inside chat_message has a known rendering bug.

    Args:
        msg: Assistant message dict with 'dataframe', 'sql', 'viz_hint' keys.
        index: Message index used as unique key for Streamlit widgets.
    """
    df = msg.get("dataframe")
    sql = msg.get("sql")
    viz_hint = msg.get("viz_hint")

    # No chart for narrative/qualitative responses (df is None)
    if df is None:
        return

    # SQL returned 0 rows — show clean empty state
    if isinstance(df, pd.DataFrame) and df.empty:
        st.caption("No se encontraron datos para los filtros seleccionados.")
        return

    # VIZ_HINT absent or type is "table" -> table fallback with notice
    if viz_hint is None or viz_hint.get("type") == "table":
        st.caption(
            "No se pudo generar la visualizacion. "
            "Mostrando datos en formato tabla."
        )
        st.dataframe(df, use_container_width=True)
    else:
        fig = build_chart(df, viz_hint)
        st.plotly_chart(fig, key=f"chart_{index}", use_container_width=True)

    # Footer buttons — Download CSV + View SQL toggle
    col1, col2 = st.columns([1, 1])
    with col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar CSV",
            data=csv_bytes,
            file_name=f"rappi_query_{index}.csv",
            mime="text/csv",
            key=f"download_{index}",
        )
    with col2:
        if sql:
            show_key = f"show_sql_{index}"
            if show_key not in st.session_state:
                st.session_state[show_key] = False
            if st.button("Ver SQL", key=f"btn_sql_{index}"):
                st.session_state[show_key] = not st.session_state[show_key]
            if st.session_state.get(show_key):
                st.code(sql, language="sql")


def handle_user_message(user_input: str) -> None:
    """Process user input through the full agent pipeline.

    Pipeline: build_enriched_query -> safe_invoke -> classify_response ->
    parse_viz_hint -> strip_viz_hint -> extract_last_sql -> run_sql_to_df.
    Stores the resulting compound message (text + df + sql + viz_hint) in
    st.session_state.messages for rendering.

    Args:
        user_input: Raw question text from the chat input widget.
    """
    agent = get_agent()

    # Add user message to history first (so it appears immediately on rerun)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "dataframe": None,
        "sql": None,
        "viz_hint": None,
    })

    # Build context-aware query with the last 3 conversation turns
    history_for_enrichment = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    enriched = build_enriched_query(user_input, history_for_enrichment)

    # Invoke agent with spinner feedback
    with st.spinner("Analizando tu pregunta..."):
        result = safe_invoke(agent, enriched)

    result = classify_response(result)
    raw_text = result.get("output", "")

    # Parse and strip VIZ_HINT from response text
    viz_hint = parse_viz_hint(raw_text)
    clean_text = strip_viz_hint(raw_text)
    sql = extract_last_sql(result)

    # Get chart data only when both SQL and viz_hint are available
    df = None
    if sql and viz_hint:
        df = run_sql_to_df(sql, DB_PATH)
        # Keep df as empty DataFrame (not None) to trigger empty-state message.
        # None means narrative (no chart at all); empty means 0 rows returned.

    # Append suggestions from classify_response when present
    if "suggestions" in result:
        clean_text += "\n\n**Sugerencias:**\n"
        for suggestion in result["suggestions"]:
            clean_text += f"- {suggestion}\n"

    # Store compound assistant message in session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": clean_text,
        "dataframe": df,
        "sql": sql,
        "viz_hint": viz_hint,
    })


# --- Page config must be the very first Streamlit call ---
# Pass the SVG as a base64 data URI so Streamlit writes it into the <head>
# favicon link — st.markdown injections go into the body and are ignored by browsers.
_page_icon: str | Path = ":bar_chart:"
if LOGO_PATH.exists():
    _svg_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    _page_icon = f"data:image/svg+xml;base64,{_svg_b64}"

st.set_page_config(
    page_title="Rappi Ops Intelligence",
    page_icon=_page_icon,
    layout="wide",
)

# Inject custom CSS for Rappi-branded chat bubbles
st.markdown(CHAT_CSS, unsafe_allow_html=True)

# Initialize session state on first load
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    # Rappi logo — render SVG inline, scaled and centered in the sidebar
    if LOGO_PATH.exists():
        svg = LOGO_PATH.read_text()
        svg = svg.replace('width="24"', 'width="160"').replace('height="24"', 'height="52"')
        st.markdown(
            f'<div style="text-align:center;padding:8px 0">{svg}</div>',
            unsafe_allow_html=True,
        )

    # App title and caption
    st.title("Rappi Ops Intelligence")
    st.caption("Sistema de Analisis Inteligente para Operaciones")

    st.divider()

    # Example questions — clicking prefills the chat input without auto-submitting
    st.subheader("Preguntas de ejemplo")
    for idx, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(question, key=f"example_{idx}", use_container_width=True):
            # Set the chat_input session state key and rerun so the widget
            # reflects the value. st.chat_input does NOT auto-submit on rerun —
            # the user must still press Enter or click Send.
            st.session_state["chat_input"] = question
            st.rerun()

    st.divider()

    # Available metrics (collapsible)
    with st.expander("Metricas disponibles"):
        for metric in AVAILABLE_METRICS:
            st.markdown(f"- {metric}")

    st.divider()

    # Data coverage context note
    st.caption("Datos: 9 paises, ~1,200 zonas, 9 semanas (L0W-L8W)")

    st.divider()

    st.subheader("Generar Reporte Semanal")
    countries = get_available_countries(get_database())
    selected_country = st.selectbox(
        "Pais",
        options=countries,
        index=0,
        key="report_country",
    )

    report_email = st.text_input(
        "Enviar reporte por email (opcional)",
        placeholder="tu@correo.com",
        key="report_email",
    )

    if st.button("Generar Reporte", key="btn_generate_report", use_container_width=True):
        with st.spinner("Generando reporte de insights..."):
            try:
                client = OpenAI(api_key=get_openai_key())
                html_report = generate_html_report(
                    db_path=get_database(),
                    country=selected_country,
                    client=client,
                )
                if "No se encontraron insights" in html_report:
                    st.warning(f"No se encontraron insights para {selected_country} esta semana.")
                else:
                    st.download_button(
                        label="Descargar Reporte HTML",
                        data=html_report.encode("utf-8"),
                        file_name=f"rappi_insights_{selected_country}.html",
                        mime="text/html",
                        key="btn_download_report",
                    )
                    if report_email:
                        try:
                            send_report_email(
                                html_content=html_report,
                                recipient=report_email,
                                country=selected_country,
                            )
                            st.success(f"Reporte enviado a {report_email}")
                        except EnvironmentError:
                            st.warning(
                                "SMTP no configurado. Agrega SMTP_HOST, SMTP_USER y "
                                "SMTP_PASSWORD en tu archivo .env para habilitar el envio."
                            )
                        except Exception as exc:
                            st.error(f"Error al enviar el email: {exc}")
            except Exception as e:
                st.error(f"Error generando reporte: {e}")

# --- Chat display loop ---
_bot_avatar = str(LOGO_PATH) if LOGO_PATH.exists() else None
for i, msg in enumerate(st.session_state.messages):
    avatar = _bot_avatar if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_chart_compound(msg, index=i)

# --- Chat input (prefill without auto-submit) ---
prompt = st.chat_input(
    "Haz una pregunta sobre tus operaciones...",
    key="chat_input",
)

if prompt:
    handle_user_message(prompt)
    st.rerun()
