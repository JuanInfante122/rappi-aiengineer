"""Streamlit entry point for Rappi Operations Intelligence.

Phase 1 stub: validates ETL pipeline by displaying row counts
per country and a preview of raw_input_metrics data.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from data.load_data import run_etl


DB_PATH = Path("data/rappi_ops.db")

st.set_page_config(page_title="Rappi Ops Intelligence", layout="wide")

# Run ETL if database does not exist — conditional to avoid rebuilding on every
# page reload. The Dockerfile CMD runs ETL first, so this is a local dev fallback.
if not DB_PATH.exists():
    with st.spinner("Cargando datos..."):
        run_etl()

conn = sqlite3.connect(DB_PATH)

st.title("Rappi Operations Intelligence")
st.caption("Sistema de Analisis Inteligente para Operaciones")

# Row counts per country — validates geographic segmentation (9 countries)
counts = pd.read_sql(
    "SELECT country, COUNT(*) as rows FROM raw_input_metrics GROUP BY country ORDER BY country",
    conn,
)
st.subheader("Datos cargados por pais")
st.dataframe(counts, use_container_width=True)

# Preview first 10 rows of raw_input_metrics — validates reshape, zone_id, week_number
preview = pd.read_sql(
    "SELECT country, city, zone, zone_id, metric_name, week_number, value "
    "FROM raw_input_metrics LIMIT 10",
    conn,
)
st.subheader("Vista previa - raw_input_metrics")
st.dataframe(preview, use_container_width=True)

conn.close()
