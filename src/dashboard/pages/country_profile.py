"""
Pagina 2 del dashboard RADAR Cibest: Perfil de Pais.

Muestra el perfil detallado de un pais seleccionado: radar chart de las
cinco dimensiones vs. mediana regional, semaforos por linea de negocio,
narrativa ejecutiva y tabla de variables.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components.charts import radar_chart, score_gauge
from src.dashboard.components.common import (
    CIBEST_PALETTE,
    SIGNAL_COLORS,
    iso3_to_name_map,
    load_configurations,
    load_latest_parquet,
    signal_to_emoji,
)


st.title("Perfil de Pais")
st.caption(
    "Diagnostico detallado del mercado seleccionado: dimensiones, senales por "
    "linea de negocio y narrativa ejecutiva."
)

configs = load_configurations()
name_map = iso3_to_name_map(configs)

global_ranking = load_latest_parquet("global_ranking_")
radar_by_line = load_latest_parquet("radar_by_line_")

if global_ranking is None or radar_by_line is None:
    st.warning("Resultados de scoring no disponibles. Ejecute el pipeline completo.")
    st.stop()

# ---------------------------------------------------------------------------
# Selector de pais
# ---------------------------------------------------------------------------
options = sorted(global_ranking.index, key=lambda x: name_map.get(x, x))
selected_iso3 = st.selectbox(
    "Seleccione un pais",
    options=options,
    format_func=lambda x: f"{name_map.get(x, x)} ({x})",
)

country_name = name_map.get(selected_iso3, selected_iso3)

# ---------------------------------------------------------------------------
# KPIs principales
# ---------------------------------------------------------------------------
rank_global = int(radar_by_line.loc[selected_iso3, "rank_global"])
score_global = float(radar_by_line.loc[selected_iso3, "GLOBAL"])

col_a, col_b, col_c = st.columns(3)
col_a.metric("Posicion en el ranking", f"#{rank_global}")
col_b.metric("Score RADAR Global", f"{score_global:.3f}")
col_c.metric("Total mercados evaluados", len(radar_by_line))

# ---------------------------------------------------------------------------
# Radar chart vs. mediana regional
# ---------------------------------------------------------------------------
st.subheader("Perfil dimensional")
dim_cols = [c for c in global_ranking.columns if c.startswith("score_") and c != "score"]
dim_names = [c.replace("score_", "") for c in dim_cols]

selected_scores = {
    name: float(global_ranking.loc[selected_iso3, f"score_{name}"])
    for name in dim_names
}
median_scores = {
    name: float(global_ranking[f"score_{name}"].median())
    for name in dim_names
}

fig_radar = radar_chart(
    dimension_scores_selected=selected_scores,
    dimension_scores_benchmark=median_scores,
    country_label=country_name,
    benchmark_label="Mediana regional",
)
st.plotly_chart(fig_radar, width="stretch")

# ---------------------------------------------------------------------------
# Senales por linea de negocio
# ---------------------------------------------------------------------------
st.subheader("Senales por linea de negocio")
bl_keys = list(configs["business_lines"]["business_lines"].keys())
bl_labels = {k: configs["business_lines"]["business_lines"][k]["label"] for k in bl_keys}

cols = st.columns(len(bl_keys))
for col, bl in zip(cols, bl_keys):
    score = float(radar_by_line.loc[selected_iso3, bl]) if bl in radar_by_line.columns else 0.0
    with col:
        st.markdown(f"**{bl_labels[bl]}**")
        st.plotly_chart(
            score_gauge(score=score, label=bl, max_score=1.0),
            width="stretch",
        )

# Tabla de senales etiquetadas (si existe el consolidado)
signals_df = load_latest_parquet("signals_consolidated_")
if signals_df is not None and selected_iso3 in signals_df["country_iso3"].values:
    row = signals_df[signals_df["country_iso3"] == selected_iso3].iloc[0]
    st.markdown("**Etiquetas de senal:**")
    badge_cols = st.columns(len(bl_keys))
    for col, bl in zip(badge_cols, bl_keys):
        sig = row.get(f"signal_{bl}", "N/D")
        emoji = signal_to_emoji(sig)
        col.markdown(
            f"<div style='text-align:center; padding:8px; "
            f"background:{SIGNAL_COLORS.get(sig, '#888')}; color:white; "
            f"border-radius:6px;'>{emoji} {bl}<br><small>{sig}</small></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Narrativa ejecutiva")
    st.info(row.get("narrative", "Narrativa no disponible."))

# ---------------------------------------------------------------------------
# Detalle por dimension
# ---------------------------------------------------------------------------
st.subheader("Detalle por dimension")
detail_rows = []
for dim in dim_names:
    detail_rows.append({
        "Dimension": dim,
        "Score": round(selected_scores[dim], 3),
        "Mediana regional": round(median_scores[dim], 3),
        "Brecha": round(selected_scores[dim] - median_scores[dim], 3),
    })
detail_df = pd.DataFrame(detail_rows)
st.dataframe(
    detail_df.style.background_gradient(subset=["Brecha"], cmap="RdYlGn"),
    width="stretch",
    hide_index=True,
)
