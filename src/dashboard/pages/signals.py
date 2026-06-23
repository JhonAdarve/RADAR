"""
Pagina 3 del dashboard RADAR Cibest: Senales por Linea de Negocio.

Para una linea de negocio seleccionada, muestra el mapa coropletico de
senales (4 niveles), la tabla de ranking especifico de esa linea y los
top-5 mercados con justificacion de la senal.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components.charts import choropleth_signals
from src.dashboard.components.common import (
    SIGNAL_COLORS,
    iso3_to_name_map,
    load_configurations,
    load_latest_parquet,
    signal_to_emoji,
)


st.title("Senales por Linea de Negocio")
st.caption(
    "Visualizacion diferenciada de oportunidades por linea de negocio. Cada "
    "linea utiliza un perfil de pesos especifico que prioriza variables "
    "criticas para su modelo operativo."
)

configs = load_configurations()
name_map = iso3_to_name_map(configs)
bl_cfg = configs["business_lines"]["business_lines"]

radar_by_line = load_latest_parquet("radar_by_line_")
signals_df = load_latest_parquet("signals_consolidated_")

if radar_by_line is None or signals_df is None:
    st.warning(
        "Resultados de scoring/senales no disponibles. Ejecute el pipeline "
        "completo y luego el modulo de senales."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Selector de linea de negocio
# ---------------------------------------------------------------------------
bl_options = list(bl_cfg.keys())
bl_labels = {k: bl_cfg[k]["label"] for k in bl_options}

selected_bl = st.radio(
    "Linea de negocio",
    options=bl_options,
    format_func=lambda k: bl_labels[k],
    horizontal=True,
)

st.markdown(f"**Descripcion:** {bl_cfg[selected_bl]['description']}")
st.markdown(f"**Logica de senal:** {bl_cfg[selected_bl]['signal_logic']}")

# ---------------------------------------------------------------------------
# Mapa de senales
# ---------------------------------------------------------------------------
st.subheader(f"Mapa de senales: {bl_labels[selected_bl]}")

signal_col = f"signal_{selected_bl}"
score_col = f"score_{selected_bl}"

if signal_col not in signals_df.columns:
    st.error(f"No hay senales para la linea {selected_bl}")
    st.stop()

map_df = signals_df[["country_iso3", signal_col, score_col]].copy()
map_df["country_name"] = map_df["country_iso3"].map(name_map)

fig_map = choropleth_signals(
    map_df, signal_col=signal_col,
    title=f"Senales de oportunidad - {bl_labels[selected_bl]}",
)
st.plotly_chart(fig_map, width="stretch")

# ---------------------------------------------------------------------------
# Top 5 mercados con justificacion
# ---------------------------------------------------------------------------
st.subheader("Top 5 mercados para esta linea")
top5 = signals_df.nlargest(5, score_col)[
    ["country_iso3", "country_name", score_col, signal_col, "narrative"]
].reset_index(drop=True)

for _, row in top5.iterrows():
    emoji = signal_to_emoji(row[signal_col])
    color = SIGNAL_COLORS.get(row[signal_col], "#888")
    st.markdown(
        f"""
        <div style='border-left: 5px solid {color};
                    padding: 10px 16px; margin-bottom: 12px;
                    background: #FAFAFA; border-radius: 4px;'>
            <h4 style='margin: 0;'>{emoji} {row['country_name']} 
                <span style='color: #666; font-size: 0.85em;'>({row['country_iso3']})</span>
            </h4>
            <p style='margin: 4px 0; font-size: 0.9em;'>
                <b>Score:</b> {row[score_col]:.3f} &nbsp;|&nbsp;
                <b>Senal:</b> {row[signal_col]}
            </p>
            <p style='margin: 6px 0 0 0; color: #444; font-size: 0.9em;'>
                {row['narrative'][:400]}...
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabla completa
# ---------------------------------------------------------------------------
st.subheader("Tabla completa para esta linea")
table = signals_df[["country_iso3", "country_name", score_col, signal_col]].copy()
table = table.sort_values(score_col, ascending=False).reset_index(drop=True)
table.insert(0, "Rank", table.index + 1)

st.dataframe(
    table.style.format({score_col: "{:.3f}"}),
    width="stretch",
    height=520,
    hide_index=True,
)

st.download_button(
    label="Descargar tabla (CSV)",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name=f"radar_senales_{selected_bl}.csv",
    mime="text/csv",
)
