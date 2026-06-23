"""
Pagina 5 del dashboard RADAR Cibest: Tendencias Historicas.

Muestra series de tiempo de variables seleccionables con un comparador
multi-pais. Util para entender la dinamica reciente de los mercados y
detectar inflexiones que ameriten alertas.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.common import (
    CIBEST_PALETTE,
    iso3_to_name_map,
    load_configurations,
    load_latest_parquet,
)


st.title("Tendencias Historicas")
st.caption(
    "Evolucion de variables clave por pais. Permite identificar dinamismo "
    "reciente, regresiones macroeconomicas o saltos institucionales."
)

configs = load_configurations()
name_map = iso3_to_name_map(configs)

master_df = load_latest_parquet("master_raw_", scores_dir="data/raw/")
if master_df is None:
    st.warning(
        "Master raw no encontrado en data/raw/. Ejecute primero el pipeline "
        "de extraccion: `python -m src.data_extraction.pipeline`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Selectores
# ---------------------------------------------------------------------------
catalog_dimensions = configs["variables"]["dimensions"]
flat_vars: dict[str, dict] = {}
for dim_key, dim_content in catalog_dimensions.items():
    for var, meta in dim_content["variables"].items():
        flat_vars[var] = {**meta, "dimension": dim_key, "dim_label": dim_content["label"]}

available_vars = sorted([v for v in flat_vars if v in master_df["variable"].unique()])

if not available_vars:
    st.error("No hay variables con datos historicos disponibles")
    st.stop()

col_selector, col_countries = st.columns([2, 3])

with col_selector:
    selected_var = st.selectbox(
        "Variable",
        options=available_vars,
        format_func=lambda v: f"{flat_vars[v]['dim_label']} - {v}",
    )

with col_countries:
    countries_in_data = sorted(master_df[master_df["variable"] == selected_var]["country_iso3"].unique())
    default_set = ["COL", "MEX", "PAN", "CHL", "BRA", "ESP"]
    default_select = [c for c in default_set if c in countries_in_data][:5]
    selected_countries = st.multiselect(
        "Paises a comparar (max recomendado: 6)",
        options=countries_in_data,
        default=default_select,
        format_func=lambda c: f"{name_map.get(c, c)} ({c})",
    )

if not selected_countries:
    st.info("Seleccione al menos un pais")
    st.stop()

# ---------------------------------------------------------------------------
# Grafico de series de tiempo
# ---------------------------------------------------------------------------
df_filtered = master_df[
    (master_df["variable"] == selected_var)
    & (master_df["country_iso3"].isin(selected_countries))
    & (master_df["year"] > 0)
].sort_values(["country_iso3", "year"])

if df_filtered.empty:
    st.warning("No hay datos disponibles para la combinacion seleccionada")
    st.stop()

fig = go.Figure()
palette = [
    CIBEST_PALETTE["gray"],
    CIBEST_PALETTE["gold"],
    CIBEST_PALETTE["gray_light"],
    CIBEST_PALETTE["gold_dark"],
    CIBEST_PALETTE["green"],
    CIBEST_PALETTE["amber"],
]

for i, country in enumerate(selected_countries):
    df_c = df_filtered[df_filtered["country_iso3"] == country]
    if df_c.empty:
        continue
    fig.add_trace(go.Scatter(
        x=df_c["year"],
        y=df_c["value"],
        mode="lines+markers",
        name=name_map.get(country, country),
        line={"color": palette[i % len(palette)], "width": 2.5},
        marker={"size": 7},
    ))

fig.update_layout(
    title=f"{flat_vars[selected_var]['description']}",
    xaxis_title="Ano",
    yaxis_title=selected_var,
    hovermode="x unified",
    height=520,
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
)
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Alertas: variaciones > 10% YoY en el ultimo ano disponible
# ---------------------------------------------------------------------------
st.subheader("Alertas de variacion")

alerts: list[dict] = []
for country in selected_countries:
    df_c = df_filtered[df_filtered["country_iso3"] == country].sort_values("year")
    if len(df_c) < 2:
        continue
    last = df_c.iloc[-1]
    prev = df_c.iloc[-2]
    if pd.notna(prev["value"]) and prev["value"] != 0:
        change_pct = (last["value"] - prev["value"]) / abs(prev["value"]) * 100
        if abs(change_pct) >= 10:
            alerts.append({
                "Pais": name_map.get(country, country),
                "Ano": int(last["year"]),
                "Valor anterior": round(float(prev["value"]), 3),
                "Valor actual": round(float(last["value"]), 3),
                "Variacion %": round(change_pct, 1),
            })

if alerts:
    alerts_df = pd.DataFrame(alerts).sort_values("Variacion %", key=abs, ascending=False)
    st.dataframe(
        alerts_df.style.background_gradient(subset=["Variacion %"], cmap="RdYlGn"),
        width="stretch", hide_index=True,
    )
else:
    st.success("Sin alertas: ninguna variacion > 10% en el ultimo ano disponible.")
