"""
Pagina 1 del dashboard RADAR Cibest: Ranking General.

Muestra el mapa coropletico de America + Espana coloreado por score
RADAR global y la tabla de ranking ordenable y filtrable.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components.charts import choropleth_americas, ranking_bar_chart
from src.dashboard.components.common import (
    CIBEST_PALETTE,
    iso3_to_name_map,
    load_configurations,
    load_latest_parquet,
)


st.title("Ranking General")
st.caption(
    "Atractivo agregado de los mercados evaluados con el enfoque multicriteria "
    "hibrido. El score combina ranking absoluto, proximidad bilateral con "
    "Colombia y dinamismo reciente."
)

configs = load_configurations()
name_map = iso3_to_name_map(configs)

radar_df = load_latest_parquet("radar_by_line_")
if radar_df is None:
    st.warning(
        "No se encontraron resultados de scoring en data/scores/. "
        "Ejecute primero el pipeline completo: `python -m src.scoring.hybrid_scorer`."
    )
    st.stop()

radar_df = radar_df.reset_index().rename(columns={"index": "country_iso3"})
radar_df["country_name"] = radar_df["country_iso3"].map(name_map)

# ---------------------------------------------------------------------------
# Filtros laterales
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")
regions = sorted({c["region"] for c in configs["settings"]["countries"]})
selected_regions = st.sidebar.multiselect(
    "Regiones", options=regions, default=regions,
)
region_map = {c["iso3"]: c["region"] for c in configs["settings"]["countries"]}
radar_df["region"] = radar_df["country_iso3"].map(region_map)
radar_df = radar_df[radar_df["region"].isin(selected_regions)]

top_n = st.sidebar.slider("Mostrar top N en grafico", 5, 30, 15)

# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------
col_map, col_bar = st.columns([3, 2])

with col_map:
    st.subheader("Mapa de atractivo")
    fig_map = choropleth_americas(
        radar_df, value_col="GLOBAL", title="Score RADAR Global por mercado",
    )
    st.plotly_chart(fig_map, width="stretch")

with col_bar:
    st.subheader(f"Top {top_n}")
    fig_bar = ranking_bar_chart(
        radar_df, score_col="GLOBAL", country_col="country_name", top_n=top_n,
    )
    st.plotly_chart(fig_bar, width="stretch")

# ---------------------------------------------------------------------------
# Tabla detallada
# ---------------------------------------------------------------------------
st.subheader("Tabla de ranking")
display_cols = ["rank_global", "country_iso3", "country_name", "GLOBAL",
                "IB", "PF", "AD", "BD", "CIB"]
present = [c for c in display_cols if c in radar_df.columns]
table = radar_df[present].sort_values("rank_global").reset_index(drop=True)

styled = table.style.format(
    {c: "{:.3f}" for c in present if c not in {"rank_global", "country_iso3", "country_name"}}
).background_gradient(
    subset=[c for c in present if c not in {"rank_global", "country_iso3", "country_name"}],
    cmap="YlGnBu",
)
st.dataframe(styled, width="stretch", height=520)

st.download_button(
    label="Descargar tabla (CSV)",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name="radar_ranking_general.csv",
    mime="text/csv",
)
