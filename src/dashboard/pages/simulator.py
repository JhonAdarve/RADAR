"""
Pagina 4 del dashboard RADAR Cibest: Simulador Que-Pasa-Si.

Permite a la alta direccion ajustar los pesos de las cinco dimensiones
en tiempo real y ver el ranking recalculado, con destacado de paises
que cambian significativamente de posicion respecto al ranking original.

Esta pagina es especialmente util durante la sesion de elicitacion BWM
y para responder preguntas estrategicas como:
    - "Que pasa si priorizamos mas la dimension digital?"
    - "Que pasa si reducimos el peso de la proximidad con Colombia?"

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components.common import (
    CIBEST_PALETTE,
    iso3_to_name_map,
    load_configurations,
    load_latest_parquet,
)
from src.scoring.hybrid_scorer import (
    compute_radar_composite,
    compute_trend_factor,
)
from src.scoring.ranking import TOPSISRanking
from src.scoring.weighting import compute_hierarchical_weights
from src.utils import get_variable_catalog, resolve_data_path


st.title("Simulador Que-Pasa-Si")
st.caption(
    "Ajuste los pesos dimensionales con los sliders y observe en tiempo real "
    "como cambia el ranking. Los paises que suben o bajan mas de 3 posiciones "
    "se destacan."
)

configs = load_configurations()
name_map = iso3_to_name_map(configs)
catalog = get_variable_catalog(configs["variables"])

# Cargar matriz de decision desde el ultimo scoring (debe persistirse en pipeline)
decision_matrix_path = resolve_data_path("data/processed") / "decision_matrix_latest.parquet"
wide_raw_path = resolve_data_path("data/processed") / "wide_raw_latest.parquet"

if not decision_matrix_path.exists():
    st.warning(
        "Matriz de decision no encontrada. Ejecute primero el pipeline de "
        "scoring que persiste decision_matrix_latest.parquet en data/processed/."
    )
    st.stop()

decision_matrix = pd.read_parquet(decision_matrix_path)
ipc_df = load_latest_parquet("ipc_")
ipc = ipc_df["ipc"] if ipc_df is not None else pd.Series(0.5, index=decision_matrix.index)

# Tendencia desde el master raw (si esta disponible) o se aproxima
trend = pd.Series(0.5, index=decision_matrix.index)

# ---------------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------------
st.sidebar.header("Pesos dimensionales")
default_weights = configs["weights"]["dimension_weights"]

dim_weights: dict[str, float] = {}
for dim, default_val in default_weights.items():
    label = configs["variables"]["dimensions"][dim]["label"]
    dim_weights[dim] = st.sidebar.slider(
        f"{label}",
        min_value=0.0,
        max_value=1.0,
        value=float(default_val),
        step=0.05,
    )

# Renormalizar
total = sum(dim_weights.values())
if total == 0:
    st.error("Los pesos no pueden sumar cero")
    st.stop()
dim_weights = {k: v / total for k, v in dim_weights.items()}

st.sidebar.markdown("### Pesos del score compuesto")
weights_comp = configs["settings"]["scoring"]["composite_weights"]
alpha = st.sidebar.slider("Alpha (TOPSIS)", 0.0, 1.0, float(weights_comp["alpha"]), 0.05)
beta = st.sidebar.slider("Beta (Proximidad)", 0.0, 1.0, float(weights_comp["beta"]), 0.05)
gamma = st.sidebar.slider("Gamma (Tendencia)", 0.0, 1.0, float(weights_comp["gamma"]), 0.05)

# ---------------------------------------------------------------------------
# Recalculo
# ---------------------------------------------------------------------------
var_weights_by_dim = configs["weights"]["variable_weights"]
var_weights = compute_hierarchical_weights(dim_weights, var_weights_by_dim)
var_weights = {k: v for k, v in var_weights.items() if k in decision_matrix.columns}
# Renormalizar tras filtrar
total_var = sum(var_weights.values())
if total_var > 0:
    var_weights = {k: v / total_var for k, v in var_weights.items()}

ranker = TOPSISRanking()
new_ranking = ranker.rank(decision_matrix, var_weights, catalog)
new_radar = compute_radar_composite(
    new_ranking["score"], ipc.reindex(new_ranking.index).fillna(ipc.median()),
    trend.reindex(new_ranking.index).fillna(trend.median()),
    alpha, beta, gamma,
)
new_radar = new_radar.sort_values(ascending=False)

# Cargar ranking original
original_radar_df = load_latest_parquet("radar_by_line_")
if original_radar_df is None or "GLOBAL" not in original_radar_df.columns:
    st.warning("Ranking original no disponible, no se puede comparar.")
    st.stop()
original_radar = original_radar_df["GLOBAL"]
original_rank = original_radar.rank(ascending=False, method="min").astype(int)

new_rank = new_radar.rank(ascending=False, method="min").astype(int)

# ---------------------------------------------------------------------------
# Tabla comparativa
# ---------------------------------------------------------------------------
comparison = pd.DataFrame({
    "country_iso3": new_radar.index,
    "country_name": [name_map.get(c, c) for c in new_radar.index],
    "rank_simulado": new_rank.reindex(new_radar.index).values,
    "rank_original": original_rank.reindex(new_radar.index).values,
    "score_simulado": new_radar.values,
})
comparison["delta_rank"] = (
    comparison["rank_original"] - comparison["rank_simulado"]
).fillna(0).astype(int)
comparison = comparison.sort_values("rank_simulado").reset_index(drop=True)


def _highlight_changes(row: pd.Series) -> list[str]:
    """Resalta filas con cambios > 3 posiciones."""
    if abs(row["delta_rank"]) >= 3:
        color = CIBEST_PALETTE["gold"] if row["delta_rank"] > 0 else "#FFD0D0"
        return [f"background-color: {color}; font-weight: 600"] * len(row)
    return [""] * len(row)


st.subheader("Comparacion de rankings")
col_orig, col_sim = st.columns(2)

with col_orig:
    st.markdown("**Ranking original (pesos por defecto)**")
    orig_display = pd.DataFrame({
        "Pais": [name_map.get(c, c) for c in original_radar.sort_values(ascending=False).index],
        "Score": original_radar.sort_values(ascending=False).round(3).values,
    }).head(15)
    orig_display.insert(0, "Rank", range(1, len(orig_display) + 1))
    st.dataframe(orig_display, width="stretch", hide_index=True, height=520)

with col_sim:
    st.markdown("**Ranking simulado (sus pesos)**")
    sim_display = comparison[["rank_simulado", "country_name", "score_simulado", "delta_rank"]].head(15).copy()
    sim_display.columns = ["Rank", "Pais", "Score", "Δ Rank"]
    st.dataframe(
        sim_display.style.apply(
            lambda r: [
                f"background-color: {CIBEST_PALETTE['gold']}; font-weight: 600" if r["Δ Rank"] >= 3
                else f"background-color: #FFD0D0; font-weight: 600" if r["Δ Rank"] <= -3
                else "" for _ in r
            ], axis=1,
        ).format({"Score": "{:.3f}"}),
        width="stretch", hide_index=True, height=520,
    )

# ---------------------------------------------------------------------------
# Mayores movimientos
# ---------------------------------------------------------------------------
st.subheader("Mayores movimientos")
movers_up = comparison.nlargest(5, "delta_rank")
movers_down = comparison.nsmallest(5, "delta_rank")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**🔼 Mayor ascenso**")
    st.dataframe(
        movers_up[["country_name", "rank_original", "rank_simulado", "delta_rank"]],
        width="stretch", hide_index=True,
    )
with c2:
    st.markdown("**🔽 Mayor descenso**")
    st.dataframe(
        movers_down[["country_name", "rank_original", "rank_simulado", "delta_rank"]],
        width="stretch", hide_index=True,
    )
