"""
Modelo gravitacional de proximidad para RADAR Cibest.

Calcula el Indice de Proximidad Compuesto (IPC) Colombia <-> destino
siguiendo la literatura gravitacional aplicada a flujos financieros
(Anderson & van Wincoop 2003; Brei & von Peter 2018).

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import ScoringError


PROXIMITY_VARIABLES_MAP: Dict[str, str] = {
    "geographic_distance_km": "negative",
    "cultural_distance_hofstede": "negative",
    "common_language_spanish": "positive",
    #"bilateral_trade_colombia": "positive",
    "colombian_diaspora_stock": "positive",
}


def _normalize_component(series: pd.Series, direction: str) -> pd.Series:
    """Normaliza una serie a [0, 1] aplicando direccion."""
    s = series.astype(float)
    rng = s.max() - s.min()
    if rng == 0 or pd.isna(rng):
        return pd.Series(0.5, index=series.index)
    norm = (s - s.min()) / rng
    if direction == "negative":
        norm = 1.0 - norm
    return norm.fillna(norm.mean())


def compute_ipc(
    wide_raw: pd.DataFrame,
    origin_country: str = "COL",
    component_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Calcula el Indice de Proximidad Compuesto Colombia -> cada pais.

    Args:
        wide_raw: Matriz ancha RAW con variables de proximidad.
        origin_country: ISO3 del origen (Colombia por defecto).
        component_weights: Pesos de cada componente. None = equiponderacion.

    Returns:
        DataFrame con ipc y scores parciales por componente.
    """
    available = [v for v in PROXIMITY_VARIABLES_MAP if v in wide_raw.columns]
    if not available:
        raise ScoringError(
            "No hay variables de proximidad en la matriz ancha. "
            "Verifique que se hayan extraido las variables complementarias."
        )

    logger.info("IPC se calculara con {n} componentes: {lst}", n=len(available), lst=available)

    if component_weights is None:
        component_weights = {v: 1.0 / len(available) for v in available}
    total = sum(component_weights.get(v, 0) for v in available)
    if total == 0:
        raise ScoringError("Pesos de componentes IPC suman 0")
    component_weights = {v: component_weights.get(v, 0) / total for v in available}

    partial_scores: Dict[str, pd.Series] = {}
    for var in available:
        direction = PROXIMITY_VARIABLES_MAP[var]
        partial_scores[var] = _normalize_component(wide_raw[var], direction)

    partial_df = pd.DataFrame(partial_scores)

    if origin_country in partial_df.index:
        partial_df.loc[origin_country] = 1.0

    weights_series = pd.Series(component_weights).reindex(partial_df.columns)
    ipc = partial_df.mul(weights_series, axis=1).sum(axis=1)
    ipc = ipc.clip(lower=0.0, upper=1.0)

    out = partial_df.add_suffix("_proximity_score")
    out["ipc"] = ipc
    out = out.sort_values("ipc", ascending=False)

    logger.info(
        "IPC calculado: top-3 afinidad -> {top}",
        top=dict(zip(out.index[:3], out["ipc"].head(3).round(3))),
    )
    return out


def compute_gravity_flow(
    wide_raw: pd.DataFrame,
    origin_country: str = "COL",
    mass_variable: str = "gdp_nominal",
) -> pd.Series:
    """Flujo gravitacional estilizado (alternativa al IPC)."""
    if mass_variable not in wide_raw.columns:
        raise ScoringError(f"Variable de masa no disponible: {mass_variable}")
    if "geographic_distance_km" not in wide_raw.columns:
        raise ScoringError("Distancia geografica no disponible")

    m_origin = wide_raw.loc[origin_country, mass_variable] if origin_country in wide_raw.index else 1.0
    m_j = wide_raw[mass_variable].replace(0, np.nan)
    dist = wide_raw["geographic_distance_km"].replace(0, np.nan)
    cultural = wide_raw.get("cultural_distance_hofstede", pd.Series(0, index=wide_raw.index))
    trade = wide_raw.get("bilateral_trade_colombia", pd.Series(0, index=wide_raw.index))

    flow = (
        np.log(m_origin * m_j.fillna(m_j.mean()))
        - 1.0 * np.log(dist.fillna(dist.mean()))
        - 0.3 * cultural.fillna(0)
        + 0.2 * np.log1p(trade.fillna(0))
    )
    flow[origin_country] = flow.max() + 1.0
    flow_norm = (flow - flow.min()) / (flow.max() - flow.min())
    return flow_norm.fillna(0.5)
