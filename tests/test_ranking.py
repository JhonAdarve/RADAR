"""
Tests unitarios para src/scoring/ranking.py.

Valida TOPSIS, VIKOR y la interfaz de patron Strategy.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preparation.normalization import normalize
from src.scoring.ranking import (
    TOPSISRanking,
    VIKORRanking,
    build_ranking_method,
)
from src.utils import get_variable_catalog


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------
def test_topsis_score_in_unit_interval(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    decision = normalize(synthetic_wide_df, method="min_max", variable_catalog=catalog)
    weights = {col: 1 / decision.shape[1] for col in decision.columns}
    ranker = TOPSISRanking()
    ranking = ranker.rank(decision, weights, catalog)
    assert "score" in ranking.columns
    assert (ranking["score"] >= 0).all() and (ranking["score"] <= 1).all()


def test_topsis_rank_is_unique(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    decision = normalize(synthetic_wide_df, method="min_max", variable_catalog=catalog)
    weights = {col: 1 / decision.shape[1] for col in decision.columns}
    ranker = TOPSISRanking()
    ranking = ranker.rank(decision, weights, catalog)
    # Numero de ranks distintos debe ser igual al numero de filas
    assert ranking["rank"].min() == 1
    assert ranking["rank"].max() == len(ranking)


def test_topsis_includes_dimension_scores(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    decision = normalize(synthetic_wide_df, method="min_max", variable_catalog=catalog)
    weights = {col: 1 / decision.shape[1] for col in decision.columns}
    ranker = TOPSISRanking()
    ranking = ranker.rank(decision, weights, catalog)
    # Debe haber al menos una columna score_<dimension>
    dim_cols = [c for c in ranking.columns if c.startswith("score_") and c != "score"]
    assert len(dim_cols) >= 3


def test_topsis_known_scenario() -> None:
    """Escenario controlado: 3 alternativas, una claramente dominante."""
    decision = pd.DataFrame({
        "v1": [1.0, 0.5, 0.0],
        "v2": [1.0, 0.5, 0.0],
    }, index=["best", "mid", "worst"])
    weights = {"v1": 0.5, "v2": 0.5}
    catalog = {
        "v1": {"dimension": "d1", "direction": "positive"},
        "v2": {"dimension": "d1", "direction": "positive"},
    }
    ranker = TOPSISRanking()
    ranking = ranker.rank(decision, weights, catalog)
    assert ranking.index[0] == "best"
    assert ranking.index[-1] == "worst"


def test_topsis_missing_weights_raises(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    decision = normalize(synthetic_wide_df, method="min_max", variable_catalog=catalog)
    weights = {col: 1 / 2 for col in list(decision.columns)[:2]}  # solo 2 pesos
    ranker = TOPSISRanking()
    with pytest.raises(Exception):
        ranker.rank(decision, weights, catalog)


# ---------------------------------------------------------------------------
# VIKOR
# ---------------------------------------------------------------------------
def test_vikor_score_in_unit_interval(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    decision = normalize(synthetic_wide_df, method="min_max", variable_catalog=catalog)
    weights = {col: 1 / decision.shape[1] for col in decision.columns}
    ranker = VIKORRanking()
    ranking = ranker.rank(decision, weights, catalog)
    assert (ranking["score"] >= 0).all() and (ranking["score"] <= 1).all()


def test_vikor_invalid_v_raises() -> None:
    with pytest.raises(Exception):
        VIKORRanking(v=1.5)


# ---------------------------------------------------------------------------
# Factoria
# ---------------------------------------------------------------------------
def test_factory_creates_correct_instance() -> None:
    assert isinstance(build_ranking_method("topsis"), TOPSISRanking)
    assert isinstance(build_ranking_method("vikor"), VIKORRanking)
    with pytest.raises(ValueError):
        build_ranking_method("promethee")


def test_topsis_vikor_correlation_high(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    """TOPSIS y VIKOR con los mismos pesos deberian correlacionar fuertemente."""
    catalog = get_variable_catalog(synthetic_variables_cfg)
    decision = normalize(synthetic_wide_df, method="min_max", variable_catalog=catalog)
    weights = {col: 1 / decision.shape[1] for col in decision.columns}

    r_t = TOPSISRanking().rank(decision, weights, catalog)
    r_v = VIKORRanking().rank(decision, weights, catalog)

    corr = r_t["score"].corr(r_v["score"], method="spearman")
    assert corr > 0.7  # correlacion al menos moderada-alta
