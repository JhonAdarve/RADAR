"""
Tests unitarios para src/signals/business_line_signals.py y country_profile.py.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.signals.business_line_signals import (
    _classify_signal,
    evaluate_signal_for_line,
    generate_signal_matrix,
)
from src.signals.country_profile import (
    classify_dimensions,
    extract_dimension_scores,
    generate_all_profiles,
)


# ---------------------------------------------------------------------------
# Clasificacion de senales
# ---------------------------------------------------------------------------
def test_classify_signal_levels() -> None:
    thresholds = {
        "ALTA_OPORTUNIDAD": 0.70,
        "OPORTUNIDAD_MODERADA": 0.45,
        "BAJA_OPORTUNIDAD": 0.25,
    }
    assert _classify_signal(0.85, False, thresholds) == "ALTA_OPORTUNIDAD"
    assert _classify_signal(0.55, False, thresholds) == "OPORTUNIDAD_MODERADA"
    assert _classify_signal(0.30, False, thresholds) == "BAJA_OPORTUNIDAD"
    assert _classify_signal(0.10, False, thresholds) == "RIESGO"


def test_risk_override_forces_riesgo() -> None:
    thresholds = {
        "ALTA_OPORTUNIDAD": 0.70,
        "OPORTUNIDAD_MODERADA": 0.45,
        "BAJA_OPORTUNIDAD": 0.25,
    }
    # Score alto pero override activo -> RIESGO
    assert _classify_signal(0.95, True, thresholds) == "RIESGO"


# ---------------------------------------------------------------------------
# Perfiles
# ---------------------------------------------------------------------------
def test_extract_dimension_scores() -> None:
    ranking = pd.DataFrame({
        "score": [0.8, 0.6, 0.4],
        "rank": [1, 2, 3],
        "score_macro": [0.9, 0.5, 0.3],
        "score_financial": [0.7, 0.7, 0.5],
    }, index=["A", "B", "C"])
    dim_scores = extract_dimension_scores(ranking)
    assert "macro" in dim_scores.columns
    assert "financial" in dim_scores.columns
    assert "score" not in dim_scores.columns


def test_classify_dimensions_basic() -> None:
    scores = pd.DataFrame({
        "macro": [0.9, 0.5, 0.1],
        "financial": [0.6, 0.5, 0.4],
    }, index=["A", "B", "C"])
    classification = classify_dimensions(scores, strength_threshold=0.10, weakness_threshold=0.10)
    assert classification.loc["A", "macro"] == "strength"
    assert classification.loc["C", "macro"] == "weakness"


# ---------------------------------------------------------------------------
# Generacion de matriz de senales
# ---------------------------------------------------------------------------
def test_evaluate_signal_for_line() -> None:
    scores = pd.Series({"A": 0.9, "B": 0.5, "C": 0.1, "D": 0.7, "E": 0.3})
    wide_raw = pd.DataFrame({
        "political_stability": [1.0, 0.5, 0.0, 0.8, 0.3],
        "control_of_corruption": [1.0, 0.5, 0.0, 0.8, 0.3],
    }, index=scores.index)
    cfg = {
        "signal_thresholds": {
            "ALTA_OPORTUNIDAD": 0.70,
            "OPORTUNIDAD_MODERADA": 0.45,
            "BAJA_OPORTUNIDAD": 0.25,
            "risk_override": {
                "political_stability_percentile": 0.15,
                "corruption_percentile": 0.15,
            },
        }
    }
    signals = evaluate_signal_for_line(scores, "IB", wide_raw, cfg)
    assert "A" in signals.index
    assert signals.loc["A"] in {"ALTA_OPORTUNIDAD", "OPORTUNIDAD_MODERADA", "BAJA_OPORTUNIDAD", "RIESGO"}


def test_generate_signal_matrix_shape() -> None:
    radar_by_line = pd.DataFrame({
        "IB": [0.9, 0.7, 0.4, 0.2],
        "PF": [0.5, 0.8, 0.3, 0.4],
        "GLOBAL": [0.7, 0.75, 0.35, 0.30],
    }, index=["A", "B", "C", "D"])
    wide_raw = pd.DataFrame({
        "political_stability": [1.0, 0.8, 0.5, 0.3],
        "control_of_corruption": [1.0, 0.8, 0.5, 0.3],
    }, index=["A", "B", "C", "D"])
    cfg = {
        "business_lines": {
            "IB": {"label": "IB"},
            "PF": {"label": "PF"},
        },
        "signal_thresholds": {
            "ALTA_OPORTUNIDAD": 0.70,
            "OPORTUNIDAD_MODERADA": 0.45,
            "BAJA_OPORTUNIDAD": 0.25,
            "risk_override": {
                "political_stability_percentile": 0.15,
                "corruption_percentile": 0.15,
            },
        },
    }
    matrix = generate_signal_matrix(radar_by_line, wide_raw, cfg)
    assert matrix.shape == (4, 2)
    assert set(matrix.columns) == {"IB", "PF"}
