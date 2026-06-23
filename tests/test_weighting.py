"""
Tests unitarios para src/scoring/weighting.py.

Valida BWM, AHP, validacion de consistencia y agregacion de expertos.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring.weighting import (
    AHPWeighting,
    BWMWeighting,
    aggregate_expert_weights,
    build_weighting_method,
    compute_hierarchical_weights,
)
from src.utils import ConsistencyError


# ---------------------------------------------------------------------------
# BWM
# ---------------------------------------------------------------------------
def test_bwm_returns_normalized_weights(bwm_judgments: dict) -> None:
    bwm = BWMWeighting()
    weights = bwm.compute_weights(bwm_judgments)
    assert set(weights.keys()) == set(bwm_judgments["criteria"])
    assert abs(sum(weights.values()) - 1.0) < 1e-3


def test_bwm_best_has_highest_weight(bwm_judgments: dict) -> None:
    bwm = BWMWeighting()
    weights = bwm.compute_weights(bwm_judgments)
    best = bwm_judgments["best"]
    # El criterio "best" deberia tener uno de los pesos mas altos
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    assert best in [k for k, _ in sorted_w[:2]]


def test_bwm_worst_has_lowest_weight(bwm_judgments: dict) -> None:
    bwm = BWMWeighting()
    weights = bwm.compute_weights(bwm_judgments)
    worst = bwm_judgments["worst"]
    sorted_w = sorted(weights.items(), key=lambda x: x[1])
    assert worst in [k for k, _ in sorted_w[:2]]


def test_bwm_consistency_passes(bwm_judgments: dict) -> None:
    bwm = BWMWeighting(max_consistency_ratio=0.20)
    consistency = bwm.validate_consistency(bwm_judgments)
    assert "ratio" in consistency
    assert consistency["ratio"] >= 0


def test_bwm_invalid_best_worst_raises() -> None:
    bwm = BWMWeighting()
    bad = {
        "criteria": ["a", "b"],
        "best": "a", "worst": "a",
        "best_to_others": {"a": 1, "b": 1},
        "others_to_worst": {"a": 1, "b": 1},
    }
    with pytest.raises(Exception):
        bwm.compute_weights(bad)


# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------
def test_ahp_uniform_matrix() -> None:
    ahp = AHPWeighting()
    n = 4
    matrix = np.ones((n, n))
    judgments = {"criteria": ["a", "b", "c", "d"], "pairwise": matrix.tolist()}
    weights = ahp.compute_weights(judgments)
    expected = 1 / n
    for w in weights.values():
        assert abs(w - expected) < 1e-6


def test_ahp_consistency_perfect_matrix() -> None:
    ahp = AHPWeighting()
    matrix = [
        [1, 2, 4],
        [0.5, 1, 2],
        [0.25, 0.5, 1],
    ]  # Perfectamente consistente
    judgments = {"criteria": ["a", "b", "c"], "pairwise": matrix}
    consistency = ahp.validate_consistency(judgments)
    assert consistency["ratio"] < 0.10


def test_ahp_inconsistent_matrix_raises() -> None:
    ahp = AHPWeighting(max_consistency_ratio=0.05)
    matrix = [
        [1, 9, 1/9],
        [1/9, 1, 9],
        [9, 1/9, 1],
    ]  # Maximamente inconsistente
    judgments = {"criteria": ["a", "b", "c"], "pairwise": matrix}
    with pytest.raises(ConsistencyError):
        ahp.validate_consistency(judgments)


# ---------------------------------------------------------------------------
# Agregacion y jerarquia
# ---------------------------------------------------------------------------
def test_aggregate_geometric_mean() -> None:
    experts = [
        {"a": 0.4, "b": 0.6},
        {"a": 0.6, "b": 0.4},
    ]
    agg = aggregate_expert_weights(experts, method="geometric_mean")
    assert abs(sum(agg.values()) - 1.0) < 1e-9
    # Por simetria deberian ser iguales
    assert abs(agg["a"] - agg["b"]) < 1e-6


def test_hierarchical_weights_sum_to_one() -> None:
    dim_w = {"d1": 0.3, "d2": 0.7}
    var_w = {
        "d1": {"v1": 0.4, "v2": 0.6},
        "d2": {"v3": 0.5, "v4": 0.5},
    }
    final = compute_hierarchical_weights(dim_w, var_w)
    assert abs(sum(final.values()) - 1.0) < 1e-9
    assert abs(final["v1"] - 0.3 * 0.4) < 1e-9
    assert abs(final["v3"] - 0.7 * 0.5) < 1e-9


def test_factory_returns_correct_class() -> None:
    bwm = build_weighting_method("bwm")
    ahp = build_weighting_method("ahp")
    assert isinstance(bwm, BWMWeighting)
    assert isinstance(ahp, AHPWeighting)
    with pytest.raises(ValueError):
        build_weighting_method("electre")
