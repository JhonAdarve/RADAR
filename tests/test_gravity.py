"""
Tests unitarios para src/scoring/gravity.py.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.scoring.gravity import compute_gravity_flow, compute_ipc


def test_ipc_in_unit_interval(synthetic_wide_df: pd.DataFrame) -> None:
    ipc_df = compute_ipc(synthetic_wide_df, origin_country="COL")
    assert "ipc" in ipc_df.columns
    assert (ipc_df["ipc"] >= 0).all() and (ipc_df["ipc"] <= 1).all()


def test_ipc_origin_is_one(synthetic_wide_df: pd.DataFrame) -> None:
    ipc_df = compute_ipc(synthetic_wide_df, origin_country="COL")
    assert abs(ipc_df.loc["COL", "ipc"] - 1.0) < 1e-6


def test_ipc_decreases_with_distance(synthetic_wide_df: pd.DataFrame) -> None:
    """Espana esta mas lejos de Colombia que Panama: deberia tener menor IPC."""
    ipc_df = compute_ipc(synthetic_wide_df, origin_country="COL")
    # Excluyendo COL (que es 1.0), comparar PAN vs ESP
    assert ipc_df.loc["PAN", "ipc"] > ipc_df.loc["ESP", "ipc"]


def test_ipc_missing_variables_raises() -> None:
    bad = pd.DataFrame({"random_col": [1, 2, 3]}, index=["A", "B", "C"])
    with pytest.raises(Exception):
        compute_ipc(bad)


def test_gravity_flow_origin_is_max(synthetic_wide_df: pd.DataFrame) -> None:
    # Necesitamos gdp_nominal en el wide; si no esta, el test se skipea
    if "gdp_nominal" not in synthetic_wide_df.columns:
        # Agregar columna sintetica
        wide = synthetic_wide_df.copy()
        wide["gdp_nominal"] = [320, 1300, 60, 280, 1400]  # USD billones
        flow = compute_gravity_flow(wide, origin_country="COL")
        assert flow.loc["COL"] == flow.max()


def test_compute_ipc_with_custom_weights(synthetic_wide_df: pd.DataFrame) -> None:
    weights = {
        "geographic_distance_km": 0.5,
        "common_language_spanish": 0.5,
    }
    ipc_df = compute_ipc(synthetic_wide_df, origin_country="COL", component_weights=weights)
    assert "ipc" in ipc_df.columns
    assert len(ipc_df) == len(synthetic_wide_df)
