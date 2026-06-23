"""
Tests unitarios para src/data_preparation/normalization.py y cleaning.py.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preparation.cleaning import (
    impute_missing,
    pivot_long_to_wide,
    validate_country_coverage,
)
from src.data_preparation.normalization import (
    apply_direction,
    normalize,
    normalize_min_max,
    normalize_vector,
    normalize_z_score,
)
from src.utils import get_variable_catalog


# ---------------------------------------------------------------------------
# Tests de pivoteo y limpieza
# ---------------------------------------------------------------------------
def test_pivot_long_to_wide_basic(synthetic_long_df: pd.DataFrame) -> None:
    wide = pivot_long_to_wide(synthetic_long_df, year_strategy="latest_available")
    assert isinstance(wide, pd.DataFrame)
    assert "COL" in wide.index
    assert "gdp_per_capita_ppp" in wide.columns
    assert wide.loc["COL", "common_language_spanish"] == 1.0


def test_pivot_takes_latest_year(synthetic_long_df: pd.DataFrame) -> None:
    wide = pivot_long_to_wide(synthetic_long_df, year_strategy="latest_available")
    # El valor mas reciente para 2024 debe ser cercano al valor base con ruido
    assert 14000 < wide.loc["COL", "gdp_per_capita_ppp"] < 18000


def test_validate_country_coverage_excludes_low(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    # Generar un DataFrame con un pais sin datos
    wide = synthetic_wide_df.copy()
    wide.loc["MEX", :] = np.nan
    wide_clean, excluded = validate_country_coverage(wide, catalog, missing_threshold=0.30)
    assert "MEX" in excluded


def test_impute_global_median_no_nulls(synthetic_wide_df: pd.DataFrame) -> None:
    wide = synthetic_wide_df.copy()
    wide.iloc[0, 0] = np.nan
    imputed = impute_missing(wide, method="global_median")
    assert imputed.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Tests de normalizacion
# ---------------------------------------------------------------------------
def test_min_max_in_unit_interval(synthetic_wide_df: pd.DataFrame) -> None:
    norm = normalize_min_max(synthetic_wide_df)
    assert (norm.values >= 0).all() and (norm.values <= 1).all()


def test_z_score_zero_mean(synthetic_wide_df: pd.DataFrame) -> None:
    norm = normalize_z_score(synthetic_wide_df)
    means = norm.mean()
    assert np.allclose(means.values, 0.0, atol=1e-9)


def test_vector_unit_norm(synthetic_wide_df: pd.DataFrame) -> None:
    norm = normalize_vector(synthetic_wide_df)
    norms = np.sqrt((norm ** 2).sum())
    assert np.allclose(norms.values, 1.0, atol=1e-6)


def test_apply_direction_inverts_negative(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    norm = normalize_min_max(synthetic_wide_df)
    oriented = apply_direction(norm, catalog)
    # Inflation_rate es negative: el pais con mas inflacion (COL) debe quedar mas bajo
    inf_col = "inflation_rate"
    if inf_col in oriented.columns:
        # COL tiene mayor inflacion (9.5) -> tras invertir debe quedar mas bajo
        assert oriented.loc["COL", inf_col] < oriented.loc["CHL", inf_col]


def test_normalize_pipeline_output_shape(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    out = normalize(synthetic_wide_df, method="min_max", variable_catalog=catalog)
    assert out.shape == synthetic_wide_df.shape
    assert (out.values >= 0).all() and (out.values <= 1).all()


@pytest.mark.parametrize("method", ["min_max", "z_score", "vector"])
def test_normalize_methods_run(synthetic_wide_df: pd.DataFrame, synthetic_variables_cfg: dict, method: str) -> None:
    catalog = get_variable_catalog(synthetic_variables_cfg)
    out = normalize(synthetic_wide_df, method=method, variable_catalog=catalog)
    assert out.shape == synthetic_wide_df.shape
    assert not out.isna().any().any()
