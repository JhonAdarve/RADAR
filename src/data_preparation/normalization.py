"""
Normalizacion de la matriz de decision para RADAR Cibest.

Implementa min-max, z-score y vector normalization con orientacion
automatica de variables segun direction declarada en variables.yaml.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import DataPreparationError


def normalize_min_max(wide: pd.DataFrame) -> pd.DataFrame:
    """Normalizacion min-max a [0, 1] por columna."""
    minv = wide.min()
    maxv = wide.max()
    rng = maxv - minv
    rng_safe = rng.replace(0, np.nan)
    out = (wide - minv) / rng_safe
    out = out.fillna(0.5)
    return out


def normalize_z_score(wide: pd.DataFrame) -> pd.DataFrame:
    """Estandarizacion z-score por columna."""
    mu = wide.mean()
    sigma = wide.std(ddof=0)
    sigma_safe = sigma.replace(0, np.nan)
    out = (wide - mu) / sigma_safe
    return out.fillna(0.0)


def normalize_vector(wide: pd.DataFrame) -> pd.DataFrame:
    """Normalizacion vectorial (norma L2 = 1 por columna)."""
    norms = np.sqrt((wide ** 2).sum())
    norms_safe = norms.replace(0, np.nan)
    out = wide / norms_safe
    return out.fillna(0.0)


def apply_direction(
    wide_normalized: pd.DataFrame,
    variable_catalog: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Aplica direccion (positive/negative) a cada variable.

    Las variables 'negative' se invierten para que mayor valor = mejor.
    """
    out = wide_normalized.copy()
    min_ = out.min().min()
    max_ = out.max().max()
    use_minmax_flip = (min_ >= 0) and (max_ <= 1.001)

    inverted: list[str] = []
    for var in out.columns:
        meta = variable_catalog.get(var)
        if meta is None:
            continue
        if meta["direction"] == "negative":
            if use_minmax_flip:
                out[var] = 1.0 - out[var]
            else:
                out[var] = -out[var]
            inverted.append(var)

    logger.info(
        "Direccion aplicada: {n} variables invertidas -> {lst}",
        n=len(inverted), lst=inverted,
    )
    return out


def normalize(
    wide: pd.DataFrame,
    method: str = "min_max",
    variable_catalog: Dict[str, Dict[str, Any]] | None = None,
    apply_direction_flag: bool = True,
) -> pd.DataFrame:
    """Pipeline de normalizacion end-to-end.

    Args:
        wide: Matriz ancha imputada.
        method: 'min_max' | 'z_score' | 'vector'.
        variable_catalog: Catalogo aplanado, requerido si apply_direction_flag.
        apply_direction_flag: Si True aplica direcciones tras normalizar.

    Returns:
        Matriz normalizada y orientada lista para scoring.
    """
    if method == "min_max":
        normalized = normalize_min_max(wide)
    elif method == "z_score":
        normalized = normalize_z_score(wide)
    elif method == "vector":
        normalized = normalize_vector(wide)
    else:
        raise DataPreparationError(f"Metodo de normalizacion invalido: {method}")

    logger.info(
        "Normalizacion {m}: {c} paises x {v} variables",
        m=method, c=len(normalized), v=normalized.shape[1],
    )

    if apply_direction_flag:
        if variable_catalog is None:
            raise DataPreparationError(
                "apply_direction_flag=True requiere variable_catalog"
            )
        normalized = apply_direction(normalized, variable_catalog)

    return normalized
