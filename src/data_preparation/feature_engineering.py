"""
Ingenieria de caracteristicas para RADAR Cibest.

Genera variables derivadas (logs, indices compuestos) que enriquecen la
matriz de decision antes de la normalizacion.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger


LOG_CANDIDATES: List[str] = [

    # Macro (skewed pero fundamentales, no se pueden excluir)
    'gdp_nominal',
    'population_total',

    # Proximidad (skewed pero clave para el IPC, no se pueden excluir)
    'geographic_distance_km',
    'colombian_diaspora_stock',  #variable de proximidad basada en stock de colombianos en el exterior

    # financieras (skewed pero relevantes para internacionalizacion, se pueden evaluar para exclusion futura si se desea un indice mas parsimonioso)
    'stock_market_cap_gdp',
    'financial_system_deposits_gdp',
    'domestic_credit_private_gdp',

    # flujos (skewed pero relevantes para internacionalizacion, se pueden evaluar para exclusion futura si se desea un indice mas parsimonioso)
    'fdi_net_inflows_gdp',
    'personal_remittances_gdp',

    # digital 
    'secure_internet_servers_per_million',

    # infraestructura 
    'atms_per_100k_adults',
    'commercial_bank_branches_per_100k_adults',

]


heavy_vars = [
    'secure_internet_servers_per_million',
    'stock_market_cap_gdp',
    'fdi_net_inflows_gdp',
]



def apply_log_transform(
    wide: pd.DataFrame,
    variables: List[str] = LOG_CANDIDATES,
    base: str = "natural",
) -> pd.DataFrame:
    """Aplica log(1+x) a variables con distribucion sesgada."""
    out = wide.copy()
    log_fn = np.log1p if base == "natural" else (lambda x: np.log10(1 + x))
    transformed: List[str] = []

    for var in variables:
        if var not in out.columns:
            continue
        values = out[var].clip(lower=0)
        out[var] = log_fn(values)
        transformed.append(var)

    logger.info("Log-transformacion ({b}) aplicada a: {lst}", b=base, lst=transformed)
    return out


def build_wgi_composite(wide: pd.DataFrame) -> pd.DataFrame:
    """Agrega un indice compuesto WGI auxiliar (no reemplaza variables individuales)."""
    wgi_vars = [
        "regulatory_quality",
        "government_effectiveness",
        "rule_of_law",
        "political_stability",
        "voice_accountability",
        "control_of_corruption",
    ]
    available = [v for v in wgi_vars if v in wide.columns]
    if len(available) < 3:
        logger.warning("Muy pocos WGI disponibles ({n}) para el compuesto", n=len(available))
        return wide

    out = wide.copy()
    out["wgi_composite"] = out[available].mean(axis=1)
    logger.info("WGI composite construido con {n} indicadores", n=len(available))
    return out

def compute_cultural_distance(df, origin_country="COL"):
    hofstede_cols = [
        'hofstede_pdi', 'hofstede_idv', 'hofstede_mas',
        'hofstede_uai', 'hofstede_lto', 'hofstede_ivr'
    ]

    origin_values = df.loc[origin_country, hofstede_cols]

    # Distancia euclidiana
    df["cultural_distance_hofstede"] = np.sqrt(
        ((df[hofstede_cols] - origin_values) ** 2).sum(axis=1)
    )

    return df


def winsorize(df, cols, lower=0.01, upper=0.99):
    for c in cols:
        if c in df.columns:
            q_low = df[c].quantile(lower)
            q_high = df[c].quantile(upper)
            df[c] = df[c].clip(q_low, q_high)
    return df


def run_feature_engineering(
    wide: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Ejecuta el pipeline completo de feature engineering."""
    out = apply_log_transform(wide)
    #out = build_wgi_composite(out) 
    # No se construye wgi_composite porque los indicadores WGI individuales ya
    # entran al TOPSIS global. Incluir ambos produciría doble conteo de la
    # dimensión institucional.
    out = winsorize(out, heavy_vars)  
    out = compute_cultural_distance(out)
     

    logger.info(
        "Feature engineering completado: {c} paises x {v} variables",
        c=len(out), v=out.shape[1],
    )
    return out
