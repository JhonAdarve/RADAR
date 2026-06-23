"""
Fixtures compartidas para los tests unitarios de RADAR Cibest.

Provee datos sinteticos reproducibles que permiten validar los modulos
sin depender de APIs externas. Las fixtures cubren:
    - Configuraciones reducidas (5 paises, 8 variables)
    - DataFrame largo de extraccion sintetico
    - Matriz ancha imputada
    - Matriz de decision normalizada

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_settings() -> Dict[str, Any]:
    """Configuracion settings.yaml reducida para tests."""
    return {
        "project": {"name": "RADAR Cibest TEST"},
        "countries": [
            {"iso3": "COL", "name": "Colombia", "region": "south_america"},
            {"iso3": "MEX", "name": "Mexico", "region": "north_america"},
            {"iso3": "PAN", "name": "Panama", "region": "central_america"},
            {"iso3": "CHL", "name": "Chile", "region": "south_america"},
            {"iso3": "ESP", "name": "Espana", "region": "europe_strategic"},
        ],
        "origin_country": "COL",
        "data": {
            "time_range": {"start_year": 2020, "end_year": 2024},
            "raw_path": "data/raw/",
            "processed_path": "data/processed/",
            "scores_path": "data/scores/",
            "cache_path": "data/cache/",
            "static_path": "data/static/",
            "cache_expiry_days": 30,
        },
        "scoring": {
            "missing_data_threshold": 0.30,
            "imputation_method": "global_median",
            "normalization_method": "min_max",
            "composite_weights": {"alpha": 0.55, "beta": 0.30, "gamma": 0.15},
            "profile_thresholds": {"strength_threshold": 0.10, "weakness_threshold": 0.10},
            "bwm": {"max_consistency_ratio": 0.10},
            "topsis": {"distance_metric": "euclidean"},
        },
    }


@pytest.fixture
def synthetic_variables_cfg() -> Dict[str, Any]:
    """Configuracion variables.yaml reducida con 5 dimensiones x 1-2 variables."""
    return {
        "dimensions": {
            "macro": {
                "label": "Macroeconomica",
                "variables": {
                    "gdp_per_capita_ppp": {
                        "source": "imf",
                        "indicator_code": "PPPPC",
                        "direction": "positive",
                        "frequency": "annual",
                        "description": "PIB per capita PPP",
                        "business_lines": ["IB", "CIB"],
                        "literature": ["Frost_2020"],
                    },
                    "inflation_rate": {
                        "source": "imf",
                        "indicator_code": "PCPIPCH",
                        "direction": "negative",
                        "frequency": "annual",
                        "description": "Inflacion",
                        "business_lines": ["IB"],
                        "literature": ["Frost_2020"],
                    },
                },
            },
            "financial": {
                "label": "Financiera",
                "variables": {
                    "domestic_credit_private_gdp": {
                        "source": "world_bank",
                        "indicator_code": "FS.AST.PRVT.GD.ZS",
                        "direction": "positive",
                        "frequency": "annual",
                        "description": "Credito privado",
                        "business_lines": ["IB"],
                        "literature": ["Cihak_et_al_2012"],
                    },
                },
            },
            "institutional": {
                "label": "Institucional",
                "variables": {
                    "regulatory_quality": {
                        "source": "wgi",
                        "indicator_code": "RQ.EST",
                        "direction": "positive",
                        "frequency": "annual",
                        "description": "Calidad regulatoria",
                        "business_lines": ["IB", "PF", "AD", "BD", "CIB"],
                        "literature": ["Kaufmann_Kraay_2024"],
                    },
                    "political_stability": {
                        "source": "wgi",
                        "indicator_code": "PV.EST",
                        "direction": "positive",
                        "frequency": "annual",
                        "description": "Estabilidad politica",
                        "business_lines": ["IB", "AD"],
                        "literature": ["Papaioannou_2009"],
                    },
                },
            },
            "digital_tech": {
                "label": "Digital",
                "variables": {
                    "internet_users_pct": {
                        "source": "world_bank",
                        "indicator_code": "IT.NET.USER.ZS",
                        "direction": "positive",
                        "frequency": "annual",
                        "description": "Usuarios internet",
                        "business_lines": ["BD", "AD"],
                        "literature": ["Frost_2020"],
                    },
                },
            },
            "proximity": {
                "label": "Proximidad",
                "variables": {
                    "geographic_distance_km": {
                        "source": "complementary",
                        "indicator_code": "CEPII_DIST",
                        "direction": "negative",
                        "frequency": "static",
                        "description": "Distancia a Bogota",
                        "business_lines": ["IB", "PF", "AD", "BD", "CIB"],
                        "literature": ["Brei_vonPeter_2018"],
                    },
                    "common_language_spanish": {
                        "source": "complementary",
                        "indicator_code": "CEPII_LANG_ES",
                        "direction": "positive",
                        "frequency": "static",
                        "description": "Idioma compartido",
                        "business_lines": ["IB", "PF"],
                        "literature": ["Ghemawat_2001"],
                    },
                },
            },
        }
    }


@pytest.fixture
def synthetic_long_df() -> pd.DataFrame:
    """DataFrame largo sintetico para 5 paises y 7 variables."""
    rng = np.random.default_rng(42)
    countries = ["COL", "MEX", "PAN", "CHL", "ESP"]
    variables = [
        "gdp_per_capita_ppp", "inflation_rate", "domestic_credit_private_gdp",
        "regulatory_quality", "political_stability", "internet_users_pct",
        "geographic_distance_km", "common_language_spanish",
    ]
    base_values = {
        "gdp_per_capita_ppp": [16000, 21000, 35000, 28000, 42000],
        "inflation_rate": [9.5, 4.2, 2.8, 7.6, 3.5],
        "domestic_credit_private_gdp": [54, 39, 78, 84, 95],
        "regulatory_quality": [0.3, 0.05, 0.4, 1.2, 1.0],
        "political_stability": [-0.5, -0.7, 0.4, 0.1, 0.5],
        "internet_users_pct": [73, 76, 82, 90, 94],
        "geographic_distance_km": [0, 3500, 600, 5800, 8400],
        "common_language_spanish": [1, 1, 1, 1, 1],
    }
    rows = []
    for var in variables:
        for i, country in enumerate(countries):
            base = base_values[var][i]
            if var in ("geographic_distance_km", "common_language_spanish"):
                rows.append({"country_iso3": country, "year": 0, "variable": var, "value": float(base)})
            else:
                for year in range(2020, 2025):
                    noise = rng.normal(0, abs(base) * 0.05)
                    rows.append({
                        "country_iso3": country, "year": year, "variable": var,
                        "value": float(base + noise),
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_wide_df(synthetic_long_df: pd.DataFrame) -> pd.DataFrame:
    """Matriz ancha country x variable derivada del long df."""
    static = synthetic_long_df[synthetic_long_df["year"] == 0]
    temporal = synthetic_long_df[synthetic_long_df["year"] > 0]
    latest = temporal.loc[temporal.groupby(["country_iso3", "variable"])["year"].idxmax()]
    combined = pd.concat([latest, static], ignore_index=True)
    wide = combined.pivot(index="country_iso3", columns="variable", values="value")
    return wide


@pytest.fixture
def bwm_judgments() -> Dict[str, Any]:
    """Juicios BWM simulados para 5 dimensiones."""
    return {
        "criteria": ["macro", "financial", "institutional", "digital_tech", "proximity"],
        "best": "institutional",
        "worst": "digital_tech",
        "best_to_others": {
            "macro": 3, "financial": 2, "institutional": 1,
            "digital_tech": 7, "proximity": 4,
        },
        "others_to_worst": {
            "macro": 5, "financial": 6, "institutional": 7,
            "digital_tech": 1, "proximity": 4,
        },
    }
