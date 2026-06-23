"""
Perfiles de fortalezas y debilidades por pais para RADAR Cibest.

A partir de los scores parciales por dimension de TOPSIS, clasifica cada
dimension como fortaleza, debilidad o neutro usando la mediana como
benchmark.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger


def extract_dimension_scores(ranking_result: pd.DataFrame) -> pd.DataFrame:
    """Extrae los scores parciales por dimension del output de TOPSIS."""
    dim_cols = [c for c in ranking_result.columns if c.startswith("score_") and c != "score"]
    if not dim_cols:
        raise ValueError("El ranking no contiene scores por dimension")
    df = ranking_result[dim_cols].copy()
    df.columns = [c.replace("score_", "") for c in dim_cols]
    return df


def classify_dimensions(
    dimension_scores: pd.DataFrame,
    strength_threshold: float = 0.10,
    weakness_threshold: float = 0.10,
) -> pd.DataFrame:
    """Clasifica cada celda como strength/weakness/neutral."""
    medians = dimension_scores.median()
    classification = pd.DataFrame(
        index=dimension_scores.index,
        columns=dimension_scores.columns,
        dtype=object,
    )

    for col in dimension_scores.columns:
        med = medians[col]
        upper = med + strength_threshold
        lower = med - weakness_threshold
        values = dimension_scores[col]
        classification[col] = pd.cut(
            values,
            bins=[-float("inf"), lower, upper, float("inf")],
            labels=["weakness", "neutral", "strength"],
        ).astype(str)

    return classification


def rank_by_dimension(dimension_scores: pd.DataFrame) -> pd.DataFrame:
    """Calcula el rank de cada pais dentro de cada dimension."""
    return dimension_scores.rank(ascending=False, method="min").astype(int)


def generate_country_profile(
    country_iso3: str,
    dimension_scores: pd.DataFrame,
    classification: pd.DataFrame,
    dimension_ranks: pd.DataFrame,
    overall_rank: int,
    overall_score: float,
) -> Dict[str, Any]:
    """Construye un diccionario estructurado con el perfil de un pais."""
    profile: Dict[str, Any] = {
        "country_iso3": country_iso3,
        "rank_overall": int(overall_rank),
        "score_overall": round(float(overall_score), 4),
        "profile": {},
    }

    for dim in dimension_scores.columns:
        profile["profile"][dim] = {
            "score": round(float(dimension_scores.loc[country_iso3, dim]), 4),
            "classification": str(classification.loc[country_iso3, dim]),
            "rank_dimension": int(dimension_ranks.loc[country_iso3, dim]),
        }

    strengths = [d for d, info in profile["profile"].items() if info["classification"] == "strength"]
    weaknesses = [d for d, info in profile["profile"].items() if info["classification"] == "weakness"]
    profile["summary"] = {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "n_strengths": len(strengths),
        "n_weaknesses": len(weaknesses),
    }

    return profile


def generate_all_profiles(
    ranking_result: pd.DataFrame,
    radar_global: pd.Series,
    strength_threshold: float = 0.10,
    weakness_threshold: float = 0.10,
) -> Dict[str, Dict[str, Any]]:
    """Genera perfiles completos para todos los paises."""
    dim_scores = extract_dimension_scores(ranking_result)
    classification = classify_dimensions(dim_scores, strength_threshold, weakness_threshold)
    dim_ranks = rank_by_dimension(dim_scores)
    rank_radar = radar_global.rank(ascending=False, method="min").astype(int)

    profiles: Dict[str, Dict[str, Any]] = {}
    for country in dim_scores.index:
        if country not in radar_global.index:
            continue
        profiles[country] = generate_country_profile(
            country_iso3=country,
            dimension_scores=dim_scores,
            classification=classification,
            dimension_ranks=dim_ranks,
            overall_rank=int(rank_radar[country]),
            overall_score=float(radar_global[country]),
        )

    logger.info("Perfiles generados para {n} paises", n=len(profiles))
    return profiles


def profiles_to_dataframe(profiles: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Aplana perfiles a DataFrame tabular."""
    rows: List[Dict[str, Any]] = []
    for country, profile in profiles.items():
        row: Dict[str, Any] = {
            "country_iso3": country,
            "rank_overall": profile["rank_overall"],
            "score_overall": profile["score_overall"],
            "n_strengths": profile["summary"]["n_strengths"],
            "n_weaknesses": profile["summary"]["n_weaknesses"],
            "strengths": "; ".join(profile["summary"]["strengths"]),
            "weaknesses": "; ".join(profile["summary"]["weaknesses"]),
        }
        for dim, info in profile["profile"].items():
            row[f"{dim}_score"] = info["score"]
            row[f"{dim}_class"] = info["classification"]
            row[f"{dim}_rank"] = info["rank_dimension"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rank_overall").reset_index(drop=True)
