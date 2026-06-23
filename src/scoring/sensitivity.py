"""
Analisis de sensibilidad para RADAR Cibest.

Evalua la robustez del ranking ante variaciones en pesos +/-20% y compara
TOPSIS vs VIKOR como validacion cruzada.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from src.scoring.ranking import RankingMethod, TOPSISRanking, VIKORRanking
from src.scoring.weighting import compute_hierarchical_weights


def perturb_dimension_weights(
    dimension_weights: Dict[str, float],
    perturbation: float,
    dimension: str,
) -> Dict[str, float]:
    """Perturba el peso de una dimension y reescala las demas (suma 1)."""
    if dimension not in dimension_weights:
        raise ValueError(f"Dimension desconocida: {dimension}")

    perturbed = dict(dimension_weights)
    perturbed[dimension] *= perturbation
    total = sum(perturbed.values())
    return {k: v / total for k, v in perturbed.items()}


def run_sensitivity_analysis(
    decision_matrix: pd.DataFrame,
    dimension_weights: Dict[str, float],
    variable_weights_by_dim: Dict[str, Dict[str, float]],
    variable_catalog: Dict[str, Dict[str, Any]],
    perturbations: List[float] = (0.80, 0.90, 1.10, 1.20),
    top_n: int = 10,
    ranker: RankingMethod | None = None,
) -> pd.DataFrame:
    """Ejecuta analisis de sensibilidad perturbando cada dimension."""
    if ranker is None:
        ranker = TOPSISRanking()

    base_weights = compute_hierarchical_weights(dimension_weights, variable_weights_by_dim)
    base_weights = {k: v for k, v in base_weights.items() if k in decision_matrix.columns}
    s = sum(base_weights.values())
    if s > 0:
        base_weights = {k: v / s for k, v in base_weights.items()}
    base_ranking = ranker.rank(decision_matrix, base_weights, variable_catalog)
    base_scores = base_ranking["score"]
    base_top = set(base_ranking.head(top_n).index)

    results: List[Dict[str, Any]] = []
    for dim in dimension_weights:
        for p in perturbations:
            perturbed_dw = perturb_dimension_weights(dimension_weights, p, dim)
            perturbed_weights = compute_hierarchical_weights(perturbed_dw, variable_weights_by_dim)
            perturbed_weights = {k: v for k, v in perturbed_weights.items() if k in decision_matrix.columns}
            s2 = sum(perturbed_weights.values())
            if s2 > 0:
                perturbed_weights = {k: v / s2 for k, v in perturbed_weights.items()}
            perturbed_ranking = ranker.rank(decision_matrix, perturbed_weights, variable_catalog)
            perturbed_scores = perturbed_ranking["score"]
            perturbed_top = set(perturbed_ranking.head(top_n).index)

            corr = base_scores.corr(perturbed_scores, method="spearman")
            overlap = len(base_top & perturbed_top)

            results.append({
                "dimension": dim,
                "perturbation": p,
                "score_corr": round(corr, 4),
                "topN_overlap": overlap,
                "topN_size": top_n,
                "countries_in": "; ".join(sorted(perturbed_top - base_top)),
                "countries_out": "; ".join(sorted(base_top - perturbed_top)),
            })

    df_results = pd.DataFrame(results)
    logger.info(
        "Sensibilidad: corr Spearman media = {c:.3f} | overlap top-{n} medio = {o:.1f}",
        c=df_results["score_corr"].mean(), n=top_n, o=df_results["topN_overlap"].mean(),
    )
    return df_results


def compare_topsis_vikor(
    decision_matrix: pd.DataFrame,
    weights: Dict[str, float],
    variable_catalog: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Compara rankings TOPSIS vs VIKOR con los mismos pesos."""
    topsis = TOPSISRanking()
    vikor = VIKORRanking()

    r_t = topsis.rank(decision_matrix, weights, variable_catalog)
    r_v = vikor.rank(decision_matrix, weights, variable_catalog)

    merged = pd.DataFrame({
        "score_topsis": r_t["score"],
        "rank_topsis": r_t["rank"],
        "score_vikor": r_v["score"],
        "rank_vikor": r_v["rank"],
    })
    merged["rank_diff"] = (merged["rank_topsis"] - merged["rank_vikor"]).abs()
    merged = merged.sort_values("rank_topsis")

    corr = merged["score_topsis"].corr(merged["score_vikor"], method="spearman")
    logger.info("TOPSIS vs VIKOR: correlacion Spearman = {c:.3f}", c=corr)
    return merged
