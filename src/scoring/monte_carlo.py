from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.scoring.ranking import TOPSISRanking
from src.scoring.weighting import compute_hierarchical_weights
from src.utils import ScoringError


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normaliza un diccionario de pesos para que sume 1."""

    total = sum(float(value) for value in weights.values())

    if total <= 0:
        raise ScoringError("La suma de pesos es cero o negativa.")

    return {
        key: float(value) / total
        for key, value in weights.items()
    }


def _perturb_weights_dirichlet(
    base_weights: Dict[str, float],
    concentration: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Perturba pesos usando una Dirichlet centrada en los pesos base.

    Args:
        base_weights: Pesos base.
        concentration: Concentración Dirichlet. Mayor valor implica menor ruido.
        rng: Generador aleatorio.

    Returns:
        Pesos perturbados y normalizados.
    """

    normalized = _normalize_weights(base_weights)

    positive_items = {
        key: value
        for key, value in normalized.items()
        if value > 0
    }

    zero_items = {
        key: 0.0
        for key, value in normalized.items()
        if value <= 0
    }

    if len(positive_items) == 1:
        only_key = next(iter(positive_items))
        return {
            **zero_items,
            only_key: 1.0,
        }

    keys = list(positive_items.keys())
    values = np.array(
        [positive_items[key] for key in keys],
        dtype=float,
    )

    alpha = values * float(concentration)
    alpha = np.clip(alpha, 1e-6, None)

    sampled = rng.dirichlet(alpha)

    perturbed = {
        key: float(value)
        for key, value in zip(keys, sampled)
    }

    perturbed.update(zero_items)

    return _normalize_weights(perturbed)


def _build_effective_variable_weights_by_dim(
    business_line_cfg: Dict[str, Any],
    variable_weights_by_dim: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Construye pesos efectivos por variable dentro de cada dimensión.

    Replica la lógica de overrides parciales:
    - parte de pesos globales de weights.yaml;
    - aplica variable_weight_overrides de business_lines.yaml;
    - renormaliza dentro de cada dimensión.
    """

    variable_overrides = (
        business_line_cfg.get("variable_weight_overrides", {}) or {}
    )

    effective_weights_by_dim: Dict[str, Dict[str, float]] = {}

    for dimension, global_weights in variable_weights_by_dim.items():
        effective_weights = dict(global_weights)

        if dimension in variable_overrides:
            for variable, weight in variable_overrides[dimension].items():
                effective_weights[variable] = float(weight)

        effective_weights_by_dim[dimension] = _normalize_weights(
            effective_weights
        )

    return effective_weights_by_dim


def _simulate_business_line_weights(
    business_line_cfg: Dict[str, Any],
    variable_weights_by_dim: Dict[str, Dict[str, float]],
    dimension_concentration: float,
    variable_concentration: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Simula pesos jerárquicos para una línea de negocio."""

    base_dimension_weights = _normalize_weights(
        dict(business_line_cfg["weight_profile"])
    )

    simulated_dimension_weights = _perturb_weights_dirichlet(
        base_weights=base_dimension_weights,
        concentration=dimension_concentration,
        rng=rng,
    )

    effective_variable_weights_by_dim = _build_effective_variable_weights_by_dim(
        business_line_cfg=business_line_cfg,
        variable_weights_by_dim=variable_weights_by_dim,
    )

    simulated_variable_weights_by_dim: Dict[str, Dict[str, float]] = {}

    for dimension, variable_weights in effective_variable_weights_by_dim.items():
        simulated_variable_weights_by_dim[dimension] = _perturb_weights_dirichlet(
            base_weights=variable_weights,
            concentration=variable_concentration,
            rng=rng,
        )

    final_variable_weights = compute_hierarchical_weights(
        simulated_dimension_weights,
        simulated_variable_weights_by_dim,
    )

    return simulated_dimension_weights, final_variable_weights


def _filter_and_normalize_weights_for_matrix(
    weights: Dict[str, float],
    decision_matrix: pd.DataFrame,
) -> Dict[str, float]:
    """Filtra pesos a variables presentes en decision_matrix y renormaliza."""

    filtered = {
        variable: weight
        for variable, weight in weights.items()
        if variable in decision_matrix.columns
    }

    missing = [
        variable
        for variable in decision_matrix.columns
        if variable not in filtered
    ]

    if missing:
        raise ScoringError(
            f"Faltan pesos simulados para variables en decision_matrix: {missing}"
        )

    return _normalize_weights(filtered)


def _ranking_to_score_series(ranking_df: pd.DataFrame) -> pd.Series:
    """Extrae scores TOPSIS indexados por país."""

    ranking = ranking_df.copy()

    if "country_iso3" in ranking.columns:
        ranking = ranking.set_index("country_iso3")

    return ranking["score"]


def _rank_scores_desc(score_series: pd.Series) -> pd.Series:
    """Rankea scores en orden descendente."""

    return score_series.rank(
        ascending=False,
        method="min",
    ).astype(int)


def _standardize_ranking_output(
    ranking_df: pd.DataFrame,
    business_line: str,
    simulation_id: int,
) -> pd.DataFrame:
    """Estandariza salida TOPSIS en formato largo."""

    ranking = ranking_df.copy()

    if "country_iso3" not in ranking.columns:
        ranking = ranking.reset_index()

        if "index" in ranking.columns:
            ranking = ranking.rename(columns={"index": "country_iso3"})

    return ranking[["country_iso3", "score", "rank"]].assign(
        business_line=business_line,
        simulation_id=simulation_id,
    )


def coerce_component_series(
    component: pd.Series | pd.DataFrame,
    value_col: Optional[str] = None,
    component_name: str = "component",
) -> pd.Series:
    """Convierte IPC o Trend a Serie indexada por country_iso3.

    Args:
        component: Serie o DataFrame con country_iso3 y una columna de valor.
        value_col: Columna de valor. Si None, se infiere.
        component_name: Nombre final de la serie.

    Returns:
        Serie indexada por country_iso3.
    """

    if isinstance(component, pd.Series):
        series = component.copy()
        series.name = component_name
        return pd.to_numeric(series, errors="coerce")

    df = component.copy()

    if "country_iso3" in df.columns:
        df = df.set_index("country_iso3")

    if value_col is None:
        candidate_cols = [
            component_name,
            "ipc",
            "ipc_score",
            "trend",
            "score",
            "value",
        ]

        value_col = next(
            (col for col in candidate_cols if col in df.columns),
            None,
        )

    if value_col is None or value_col not in df.columns:
        raise ValueError(
            f"No se pudo inferir columna de valor para {component_name}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    series = pd.to_numeric(df[value_col], errors="coerce")
    series.name = component_name

    return series


def _align_component_to_countries(
    series: pd.Series,
    countries: List[str],
    component_name: str,
) -> pd.Series:
    """Alinea una serie de componente a los países objetivo."""

    aligned = series.reindex(countries)

    missing = aligned[aligned.isna()].index.tolist()

    if missing:
        raise ScoringError(
            f"El componente {component_name} no tiene valores para países: {missing}"
        )

    return aligned.astype(float)


def _simulate_composite_weights(
    base_composite_weights: Dict[str, float],
    concentration: float,
    rng: np.random.Generator,
    perturb: bool = True,
) -> Dict[str, float]:
    """Simula o retorna pesos compuestos alpha/beta/gamma."""

    required = ["alpha", "beta", "gamma"]

    missing = [
        key for key in required
        if key not in base_composite_weights
    ]

    if missing:
        raise ScoringError(
            f"Faltan pesos compuestos requeridos: {missing}"
        )

    base = {
        key: float(base_composite_weights[key])
        for key in required
    }

    if not perturb:
        return _normalize_weights(base)

    return _perturb_weights_dirichlet(
        base_weights=base,
        concentration=concentration,
        rng=rng,
    )


def _compute_radar_score(
    topsis_scores: pd.Series,
    ipc_scores: pd.Series,
    trend_scores: pd.Series,
    composite_weights: Dict[str, float],
) -> pd.Series:
    """Calcula score RADAR compuesto."""

    weights = _normalize_weights(composite_weights)

    radar_score = (
        weights["alpha"] * topsis_scores
        + weights["beta"] * ipc_scores
        + weights["gamma"] * trend_scores
    )

    radar_score.name = "radar_score"

    return radar_score


def summarize_rank_robustness(
    simulation_long: pd.DataFrame,
    score_col: str,
    rank_col: str,
) -> pd.DataFrame:
    """Resume robustez de ranking por país y línea."""

    return (
        simulation_long
        .groupby(["business_line", "country_iso3"], as_index=False)
        .agg(
            mean_rank=(rank_col, "mean"),
            median_rank=(rank_col, "median"),
            std_rank=(rank_col, "std"),
            min_rank=(rank_col, "min"),
            max_rank=(rank_col, "max"),
            p10_rank=(rank_col, lambda x: x.quantile(0.10)),
            p90_rank=(rank_col, lambda x: x.quantile(0.90)),
            mean_score=(score_col, "mean"),
            std_score=(score_col, "std"),
            p10_score=(score_col, lambda x: x.quantile(0.10)),
            p90_score=(score_col, lambda x: x.quantile(0.90)),
        )
        .sort_values(["business_line", "mean_rank"])
        .reset_index(drop=True)
    )


def summarize_topn_probabilities(
    simulation_long: pd.DataFrame,
    rank_col: str,
    top_values: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Calcula probabilidades de pertenecer a Top-N."""

    if top_values is None:
        top_values = [3, 5, 10, 15]

    rows: List[Dict[str, Any]] = []

    grouped = simulation_long.groupby(["business_line", "country_iso3"])

    for (business_line, country), group in grouped:
        row: Dict[str, Any] = {
            "business_line": business_line,
            "country_iso3": country,
        }

        for top_n in top_values:
            row[f"prob_top_{top_n}"] = float((group[rank_col] <= top_n).mean())

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["business_line", "prob_top_5"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _rank_to_tier(rank: int, n_countries: int) -> str:
    """Convierte ranking en banda ordinal de atractividad."""

    percentile_position = rank / n_countries

    if percentile_position <= 0.20:
        return "Alta"
    if percentile_position <= 0.40:
        return "Media-alta"
    if percentile_position <= 0.60:
        return "Media"
    return "Baja"


def summarize_tier_probabilities(
    simulation_long: pd.DataFrame,
    rank_col: str,
) -> pd.DataFrame:
    """Calcula probabilidad de pertenecer a cada banda."""

    df = simulation_long.copy()

    n_countries_by_line_sim = (
        df
        .groupby(["business_line", "simulation_id"])["country_iso3"]
        .transform("nunique")
    )

    df["tier"] = [
        _rank_to_tier(
            rank=int(rank),
            n_countries=int(n_countries),
        )
        for rank, n_countries in zip(df[rank_col], n_countries_by_line_sim)
    ]

    tier_counts = (
        df
        .groupby(["business_line", "country_iso3", "tier"])
        .size()
        .rename("n")
        .reset_index()
    )

    total_counts = (
        df
        .groupby(["business_line", "country_iso3"])
        .size()
        .rename("total")
        .reset_index()
    )

    out = tier_counts.merge(
        total_counts,
        on=["business_line", "country_iso3"],
        how="left",
    )

    out["probability"] = out["n"] / out["total"]

    pivot = (
        out
        .pivot_table(
            index=["business_line", "country_iso3"],
            columns="tier",
            values="probability",
            fill_value=0.0,
        )
        .reset_index()
    )

    for tier in ["Alta", "Media-alta", "Media", "Baja"]:
        if tier not in pivot.columns:
            pivot[tier] = 0.0

    return pivot[
        [
            "business_line",
            "country_iso3",
            "Alta",
            "Media-alta",
            "Media",
            "Baja",
        ]
    ].sort_values(
        ["business_line", "Alta", "Media-alta"],
        ascending=[True, False, False],
    )


def summarize_line_correlation_robustness(
    line_correlation_long: pd.DataFrame,
) -> pd.DataFrame:
    """Resume robustez de correlaciones entre líneas."""

    return (
        line_correlation_long
        .groupby(["line_a", "line_b"], as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            std_spearman=("spearman", "std"),
            p10_spearman=("spearman", lambda x: x.quantile(0.10)),
            p90_spearman=("spearman", lambda x: x.quantile(0.90)),
            min_spearman=("spearman", "min"),
            max_spearman=("spearman", "max"),
        )
        .sort_values("mean_spearman", ascending=False)
        .reset_index(drop=True)
    )


def summarize_composite_weight_distribution(
    simulation_long: pd.DataFrame,
) -> pd.DataFrame:
    """Resume distribución simulada de alpha, beta y gamma."""

    return (
        simulation_long[
            ["simulation_id", "alpha", "beta", "gamma"]
        ]
        .drop_duplicates()
        .agg(
            {
                "alpha": ["mean", "std", "min", "max"],
                "beta": ["mean", "std", "min", "max"],
                "gamma": ["mean", "std", "min", "max"],
            }
        )
    )


def run_monte_carlo_topsis_robustness(
    decision_matrix: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
    variable_catalog: Dict[str, Dict[str, Any]],
    n_simulations: int = 1000,
    dimension_concentration: float = 150.0,
    variable_concentration: float = 100.0,
    random_seed: int = 42,
    business_lines: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Ejecuta Monte Carlo solo sobre ranking TOPSIS estructural."""

    rng = np.random.default_rng(random_seed)

    business_line_cfgs = configs["business_lines"]["business_lines"]
    variable_weights_by_dim = configs["weights"]["variable_weights"]

    if business_lines is None:
        business_lines = list(business_line_cfgs.keys())


    origin_country = configs["settings"].get("origin_country", "COL")
    exclude_origin_country = bool(
        configs["settings"]
        .get("scoring", {})
        .get("exclude_origin_country", True)
    )

    if exclude_origin_country and origin_country in decision_matrix.index:
        logger.info(
            "País origen excluido de Monte Carlo TOPSIS: {origin}",
            origin=origin_country,
        )

        decision_matrix = decision_matrix.drop(index=origin_country)


    ranker = TOPSISRanking(
        distance_metric=configs["settings"]["scoring"]["topsis"].get(
            "distance_metric",
            "euclidean",
        )
    )

    simulation_frames: List[pd.DataFrame] = []
    correlation_frames: List[pd.DataFrame] = []

    for simulation_id in range(n_simulations):
        ranks_by_line: Dict[str, pd.Series] = {}

        for business_line in business_lines:
            business_line_cfg = business_line_cfgs[business_line]

            _, simulated_weights = _simulate_business_line_weights(
                business_line_cfg=business_line_cfg,
                variable_weights_by_dim=variable_weights_by_dim,
                dimension_concentration=dimension_concentration,
                variable_concentration=variable_concentration,
                rng=rng,
            )

            final_weights = _filter_and_normalize_weights_for_matrix(
                weights=simulated_weights,
                decision_matrix=decision_matrix,
            )

            ranking = ranker.rank(
                decision_matrix,
                final_weights,
                variable_catalog,
            )

            standardized = _standardize_ranking_output(
                ranking_df=ranking,
                business_line=business_line,
                simulation_id=simulation_id,
            )

            simulation_frames.append(standardized)

            ranks_by_line[business_line] = standardized.set_index(
                "country_iso3"
            )["rank"]

        rank_matrix = pd.DataFrame(ranks_by_line)
        corr_matrix = rank_matrix.corr(method="spearman")

        corr_rows = []

        for line_a, line_b in combinations(business_lines, 2):
            corr_rows.append(
                {
                    "simulation_id": simulation_id,
                    "line_a": line_a,
                    "line_b": line_b,
                    "spearman": corr_matrix.loc[line_a, line_b],
                }
            )

        correlation_frames.append(pd.DataFrame(corr_rows))

        if (simulation_id + 1) % 100 == 0:
            logger.info(
                "Monte Carlo TOPSIS progreso: {done}/{total}",
                done=simulation_id + 1,
                total=n_simulations,
            )

    simulation_long = pd.concat(simulation_frames, ignore_index=True)
    line_correlation_long = pd.concat(correlation_frames, ignore_index=True)

    return {
        "simulation_long": simulation_long,
        "rank_robustness": summarize_rank_robustness(
            simulation_long,
            score_col="score",
            rank_col="rank",
        ),
        "topn_probabilities": summarize_topn_probabilities(
            simulation_long,
            rank_col="rank",
        ),
        "tier_probabilities": summarize_tier_probabilities(
            simulation_long,
            rank_col="rank",
        ),
        "line_correlation_robustness": summarize_line_correlation_robustness(
            line_correlation_long
        ),
    }


def run_monte_carlo_radar_robustness(
    decision_matrix: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
    variable_catalog: Dict[str, Dict[str, Any]],
    ipc_scores: pd.Series,
    trend_scores: pd.Series,
    n_simulations: int = 1000,
    dimension_concentration: float = 150.0,
    variable_concentration: float = 100.0,
    composite_concentration: float = 150.0,
    perturb_composite_weights: bool = True,
    random_seed: int = 42,
    business_lines: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Ejecuta Monte Carlo sobre el score RADAR completo.

    Simula:
    - pesos TOPSIS por dimensión y variable;
    - pesos compuestos alpha/beta/gamma.

    Mantiene fijos:
    - IPC;
    - Trend.

    Args:
        decision_matrix: Matriz normalizada país x variable.
        configs: Configuraciones del proyecto.
        variable_catalog: Catálogo de variables.
        ipc_scores: Serie IPC indexada por país.
        trend_scores: Serie Trend indexada por país.
        n_simulations: Número de simulaciones.
        dimension_concentration: Concentración Dirichlet para dimensiones.
        variable_concentration: Concentración Dirichlet para variables.
        composite_concentration: Concentración Dirichlet para alpha/beta/gamma.
        perturb_composite_weights: Si False, usa alpha/beta/gamma fijos.
        random_seed: Semilla.
        business_lines: Líneas a simular.

    Returns:
        Diccionario con simulaciones y resúmenes de robustez RADAR.
    """

    rng = np.random.default_rng(random_seed)

    business_line_cfgs = configs["business_lines"]["business_lines"]
    variable_weights_by_dim = configs["weights"]["variable_weights"]
    base_composite_weights = configs["settings"]["scoring"]["composite_weights"]

    if business_lines is None:
        business_lines = list(business_line_cfgs.keys())


    origin_country = configs["settings"].get("origin_country", "COL")
    exclude_origin_country = bool(
        configs["settings"]
        .get("scoring", {})
        .get("exclude_origin_country", True)
    )

    if exclude_origin_country and origin_country in decision_matrix.index:
        logger.info(
            "País origen excluido de Monte Carlo RADAR: {origin}",
            origin=origin_country,
        )

        decision_matrix = decision_matrix.drop(index=origin_country)


    countries = list(decision_matrix.index)

    ipc_aligned = _align_component_to_countries(
        ipc_scores,
        countries,
        component_name="ipc",
    )

    trend_aligned = _align_component_to_countries(
        trend_scores,
        countries,
        component_name="trend",
    )

    ranker = TOPSISRanking(
        distance_metric=configs["settings"]["scoring"]["topsis"].get(
            "distance_metric",
            "euclidean",
        )
    )

    simulation_frames: List[pd.DataFrame] = []
    correlation_frames: List[pd.DataFrame] = []

    for simulation_id in range(n_simulations):
        composite_weights = _simulate_composite_weights(
            base_composite_weights=base_composite_weights,
            concentration=composite_concentration,
            rng=rng,
            perturb=perturb_composite_weights,
        )

        radar_ranks_by_line: Dict[str, pd.Series] = {}

        for business_line in business_lines:
            business_line_cfg = business_line_cfgs[business_line]

            _, simulated_weights = _simulate_business_line_weights(
                business_line_cfg=business_line_cfg,
                variable_weights_by_dim=variable_weights_by_dim,
                dimension_concentration=dimension_concentration,
                variable_concentration=variable_concentration,
                rng=rng,
            )

            final_weights = _filter_and_normalize_weights_for_matrix(
                weights=simulated_weights,
                decision_matrix=decision_matrix,
            )

            topsis_ranking = ranker.rank(
                decision_matrix,
                final_weights,
                variable_catalog,
            )

            topsis_scores = _ranking_to_score_series(topsis_ranking)
            topsis_scores = topsis_scores.reindex(countries)

            topsis_ranks = _rank_scores_desc(topsis_scores)

            radar_scores = _compute_radar_score(
                topsis_scores=topsis_scores,
                ipc_scores=ipc_aligned,
                trend_scores=trend_aligned,
                composite_weights=composite_weights,
            )

            radar_ranks = _rank_scores_desc(radar_scores)

            frame = pd.DataFrame(
                {
                    "country_iso3": countries,
                    "business_line": business_line,
                    "simulation_id": simulation_id,
                    "topsis_score": topsis_scores.values,
                    "topsis_rank": topsis_ranks.values,
                    "ipc_score": ipc_aligned.values,
                    "trend_score": trend_aligned.values,
                    "alpha": composite_weights["alpha"],
                    "beta": composite_weights["beta"],
                    "gamma": composite_weights["gamma"],
                    "radar_score": radar_scores.values,
                    "radar_rank": radar_ranks.values,
                }
            )

            simulation_frames.append(frame)

            radar_ranks_by_line[business_line] = radar_ranks

        radar_rank_matrix = pd.DataFrame(radar_ranks_by_line)
        corr_matrix = radar_rank_matrix.corr(method="spearman")

        corr_rows = []

        for line_a, line_b in combinations(business_lines, 2):
            corr_rows.append(
                {
                    "simulation_id": simulation_id,
                    "line_a": line_a,
                    "line_b": line_b,
                    "spearman": corr_matrix.loc[line_a, line_b],
                }
            )

        correlation_frames.append(pd.DataFrame(corr_rows))

        if (simulation_id + 1) % 100 == 0:
            logger.info(
                "Monte Carlo RADAR progreso: {done}/{total}",
                done=simulation_id + 1,
                total=n_simulations,
            )

    simulation_long = pd.concat(simulation_frames, ignore_index=True)
    line_correlation_long = pd.concat(correlation_frames, ignore_index=True)

    return {
        "simulation_long": simulation_long,
        "rank_robustness": summarize_rank_robustness(
            simulation_long,
            score_col="radar_score",
            rank_col="radar_rank",
        ),
        "topn_probabilities": summarize_topn_probabilities(
            simulation_long,
            rank_col="radar_rank",
        ),
        "tier_probabilities": summarize_tier_probabilities(
            simulation_long,
            rank_col="radar_rank",
        ),
        "line_correlation_robustness": summarize_line_correlation_robustness(
            line_correlation_long
        ),
        "composite_weight_distribution": summarize_composite_weight_distribution(
            simulation_long
        ),
    }