"""
Orquestador del scoring hibrido RADAR Cibest.

Integra BWM (pesos), TOPSIS (ranking) y modelo gravitacional (proximidad)
en un score RADAR compuesto:
    RADAR(j, l) = alpha * CC_TOPSIS(j, l) + beta * IPC(j) + gamma * Tendencia(j)

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data_preparation.cleaning import run_cleaning
from src.data_preparation.feature_engineering import run_feature_engineering
from src.data_preparation.normalization import normalize
from src.scoring.gravity import compute_ipc
from src.scoring.ranking import TOPSISRanking
from src.scoring.weighting import compute_hierarchical_weights
from src.utils import (
    ScoringError,
    get_country_list,
    get_variable_catalog,
    load_all_configs,
    resolve_data_path,
    setup_logger,
)


def prepare_decision_matrix(
    df_long: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Ejecuta cleaning + feature engineering + normalizacion.

    Nota metodológica:
    - wide_enriched conserva todas las variables necesarias para IPC, Trend,
      auditoría y dashboard.
    - decision_matrix excluye variables de proximidad, auxiliares de feature
      engineering y variables marcadas con include_in_topsis: false.
    - TOPSIS debe usar únicamente variables estructurales elegibles.
    """

    settings = configs["settings"]
    variables_cfg = configs["variables"]
    catalog = get_variable_catalog(variables_cfg)
    norm_method = settings["scoring"]["normalization_method"]

    # 1. Cleaning
    wide_clean, excluded = run_cleaning(df_long, configs)

    # 2. Feature engineering
    # Aquí se construye, por ejemplo, cultural_distance_hofstede.
    wide_enriched = run_feature_engineering(wide_clean, configs)

    # 3. Variables que NO deben entrar a TOPSIS por configuración del catálogo.
    # Ejemplo: gdp_growth corresponde al componente Trend del score RADAR,
    # por tanto no debe entrar a TOPSIS para evitar doble conteo.
    exclude_by_catalog = [
        variable
        for variable, meta in catalog.items()
        if meta.get("include_in_topsis") is False
    ]

    # 4. Variables de proximidad / IPC.
    proximity_vars = [
        "geographic_distance_km",
        "common_language_spanish",
        "cultural_distance_hofstede",
        "bilateral_trade_colombia",
        "colombian_diaspora_stock",
    ]

    # 5. Variables auxiliares usadas para construir features,
    # pero que no deben entrar directamente al TOPSIS.
    feature_engineering_aux_vars = [
        "hofstede_pdi",
        "hofstede_idv",
        "hofstede_mas",
        "hofstede_uai",
        "hofstede_lto",
        "hofstede_ivr",
        "salidas_colombianos",
    ]

    # 6. Lista final de exclusión para TOPSIS.
    exclude_from_topsis = sorted(
        set(
            proximity_vars
            + feature_engineering_aux_vars
            + exclude_by_catalog
        )
    )

    exclude_from_topsis_present = [
        variable
        for variable in exclude_from_topsis
        if variable in wide_enriched.columns
    ]

    if exclude_from_topsis_present:
        logger.info(
            "Variables excluidas de TOPSIS: {vars}",
            vars=exclude_from_topsis_present,
        )

    # 7. Matriz estructural para TOPSIS.
    wide_structural = wide_enriched.drop(
        columns=exclude_from_topsis_present,
        errors="ignore",
    )

    # 8. Normalización solo de variables estructurales.
    decision_matrix = normalize(
        wide_structural,
        method=norm_method,
        variable_catalog=catalog,
        apply_direction_flag=True,
    )

    # 9. Persistir matrices intermedias.
    processed_dir = resolve_data_path(settings["data"]["processed_path"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    decision_matrix.to_parquet(
        processed_dir / "decision_matrix_latest.parquet"
    )

    wide_enriched.to_parquet(
        processed_dir / "wide_raw_latest.parquet"
    )

    return wide_enriched, decision_matrix, excluded



def _build_business_line_weights(
    business_line_cfg: Dict[str, Any],
    variable_weights_by_dim: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Construye pesos finales por variable para una línea de negocio.

    Lógica:
    1. Usa los pesos por dimensión de la línea.
    2. Parte de los pesos globales de variables definidos en weights.yaml.
    3. Aplica variable_weight_overrides como ajustes parciales.
    4. Renormaliza los pesos de variables dentro de cada dimensión.
    5. Calcula pesos jerárquicos finales.

    Nota metodológica:
    Los overrides no reemplazan toda la dimensión. Solo modifican las variables
    explícitamente declaradas para la línea de negocio. Las demás variables
    conservan su peso global.
    """

    dim_weights = dict(business_line_cfg["weight_profile"])

    total_dim = sum(dim_weights.values())
    if total_dim <= 0:
        raise ScoringError("Los pesos de dimensiones de la línea suman 0.")

    if abs(total_dim - 1.0) > 1e-6:
        dim_weights = {k: v / total_dim for k, v in dim_weights.items()}

    variable_overrides = business_line_cfg.get("variable_weight_overrides", {})

    effective_variable_weights_by_dim: Dict[str, Dict[str, float]] = {}

    for dim, global_var_weights in variable_weights_by_dim.items():
        # Partir siempre de los pesos globales
        effective_weights = dict(global_var_weights)

        # Aplicar overrides parciales si existen
        if dim in variable_overrides:
            for var, weight in variable_overrides[dim].items():
                effective_weights[var] = weight

        total_var = sum(effective_weights.values())
        if total_var <= 0:
            raise ScoringError(
                f"Los pesos de variables para dimensión '{dim}' suman 0."
            )

        effective_variable_weights_by_dim[dim] = {
            k: v / total_var
            for k, v in effective_weights.items()
        }

    var_weights = compute_hierarchical_weights(
        dim_weights,
        effective_variable_weights_by_dim,
    )

    return dim_weights, var_weights


def score_business_line(
    decision_matrix: pd.DataFrame,
    business_line_key: str,
    configs: Dict[str, Dict[str, Any]],
    variable_catalog: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Calcula el ranking TOPSIS para una linea de negocio especifica."""
    bl_cfg_all = configs["business_lines"]["business_lines"]
    if business_line_key not in bl_cfg_all:
        raise ScoringError(f"Linea de negocio desconocida: {business_line_key}")
    bl_cfg = bl_cfg_all[business_line_key]

    variable_weights_by_dim = configs["weights"]["variable_weights"]
    _, var_weights = _build_business_line_weights(bl_cfg, variable_weights_by_dim)
    # Filtrar a columnas disponibles y renormalizar
    var_weights = {k: v for k, v in var_weights.items() if k in decision_matrix.columns}
    s = sum(var_weights.values())
    if s > 0:
        var_weights = {k: v / s for k, v in var_weights.items()}

    ranker = TOPSISRanking(
        distance_metric=configs["settings"]["scoring"]["topsis"].get("distance_metric", "euclidean")
    )
    ranking = ranker.rank(decision_matrix, var_weights, variable_catalog)
    return ranking.assign(business_line=business_line_key)


def score_all_business_lines(
    decision_matrix: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
    variable_catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """Ejecuta TOPSIS por cada linea de negocio."""
    bl_cfg_all = configs["business_lines"]["business_lines"]
    results: Dict[str, pd.DataFrame] = {}
    for bl_key in bl_cfg_all:
        logger.info("--- TOPSIS linea {bl} ---", bl=bl_key)
        results[bl_key] = score_business_line(decision_matrix, bl_key, configs, variable_catalog)
    return results


def compute_global_topsis(
    decision_matrix: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
    variable_catalog: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Ranking TOPSIS global con pesos agregados del BWM."""
    dim_weights = configs["weights"]["dimension_weights"]
    var_weights_by_dim = configs["weights"]["variable_weights"]
    var_weights = compute_hierarchical_weights(dim_weights, var_weights_by_dim)
    var_weights = {k: v for k, v in var_weights.items() if k in decision_matrix.columns}
    s = sum(var_weights.values())
    if s > 0:
        var_weights = {k: v / s for k, v in var_weights.items()}

    ranker = TOPSISRanking(
        distance_metric=configs["settings"]["scoring"]["topsis"].get("distance_metric", "euclidean")
    )
    return ranker.rank(decision_matrix, var_weights, variable_catalog)


def build_asof_snapshot(
    df_long: pd.DataFrame,
    snapshot_year: int,
) -> pd.DataFrame:
    """Construye un corte histórico usando el último dato disponible hasta snapshot_year.

    Para cada país-variable toma el último valor disponible con year <= snapshot_year.
    Esto permite reconstruir matrices históricas comparables sin exigir que todas
    las variables tengan dato exactamente en el mismo año.
    """

    df = df_long.copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country_iso3", "variable", "year", "value"])
    df["year"] = df["year"].astype(int)

    snapshot = (
        df[df["year"] <= snapshot_year]
        .sort_values(["country_iso3", "variable", "year"])
        .groupby(["country_iso3", "variable"], as_index=False)
        .tail(1)
    )

    return snapshot

def append_origin_reference_rows(
    df_snapshot: pd.DataFrame,
    df_long: pd.DataFrame,
    origin_country: str = "COL",
) -> pd.DataFrame:
    """Asegura que el país origen exista en snapshots históricos.

    En el cálculo de tendencia histórica se construyen snapshots tipo as-of
    para años anteriores. En esos cortes, el país origen puede ser excluido
    posteriormente por cobertura insuficiente dentro de run_cleaning().

    Sin embargo, Colombia no debe competir en el ranking, pero sí debe estar
    disponible durante feature_engineering porque se usa como referencia para
    construir variables relativas, por ejemplo cultural_distance_hofstede.

    Por eso se anexan al snapshot las últimas observaciones disponibles del
    país origen desde el df_long completo. Estas filas permiten calcular las
    features relativas al origen y luego COL se elimina del decision_matrix
    antes del TOPSIS histórico.
    """

    if df_long.empty:
        return df_snapshot

    origin_rows = df_long[df_long["country_iso3"] == origin_country].copy()

    if origin_rows.empty:
        logger.warning(
            "No se encontraron filas para país origen {origin} en df_long. "
            "El feature engineering histórico podría fallar.",
            origin=origin_country,
        )
        return df_snapshot

    origin_rows["year"] = pd.to_numeric(origin_rows["year"], errors="coerce")
    origin_rows = origin_rows.dropna(
        subset=["country_iso3", "variable", "year", "value"]
    )

    if origin_rows.empty:
        logger.warning(
            "Filas del país origen {origin} sin datos válidos. "
            "El feature engineering histórico podría fallar.",
            origin=origin_country,
        )
        return df_snapshot

    origin_rows["year"] = origin_rows["year"].astype(int)

    origin_latest = (
        origin_rows
        .sort_values(["country_iso3", "variable", "year"])
        .groupby(["country_iso3", "variable"], as_index=False)
        .tail(1)
    )

    out = pd.concat(
        [df_snapshot, origin_latest],
        ignore_index=True,
    )

    return out

def compute_trend_from_global_score_yoy(
    df_long: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
    variable_catalog: Dict[str, Dict[str, Any]],
    origin_country: str = "COL",
    target_countries: list | None = None,
    n_years: int = 3,
    end_year: int | None = None,
    neutral_value: float = 0.5,
) -> pd.Series:
    """Calcula Tendencia(j) como variación interanual promedio del score TOPSIS global.

    Metodología:
        1. Construye snapshots históricos tipo as-of para los últimos n_years.
        2. Calcula TOPSIS global para cada snapshot.
        3. Construye un panel país x año con scores TOPSIS.
        4. Calcula variaciones interanuales del score.
        5. Promedia las variaciones interanuales.
        6. Normaliza el resultado a [0, 1].

    La tendencia captura dinamismo competitivo:
        - valores altos: países cuyo score TOPSIS viene mejorando
        - valores bajos: países cuyo score TOPSIS viene deteriorándose
    """

    df = df_long.copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    if end_year is None:
        end_year = int(df["year"].max())

    years = list(range(end_year - n_years + 1, end_year + 1))

    if len(years) < 2:
        logger.warning(
            "Ventana insuficiente para calcular tendencia TOPSIS. Se asigna neutral={neutral}",
            neutral=neutral_value,
        )
        return pd.Series(neutral_value, index=target_countries, name="trend")

    scores_by_year: Dict[int, pd.Series] = {}

    for year in years:
        logger.info(
            "Calculando snapshot histórico para Trend TOPSIS: año={year}",
            year=year,
        )


        df_snapshot = build_asof_snapshot(df, year)

        if df_snapshot.empty:
            logger.warning(
                "Snapshot vacío para año={year}. Se omite.",
                year=year,
            )
            continue

        df_snapshot = append_origin_reference_rows(
            df_snapshot=df_snapshot,
            df_long=df_long,
            origin_country=origin_country,
        )

        # IMPORTANTE:
        # Para el cálculo histórico de tendencia, Colombia debe existir durante
        # feature_engineering porque algunas variables relativas, como
        # cultural_distance_hofstede, se calculan tomando a COL como referencia.
        # Sin embargo, COL no debe competir en el ranking TOPSIS histórico.
        # Por eso se conserva en el snapshot antes de prepare_decision_matrix()
        # y se elimina del decision_matrix inmediatamente después.

        wide_raw_y, decision_matrix_y, _ = prepare_decision_matrix(
            df_snapshot,
            configs,
        )


        # Colombia debe existir para feature engineering, pero no competir
        if origin_country in decision_matrix_y.index:
            decision_matrix_y = decision_matrix_y.drop(index=origin_country)

        # Mantener universo comparable con el scoring actual
        if target_countries is not None:
            valid_countries = [
                c for c in target_countries
                if c in decision_matrix_y.index
            ]
            decision_matrix_y = decision_matrix_y.loc[valid_countries]

        if decision_matrix_y.empty:
            logger.warning(
                "Decision matrix histórica vacía para año={year}. Se omite.",
                year=year,
            )
            continue

        ranking_y = compute_global_topsis(
            decision_matrix_y,
            configs,
            variable_catalog,
        )

        score_y = _score_series_from_ranking(ranking_y)
        score_y.name = year

        scores_by_year[year] = score_y

    if len(scores_by_year) < 2:
        logger.warning(
            "No se pudieron calcular al menos dos scores históricos TOPSIS. "
            "Se asigna tendencia neutral={neutral}",
            neutral=neutral_value,
        )
        return pd.Series(neutral_value, index=target_countries, name="trend")

    score_panel = pd.DataFrame(scores_by_year).sort_index(axis=1)

    if target_countries is not None:
        score_panel = score_panel.reindex(target_countries)

    # Relleno temporal para países con huecos puntuales
    score_panel = score_panel.ffill(axis=1).bfill(axis=1)

    # Variación interanual del score TOPSIS
    yoy_changes = score_panel.diff(axis=1)

    # Promedio de variaciones interanuales
    trend_raw = yoy_changes.iloc[:, 1:].mean(axis=1)

    if trend_raw.dropna().empty:
        logger.warning(
            "Trend raw vacío. Se asigna tendencia neutral={neutral}",
            neutral=neutral_value,
        )
        return pd.Series(neutral_value, index=target_countries, name="trend")

    min_v = trend_raw.min()
    max_v = trend_raw.max()
    value_range = max_v - min_v

    if pd.isna(value_range) or value_range == 0:
        trend_norm = pd.Series(
            neutral_value,
            index=trend_raw.index,
            name="trend",
        )
    else:
        trend_norm = (trend_raw - min_v) / value_range
        trend_norm = trend_norm.fillna(neutral_value)
        trend_norm.name = "trend"

    logger.info(
        "Trend TOPSIS calculado con variación interanual promedio. "
        "Años={years}, países={countries}",
        years=years,
        countries=len(trend_norm),
    )

    return trend_norm


def compute_trend_factor(
    df_long: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
    target_countries: list | None = None,
) -> pd.Series:
    """Calcula el factor de tendencia por país usando una variable configurada.

    Por defecto usa gdp_growth.

    Metodología:
    1. Toma la variable configurada en settings.scoring.trend.variable.
    2. Usa la ventana [end_year - n_years + 1, end_year].
    3. Calcula el promedio por país en esa ventana.
    4. Si la ventana no tiene datos suficientes, usa el último dato disponible
       de la variable por país.
    5. Normaliza el resultado a [0, 1]


    La tendencia actual se interpreta como un proxy de momentum macroeconómico.

    La salida se normaliza entre 0 y 1 para combinarla con TOPSIS e IPC en:

        RADAR(j,l) = alpha * CC_TOPSIS(j,l) + beta * IPC(j) + gamma * Tendencia(j)

    Args:
        df_long: Dataset largo con columnas country_iso3, year, variable, value.
        configs: Diccionario completo de configuración.
        target_countries: Lista de países que entraron efectivamente a la matriz
            de decisión. Si None, se usa la lista de países de settings.

    Returns:
        Serie indexada por country_iso3 con valores normalizados de tendencia.
    """


    settings = configs["settings"]

    trend_cfg = (
        settings
        .get("scoring", {})
        .get("trend", {})
    )

    growth_variable = trend_cfg.get("variable", "gdp_growth")
    n_years = int(trend_cfg.get("n_years", 3))
    neutral_value = float(trend_cfg.get("neutral_value", 0.5))

    end_year_cfg = trend_cfg.get("end_year")
    end_year = int(end_year_cfg) if end_year_cfg is not None else None

    if target_countries is None:
        target_countries = get_country_list(settings)

    df_g = df_long[df_long["variable"] == growth_variable].copy()

    logger.info(
        "Calculando tendencia con variable={var}. Filas encontradas={rows}",
        var=growth_variable,
        rows=len(df_g),
    )

    if df_g.empty:
        logger.warning(
            "Sin datos para tendencia ({var}). Se asigna valor neutral {neutral}",
            var=growth_variable,
            neutral=neutral_value,
        )

        return pd.Series(
            neutral_value,
            index=target_countries,
            name="trend",
        )

    df_g["year"] = pd.to_numeric(df_g["year"], errors="coerce")
    df_g["value"] = pd.to_numeric(df_g["value"], errors="coerce")

    df_g = df_g.dropna(
        subset=["country_iso3", "year", "value"])

    if df_g.empty:
        logger.warning(
            "Datos inválidos para tendencia ({var}). Se asigna valor neutral {neutral}",
            var=growth_variable,
            neutral=neutral_value,
        )

        return pd.Series(
            neutral_value,
            index=target_countries,
            name="trend",
        )

    df_g["year"] = df_g["year"].astype(int)

    if end_year is None:
        end_year = int(df_g["year"].max())

    min_year = end_year - n_years + 1

    logger.info(
        "Ventana tendencia {var}: {min_year}-{end_year}",
        var=growth_variable,
        min_year=min_year,
        end_year=end_year,
    )

    df_recent = df_g[
        (df_g["year"] >= min_year)
        & (df_g["year"] <= end_year)
    ].copy()

    logger.info(
        "Filas en ventana de tendencia: {rows}",
        rows=len(df_recent),
    )

    if not df_recent.empty:
        trend_raw = (
            df_recent
            .groupby("country_iso3")["value"]
            .mean()
        )
    else:
        logger.warning(
            "No hay datos de {var} en la ventana {min_year}-{end_year}. "
            "Se usará último dato disponible por país.",
            var=growth_variable,
            min_year=min_year,
            end_year=end_year,
        )

        trend_raw = (
            df_g
            .sort_values(["country_iso3", "year"])
            .groupby("country_iso3")["value"]
            .last()
        )

    trend_raw = trend_raw.reindex(target_countries)

    # Si algunos países no tienen dato, se imputan con la mediana de la muestra.
    if trend_raw.dropna().empty:
        logger.warning(
            "Todos los países objetivo quedaron sin tendencia. "
            "Se asigna valor neutral {neutral}",
            neutral=neutral_value,
        )

        return pd.Series(
            neutral_value,
            index=target_countries,
            name="trend",
        )

    trend_raw = trend_raw.fillna(trend_raw.median())

    # ---------------------------------------------------------------------
    # Winsorización antes de normalizar
    # ---------------------------------------------------------------------
    # IMPORTANTE:
    # Se aplica antes del min-max para evitar que países con crecimiento atípico
    # extremo, por ejemplo Guyana, dominen completamente el rango de normalización.
    # Esto conserva el orden relativo general, pero reduce la influencia de outliers.
    lower_q = float(trend_cfg.get("winsor_lower_quantile", 0.05))
    upper_q = float(trend_cfg.get("winsor_upper_quantile", 0.95))

    lower_bound = trend_raw.quantile(lower_q)
    upper_bound = trend_raw.quantile(upper_q)

    trend_raw = trend_raw.clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    logger.info(
        "Winsorización aplicada a tendencia {var}: q_low={q_low}, q_high={q_high}, "
        "lower={lower}, upper={upper}",
        var=growth_variable,
        q_low=lower_q,
        q_high=upper_q,
        lower=round(float(lower_bound), 4),
        upper=round(float(upper_bound), 4),
    )

    # ---------------------------------------------------------------------
    # Normalización min-max posterior a winsorización
    # ---------------------------------------------------------------------
    min_v = trend_raw.min()
    max_v = trend_raw.max()
    value_range = max_v - min_v

    logger.info(
        "Resumen trend_raw {var}: min={min_v}, max={max_v}, rango={range_v}",
        var=growth_variable,
        min_v=round(float(min_v), 4),
        max_v=round(float(max_v), 4),
        range_v=round(float(value_range), 4),
    )

    if pd.isna(value_range) or value_range == 0:
        logger.warning(
            "La variable {var} no tiene variación entre países. "
            "Se asigna valor neutral {neutral}",
            var=growth_variable,
            neutral=neutral_value,
        )

        trend_norm = pd.Series(
            neutral_value,
            index=target_countries,
        )
    else:
        trend_norm = (trend_raw - min_v) / value_range
        trend_norm = trend_norm.fillna(neutral_value)

    trend_norm.name = "trend"

    logger.info(
        "Factor de tendencia calculado con {var}, ventana={n} años, países={c}",
        var=growth_variable,
        n=n_years,
        c=len(trend_norm),
    )

    return trend_norm


def compute_radar_composite(
    topsis_scores: pd.Series,
    ipc: pd.Series,
    trend: pd.Series,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.Series:
    """RADAR(j) = alpha * CC + beta * IPC + gamma * Tendencia."""
    total_w = alpha + beta + gamma
    if abs(total_w - 1.0) > 1e-6:
        alpha, beta, gamma = alpha / total_w, beta / total_w, gamma / total_w

    idx = topsis_scores.index
    ipc_aligned = ipc.reindex(idx).fillna(ipc.median())
    trend_aligned = trend.reindex(idx).fillna(trend.median())

    radar = alpha * topsis_scores + beta * ipc_aligned + gamma * trend_aligned
    return radar.sort_values(ascending=False)


def _score_series_from_ranking(ranking_df: pd.DataFrame) -> pd.Series:
    """Extrae una Serie score indexada por country_iso3."""

    if "country_iso3" in ranking_df.columns:
        return ranking_df.set_index("country_iso3")["score"]

    if ranking_df.index.name == "country_iso3":
        return ranking_df["score"]

    return ranking_df["score"]


def run_full_scoring(
    df_long: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
    persist: bool = True,
) -> Dict[str, Any]:
    """Ejecuta el scoring completo del RADAR Cibest.

    Flujo:
    1. Excluye el país origen si settings.scoring.exclude_origin_country = true.
    2. Prepara matriz de decisión.
    3. Calcula ranking global TOPSIS.
    4. Calcula rankings por línea de negocio.
    5. Calcula IPC / afinidad.
    6. Persiste resultados si persist=True.

    Nota metodológica:
    Si el holding está basado en Colombia, Colombia no debe competir como país
    candidato de internacionalización. Se conserva como país origen para cálculo
    de afinidad, pero se excluye del universo evaluado.


    Fórmula:
        RADAR(j,l) = alpha * CC_TOPSIS(j,l) + beta * IPC(j) + gamma * Tendencia(j)
    """

    settings = configs["settings"]
    variable_catalog = get_variable_catalog(configs["variables"])

    origin_country = settings.get("origin_country", "COL")
    exclude_origin_country = (
        settings
        .get("scoring", {})
        .get("exclude_origin_country", False)
    )

    df_scoring = df_long.copy()

    # if exclude_origin_country:
    #     n_before = df_scoring["country_iso3"].nunique()

    #     df_scoring = df_scoring[
    #         df_scoring["country_iso3"] != origin_country
    #     ].copy()

    #     n_after = df_scoring["country_iso3"].nunique()

    #     logger.info(
    #         "País origen excluido del scoring: {origin}. Países {before} -> {after}",
    #         origin=origin_country,
    #         before=n_before,
    #         after=n_after,
    #     )

    wide_raw, decision_matrix, excluded = prepare_decision_matrix(df_scoring, configs)
    # EXCLUIR COLOMBIA DESPUÉS del feature engineering
    if exclude_origin_country:
        n_before = decision_matrix.shape[0]

        if origin_country in decision_matrix.index:
            decision_matrix = decision_matrix.drop(index=origin_country)

        if origin_country in wide_raw.index:
            wide_raw = wide_raw.drop(index=origin_country)

        n_after = decision_matrix.shape[0]

        logger.info(
            "País origen excluido del scoring (post-feature engineering): {origin}. Países {before} -> {after}",
            origin=origin_country,
            before=n_before,
            after=n_after,
        )

    global_ranking = compute_global_topsis(decision_matrix, configs, variable_catalog)
    bl_rankings = score_all_business_lines(decision_matrix, configs, variable_catalog)

    ipc_df = compute_ipc(wide_raw, origin_country=origin_country, component_weights=configs["weights"].get("ipc_component_weights"))

    if "country_iso3" in ipc_df.columns:
        ipc = ipc_df.set_index("country_iso3")["ipc"]
    else:
        ipc = ipc_df["ipc"]

    if exclude_origin_country:
        ipc = ipc.drop(index=origin_country, errors="ignore")
        if "country_iso3" in ipc_df.columns:
            ipc_df = ipc_df[ipc_df["country_iso3"] != origin_country].copy()
        else:
            ipc_df = ipc_df.drop(index=origin_country, errors="ignore")

    # trend = compute_trend_factor(
    #     df_scoring,
    #     configs,
    #     target_countries=decision_matrix.index.tolist(),
    # )
    trend_cfg = (
        settings
        .get("scoring", {})
        .get("trend", {})
    )

    trend_enabled = bool(trend_cfg.get("enabled", True))
    trend_method = trend_cfg.get("method", "gdp_growth")

    
    logger.info(
        "Método de tendencia configurado: {method} | enabled={enabled} | variable={var}",
        method=trend_method,
        enabled=trend_enabled,
        var=trend_cfg.get("variable", "gdp_growth"),
    )


    if not trend_enabled:
        trend = pd.Series(
            float(trend_cfg.get("neutral_value", 0.5)),
            index=decision_matrix.index.tolist(),
            name="trend",
        )


    
    elif trend_method == "global_score_yoy":
        end_year_cfg = trend_cfg.get("end_year")
        if end_year_cfg is not None:
            end_year_cfg = int(end_year_cfg)

        trend = compute_trend_from_global_score_yoy(
            
            # IMPORTANTE:
            # Para calcular la tendencia histórica del score TOPSIS se usa df_long
            # completo y NO df_scoring. Esto garantiza que el país origen (COL)
            # siga disponible durante los snapshots históricos y el feature engineering.
            # COL puede excluirse después del decision_matrix para que no compita,
            # pero debe existir como referencia metodológica.

            df_long=df_long,
            configs=configs,
            variable_catalog=variable_catalog,
            origin_country=origin_country,
            target_countries=decision_matrix.index.tolist(),
            n_years=int(trend_cfg.get("n_years", 3)),
            end_year=end_year_cfg,
            neutral_value=float(trend_cfg.get("neutral_value", 0.5)),
        )
    else:
        trend = compute_trend_factor(
            
            # Tendencia basada en gdp_growth.
            # Esta ruta no recalcula TOPSIS histórico; solo usa la variable configurada
            # en settings.scoring.trend.variable, por defecto gdp_growth.
            # Es más robusta cuando la cobertura histórica multivariable es baja.

            df_long,
            configs,
            target_countries=decision_matrix.index.tolist(),
        )

    weights_comp = settings["scoring"]["composite_weights"]
    alpha = float(weights_comp["alpha"])
    beta = float(weights_comp["beta"])
    gamma = float(weights_comp["gamma"])

    global_score = _score_series_from_ranking(global_ranking)

    radar_global = compute_radar_composite(
        global_score,
        ipc,
        trend,
        alpha,
        beta,
        gamma,
    )

    radar_global_df = (
        radar_global
        .rename("radar_score")
        .reset_index()
        .rename(columns={"index": "country_iso3"})
    )

    radar_global_df["rank"] = (
        radar_global_df["radar_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    radar_global_df = radar_global_df.sort_values("rank")

    radar_by_line_data: Dict[str, pd.Series] = {}

    for bl_key, bl_df in bl_rankings.items():
        bl_score = _score_series_from_ranking(bl_df)

        radar_by_line_data[bl_key] = compute_radar_composite(
            bl_score,
            ipc,
            trend,
            alpha,
            beta,
            gamma,
        )

    radar_by_line = (
        pd.DataFrame(radar_by_line_data)
        .reset_index()
        .rename(columns={"index": "country_iso3"})
    )

    radar_by_line["GLOBAL"] = radar_by_line["country_iso3"].map(
        radar_global.to_dict()
    )

    radar_by_line["rank_global"] = (
        radar_by_line["GLOBAL"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    radar_by_line = radar_by_line.sort_values("rank_global")

    trend_df = (
        trend
        .rename("trend")
        .reset_index()
        .rename(columns={"index": "country_iso3"})
    )

    results = {
        "decision_matrix": decision_matrix,
        "wide_raw": wide_raw,
        "global_ranking": global_ranking,
        "business_line_rankings": bl_rankings,
        "ipc": ipc_df,
        "trend": trend_df,
        "radar_global": radar_global_df,
        "radar_by_line": radar_by_line,
        "excluded_countries": excluded,
        "origin_country": origin_country,
        "origin_country_excluded": bool(exclude_origin_country),
        "composite_weights": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        },
    }

    if persist:
        _persist_results(results, settings)

    return results

def _persist_results(results: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """Persiste resultados en data/scores/ con timestamp."""
    scores_dir = resolve_data_path(settings["data"].get("scores_path", "data/scores"))
    scores_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d")

    results["global_ranking"].to_parquet(
        scores_dir / f"global_ranking_{stamp}.parquet",
        index=False,
    )

    results["radar_global"].to_parquet(
        scores_dir / f"radar_global_{stamp}.parquet",
        index=False,
    )

    results["radar_by_line"].to_parquet(
        scores_dir / f"radar_by_line_{stamp}.parquet",
        index=False,
    )

    results["ipc"].to_parquet(
        scores_dir / f"ipc_{stamp}.parquet",
        index=False,
    )

    results["trend"].to_parquet(
        scores_dir / f"trend_{stamp}.parquet",
        index=False,
    )

    for bl_key, bl_df in results["business_line_rankings"].items():
        bl_df.to_parquet(
            scores_dir / f"ranking_{bl_key}_{stamp}.parquet",
            index=False,
        )

    logger.info(
        "Resultados persistidos en {d} con timestamp {s}",
        d=scores_dir,
        s=stamp,
    )


def main() -> None:
    """Entry point CLI: lee master_raw mas reciente y ejecuta scoring."""
    configs = load_all_configs()
    setup_logger(configs["settings"].get("logging"))

    raw_dir = resolve_data_path(configs["settings"]["data"]["raw_path"])
    candidates = sorted(raw_dir.glob("master_raw_*.parquet"), reverse=True)
    if not candidates:
        from src.data_extraction.pipeline import run_extraction
        master, _ = run_extraction(configs=configs, save_intermediate=True)
    else:
        master = pd.read_parquet(candidates[0])
        logger.info("Usando master raw existente: {f}", f=candidates[0].name)

    if master.empty:
        logger.error("Master vacio, abortando scoring")
        return

    results = run_full_scoring(master, configs, persist=True)
    logger.info("Scoring finalizado: {n} paises evaluados", n=len(results["radar_global"]))


if __name__ == "__main__":
    main()
