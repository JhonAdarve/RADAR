"""
Limpieza y validacion de datos crudos para RADAR Cibest.

Funcionalidades:
    - Pivoteo del formato largo a matriz ancha (country x variable)
    - Manejo de valores faltantes con imputacion configurable
    - Validacion de cobertura por pais contra umbral minimo
    - Deteccion de outliers extremos (opcional)

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import InsufficientDataError, get_country_list, get_variable_catalog


def pivot_long_to_wide(
    df_long: pd.DataFrame,
    year_strategy: str = "latest_available",
) -> pd.DataFrame:
    """Pivota de formato largo a matriz ancha country_iso3 x variable.

    Args:
        df_long: DataFrame largo con columnas country_iso3, year, variable, value.
        year_strategy: 'latest_available' | 'mean_3y' | 'mean_5y'.

    Returns:
        DataFrame ancho indexado por country_iso3.
    """
    df_clean = df_long.dropna(subset=["value"]).copy()

    static_mask = df_clean["year"] == 0
    df_static = df_clean[static_mask]
    df_temporal = df_clean[~static_mask]

    if year_strategy == "latest_available":
        if not df_temporal.empty:
            idx = df_temporal.groupby(["country_iso3", "variable"])["year"].idxmax()
            df_selected = df_temporal.loc[idx]
        else:
            df_selected = df_temporal
    elif year_strategy in ("mean_3y", "mean_5y"):
        n_years = 3 if year_strategy == "mean_3y" else 5
        latest = df_temporal.groupby(["country_iso3", "variable"])["year"].transform("max")
        df_selected = df_temporal[df_temporal["year"] > (latest - n_years)]
        df_selected = (
            df_selected.groupby(["country_iso3", "variable"], as_index=False)["value"]
            .mean()
        )
    else:
        raise ValueError(f"year_strategy desconocido: {year_strategy}")

    df_combined = pd.concat(
        [
            df_selected[["country_iso3", "variable", "value"]],
            df_static[["country_iso3", "variable", "value"]],
        ],
        ignore_index=True,
    )
    df_combined = df_combined.drop_duplicates(subset=["country_iso3", "variable"], keep="last")

    wide = df_combined.pivot(index="country_iso3", columns="variable", values="value")
    logger.info(
        "Pivoteo ancho: {c} paises x {v} variables (estrategia={s})",
        c=len(wide), v=wide.shape[1], s=year_strategy,
    )
    return wide

def pivot_latest_value_and_year(
    df_long: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construye matrices wide de último valor y año disponible.

    Para cada país-variable conserva la observación más reciente disponible.

    Returns:
        wide_values: matriz país x variable con el último valor disponible.
        wide_years: matriz país x variable con el año asociado al último valor.
    """

    df = df_long.copy()

    required_cols = {"country_iso3", "year", "variable", "value"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise InsufficientDataError(
            f"df_long no contiene columnas requeridas: {sorted(missing_cols)}"
        )

    df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
    df["variable"] = df["variable"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(
        subset=["country_iso3", "variable", "year", "value"]
    )

    if df.empty:
        raise InsufficientDataError(
            "No hay datos válidos para construir matriz wide."
        )

    df["year"] = df["year"].astype(int)

    latest = (
        df
        .sort_values(["country_iso3", "variable", "year"])
        .groupby(["country_iso3", "variable"], as_index=False)
        .tail(1)
    )

    wide_values = latest.pivot(
        index="country_iso3",
        columns="variable",
        values="value",
    )

    wide_years = latest.pivot(
        index="country_iso3",
        columns="variable",
        values="year",
    )

    logger.info(
        "Pivoteo ancho: {c} paises x {v} variables (estrategia=latest_available)",
        c=wide_values.shape[0],
        v=wide_values.shape[1],
    )

    return wide_values, wide_years


def apply_freshness_filter(
    wide_values: pd.DataFrame,
    wide_years: pd.DataFrame,
    variable_catalog: Dict[str, Dict[str, Any]],
    settings: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Marca como faltantes los datos cuyo último año disponible es demasiado antiguo.

    Una celda se considera stale si:

        freshness_reference_year - latest_year > max_data_age_years

    Las variables estáticas o explícitamente exentas no se castigan por antigüedad.
    """

    dq_cfg = settings.get("data_quality", {})

    max_age = int(dq_cfg.get("max_data_age_years", 5))

    reference_year = dq_cfg.get("freshness_reference_year")

    if reference_year is None:
        reference_year = int(
            pd.to_numeric(wide_years.stack(), errors="coerce").max()
        )
    else:
        reference_year = int(reference_year)

    exempt_variables = set(dq_cfg.get("freshness_exempt_variables", []))
    exempt_frequencies = set(
        dq_cfg.get("freshness_exempt_frequencies", ["static"])
    )

    # Eximir automáticamente variables con frecuencia estática.
    for variable, meta in variable_catalog.items():
        if meta.get("frequency") in exempt_frequencies:
            exempt_variables.add(variable)

    age_matrix = reference_year - wide_years

    stale_mask = age_matrix > max_age

    # No castigar variables estructurales/estáticas o explícitamente exentas.
    for variable in exempt_variables:
        if variable in stale_mask.columns:
            stale_mask[variable] = False

    wide_fresh = wide_values.mask(stale_mask)

    stale_report = pd.DataFrame(
        {
            "n_stale_variables": stale_mask.sum(axis=1),
            "pct_stale_variables": stale_mask.mean(axis=1),
        }
    )

    total_stale = int(stale_mask.sum().sum())

    logger.info(
        "Control de antigüedad aplicado: reference_year={ref}, max_age={age}, "
        "celdas stale={n}",
        ref=reference_year,
        age=max_age,
        n=total_stale,
    )

    if total_stale > 0:
        top_stale_vars = (
            stale_mask
            .sum(axis=0)
            .sort_values(ascending=False)
            .head(15)
            .to_dict()
        )

        top_stale_countries = (
            stale_mask
            .sum(axis=1)
            .sort_values(ascending=False)
            .head(15)
            .to_dict()
        )

        logger.info(
            "Variables con más datos stale: {vars}",
            vars=top_stale_vars,
        )

        logger.info(
            "Países con más datos stale: {countries}",
            countries=top_stale_countries,
        )

    return wide_fresh, stale_report


def validate_country_coverage(
    wide: pd.DataFrame,
    variable_catalog: Dict[str, Dict[str, Any]],
    missing_threshold: float = 0.30,
) -> Tuple[pd.DataFrame, List[str]]:
    """Filtra países que no cumplan cobertura mínima de variables.

    La cobertura debe calcularse después de aplicar reglas de frescura.
    Por tanto, una variable stale ya cuenta como faltante efectivo.
    """

    expected_vars = [
        variable
        for variable in variable_catalog
        if variable in wide.columns
    ]

    if not expected_vars:
        raise InsufficientDataError(
            "La matriz ancha no contiene variables del catálogo."
        )

    n_expected = len(expected_vars)

    missing_ratio = wide[expected_vars].isna().sum(axis=1) / n_expected

    excluded = (
        missing_ratio[missing_ratio > missing_threshold]
        .index
        .tolist()
    )

    if excluded:
        excluded_detail = (
            missing_ratio
            .loc[excluded]
            .sort_values(ascending=False)
            .round(3)
            .to_dict()
        )

        logger.warning(
            "Paises excluidos por >{p:.0%} variables faltantes efectivas "
            "(missing + stale): {lst}",
            p=missing_threshold,
            lst=excluded_detail,
        )

    wide_clean = wide.drop(index=excluded, errors="ignore")

    return wide_clean, excluded


def impute_missing(
    wide: pd.DataFrame,
    method: str = "regional_median",
    settings: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Imputa valores faltantes segun estrategia configurada."""
    wide_out = wide.copy()
    n_missing_before = wide_out.isna().sum().sum()

    if method == "global_median":
        medians = wide_out.median(numeric_only=True)
        wide_out = wide_out.fillna(medians)

    elif method == "regional_median":
        if settings is None:
            raise ValueError("regional_median requiere settings con lista de paises")
        region_map = {c["iso3"]: c["region"] for c in settings["countries"]}
        wide_out["_region"] = wide_out.index.map(region_map)
        for region, group in wide_out.groupby("_region"):
            region_medians = group.drop(columns="_region").median(numeric_only=True)
            mask = wide_out["_region"] == region
            for col in wide_out.columns:
                if col == "_region":
                    continue
                wide_out.loc[mask, col] = wide_out.loc[mask, col].fillna(region_medians[col])
        wide_out = wide_out.drop(columns="_region")
        wide_out = wide_out.fillna(wide_out.median(numeric_only=True))

    elif method == "knn":
        try:
            from sklearn.impute import KNNImputer  # noqa: WPS433
        except ImportError as exc:
            raise ImportError("sklearn es requerido para method='knn'") from exc
        imputer = KNNImputer(n_neighbors=3)
        wide_out = pd.DataFrame(
            imputer.fit_transform(wide_out),
            index=wide_out.index,
            columns=wide_out.columns,
        )

    else:
        raise ValueError(f"Metodo de imputacion desconocido: {method}")

    n_missing_after = wide_out.isna().sum().sum()
    logger.info(
        "Imputacion ({m}): {b} -> {a} celdas faltantes",
        m=method, b=int(n_missing_before), a=int(n_missing_after),
    )
    return wide_out


def flag_outliers_iqr(wide: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Genera mascara booleana de outliers extremos (regla IQR)."""
    q1 = wide.quantile(0.25)
    q3 = wide.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (wide < lower) | (wide > upper)
    n_out = mask.sum().sum()
    logger.info("Outliers detectados (IQR*{k}): {n} celdas", k=k, n=int(n_out))
    return mask


def run_cleaning(
    df_long: pd.DataFrame,
    configs: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[str]]:
    """Ejecuta el flujo completo de limpieza de extremo a extremo.

    Política metodológica:
    1. Se pivotea el dataset largo a matriz país × variable usando el último dato disponible.
    2. Se aplica filtro de vigencia: datos obsoletos se convierten en NaN.
    3. Se calcula cobertura efectiva antes de imputar.
    4. Se excluyen países que superan el umbral de faltantes efectivos.
    5. Solo después se imputan faltantes residuales en países admisibles.

    Esta secuencia evita que la imputación "rescate" países con baja cobertura
    estructural y reduce el riesgo de sesgar TOPSIS, cuyos ideales positivo y
    negativo dependen del universo de países incluidos.
    """
    settings = configs["settings"]
    variables_cfg = configs["variables"]

    # Mantener compatibilidad con la configuración actual.
    threshold = settings["scoring"]["missing_data_threshold"]
    method = settings["scoring"]["imputation_method"]

    catalog = get_variable_catalog(variables_cfg)

    # ---------------------------------------------------------------------
    # 1. Matriz latest_available con valores y años
    # ---------------------------------------------------------------------
    # Preferido: usar función que preserve el año de cada último dato para
    # poder aplicar regla de vigencia.
    wide_values, wide_years = pivot_latest_value_and_year(df_long)

    # Asegurar que solo se evalúen variables del catálogo que existen en datos.
    ordered_catalog_cols = [var for var in catalog if var in wide_values.columns]
    wide_values = wide_values[ordered_catalog_cols].copy()
    wide_years = wide_years[ordered_catalog_cols].copy()

    # ---------------------------------------------------------------------
    # 2. Aplicar freshness filter antes de cobertura
    # ---------------------------------------------------------------------
    # Esta función debe convertir en NaN los datos cuyo año excede
    # max_data_age_years frente a freshness_reference_year.
    wide_fresh, freshness_audit = apply_freshness_filter(
        wide_values=wide_values,
        wide_years=wide_years,
        variable_catalog=catalog,
        settings=settings,
    )

    # ---------------------------------------------------------------------
    # 3. Excluir países por faltantes efectivos antes de imputar
    # ---------------------------------------------------------------------
    wide_valid, excluded = validate_country_coverage(
        wide_fresh,
        catalog,
        missing_threshold=threshold,
    )

    excluded = sorted(set(excluded))

    if excluded:
        logger.warning(
            "Países excluidos antes de imputación por faltantes efectivos > {t:.1%}: {countries}",
            t=threshold,
            countries=excluded,
        )
    else:
        logger.info(
            "Ningún país supera el umbral de faltantes efectivos antes de imputación: {t:.1%}",
            t=threshold,
        )

    # ---------------------------------------------------------------------
    # 4. Imputar solo países que pasaron la regla de cobertura
    # ---------------------------------------------------------------------
    wide_imputed = impute_missing(
        wide_valid,
        method=method,
        settings=settings,
    )

    # ---------------------------------------------------------------------
    # 5. Ordenar columnas según catálogo
    # ---------------------------------------------------------------------
    ordered_cols = [v for v in catalog if v in wide_imputed.columns]
    wide_imputed = wide_imputed[ordered_cols].copy()

    # ---------------------------------------------------------------------
    # 6. Validación defensiva post-imputación
    # ---------------------------------------------------------------------
    residual_missing = wide_imputed.isna().sum().sum()

    if residual_missing > 0:
        logger.warning(
            "La matriz imputada conserva {n} faltantes residuales.",
            n=int(residual_missing),
        )

    logger.info(
        "Limpieza completada: {c} países x {v} variables | excluidos={excluded}",
        c=len(wide_imputed),
        v=wide_imputed.shape[1],
        excluded=excluded,
    )

    return wide_imputed, excluded
