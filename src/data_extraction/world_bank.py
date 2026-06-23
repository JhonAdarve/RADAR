"""
Conector World Bank para RADAR Cibest.

Responsabilidad
----------------
Extraer las variables World Bank activas del proyecto y devolverlas en el
formato estándar del pipeline:

    country_iso3 | year | variable | value

Alcance vigente
---------------
- WDI / Findex: `db=2` mediante `wbgapi`.
- WGI: `db=3` mediante `wbgapi` y códigos DataBank `GOV_WGI_*`.
- GFDD: `db=32` mediante `wbgapi`.

Regla metodológica
------------------
Se usa el último dato disponible por país e indicador (`mrnev=1`). No se fuerza
un año común, para evitar pérdida sistemática de información.

Trazabilidad
------------
Este módulo no consume fuentes complementarias ni Damodaran. Esas fuentes están
en `complementary.py` y `damodaran_country_risk_premium.py`.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import wbgapi as wb
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DataExtractionError, get_world_bank_variable_catalog, infer_world_bank_db

AMERICAS_REGIONS = ("LCN", "NAC")
EXTRA_COUNTRIES = {"ESP"}


def get_americas_countries(include_extra: Optional[Set[str]] = None) -> List[str]:
    """Obtiene países de América desde wbgapi: LCN + NAC + extras."""
    include_extra = EXTRA_COUNTRIES if include_extra is None else include_extra
    countries: Set[str] = set()
    for region in AMERICAS_REGIONS:
        countries |= set(wb.region.members(region))
    countries |= set(include_extra)
    return sorted(countries)


def _is_cache_valid(cache_file: Path, expiry_days: int) -> bool:
    """Indica si un parquet de caché existe y aún no expira."""
    return cache_file.exists() and (
        datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
    ) < timedelta(days=expiry_days)


def _year_to_int(year_value: Any) -> int:
    """Convierte años tipo `YR2024` o `2024` a entero."""
    return int(str(year_value).replace("YR", ""))


def _wbgapi_dataframe(
    indicator_codes: List[str],
    countries: List[str],
    db: int,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Ejecuta la consulta a World Bank con `wbgapi` y reintentos.

    Esta es la única función del módulo que llama a World Bank. Todas las
    variables se consultan usando el último dato no vacío (`mrnev=1`).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return (
                wb.data.DataFrame(
                    indicator_codes,
                    economy=set(countries),
                    time="all",
                    mrnev=1,
                    skipAggs=True,
                    index=["economy", "time"],
                    columns="series",
                    labels=False,
                    db=db,
                )
                .reset_index()
                .rename(columns={"economy": "country_iso3", "time": "year"})
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            backoff = 2 ** attempt
            logger.warning(
                "wbgapi falló intento {a}/{m}, db={db}, indicadores={i}: {e}. Reintentando {b}s",
                a=attempt, m=max_retries, db=db, i=indicator_codes, e=str(exc), b=backoff,
            )
            time.sleep(backoff)
    raise DataExtractionError(f"Fallo extracción wbgapi db={db}: {indicator_codes}") from last_exc


def _wide_to_latest_long(df_raw: pd.DataFrame, vars_dict: Dict[str, str]) -> pd.DataFrame:
    """Convierte salida wide de `wbgapi` a formato largo estándar."""
    output_cols = ["country_iso3", "year", "variable_code", "value", "variable_description"]
    if df_raw.empty:
        return pd.DataFrame(columns=output_cols)

    missing = [col for col in ["country_iso3", "year"] if col not in df_raw.columns]
    if missing:
        raise DataExtractionError(f"Respuesta wbgapi sin columnas esperadas: {missing}")

    df_raw = df_raw.copy()
    df_raw["year"] = df_raw["year"].map(_year_to_int)
    value_vars = [code for code in vars_dict if code in df_raw.columns]
    if not value_vars:
        return pd.DataFrame(columns=output_cols)

    df_long = (
        df_raw
        .melt(
            id_vars=["country_iso3", "year"],
            value_vars=value_vars,
            var_name="variable_code",
            value_name="value",
        )
        .dropna(subset=["value"])
    )
    if df_long.empty:
        return pd.DataFrame(columns=output_cols)

    return (
        df_long
        .sort_values("year", ascending=False)
        .groupby(["country_iso3", "variable_code"], as_index=False)
        .first()
        .assign(variable_description=lambda d: d["variable_code"].map(vars_dict))
    )


def get_latest_indicators(
    vars_dict: Dict[str, str],
    db: int,
    countries: Optional[List[str]] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Extrae el último dato disponible por país e indicador para una base World Bank."""
    if not vars_dict:
        return pd.DataFrame(columns=["country_iso3", "year", "variable_code", "value", "variable_description"])
    countries = get_americas_countries() if countries is None else countries
    df_raw = _wbgapi_dataframe(list(vars_dict.keys()), countries, db=db, max_retries=max_retries)
    return _wide_to_latest_long(df_raw, vars_dict)


def fetch_indicator(
    indicator_code: str,
    countries: Optional[List[str]] = None,
    source_cfg: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
    cache_expiry_days: int = 30,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Extrae un único indicador y retorna `country_iso3, year, variable, value`."""
    countries = get_americas_countries() if countries is None else countries
    db = infer_world_bank_db(indicator_code, source_cfg)
    cache_file = None if cache_dir is None else cache_dir / f"wb_{indicator_code.replace('.', '_')}_db{db}.parquet"
    if cache_file and _is_cache_valid(cache_file, cache_expiry_days):
        return pd.read_parquet(cache_file)

    df = (
        get_latest_indicators({indicator_code: indicator_code}, db=db, countries=countries, max_retries=max_retries)
        .rename(columns={"variable_code": "variable"})[["country_iso3", "year", "variable", "value"]]
    )
    if cache_file and not df.empty:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
    return df


def fetch_all_indicators(
    variables_cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    countries: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    cache_expiry_days: int = 30,
) -> pd.DataFrame:
    """Extrae todas las variables World Bank/WGI/GFDD/Findex declaradas en variables.yaml.

    Prioridad para definir la base wbgapi:
    1. db / wb_db definido explícitamente en variables.yaml.
    2. source_cfg["indicator_databases"] en data_sources.yaml.
    3. infer_world_bank_db().
    4. default_db.
    """

    catalog = get_world_bank_variable_catalog(variables_cfg, source_cfg)
    countries = get_americas_countries() if countries is None else countries

    grouped: Dict[int, Dict[str, str]] = {}
    code_to_variable: Dict[str, str] = {}

    for variable, meta in catalog.items():
        code = (
            meta.get("indicator_code")
            or meta.get("code")
            or meta.get("indicator")
        )

        if code is None:
            logger.warning(
                "Variable World Bank sin indicator_code: {var}",
                var=variable,
            )
            continue

        code = str(code).strip()

        # ------------------------------------------------------------------
        # Prioridad de base:
        # 1. db/wb_db explícito en variables.yaml.
        # 2. indicator_databases en data_sources.yaml.
        # 3. infer_world_bank_db().
        # ------------------------------------------------------------------
        db_value = (
            meta.get("db")
            or meta.get("wb_db")
        )

        if db_value is not None:
            db = int(db_value)
        else:
            db = infer_world_bank_db(
                indicator_code=code,
                source_cfg=source_cfg,
            )

        grouped.setdefault(db, {})[code] = meta.get("description", variable)
        code_to_variable[code] = variable

    if not grouped:
        logger.warning("No hay indicadores World Bank activos para extraer.")
        return pd.DataFrame(
            columns=["country_iso3", "year", "variable", "value"]
        )

    logger.info(
        "Indicadores World Bank agrupados por base: {summary}",
        summary={db: len(vars_dict) for db, vars_dict in grouped.items()},
    )

    frames: List[pd.DataFrame] = []

    max_retries = (
        int(source_cfg.get("retry", {}).get("max_attempts", 3))
        if source_cfg
        else 3
    )

    for db, vars_dict in grouped.items():
        logger.info(
            "Extrayendo indicadores World Bank: db={db}, n_indicadores={n}",
            db=db,
            n=len(vars_dict),
        )

        latest = get_latest_indicators(
            vars_dict=vars_dict,
            db=db,
            countries=countries,
            max_retries=max_retries,
        )

        if latest.empty:
            logger.warning(
                "Extracción World Bank vacía para db={db}",
                db=db,
            )
            continue

        latest["variable"] = latest["variable_code"].map(code_to_variable)

        unmapped = latest[latest["variable"].isna()]["variable_code"].unique()

        if len(unmapped) > 0:
            logger.warning(
                "Indicadores sin mapeo variable interna: {codes}",
                codes=list(unmapped),
            )

        latest = latest.dropna(subset=["variable"])

        frames.append(
            latest[
                ["country_iso3", "year", "variable", "value"]
            ]
        )

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(
            columns=["country_iso3", "year", "variable", "value"]
        )

    if not df.empty:
        df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
        df["variable"] = df["variable"].astype(str).str.strip()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df.dropna(
            subset=["country_iso3", "year", "variable", "value"]
        )

        df["year"] = df["year"].astype(int)

        df = df.drop_duplicates(
            subset=["country_iso3", "year", "variable"],
            keep="last",
        )

    if cache_dir and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            cache_dir / "wb_all_latest.parquet",
            index=False,
        )

    return (
        df
        .sort_values(["country_iso3", "variable", "year"])
        .reset_index(drop=True)
    )


def _wide_to_historical_long(
    df_raw: pd.DataFrame,
    vars_dict: Dict[str, str],
) -> pd.DataFrame:
    """Convierte salida wide de wbgapi a formato largo histórico.

    Salida:
        country_iso3 | year | variable_code | value | variable_description
    """

    output_cols = [
        "country_iso3",
        "year",
        "variable_code",
        "value",
        "variable_description",
    ]

    if df_raw.empty:
        return pd.DataFrame(columns=output_cols)

    df = df_raw.copy()

    # wbgapi suele devolver un índice por país. Lo convertimos a columna.
    if df.index.name is not None:
        df = df.reset_index()
    else:
        df = df.reset_index()

    # Normalizar nombre de país ISO3
    possible_country_cols = ["economy", "country", "Country", "index"]

    country_col = None
    for c in possible_country_cols:
        if c in df.columns:
            country_col = c
            break

    if country_col is None:
        country_col = df.columns[0]

    df = df.rename(columns={country_col: "country_iso3"})

    # Identificar columnas tipo YR2022, YR2023, 2022, 2023
    year_cols = [
        c for c in df.columns
        if str(c).startswith("YR") or str(c).isdigit()
    ]

    if not year_cols:
        return pd.DataFrame(columns=output_cols)

    # Si viene una sola variable, vars_dict tendrá un código.
    # Si vienen varias, wbgapi puede devolver columnas multi-index dependiendo de la consulta.
    # Esta función cubre el caso estándar por indicador.
    records = []

    for variable_code, variable_description in vars_dict.items():
        temp = df[["country_iso3"] + year_cols].copy()

        temp = temp.melt(
            id_vars=["country_iso3"],
            value_vars=year_cols,
            var_name="year",
            value_name="value",
        )

        temp["variable_code"] = variable_code
        temp["variable_description"] = variable_description
        temp["year"] = (
            temp["year"]
            .astype(str)
            .str.replace("YR", "", regex=False)
            .astype(int)
        )

        records.append(temp)

    out = pd.concat(records, ignore_index=True)

    out = (
        out
        .dropna(subset=["country_iso3", "year", "value"])
        [output_cols]
    )

    return out

def get_historical_indicators(
    vars_dict: Dict[str, str],
    db: int,
    countries: Optional[List[str]] = None,
    start_year: int = 2022,
    end_year: int = 2024,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Extrae histórico anual por país e indicador para una base World Bank.

    A diferencia de get_latest_indicators(), esta función conserva múltiples años
    por país-variable. Se usa para calcular la tendencia del score TOPSIS global.
    """

    output_cols = [
        "country_iso3",
        "year",
        "variable_code",
        "value",
        "variable_description",
    ]

    if not vars_dict:
        return pd.DataFrame(columns=output_cols)

    countries = get_americas_countries() if countries is None else countries

    frames = []

    for indicator_code, description in vars_dict.items():
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Extrayendo histórico WB: indicador={ind}, db={db}, años={start}-{end}",
                    ind=indicator_code,
                    db=db,
                    start=start_year,
                    end=end_year,
                )

                # Importante:
                # Aquí NO se usa mrnev=1. Se pide un rango anual explícito.
                df_raw = wb.data.DataFrame(
                    indicator_code,
                    countries,
                    time=range(start_year, end_year + 1),
                    db=db,
                    labels=False,
                )

                if df_raw.empty:
                    break

                df = df_raw.copy()

                if df.index.name is not None:
                    df = df.reset_index()
                else:
                    df = df.reset_index()

                # En wbgapi normalmente la columna país puede llamarse economy
                country_col = None
                for c in ["economy", "country", "Country", "index"]:
                    if c in df.columns:
                        country_col = c
                        break

                if country_col is None:
                    country_col = df.columns[0]

                df = df.rename(columns={country_col: "country_iso3"})

                year_cols = [
                    c for c in df.columns
                    if str(c).startswith("YR") or str(c).isdigit()
                ]

                if not year_cols:
                    break

                long_df = df.melt(
                    id_vars=["country_iso3"],
                    value_vars=year_cols,
                    var_name="year",
                    value_name="value",
                )

                long_df["year"] = (
                    long_df["year"]
                    .astype(str)
                    .str.replace("YR", "", regex=False)
                    .astype(int)
                )

                long_df["variable_code"] = indicator_code
                long_df["variable_description"] = description

                long_df = long_df[
                    [
                        "country_iso3",
                        "year",
                        "variable_code",
                        "value",
                        "variable_description",
                    ]
                ].dropna(subset=["value"])

                frames.append(long_df)
                break

            except Exception as exc:
                logger.warning(
                    "Error extrayendo histórico WB indicador={ind}, intento={attempt}: {err}",
                    ind=indicator_code,
                    attempt=attempt + 1,
                    err=exc,
                )

                if attempt == max_retries - 1:
                    raise DataExtractionError(
                        f"No fue posible extraer histórico WB para {indicator_code}"
                    ) from exc

                time.sleep(1)

    if not frames:
        return pd.DataFrame(columns=output_cols)

    return pd.concat(frames, ignore_index=True)

def fetch_indicator_history(
    variable_name: str,
    variables_cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    countries: Optional[List[str]] = None,
    start_year: int = 2022,
    end_year: int = 2024,
    cache_dir: Optional[Path] = None,
    cache_expiry_days: int = 30,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Extrae histórico anual de una variable World Bank específica.

    Ejemplo de uso:
        fetch_indicator_history(
            variable_name="gdp_growth",
            start_year=2022,
            end_year=2024,
        )

    Retorna:
        country_iso3 | year | variable | value
    """

    catalog = get_world_bank_variable_catalog(variables_cfg, source_cfg)

    if variable_name not in catalog:
        raise DataExtractionError(
            f"La variable {variable_name} no está en el catálogo World Bank."
        )

    meta = catalog[variable_name]

    indicator_code = (
        meta.get("code")
        or meta.get("indicator")
        or meta.get("indicator_code")
    )

    if indicator_code is None:
        raise DataExtractionError(
            f"No se encontró código World Bank para la variable {variable_name}."
        )

    countries = get_americas_countries() if countries is None else countries
    db = infer_world_bank_db(indicator_code, source_cfg)

    cache_file = None

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = (
            cache_dir
            / f"wb_history_{variable_name}_{start_year}_{end_year}.parquet"
        )

    if cache_file and _is_cache_valid(cache_file, cache_expiry_days):
        logger.info(
            "Usando caché histórico WB para {var}: {file}",
            var=variable_name,
            file=cache_file,
        )
        return pd.read_parquet(cache_file)

    frames = []

    for attempt in range(max_retries):
        try:
            logger.info(
                "Extrayendo histórico WB para {var} ({code}), años {start}-{end}",
                var=variable_name,
                code=indicator_code,
                start=start_year,
                end=end_year,
            )

            # Importante:
            # No se usa mrnev=1 porque queremos años explícitos.
            df_raw = wb.data.DataFrame(
                indicator_code,
                countries,
                time=range(start_year, end_year + 1),
                db=db,
                labels=False,
            )

            if df_raw.empty:
                logger.warning(
                    "World Bank no retornó datos para {var} en {start}-{end}",
                    var=variable_name,
                    start=start_year,
                    end=end_year,
                )
                return pd.DataFrame(
                    columns=["country_iso3", "year", "variable", "value"]
                )

            df = df_raw.copy().reset_index()

            # wbgapi normalmente usa 'economy' como columna de país.
            country_col = None
            for c in ["economy", "country", "Country", "index"]:
                if c in df.columns:
                    country_col = c
                    break

            if country_col is None:
                country_col = df.columns[0]

            df = df.rename(columns={country_col: "country_iso3"})

            year_cols = [
                c for c in df.columns
                if str(c).startswith("YR") or str(c).isdigit()
            ]

            if not year_cols:
                logger.warning(
                    "No se encontraron columnas de año para {var}",
                    var=variable_name,
                )
                return pd.DataFrame(
                    columns=["country_iso3", "year", "variable", "value"]
                )

            out = df.melt(
                id_vars=["country_iso3"],
                value_vars=year_cols,
                var_name="year",
                value_name="value",
            )

            out["year"] = (
                out["year"]
                .astype(str)
                .str.replace("YR", "", regex=False)
                .astype(int)
            )

            out["variable"] = variable_name
            out["value"] = pd.to_numeric(out["value"], errors="coerce")

            out = (
                out[["country_iso3", "year", "variable", "value"]]
                .dropna(subset=["country_iso3", "year", "variable", "value"])
                .sort_values(["country_iso3", "year"])
                .reset_index(drop=True)
            )

            if cache_file:
                out.to_parquet(cache_file, index=False)

            return out

        except Exception as exc:
            logger.warning(
                "Error extrayendo histórico WB {var}, intento {attempt}: {err}",
                var=variable_name,
                attempt=attempt + 1,
                err=exc,
            )

            if attempt == max_retries - 1:
                raise DataExtractionError(
                    f"No fue posible extraer histórico WB para {variable_name}"
                ) from exc

            time.sleep(1)

    return pd.DataFrame(
        columns=["country_iso3", "year", "variable", "value"]
    )

def fetch_all_indicators_historical(
    variables_cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    countries: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    cache_expiry_days: int = 30,
    start_year: int = 2022,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Extrae histórico anual de variables World Bank/WGI/GFDD.

    Esta función se usa para construir master histórico y calcular:
        Tendencia(j) = variación interanual promedio del score TOPSIS global.
    """

    catalog = get_world_bank_variable_catalog(variables_cfg, source_cfg)
    countries = get_americas_countries() if countries is None else countries

    if not catalog:
        return pd.DataFrame(
            columns=["country_iso3", "year", "variable", "value"]
        )

    # Agrupar indicadores por base World Bank
    by_db: Dict[int, Dict[str, str]] = {}

    for variable_name, meta in catalog.items():
        indicator_code = meta.get("code") or meta.get("indicator") or meta.get("indicator_code")

        if indicator_code is None:
            continue

        db = infer_world_bank_db(indicator_code, source_cfg)

        by_db.setdefault(db, {})[indicator_code] = variable_name

    frames = []

    for db, vars_dict in by_db.items():
        hist = get_historical_indicators(
            vars_dict=vars_dict,
            db=db,
            countries=countries,
            start_year=start_year,
            end_year=end_year,
        )

        if hist.empty:
            continue

        # variable_code -> nombre interno del modelo
        code_to_variable = {
            code: variable_name
            for code, variable_name in vars_dict.items()
        }

        hist["variable"] = hist["variable_code"].map(code_to_variable)

        hist = hist[
            ["country_iso3", "year", "variable", "value"]
        ].dropna(subset=["variable", "value"])

        frames.append(hist)

    if not frames:
        return pd.DataFrame(
            columns=["country_iso3", "year", "variable", "value"]
        )

    out = pd.concat(frames, ignore_index=True)

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["country_iso3", "year", "variable", "value"])

    return (
        out
        .sort_values(["country_iso3", "variable", "year"])
        .reset_index(drop=True)
    )


def save_raw_data(df: pd.DataFrame, raw_dir: Path, prefix: str = "world_bank") -> Path:
    """Persiste resultados World Bank raw en parquet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d')}.parquet"
    df.to_parquet(file_path, index=False)
    logger.info("World Bank raw guardado: {f}", f=file_path)
    return file_path


if __name__ == "__main__":
    sample = get_latest_indicators({"NY.GDP.MKTP.CD": "GDP nominal"}, db=2, countries=["COL", "MEX", "USA"])
    print(sample.head())
    print("filas:", len(sample))
