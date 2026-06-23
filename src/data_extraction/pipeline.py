"""
Orquestador de extracción para RADAR Cibest.

Responsabilidad
----------------
Ejecutar las fuentes activas, consolidar resultados en un DataFrame maestro y
construir un reporte de cobertura por país.

Fuentes activas
---------------
- `world_bank.py`: WDI/Findex db=2, WGI db=3 con `GOV_WGI_*`, GFDD db=32.
- `damodaran_country_risk_premium.py`: Country Risk Premium.
- `complementary.py`: CEPII, Hofstede, Heritage y salidas de colombianos.

Salida estándar
---------------
    country_iso3 | year | variable | value | source
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_extraction import complementary, damodaran_country_risk_premium, world_bank
from src.utils import DataExtractionError, get_country_list, get_variable_catalog, load_all_configs, resolve_data_path, setup_logger


def _append_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Agrega columna source si el DataFrame contiene datos."""
    return df if df.empty else df.assign(source=source_name)


def run_extraction(
    configs: Dict[str, Dict[str, Any]] | None = None,
    save_intermediate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta extracción completa de fuentes habilitadas.

    Nota metodológica:
    - Si settings.data_extraction.keep_history = False:
        World Bank se extrae en modo latest_available, útil para el ranking actual.
    - Si settings.data_extraction.keep_history = True:
        World Bank se extrae en modo histórico anual, necesario para calcular
        la tendencia del score TOPSIS global en los últimos N años.
    """

    if configs is None:
        configs = load_all_configs()

    settings = configs["settings"]
    variables_cfg = configs["variables"]
    sources_cfg = configs["data_sources"]["sources"]

    setup_logger(settings.get("logging"))

    countries = get_country_list(settings)
    cache_dir = resolve_data_path(settings["data"].get("cache_path", "data/cache/"))
    raw_dir = resolve_data_path(settings["data"].get("raw_path", "data/raw/"))
    cache_expiry = int(settings["data"].get("cache_expiry_days", 30))

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Configuración de extracción histórica
    # -------------------------------------------------------------------------
    data_extraction_cfg = settings.get("data_extraction", {})

    keep_history = bool(data_extraction_cfg.get("keep_history", False))
    historical_start_year = int(
        data_extraction_cfg.get("historical_start_year", 2022)
    )
    historical_end_year = int(
        data_extraction_cfg.get("historical_end_year", 2024)
    )

    frames: List[pd.DataFrame] = []
    sources_failed: List[str] = []

    # -------------------------------------------------------------------------
    # 1. World Bank / WGI / GFDD
    # -------------------------------------------------------------------------
    wb_cfg = sources_cfg.get("world_bank", {})

    if wb_cfg.get("enabled", False):
        try:
            if keep_history:
                logger.info(
                    "Fuente world_bank en modo histórico: años {start}-{end}",
                    start=historical_start_year,
                    end=historical_end_year,
                )

                df = world_bank.fetch_all_indicators_historical(
                    variables_cfg=variables_cfg,
                    source_cfg=wb_cfg,
                    countries=countries,
                    cache_dir=cache_dir,
                    cache_expiry_days=cache_expiry,
                    start_year=historical_start_year,
                    end_year=historical_end_year,
                )

                wb_prefix = "world_bank_historical"

            else:
                logger.info("Fuente world_bank en modo latest_available")

                df = world_bank.fetch_all_indicators(
                    variables_cfg=variables_cfg,
                    source_cfg=wb_cfg,
                    countries=countries,
                    cache_dir=cache_dir,
                    cache_expiry_days=cache_expiry,
                )

                # -------------------------------------------------------------
                # Histórico específico para Trend basado en gdp_growth
                # -------------------------------------------------------------
                trend_cfg = (
                    settings
                    .get("scoring", {})
                    .get("trend", {})
                )

                trend_method = trend_cfg.get("method", "gdp_growth")
                trend_variable = trend_cfg.get("variable", "gdp_growth")
                trend_n_years = int(trend_cfg.get("n_years", 3))
                trend_end_year = int(
                    trend_cfg.get("end_year", datetime.now().year)
                )
                trend_start_year = trend_end_year - trend_n_years + 1

                if trend_method == "gdp_growth" and trend_variable == "gdp_growth":
                    logger.info(
                        "Anexando histórico específico de {var} para Trend: años {start}-{end}",
                        var=trend_variable,
                        start=trend_start_year,
                        end=trend_end_year,
                    )

                    trend_hist = world_bank.fetch_indicator_history(
                        variable_name=trend_variable,
                        variables_cfg=variables_cfg,
                        source_cfg=wb_cfg,
                        countries=countries,
                        start_year=trend_start_year,
                        end_year=trend_end_year,
                        cache_dir=cache_dir,
                        cache_expiry_days=cache_expiry,
                    )

                    if not trend_hist.empty:
                        df = pd.concat([df, trend_hist], ignore_index=True)

                        df = df.drop_duplicates(
                            subset=["country_iso3", "year", "variable"],
                            keep="last",
                        )

                wb_prefix = "world_bank"

            if not df.empty:
                df = _append_source(df, "world_bank")
                frames.append(df)

                if save_intermediate:
                    world_bank.save_raw_data(
                        df,
                        raw_dir,
                        prefix=wb_prefix,
                    )

        except (DataExtractionError, Exception) as exc:  # noqa: BLE001
            logger.error("Fuente world_bank falló: {e}", e=str(exc))
            sources_failed.append("world_bank")

        # -------------------------------------------------------------------------
        # 2. Damodaran Country Risk Premium
        # -------------------------------------------------------------------------
        damodaran_cfg = sources_cfg.get("damodaran_country_risk_premium", {})

        if damodaran_cfg.get("enabled", False):
            try:
                df = damodaran_country_risk_premium.fetch_all_indicators(
                    countries=countries,
                    variables_cfg=variables_cfg,
                    source_cfg=damodaran_cfg,
                    cache_dir=cache_dir,
                    cache_expiry_days=cache_expiry,
                )

                if not df.empty:
                    df = _append_source(df, "damodaran_country_risk_premium")
                    frames.append(df)

                    if save_intermediate:
                        damodaran_country_risk_premium.save_raw_data(df, raw_dir)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Fuente damodaran_country_risk_premium falló: {e}",
                    e=str(exc),
                )
                sources_failed.append("damodaran_country_risk_premium")

    # -------------------------------------------------------------------------
    # 3. Complementarias
    # -------------------------------------------------------------------------
    complementary_cfg = sources_cfg.get("complementary", {})

    if complementary_cfg.get("enabled", False):
        try:
            df = complementary.fetch_all_indicators(
                countries=countries,
                variables_cfg=variables_cfg,
                source_cfg=complementary_cfg,
            )

            if not df.empty:
                df = _append_source(df, "complementary")
                frames.append(df)

                if save_intermediate:
                    complementary.save_raw_data(df, raw_dir)

        except (DataExtractionError, Exception) as exc:  # noqa: BLE001
            logger.error("Fuente complementary falló: {e}", e=str(exc))
            sources_failed.append("complementary")

    # -------------------------------------------------------------------------
    # 4. Consolidación master
    # -------------------------------------------------------------------------
    if not frames:
        logger.error("Ninguna fuente produjo datos")
        return (
            pd.DataFrame(
                columns=[
                    "country_iso3",
                    "year",
                    "variable",
                    "value",
                    "source",
                ]
            ),
            pd.DataFrame(),
        )

    master = pd.concat(frames, ignore_index=True)

    master = master[
        ["country_iso3", "year", "variable", "value", "source"]
    ].copy()

    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    master["value"] = pd.to_numeric(master["value"], errors="coerce")

    master = master.dropna(
        subset=["country_iso3", "year", "variable", "value"]
    )

    master["year"] = master["year"].astype(int)

    # En modo histórico, este drop_duplicates conserva un registro por
    # país-variable-año. No colapsa la serie temporal.
    master = master.drop_duplicates(
        subset=["country_iso3", "year", "variable"],
        keep="last",
    )

    master = (
        master
        .sort_values(["country_iso3", "variable", "year"])
        .reset_index(drop=True)
    )

    coverage = build_coverage_report(master, variables_cfg, countries)

    # -------------------------------------------------------------------------
    # 5. Persistencia
    # -------------------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d")

    if keep_history:
        master_file = raw_dir / f"master_raw_historical_{stamp}.parquet"
        coverage_file = raw_dir / f"coverage_report_historical_{stamp}.parquet"
    else:
        master_file = raw_dir / f"master_raw_{stamp}.parquet"
        coverage_file = raw_dir / f"coverage_report_{stamp}.parquet"

    master.to_parquet(master_file, index=False)
    coverage.to_parquet(coverage_file, index=False)

    logger.info(
        "Master guardado: {file} | filas={rows} | países={countries} | variables={vars}",
        file=master_file,
        rows=len(master),
        countries=master["country_iso3"].nunique(),
        vars=master["variable"].nunique(),
    )

    if keep_history:
        logger.info(
            "Cobertura histórica por año: {summary}",
            summary=master.groupby("year").size().to_dict(),
        )

    if sources_failed:
        logger.warning("Fuentes fallidas: {lst}", lst=sources_failed)

    return master, coverage


def build_coverage_report(master: pd.DataFrame, variables_cfg: Dict[str, Any], countries: List[str]) -> pd.DataFrame:
    """Construye reporte de cobertura por país."""
    catalog = get_variable_catalog(variables_cfg)
    expected_vars = set(catalog.keys())
    n_expected = len(expected_vars)
    rows: List[Dict[str, Any]] = []
    for country in countries:
        available = set(master.loc[master["country_iso3"] == country].dropna(subset=["value"])["variable"].unique())
        missing = expected_vars - available
        rows.append({
            "country_iso3": country,
            "n_variables_total": n_expected,
            "n_variables_disponibles": len(available & expected_vars),
            "pct_cobertura": round((len(available & expected_vars) / n_expected) * 100, 2) if n_expected else 0,
            "variables_faltantes": "; ".join(sorted(missing)) if missing else "",
        })
    return pd.DataFrame(rows).sort_values("pct_cobertura", ascending=False).reset_index(drop=True)


def main() -> None:
    """Entry point de consola."""
    master, coverage = run_extraction(save_intermediate=True)
    print(master.head())
    print(coverage.head())


if __name__ == "__main__":
    main()
