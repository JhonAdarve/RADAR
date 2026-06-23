"""
Extractor de Country Risk Premium - Damodaran / NYU Stern para RADAR Cibest.

Descarga el Excel público de Damodaran, lee la hoja configurada, normaliza el
resultado y retorna el formato estándar del pipeline:

    country_iso3 | year | variable | value

Trazabilidad:
- Fuente: Damodaran / NYU Stern.
- Variable: country_risk_premium.
- Dirección metodológica: negativa; mayor prima implica mayor riesgo.
- Esta fuente no pertenece a World Bank; por eso usa requests y pandas.read_excel.
"""
from __future__ import annotations

import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import pycountry
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DataExtractionError

DEFAULT_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx"
DEFAULT_SHEET = "ERPs by country"
DEFAULT_VARIABLE_NAME = "country_risk_premium"


def to_iso3(country_name: Any) -> Optional[str]:
    """Convierte nombre de país a código ISO3 usando pycountry."""
    if pd.isna(country_name):
        return None
    try:
        return pycountry.countries.lookup(str(country_name).strip()).alpha_3
    except LookupError:
        return None


def _download_excel(url: str, timeout: int = 30, verify_ssl: bool = False) -> bytes:
    """Descarga el archivo Excel remoto de Damodaran."""
    logger.info("Descargando Country Risk Premium desde Damodaran: {url}", url=url)
    response = requests.get(url, timeout=timeout, verify=verify_ssl)
    response.raise_for_status()
    return response.content


def _parse_country_risk_premium_excel(
    content: bytes,
    sheet: str,
    skiprows: int,
    country_column: str,
    value_column: str,
    countries: Optional[Set[str]],
    variable_name: str,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """Parsea y normaliza el Excel de Damodaran."""
    xls = pd.ExcelFile(BytesIO(content))
    raw = pd.read_excel(xls, sheet_name=sheet, skiprows=skiprows)

    required = [country_column, value_column]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise DataExtractionError(f"Damodaran sin columnas requeridas: {missing}")

    df = raw[[country_column, value_column]].copy()
    df["country_iso3"] = df[country_column].apply(to_iso3)
    df["value"] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=["country_iso3", "value"])

    if countries:
        df = df[df["country_iso3"].isin(countries)]

    return (
        df.assign(
            year=datetime.now().year if year is None else int(year),
            variable=variable_name,
        )[["country_iso3", "year", "variable", "value"]]
        .sort_values("country_iso3")
        .reset_index(drop=True)
    )


def fetch_country_risk_premium(
    source_cfg: Dict[str, Any],
    countries: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extrae Country Risk Premium desde Damodaran / NYU Stern."""
    url = source_cfg.get("url") or source_cfg.get("base_url") or DEFAULT_URL
    sheet = source_cfg.get("sheet", DEFAULT_SHEET)
    skiprows = int(source_cfg.get("skiprows", 7))
    timeout = int(source_cfg.get("timeout_seconds", 30))
    verify_ssl = bool(source_cfg.get("verify_ssl", False))
    country_column = source_cfg.get("country_column", "Country")
    value_column = source_cfg.get("value_column", "Country Risk Premium")
    variable_name = source_cfg.get("variable_name", DEFAULT_VARIABLE_NAME)

    content = _download_excel(url=url, timeout=timeout, verify_ssl=verify_ssl)
    return _parse_country_risk_premium_excel(
        content=content,
        sheet=sheet,
        skiprows=skiprows,
        country_column=country_column,
        value_column=value_column,
        countries=set(countries) if countries else None,
        variable_name=variable_name,
    )


def fetch_all_indicators(
    countries: List[str],
    variables_cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    cache_dir: Optional[Path] = None,
    cache_expiry_days: int = 30,
) -> pd.DataFrame:
    """Firma compatible con el contrato general de extractores del pipeline."""
    del variables_cfg, cache_dir, cache_expiry_days
    return fetch_country_risk_premium(source_cfg=source_cfg, countries=countries)


def save_raw_data(df: pd.DataFrame, raw_dir: Path, prefix: str = "damodaran_country_risk_premium") -> Path:
    """Persiste el resultado raw de Damodaran en parquet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d')}.parquet"
    df.to_parquet(file_path, index=False)
    logger.info("Damodaran raw guardado: {f}", f=file_path)
    return file_path
