"""
Conector de variables complementarias para RADAR Cibest.

Consolida fuentes complementarias fuera de World Bank y Damodaran:
CEPII, Hofstede, Heritage EFI y salidas de colombianos.

Salida estándar:
    country_iso3 | year | variable | value

Nota metodológica:
No se fuerza rango temporal común. Las fuentes estáticas usan un año estructural
de referencia y las temporales conservan el último dato disponible.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pycountry
from loguru import logger

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    def unidecode(value: str) -> str:
        return value

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DataExtractionError, get_variable_catalog

STATIC_FILE_TO_VARIABLE = {
    "cepii_geodist": "geographic_distance_km",
    "cepii_language_spanish": "common_language_spanish",
}


HERITAGE_COUNTRY_TO_ISO3 = {
    "Argentina": "ARG", "Barbados": "BRB", "Belize": "BLZ", "Bolivia": "BOL", "Brazil": "BRA",
    "Canada": "CAN", "Chile": "CHL", "Colombia": "COL", "Costa Rica": "CRI", "Cuba": "CUB",
    "Dominica": "DMA", "Dominican Republic": "DOM", "Ecuador": "ECU", "El Salvador": "SLV",
    "Guatemala": "GTM", "Guyana": "GUY", "Haiti": "HTI", "Honduras": "HND", "Jamaica": "JAM",
    "Mexico": "MEX", "Nicaragua": "NIC", "Panama": "PAN", "Paraguay": "PRY", "Peru": "PER",
    "Saint Lucia": "LCA", "Saint Vincent and the Grenadines": "VCT", "Suriname": "SUR",
    "The Bahamas": "BHS", "Trinidad and Tobago": "TTO", "United States": "USA", "Uruguay": "URY", "Venezuela": "VEN",
}

MIGRACION_MANUAL_COUNTRY_MAP_RAW = {
    "Estados Unidos de America": "USA",
    "Estados Unidos De America": "USA",
    "Estados Unidos de América": "USA",
    "Estados Unidos": "USA",
    "Mexico": "MEX",
    "Brasil": "BRA",
    "Canada": "CAN",
    "Peru": "PER",
    "Panama": "PAN",
    "El Salvador": "SLV",
    "Republica Dominicana": "DOM",
    "Bolivia (Estado Plurinacional de)": "BOL",
    "Venezuela (Republica Bolivariana de)": "VEN",
    "Surinam": "SUR",
    "Belice": "BLZ",
    "Trinidad Y Tobago": "TTO",
    "Trinidad y Tobago": "TTO",
    "Espana": "ESP",
}

M49_TO_ISO3 = {
    32: "ARG",
    68: "BOL",
    76: "BRA",
    152: "CHL",
    170: "COL",
    188: "CRI",
    192: "CUB",
    214: "DOM",
    218: "ECU",
    222: "SLV",
    320: "GTM",
    340: "HND",
    388: "JAM",
    484: "MEX",
    558: "NIC",
    591: "PAN",
    600: "PRY",
    604: "PER",
    858: "URY",
    862: "VEN",
    124: "CAN",
    840: "USA",
    332: "HTI",
    84: "BLZ",
    328: "GUY",
    740: "SUR",
    780: "TTO",
    724: "ESP",
    44: "BHS",
    52: "BRB",
}

def normalize_country_name(value: Any) -> Optional[str]:
    """Normaliza nombres de países para mapeo robusto.

    Convierte a ASCII, elimina espacios redundantes y estandariza mayúsculas.
    """
    if pd.isna(value):
        return None

    text = unidecode(str(value).strip())
    text = " ".join(text.split())
    return text.upper()


MIGRACION_MANUAL_COUNTRY_MAP = {
    normalize_country_name(country): iso3
    for country, iso3 in MIGRACION_MANUAL_COUNTRY_MAP_RAW.items()
}

def to_iso3_from_m49(value: Any) -> Optional[str]:
    """Convierte código M49 a ISO3 si está disponible."""
    if pd.isna(value):
        return None

    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return None

    return M49_TO_ISO3.get(code)


def to_iso3(country_name: Any) -> Optional[str]:
    """Convierte nombre de país a código ISO3."""
    if pd.isna(country_name):
        return None
    try:
        return pycountry.countries.lookup(str(country_name).strip()).alpha_3
    except LookupError:
        return None


def to_iso3_spanish(country_name: Any) -> Optional[str]:
    """Convierte nombres de países en español/inglés a ISO3.

    Usa primero un mapa manual normalizado y luego intenta pycountry.
    """
    normalized = normalize_country_name(country_name)

    if normalized is None:
        return None

    if normalized in MIGRACION_MANUAL_COUNTRY_MAP:
        return MIGRACION_MANUAL_COUNTRY_MAP[normalized]

    # pycountry funciona mejor con Title Case que con upper.
    try:
        return pycountry.countries.lookup(normalized.title()).alpha_3
    except LookupError:
        return None


def _resolve_file(path_value: str | Path) -> Path:
    """Resuelve rutas absolutas o relativas contra la raíz del proyecto."""
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _filter_countries(df: pd.DataFrame, countries: List[str]) -> pd.DataFrame:
    """Filtra por países ISO3 objetivo."""
    return df[df["country_iso3"].isin(set(countries))].copy()


def load_static_standard_csv(file_path: str | Path, variable_name: str, countries: List[str], year_reference: int = 2020) -> pd.DataFrame:
    """Carga CSV simple con columnas country_iso3,value y year opcional."""
    path = _resolve_file(file_path)
    if not path.exists():
        logger.warning("CSV complementario no existe: {p}", p=path)
        return pd.DataFrame(columns=["country_iso3", "year", "variable", "value"])

    df = pd.read_csv(path)
    missing = [col for col in ["country_iso3", "value"] if col not in df.columns]
    if missing:
        raise DataExtractionError(f"{path.name} no contiene columnas requeridas: {missing}")
    if "year" not in df.columns:
        df["year"] = year_reference

    out = (
        df.assign(variable=variable_name)
        [["country_iso3", "year", "variable", "value"]]
        .assign(
            year=lambda d: pd.to_numeric(d["year"], errors="coerce"),
            value=lambda d: pd.to_numeric(d["value"], errors="coerce"),
        )
        .dropna(subset=["country_iso3", "year", "value"])
        .assign(year=lambda d: d["year"].astype(int))
    )
    return _filter_countries(out, countries)


def load_hofstede_scores(file_path: str | Path, countries: List[str], year_reference: int = 2020) -> pd.DataFrame:
    """Carga Hofstede y genera variables hofstede_* en formato largo."""
    path = _resolve_file(file_path)
    if not path.exists():
        logger.warning("CSV Hofstede no existe: {p}", p=path)
        return pd.DataFrame(columns=["country_iso3", "year", "variable", "value"])

    df = pd.read_csv(path)
    if "country" not in df.columns:
        raise DataExtractionError("Hofstede debe contener columna 'country'")

    df["country_iso3"] = df["country"].astype(str).str.strip().apply(to_iso3)
    value_vars = [v for v in ["pdi", "idv", "mas", "uai", "lto", "ivr"] if v in df.columns]
    if not value_vars:
        raise DataExtractionError("Hofstede no contiene pdi/idv/mas/uai/lto/ivr")

    out = (
        df.dropna(subset=["country_iso3"])
        .melt(id_vars=["country_iso3"], value_vars=value_vars, var_name="hofstede_code", value_name="value")
        .assign(variable=lambda d: "hofstede_" + d["hofstede_code"], year=year_reference)
        [["country_iso3", "year", "variable", "value"]]
        .assign(value=lambda d: pd.to_numeric(d["value"], errors="coerce"))
        .dropna(subset=["value"])
    )
    return _filter_countries(out, countries)


def load_heritage_efi(file_path: str | Path, countries: List[str], variable_name: str = "heritage_efi") -> pd.DataFrame:
    """Carga Heritage EFI desde CSV fuente con metadata inicial."""
    path = _resolve_file(file_path)
    if not path.exists():
        logger.warning("CSV Heritage no existe: {p}", p=path)
        return pd.DataFrame(columns=["country_iso3", "year", "variable", "value"])

    df = pd.read_csv(path, skiprows=3)
    required = ["Country", "Index Year", "Overall Score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataExtractionError(f"Heritage no contiene columnas requeridas: {missing}")

    out = (
        df[required]
        .rename(columns={"Country": "country", "Index Year": "year", "Overall Score": "value"})
        .assign(country_iso3=lambda d: d["country"].map(HERITAGE_COUNTRY_TO_ISO3), variable=variable_name)
        [["country_iso3", "year", "variable", "value"]]
        .dropna(subset=["country_iso3", "year", "value"])
        .assign(
            year=lambda d: pd.to_numeric(d["year"], errors="coerce").astype(int),
            value=lambda d: pd.to_numeric(d["value"], errors="coerce"),
        )
        .dropna(subset=["value"])
    )
    return _filter_countries(out, countries)


def fetch_colombian_outbound_travel(
    source_cfg: Dict[str, Any],
    countries: List[str],
) -> pd.DataFrame:
    """Extrae salidas de colombianos desde datos.gov.co y calcula total anual.

    La función:
    1. Lee la fuente configurada.
    2. Normaliza columnas.
    3. Mapea país destino a ISO3 usando nombre y, si existe, código M49.
    4. Filtra países objetivo.
    5. Agrega meses a total anual.
    6. Conserva último año disponible por país.
    """

    output_cols = ["country_iso3", "year", "variable", "value"]

    if not source_cfg.get("enabled", True):
        return pd.DataFrame(columns=output_cols)

    url = source_cfg["url"]

    df_raw = pd.read_csv(url)
    df_raw.columns = [unidecode(str(col)).strip() for col in df_raw.columns]

    required_cols = {"pa_s_destino", "a_o", "total"}
    missing_cols = required_cols - set(df_raw.columns)

    if missing_cols:
        raise DataExtractionError(
            f"Fuente de migración no contiene columnas requeridas: {sorted(missing_cols)}. "
            f"Columnas disponibles: {list(df_raw.columns)}"
        )

    df = df_raw.rename(
        columns={
            "pa_s_destino": "country",
            "a_o": "year",
            "total": "value",
            "codigo_m49": "codigo_m49",
        }
    ).copy()

    # Mapeo principal por nombre.
    df["country_iso3_name"] = df["country"].apply(to_iso3_spanish)

    # Fallback por M49 si está disponible.
    if "codigo_m49" in df.columns:
        df["country_iso3_m49"] = df["codigo_m49"].apply(to_iso3_from_m49)
    else:
        df["country_iso3_m49"] = None

    df["country_iso3"] = df["country_iso3_name"].fillna(df["country_iso3_m49"])

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = (
        df
        .dropna(subset=["country_iso3", "year", "value"])
        .assign(year=lambda d: d["year"].astype(int))
    )

    df = _filter_countries(df, countries)

    annual = (
        df
        .groupby(["country_iso3", "year"], as_index=False)
        .agg(value=("value", "sum"))
    )

    latest = (
        annual
        .sort_values(["country_iso3", "year"], ascending=[True, False])
        .groupby("country_iso3", as_index=False)
        .first()
    )

    out = (
        latest
        .assign(
            variable=source_cfg.get(
                "variable_name",
                "colombian_diaspora_stock",
            )
        )
        [["country_iso3", "year", "variable", "value"]]
        .sort_values(["country_iso3", "year"])
        .reset_index(drop=True)
    )

    logger.info(
        "Salidas de colombianos cargadas: países={n_countries}, filas={n_rows}, año_min={ymin}, año_max={ymax}",
        n_countries=out["country_iso3"].nunique(),
        n_rows=len(out),
        ymin=out["year"].min() if not out.empty else None,
        ymax=out["year"].max() if not out.empty else None,
    )

    return out


def fetch_all_indicators(countries: List[str], variables_cfg: Dict[str, Any], source_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Carga todas las variables complementarias activas."""
    catalog = get_variable_catalog(variables_cfg)
    active_variables = set(catalog.keys())
    year_reference = int(source_cfg.get("structural_year_reference", 2020))
    static_files = source_cfg.get("static_files", {}) or {}
    online_sources = source_cfg.get("online_sources", {}) or {}
    frames: List[pd.DataFrame] = []

    for file_key, variable_name in STATIC_FILE_TO_VARIABLE.items():
        if variable_name in active_variables and file_key in static_files:
            frames.append(load_static_standard_csv(static_files[file_key], variable_name, countries, year_reference))

    if "hofstede_country_scores" in static_files:
        frames.append(load_hofstede_scores(static_files["hofstede_country_scores"], countries, year_reference))
    if "heritage_efi_source" in static_files and "heritage_efi" in active_variables:
        frames.append(load_heritage_efi(static_files["heritage_efi_source"], countries))
    if "colombian_diaspora_stock" in online_sources and "colombian_diaspora_stock" in active_variables:
        frames.append(fetch_colombian_outbound_travel(online_sources["colombian_diaspora_stock"], countries))

    valid = [df for df in frames if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame(columns=["country_iso3", "year", "variable", "value"])

    out = pd.concat(valid, ignore_index=True)
    out = out[out["variable"].isin(active_variables)]
    return out.sort_values(["country_iso3", "variable", "year"]).reset_index(drop=True)


def save_raw_data(df: pd.DataFrame, raw_dir: Path, prefix: str = "complementary") -> Path:
    """Persiste resultados complementarios raw en parquet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d')}.parquet"
    df.to_parquet(file_path, index=False)
    logger.info("Complementary raw guardado: {f}", f=file_path)
    return file_path
