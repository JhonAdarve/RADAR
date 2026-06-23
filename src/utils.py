"""
Utilidades transversales para RADAR Cibest.

- World Bank se extrae con wbgapi: WDI/Findex db=2, WGI db=3 y GFDD db=32.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import yaml

try:
    from loguru import logger
except ImportError:  # pragma: no cover - fallback para entornos sin loguru instalado
    import logging

    logger = logging.getLogger("radar_cibest")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


# ============================================================================
# Constantes globales
# ============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"

WORLD_BANK_ALLOWED_DBS: Set[int] = {2, 3, 32, 28}
WORLD_BANK_ALLOWED_SOURCES: Set[str] = {"world_bank", "wgi"}
WGI_PREFIXES: Sequence[str] = ("RQ.", "GE.", "RL.", "PV.", "VA.", "CC.")


COMPLEMENTARY_ALLOWED_FILES: Set[str] = {
    "cepii_geodist",
    "cepii_language_spanish",
    "hofstede_country_scores",
    "heritage_efi_source",
}


COMPLEMENTARY_ALLOWED_ONLINE: Set[str] = {"colombian_diaspora_stock"}


# ============================================================================
# Jerarquia de excepciones
# ============================================================================
class RadarCibestError(Exception):
    """Excepcion base de la familia RADAR Cibest."""


class DataExtractionError(RadarCibestError):
    """Error en la extraccion desde una fuente externa."""


class DataPreparationError(RadarCibestError):
    """Error en la preparacion, limpieza, normalizacion o features."""


class ConsistencyError(RadarCibestError):
    """Inconsistencia en los juicios de expertos BWM/AHP."""


class InsufficientDataError(RadarCibestError):
    """Datos insuficientes para ejecutar la operacion solicitada."""


class ConfigurationError(RadarCibestError):
    """Error en la configuracion: YAML invalido, claves faltantes o alcance invalido."""


class ScoringError(RadarCibestError):
    """Error durante el calculo del score: TOPSIS, gravity o hibrido."""


# ============================================================================
# Carga de configuraciones
# ============================================================================
def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Carga un archivo YAML con manejo de errores claro."""
    p = Path(path)
    if not p.exists():
        raise ConfigurationError(f"Archivo de configuracion no encontrado: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"YAML invalido en {p}: {exc}") from exc

    if data is None:
        raise ConfigurationError(f"Archivo YAML vacio: {p}")
    if not isinstance(data, dict):
        raise ConfigurationError(f"El YAML debe tener una raiz tipo diccionario: {p}")
    return data


def save_yaml(data: Dict[str, Any], path: Path | str) -> Path:
    """Persiste un diccionario como YAML legible."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )
    return p.resolve()


def load_all_configs(config_dir: Path | str | None = None) -> Dict[str, Dict[str, Any]]:
    """Carga las configuraciones YAML centrales del proyecto."""
    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR
    files = {
        "settings": "settings.yaml",
        "variables": "variables.yaml",
        "business_lines": "business_lines.yaml",
        "data_sources": "data_sources.yaml",
        "weights": "weights.yaml",
    }
    return {key: load_yaml(cfg_dir / filename) for key, filename in files.items()}


# ============================================================================
# Logger
# ============================================================================
def setup_logger(logging_config: Dict[str, Any] | None = None) -> None:
    """Configura loguru segun settings.yaml; si no existe loguru usa fallback basico."""
    cfg = logging_config or {}
    if not hasattr(logger, "remove") or not hasattr(logger, "add"):
        return

    logger.remove()
    level = cfg.get("level", "INFO")
    fmt = cfg.get(
        "format",
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    logger.add(sys.stderr, level=level, format=fmt, colorize=True)

    file_path = cfg.get("file_path")
    if file_path:
        log_file = PROJECT_ROOT / file_path if not Path(file_path).is_absolute() else Path(file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format=fmt,
            rotation=cfg.get("rotation", "10 MB"),
            retention=cfg.get("retention", "30 days"),
            encoding="utf-8",
        )


# ============================================================================
# Helpers de paises y rutas
# ============================================================================
def get_country_list(settings: Dict[str, Any]) -> List[str]:
    """Devuelve la lista de codigos ISO3 en alcance desde settings.yaml.

    Soporta lista directa de strings o lista de diccionarios con clave iso3.
    """
    countries = settings.get("countries", [])
    output: List[str] = []
    for item in countries:
        if isinstance(item, str):
            output.append(item.upper())
        elif isinstance(item, Mapping) and item.get("iso3"):
            output.append(str(item["iso3"]).upper())
    return sorted(dict.fromkeys(output))


def resolve_project_path(relative_path: str | Path, create_parent: bool = False) -> Path:
    """Resuelve una ruta relativa al proyecto."""
    p = Path(relative_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if create_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_data_path(relative_path: str | Path) -> Path:
    """Resuelve una ruta de datos y crea directorio o padre segun corresponda."""
    p = resolve_project_path(relative_path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================
# Helpers de fuentes y catalogos
# ============================================================================
def normalize_source_name(source: str | None) -> str:
    """Normaliza nombres de fuentes usados en variables.yaml/data_sources.yaml."""
    return "" if not source else str(source).strip().lower()


def infer_world_bank_db(
    indicator_code: str,
    source_cfg: Optional[Dict[str, Any]] = None,
) -> int:
    """Infiere la base wbgapi para un indicador World Bank/DataBank.

    Prioridad:
    1. source_cfg['indicator_databases'] con match exacto y normalizado.
    2. Reglas explícitas para indicadores especiales, por ejemplo G20/Findex.
    3. source_cfg['dimensions'][*]['db'] si el indicador está declarado allí.
    4. Prefijos conocidos: GFDD -> 32, WGI -> 3, resto -> default_db o 2.

    Nota:
    g20.any pertenece a la base 28 de G20 Financial Inclusion / Findex,
    por lo que no debe caer al default_db=2 aunque esté mezclado en una
    dimensión tecnológica junto con indicadores WDI.
    """

    if indicator_code is None:
        return 2

    code = str(indicator_code).strip()
    code_lower = code.lower()

    if source_cfg is None:
        source_cfg = {}

    default_db = int(source_cfg.get("default_db", 2))

    # ------------------------------------------------------------------
    # 1. Overrides explícitos por indicador en data_sources.yaml
    # ------------------------------------------------------------------
    indicator_databases = source_cfg.get("indicator_databases", {}) or {}

    # Match exacto
    if code in indicator_databases:
        return int(indicator_databases[code])

    # Match normalizado case-insensitive
    indicator_databases_normalized = {
        str(k).strip().lower(): v
        for k, v in indicator_databases.items()
    }

    if code_lower in indicator_databases_normalized:
        return int(indicator_databases_normalized[code_lower])

    # ------------------------------------------------------------------
    # 2. Reglas explícitas para indicadores especiales
    # ------------------------------------------------------------------
    # G20 Financial Inclusion / Global Findex.
    # el código g20.any debe ir a db=28.
    g20_findex_codes = {
        "g20.any",
    }

    if code_lower in g20_findex_codes:
        return 28

    # ------------------------------------------------------------------
    # 3. Buscar en bloques de dimensiones declarados en data_sources.yaml
    # ------------------------------------------------------------------
    for dim_cfg in (source_cfg.get("dimensions", {}) or {}).values():
        indicators = dim_cfg.get("indicators", {}) or {}

        # Match exacto
        if code in indicators:
            return int(dim_cfg.get("db", default_db))

        # Match normalizado
        indicators_normalized = {
            str(k).strip().lower()
            for k in indicators.keys()
        }

        if code_lower in indicators_normalized:
            return int(dim_cfg.get("db", default_db))

    # ------------------------------------------------------------------
    # 4. Reglas por prefijo
    # ------------------------------------------------------------------
    if code.startswith("GFDD."):
        return 32

    if code.startswith(tuple(WGI_PREFIXES)):
        return 3

    return default_db


def get_world_bank_source_cfg(data_sources_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna sources.world_bank validado desde data_sources.yaml."""
    sources = data_sources_cfg.get("sources", data_sources_cfg)
    world_bank_cfg = sources.get("world_bank")
    if not isinstance(world_bank_cfg, dict):
        raise ConfigurationError("data_sources.yaml debe incluir sources.world_bank")
    if not world_bank_cfg.get("enabled", True):
        raise ConfigurationError("La fuente world_bank esta deshabilitada")
    return world_bank_cfg


def _coerce_variable_meta(var_name: str, meta: Any, dimension: str) -> Dict[str, Any]:
    """Normaliza una entrada de variables.yaml a metadata dict."""
    normalized = dict(meta) if isinstance(meta, Mapping) else {"indicator_code": str(meta)}
    normalized.setdefault("name", var_name)
    normalized["dimension"] = dimension

    if "indicator_code" not in normalized:
        for alt_key in ["code", "wb_code", "series", "variable_code"]:
            if alt_key in normalized:
                normalized["indicator_code"] = normalized[alt_key]
                break

    source = normalize_source_name(normalized.get("source"))
    indicator_code = str(normalized.get("indicator_code", ""))

    if not source and indicator_code:
        source = "wgi" if indicator_code.startswith(tuple(WGI_PREFIXES)) else "world_bank"
        normalized["source"] = source
    elif source:
        normalized["source"] = source

    if indicator_code and source in WORLD_BANK_ALLOWED_SOURCES and "db" not in normalized:
        normalized["db"] = infer_world_bank_db(indicator_code)

    return normalized


def get_variable_catalog(variables_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Aplana variables.yaml a un catalogo {variable: metadata}.

    Soporta:
    - dimensions.<dimension>.variables.<variable_name>: metadata
    - dimensions.<dimension>.indicators.<indicator_code>: descripcion
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    for dim_key, dim_content in (variables_cfg.get("dimensions", {}) or {}).items():
        if not isinstance(dim_content, Mapping):
            continue

        for var_name, meta in (dim_content.get("variables", {}) or {}).items():
            catalog[var_name] = _coerce_variable_meta(var_name, meta, dim_key)

        for indicator_code, description in (dim_content.get("indicators", {}) or {}).items():
            var_name = str(indicator_code).lower().replace(".", "_")
            if var_name in catalog:
                continue
            source = "wgi" if str(indicator_code).startswith(tuple(WGI_PREFIXES)) else "world_bank"
            catalog[var_name] = {
                "name": var_name,
                "description": description,
                "indicator_code": str(indicator_code),
                "source": source,
                "db": int(dim_content.get("db", infer_world_bank_db(str(indicator_code)))),
                "dimension": dim_key,
            }
    return catalog


def filter_catalog_by_sources(
    catalog: Dict[str, Dict[str, Any]],
    allowed_sources: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Filtra un catalogo por fuentes permitidas."""
    allowed = {normalize_source_name(s) for s in allowed_sources}
    return {
        variable: meta
        for variable, meta in catalog.items()
        if normalize_source_name(meta.get("source")) in allowed
    }


def get_world_bank_variable_catalog(
    variables_cfg: Dict[str, Any],
    source_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Retorna solo variables World Bank/WGI con db resuelta para wbgapi.

    Prioridad:
    1. db explícito en variables.yaml.
    2. wb_db explícito, si existe.
    3. infer_world_bank_db(indicator_code, source_cfg).
    """

    catalog = filter_catalog_by_sources(
        get_variable_catalog(variables_cfg),
        WORLD_BANK_ALLOWED_SOURCES,
    )

    resolved: Dict[str, Dict[str, Any]] = {}

    for variable, meta in catalog.items():
        indicator_code = meta.get("indicator_code")

        if not indicator_code:
            logger.warning(
                "Variable {v} omitida: no tiene indicator_code",
                v=variable,
            )
            continue

        indicator_code = str(indicator_code).strip()

        db = int(
            meta.get("db")
            or meta.get("wb_db")
            or infer_world_bank_db(indicator_code, source_cfg)
        )

        if db not in WORLD_BANK_ALLOWED_DBS:
            raise ConfigurationError(
                f"Variable {variable} usa db={db}, fuera de bases permitidas "
                f"{sorted(WORLD_BANK_ALLOWED_DBS)}. "
                f"Revise WORLD_BANK_ALLOWED_DBS en utils.py. "
                f"Archivo utils cargado desde: {__file__}"
            )

        resolved[variable] = {
            **meta,
            "indicator_code": indicator_code,
            "db": db,
        }

    return resolved


def get_enabled_complementary_sources(data_sources_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna fuentes complementarias permitidas y habilitadas."""
    sources = data_sources_cfg.get("sources", data_sources_cfg)
    complementary = sources.get("complementary", {}) or {}
    if not complementary.get("enabled", True):
        return {"static_files": {}, "online_sources": {}}

    static_files = {
        key: value
        for key, value in (complementary.get("static_files", {}) or {}).items()
        if key in COMPLEMENTARY_ALLOWED_FILES
    }
    online_sources = {
        key: value
        for key, value in (complementary.get("online_sources", {}) or {}).items()
        if key in COMPLEMENTARY_ALLOWED_ONLINE and bool(value.get("enabled", True))
    }
    return {"static_files": static_files, "online_sources": online_sources}