"""
Versionado de resultados del scoring para RADAR Cibest.

Cada ejecucion del scoring genera archivos Parquet versionados con la
fecha. Este modulo mantiene un indice YAML en data/scores/index.yaml con:

    - Fecha y hora de cada ejecucion
    - Hash SHA1 de los datos de entrada (master_raw)
    - Hash de la configuracion de pesos usada
    - Lista de archivos generados
    - Metricas resumen (n paises, score top, etc.)

Permite reproducibilidad total: cualquier resultado pasado puede
regenerarse con los mismos inputs y configuracion.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

from src.utils import resolve_data_path


def _file_hash(path: Path) -> str:
    """Calcula hash SHA1 del contenido binario de un archivo."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _dict_hash(d: Dict[str, Any]) -> str:
    """Hash determinista de un diccionario serializable."""
    serialized = yaml.safe_dump(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()[:16]


def register_scoring_run(
    settings: Dict[str, Any],
    weights: Dict[str, Any],
    master_raw_file: Path,
    output_files: List[Path],
    summary_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Registra una ejecucion de scoring en el indice versionado.

    Args:
        settings: Configuracion settings.yaml.
        weights: Configuracion weights.yaml.
        master_raw_file: Archivo de datos crudos usado como input.
        output_files: Lista de archivos generados por el scoring.
        summary_metrics: Diccionario opcional con metricas resumen.

    Returns:
        Diccionario con la entrada registrada.
    """
    scores_dir = resolve_data_path(settings["data"]["scores_path"])
    index_path = scores_dir / "index.yaml"

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_hash": _file_hash(master_raw_file) if master_raw_file.exists() else None,
        "weights_hash": _dict_hash(weights),
        "input_file": master_raw_file.name,
        "output_files": [f.name for f in output_files],
        "summary_metrics": summary_metrics or {},
    }

    index: List[Dict[str, Any]] = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = yaml.safe_load(f) or []

    index.insert(0, entry)
    with index_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(index, f, sort_keys=False, allow_unicode=True)

    logger.info("Ejecucion registrada en {p}", p=index_path)
    return entry


def list_scoring_runs(scores_path: str = "data/scores/") -> pd.DataFrame:
    """Lista las ejecuciones registradas en el indice.

    Args:
        scores_path: Directorio del indice.

    Returns:
        DataFrame con columnas timestamp, input_hash, weights_hash y metricas.
    """
    scores_dir = resolve_data_path(scores_path)
    index_path = scores_dir / "index.yaml"

    if not index_path.exists():
        return pd.DataFrame()

    with index_path.open("r", encoding="utf-8") as f:
        index = yaml.safe_load(f) or []

    rows: List[Dict[str, Any]] = []
    for entry in index:
        row = {
            "timestamp": entry.get("timestamp"),
            "input_hash": entry.get("input_hash"),
            "weights_hash": entry.get("weights_hash"),
            "input_file": entry.get("input_file"),
            "n_outputs": len(entry.get("output_files", [])),
        }
        row.update(entry.get("summary_metrics", {}))
        rows.append(row)

    return pd.DataFrame(rows)


def compare_two_runs(
    run_a_date: str,
    run_b_date: str,
    scores_path: str = "data/scores/",
    file_prefix: str = "radar_by_line_",
) -> pd.DataFrame:
    """Compara dos ejecuciones de scoring por su fecha (YYYYMMDD).

    Args:
        run_a_date: Fecha de la primera ejecucion (YYYYMMDD).
        run_b_date: Fecha de la segunda ejecucion.
        scores_path: Directorio de scores.
        file_prefix: Prefijo del archivo a comparar.

    Returns:
        DataFrame con country_iso3, score_A, score_B, delta_score.
    """
    scores_dir = resolve_data_path(scores_path)
    file_a = scores_dir / f"{file_prefix}{run_a_date}.parquet"
    file_b = scores_dir / f"{file_prefix}{run_b_date}.parquet"

    if not file_a.exists() or not file_b.exists():
        raise FileNotFoundError(f"No se encontraron ambos archivos: {file_a}, {file_b}")

    df_a = pd.read_parquet(file_a)
    df_b = pd.read_parquet(file_b)

    score_col = "GLOBAL" if "GLOBAL" in df_a.columns else df_a.columns[0]
    merged = pd.DataFrame({
        f"score_{run_a_date}": df_a[score_col],
        f"score_{run_b_date}": df_b[score_col],
    })
    merged["delta_score"] = merged[f"score_{run_b_date}"] - merged[f"score_{run_a_date}"]
    merged = merged.sort_values("delta_score", key=abs, ascending=False)
    return merged
