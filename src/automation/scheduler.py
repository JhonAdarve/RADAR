"""
Programador de ejecuciones automaticas para RADAR Cibest.

Implementa scheduling simple con la libreria schedule, deliberadamente
ligera para mantener el footprint operacional bajo (Airflow seria
sobredimensionado para el alcance de 2 personas a tiempo parcial).

Soporta dos jobs:
    - monthly_full     : pipeline completo (extraccion + scoring + senales)
    - on_demand_rescore: re-ejecutar scoring con pesos nuevos sin re-extraer

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

import schedule
from loguru import logger

from src.utils import load_all_configs, setup_logger


def job_full_pipeline() -> None:
    """Ejecuta el pipeline completo: extraccion -> preparacion -> scoring -> senales."""
    logger.info("[JOB] Iniciando pipeline completo RADAR Cibest")
    start = datetime.now()
    try:
        from src.data_extraction.pipeline import run_extraction
        from src.scoring.hybrid_scorer import run_full_scoring

        configs = load_all_configs()
        master, _coverage = run_extraction(configs=configs, save_intermediate=True)
        if master.empty:
            logger.error("Master vacio, abortando scoring")
            return
        results = run_full_scoring(master, configs, persist=True)
        elapsed = (datetime.now() - start).total_seconds()
        logger.info(
            "[JOB] Pipeline completo finalizado en {s:.1f}s | {c} paises evaluados",
            s=elapsed, c=len(results["radar_global"]),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[JOB] Pipeline completo fallo: {e}", e=str(exc))


def job_rescore_only(configs: Optional[Dict[str, Any]] = None) -> None:
    """Re-ejecuta solo el scoring usando el ultimo master raw disponible."""
    logger.info("[JOB] Iniciando rescoring on-demand")
    start = datetime.now()
    try:
        import pandas as pd
        from src.scoring.hybrid_scorer import run_full_scoring
        from src.utils import resolve_data_path

        if configs is None:
            configs = load_all_configs()
        raw_dir = resolve_data_path(configs["settings"]["data"]["raw_path"])
        candidates = sorted(raw_dir.glob("master_raw_*.parquet"), reverse=True)
        if not candidates:
            logger.error("No hay master_raw disponible para rescoring")
            return
        master = pd.read_parquet(candidates[0])
        results = run_full_scoring(master, configs, persist=True)
        elapsed = (datetime.now() - start).total_seconds()
        logger.info(
            "[JOB] Rescoring finalizado en {s:.1f}s usando {f}",
            s=elapsed, f=candidates[0].name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[JOB] Rescoring fallo: {e}", e=str(exc))


def start_scheduler(monthly_day: int = 5, monthly_time: str = "06:00") -> None:
    """Inicia el scheduler en modo bloqueante.

    Args:
        monthly_day: Dia del mes (1-28) para ejecutar pipeline completo.
        monthly_time: Hora HH:MM del job mensual.

    Note:
        Ejecutar como proceso de larga duracion (systemd, supervisor, etc.).
        Para entornos productivos considerar un orquestador formal.
    """
    setup_logger(load_all_configs()["settings"].get("logging"))
    logger.info("Scheduler RADAR Cibest iniciado")

    schedule.every().day.at(monthly_time).do(_run_if_target_day, target_day=monthly_day)

    while True:
        schedule.run_pending()
        time.sleep(60)


def _run_if_target_day(target_day: int) -> None:
    """Wrapper que solo ejecuta el job mensual si hoy coincide con target_day."""
    if datetime.now().day == target_day:
        job_full_pipeline()


if __name__ == "__main__":
    start_scheduler()
