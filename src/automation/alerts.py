"""
Sistema de alertas para RADAR Cibest.

Detecta y reporta cambios significativos entre ejecuciones consecutivas
del pipeline. Dos tipos de alertas:

    - Alertas de variable: variacion >10% en una variable critica
    - Alertas de ranking: salto >3 posiciones en el score RADAR

Las alertas se loguean y opcionalmente se envian por correo via SMTP.
La configuracion SMTP se externaliza en settings.yaml.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


def detect_variable_changes(
    df_current: pd.DataFrame,
    df_previous: pd.DataFrame,
    threshold_pct: float = 10.0,
) -> pd.DataFrame:
    """Detecta variables con variacion porcentual superior al umbral.

    Args:
        df_current: DataFrame largo de la ejecucion actual.
        df_previous: DataFrame largo de la ejecucion anterior.
        threshold_pct: Umbral de variacion porcentual absoluta.

    Returns:
        DataFrame con country_iso3, variable, valor anterior, valor actual,
        variacion porcentual.
    """
    common_cols = ["country_iso3", "year", "variable", "value"]
    cur = df_current[common_cols].rename(columns={"value": "value_current"})
    prev = df_previous[common_cols].rename(columns={"value": "value_previous"})

    merged = cur.merge(prev, on=["country_iso3", "year", "variable"], how="inner")
    merged = merged.dropna(subset=["value_current", "value_previous"])
    merged = merged[merged["value_previous"] != 0]

    merged["variation_pct"] = (
        (merged["value_current"] - merged["value_previous"]) / merged["value_previous"].abs() * 100
    )
    alerts = merged[merged["variation_pct"].abs() >= threshold_pct].copy()
    alerts = alerts.sort_values("variation_pct", key=abs, ascending=False)

    logger.info("Alertas de variable: {n} con variacion >= {t}%", n=len(alerts), t=threshold_pct)
    return alerts


def detect_ranking_changes(
    radar_current: pd.Series,
    radar_previous: pd.Series,
    rank_threshold: int = 3,
) -> pd.DataFrame:
    """Detecta paises con cambios significativos en el ranking.

    Args:
        radar_current: Serie de scores actuales.
        radar_previous: Serie de scores anteriores.
        rank_threshold: Numero minimo de posiciones para alerta.

    Returns:
        DataFrame con country_iso3, rank actual, rank anterior, delta_rank.
    """
    rank_cur = radar_current.rank(ascending=False, method="min").astype(int)
    rank_prev = radar_previous.rank(ascending=False, method="min").astype(int)
    df = pd.DataFrame({
        "country_iso3": rank_cur.index,
        "rank_current": rank_cur.values,
        "rank_previous": rank_prev.reindex(rank_cur.index).values,
    })
    df = df.dropna(subset=["rank_previous"])
    df["delta_rank"] = (df["rank_previous"] - df["rank_current"]).astype(int)
    alerts = df[df["delta_rank"].abs() >= rank_threshold].copy()
    alerts = alerts.sort_values("delta_rank", key=abs, ascending=False)
    logger.info(
        "Alertas de ranking: {n} con cambio >= {t} posiciones",
        n=len(alerts), t=rank_threshold,
    )
    return alerts


def format_alerts_email(
    var_alerts: pd.DataFrame,
    rank_alerts: pd.DataFrame,
    timestamp: Optional[datetime] = None,
) -> str:
    """Formatea las alertas como cuerpo HTML de correo.

    Args:
        var_alerts: DataFrame de alertas de variable.
        rank_alerts: DataFrame de alertas de ranking.
        timestamp: Fecha de la ejecucion (default ahora).

    Returns:
        String HTML.
    """
    ts = timestamp or datetime.now()
    body = f"""
    <html>
      <head><style>
        body {{ font-family: Arial, sans-serif; color: #1A1A1A; }}
        h2 {{ color: #0D1B2A; border-bottom: 2px solid #FDD923; padding-bottom: 4px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #DDD; }}
        th {{ background: #0D1B2A; color: #FDD923; }}
      </style></head>
      <body>
        <h1>RADAR Cibest - Reporte de Alertas</h1>
        <p>Generado: {ts.strftime("%Y-%m-%d %H:%M")}</p>

        <h2>Alertas de variables ({len(var_alerts)})</h2>
        {var_alerts.head(20).to_html(index=False) if not var_alerts.empty else "<p>Sin alertas</p>"}

        <h2>Alertas de ranking ({len(rank_alerts)})</h2>
        {rank_alerts.head(20).to_html(index=False) if not rank_alerts.empty else "<p>Sin alertas</p>"}

        <hr>
        <p style="font-size: 0.8em; color: #888;">
          Sistema RADAR Cibest | Direccion de Estrategia | Grupo Cibest
        </p>
      </body>
    </html>
    """
    return body


def send_email_alert(
    html_body: str,
    smtp_config: Dict[str, Any],
    subject: str = "RADAR Cibest - Alertas",
) -> bool:
    """Envia un correo SMTP con el reporte de alertas.

    Args:
        html_body: Cuerpo HTML.
        smtp_config: Diccionario con host, port, user, password, from, to.
        subject: Asunto del correo.

    Returns:
        True si se envio exitosamente.
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_config["from"]
        msg["To"] = ", ".join(smtp_config["to"]) if isinstance(smtp_config["to"], list) else smtp_config["to"]
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
            if smtp_config.get("use_tls", True):
                server.starttls()
            if smtp_config.get("user") and smtp_config.get("password"):
                server.login(smtp_config["user"], smtp_config["password"])
            server.send_message(msg)
        logger.info("Correo de alertas enviado a {t}", t=msg["To"])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo envio SMTP: {e}", e=str(exc))
        return False
