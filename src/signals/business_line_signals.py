"""
Motor de senales por linea de negocio para RADAR Cibest.

Componente diferenciador frente a literatura existente: ningun modelo
publicado genera senales diferenciadas por tipo de negocio financiero
(Gap 2 de la revision sistematica).

Asigna a cada par (pais, linea) una de cuatro etiquetas:
    ALTA_OPORTUNIDAD, OPORTUNIDAD_MODERADA, BAJA_OPORTUNIDAD, RIESGO

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


def _classify_signal(
    percentile: float,
    risk_override_active: bool,
    thresholds: Dict[str, float],
) -> str:
    """Clasifica un score parcial en una de las cuatro categorias."""
    if risk_override_active:
        return "RIESGO"
    if percentile >= thresholds["ALTA_OPORTUNIDAD"]:
        return "ALTA_OPORTUNIDAD"
    if percentile >= thresholds["OPORTUNIDAD_MODERADA"]:
        return "OPORTUNIDAD_MODERADA"
    if percentile >= thresholds["BAJA_OPORTUNIDAD"]:
        return "BAJA_OPORTUNIDAD"
    return "RIESGO"


def _risk_override_by_profile(
    country_iso3: str,
    wide_raw: pd.DataFrame,
    risk_cfg: Dict[str, float],
) -> bool:
    """Determina si un pais debe forzar senal RIESGO por indicadores criticos."""
    if country_iso3 not in wide_raw.index:
        return False

    if "political_stability" in wide_raw.columns:
        pct = wide_raw["political_stability"].rank(pct=True)
        if pct.loc[country_iso3] <= risk_cfg.get("political_stability_percentile", 0.15):
            return True

    if "control_of_corruption" in wide_raw.columns:
        pct = wide_raw["control_of_corruption"].rank(pct=True)
        if pct.loc[country_iso3] <= risk_cfg.get("corruption_percentile", 0.15):
            return True

    return False


def evaluate_signal_for_line(
    business_line_scores: pd.Series,
    business_line_key: str,
    wide_raw: pd.DataFrame,
    business_lines_cfg: Dict[str, Any],
) -> pd.Series:
    """Evalua la senal para una linea de negocio en todos los paises."""
    thresholds = business_lines_cfg["signal_thresholds"]
    risk_cfg = thresholds.get("risk_override", {})

    percentiles = business_line_scores.rank(pct=True)
    signals: Dict[str, str] = {}
    for country in business_line_scores.index:
        override = _risk_override_by_profile(country, wide_raw, risk_cfg)
        signals[country] = _classify_signal(
            percentile=float(percentiles.loc[country]),
            risk_override_active=override,
            thresholds=thresholds,
        )

    return pd.Series(signals, name=f"signal_{business_line_key}")


def generate_signal_matrix(
    radar_by_line: pd.DataFrame,
    wide_raw: pd.DataFrame,
    business_lines_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Genera la matriz completa pais x linea con senales etiquetadas."""
    bl_keys = list(business_lines_cfg["business_lines"].keys())
    matrix = pd.DataFrame(index=radar_by_line.index)
    for bl in bl_keys:
        if bl not in radar_by_line.columns:
            logger.warning("Linea {b} ausente en radar_by_line", b=bl)
            continue
        matrix[bl] = evaluate_signal_for_line(
            business_line_scores=radar_by_line[bl],
            business_line_key=bl,
            wide_raw=wide_raw,
            business_lines_cfg=business_lines_cfg,
        )

    logger.info(
        "Matriz de senales generada: {c} paises x {l} lineas",
        c=len(matrix), l=matrix.shape[1],
    )
    return matrix


_SIGNAL_NARRATIVE_MAP = {
    "ALTA_OPORTUNIDAD": "presenta alta oportunidad",
    "OPORTUNIDAD_MODERADA": "muestra oportunidad moderada",
    "BAJA_OPORTUNIDAD": "ofrece baja oportunidad",
    "RIESGO": "concentra riesgos significativos",
}

_BUSINESS_LINE_NARRATIVE_LABELS = {
    "IB": "intermediacion bancaria",
    "PF": "Pagos y Flujos",
    "AD": "activos digitales",
    "BD": "banca digital",
    "CIB": "corporate & investment banking",
}


def generate_country_narrative(
    country_iso3: str,
    country_profile: Dict[str, Any],
    signals: pd.Series,
    country_name: Optional[str] = None,
) -> str:
    """Genera narrativa breve sobre el perfil y senales de un pais."""
    name = country_name or country_iso3

    strengths = country_profile["summary"]["strengths"]
    weaknesses = country_profile["summary"]["weaknesses"]

    intro = (
        f"{name} ocupa la posicion #{country_profile['rank_overall']} del ranking RADAR "
        f"con un score de {country_profile['score_overall']:.2f}."
    )

    profile_sentences: List[str] = []
    if strengths:
        profile_sentences.append(
            "Sus fortalezas relativas se concentran en " + ", ".join(strengths) + "."
        )
    if weaknesses:
        profile_sentences.append(
            "Sus brechas principales estan en " + ", ".join(weaknesses) + "."
        )

    signals_text = ""
    if country_iso3 in signals.index or len(signals.index) > 0:
        signal_sentences: List[str] = []
        for bl, lbl in _BUSINESS_LINE_NARRATIVE_LABELS.items():
            if bl in signals.index and isinstance(signals.loc[bl], str):
                sig = signals.loc[bl]
                narrative = _SIGNAL_NARRATIVE_MAP.get(sig, sig.lower())
                signal_sentences.append(f"el mercado {narrative} para {lbl}")
        if signal_sentences:
            signals_text = "Por linea: " + "; ".join(signal_sentences) + "."

    return " ".join([intro] + profile_sentences + ([signals_text] if signals_text else [])).strip()


def generate_all_narratives(
    profiles: Dict[str, Dict[str, Any]],
    signal_matrix: pd.DataFrame,
    country_names: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Genera narrativas para todos los paises."""
    narratives: Dict[str, str] = {}
    country_names = country_names or {}
    for country, profile in profiles.items():
        if country not in signal_matrix.index:
            continue
        narratives[country] = generate_country_narrative(
            country_iso3=country,
            country_profile=profile,
            signals=signal_matrix.loc[country],
            country_name=country_names.get(country, country),
        )

    logger.info("Narrativas generadas: {n}", n=len(narratives))
    return narratives


def consolidate_signals_output(
    profiles: Dict[str, Dict[str, Any]],
    signal_matrix: pd.DataFrame,
    narratives: Dict[str, str],
    radar_by_line: pd.DataFrame,
    country_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Consolida senales, scores y narrativas en una tabla final exportable."""
    country_names = country_names or {}
    rows: List[Dict[str, Any]] = []

    for country, profile in profiles.items():
        if country not in signal_matrix.index:
            continue
        row: Dict[str, Any] = {
            "country_iso3": country,
            "country_name": country_names.get(country, country),
            "rank_global": profile["rank_overall"],
            "score_global": profile["score_overall"],
            "strengths": "; ".join(profile["summary"]["strengths"]),
            "weaknesses": "; ".join(profile["summary"]["weaknesses"]),
        }
        for bl in signal_matrix.columns:
            row[f"signal_{bl}"] = signal_matrix.loc[country, bl]
            if bl in radar_by_line.columns:
                row[f"score_{bl}"] = round(float(radar_by_line.loc[country, bl]), 4)
        row["narrative"] = narratives.get(country, "")
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("rank_global").reset_index(drop=True)
    return df
