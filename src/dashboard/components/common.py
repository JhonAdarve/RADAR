"""
Componentes reutilizables del dashboard RADAR Cibest.

Define la paleta corporativa, funciones de carga en cache y helpers
que se comparten entre las paginas del dashboard.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.utils import load_all_configs, resolve_data_path


CIBEST_PALETTE: Dict[str, str] = {
    "gray": "#2C2A28",
    "gray_light": "#CCCAC7",
    "gold": "#E8B931",
    "gold_dark": "#B38A1E",
    "dark": "#1A1A1A",
    "yellow": "#FDD923",
    "gray_bg": "#F5F5F5",
    "gray_border": "#D0D0D0",
    "white": "#FFFFFF",
    "green": "#2E7D32",
    "amber": "#EF6C00",
    "red": "#C62828",
}

SIGNAL_COLORS: Dict[str, str] = {
    "ALTA_OPORTUNIDAD": CIBEST_PALETTE["green"],
    "OPORTUNIDAD_MODERADA": CIBEST_PALETTE["gold"],
    "BAJA_OPORTUNIDAD": CIBEST_PALETTE["amber"],
    "RIESGO": CIBEST_PALETTE["red"],
}


@st.cache_data(show_spinner=False)
def load_configurations() -> Dict[str, Dict[str, Any]]:
    """Carga las cinco configuraciones YAML con cache de Streamlit."""
    return load_all_configs()


@st.cache_data(show_spinner=True)
def load_latest_parquet(pattern_prefix: str, scores_dir: str = "data/scores/") -> Optional[pd.DataFrame]:
    """Carga el Parquet mas reciente que coincida con un prefijo."""
    path = resolve_data_path(scores_dir)
    candidates = sorted(path.glob(f"{pattern_prefix}*.parquet"), reverse=True)
    if not candidates:
        return None
    return pd.read_parquet(candidates[0])


def iso3_to_name_map(configs: Dict[str, Any]) -> Dict[str, str]:
    """Diccionario ISO3 -> nombre de pais desde settings.yaml."""
    return {c["iso3"]: c["name"] for c in configs["settings"]["countries"]}


def signal_to_emoji(signal: str) -> str:
    """Mapea etiqueta de senal a emoji visual."""
    return {
        "ALTA_OPORTUNIDAD": "🟢",
        "OPORTUNIDAD_MODERADA": "🟡",
        "BAJA_OPORTUNIDAD": "🟠",
        "RIESGO": "🔴",
    }.get(signal, "⚪")


def apply_custom_css() -> None:
    """Inyecta CSS corporativo al dashboard."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {CIBEST_PALETTE['white']};
        }}
        h1, h2, h3 {{
            color: {CIBEST_PALETTE['gray']};
            font-family: Arial, sans-serif;
        }}
        .metric-card {{
            background: {CIBEST_PALETTE['gray_bg']};
            border-left: 4px solid {CIBEST_PALETTE['gold']};
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 8px;
        }}
        .signal-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            color: white;
            font-size: 0.85em;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
