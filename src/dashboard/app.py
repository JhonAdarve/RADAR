"""
Dashboard principal RADAR Cibest.

Aplicacion Streamlit multipagina con cinco paginas:
    1. Ranking general
    2. Perfil de pais
    3. Senales por linea de negocio
    4. Simulador que-pasa-si
    5. Tendencias historicas

Ejecucion: streamlit run src/dashboard/app.py

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations


import sys
from pathlib import Path

# Agrega la raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))


import streamlit as st

from src.dashboard.components.common import CIBEST_PALETTE, apply_custom_css


def _header() -> None:
    """Encabezado corporativo."""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(
            f"""
            <div style='background: {CIBEST_PALETTE["gray"]};
                        color: {CIBEST_PALETTE["gold"]};
                        padding: 16px; border-radius: 6px;
                        text-align: center; font-weight: bold;
                        font-size: 1.4em;'>
                RADAR
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <h2 style='color: {CIBEST_PALETTE["gray"]}; margin-bottom: 0;'>
                Sistema Analitico RADAR Cibest
            </h2>
            <p style='color: {CIBEST_PALETTE["gray_light"]}; margin-top: 4px;'>
                Ranking de Atractivo y Diagnostico Analitico Regional &nbsp;|&nbsp;
                Direccion de Estrategia &nbsp;|&nbsp; Grupo Cibest
            </p>
            """,
            unsafe_allow_html=True,
        )
    st.divider()


def main() -> None:
    """Punto de entrada Streamlit."""
    st.set_page_config(
        page_title="RADAR Cibest",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()
    _header()

    pages = {
        "Analisis": [
            st.Page("pages/ranking.py",
                    title="Ranking General", icon="🌎", default=True),
            st.Page("pages/country_profile.py",
                    title="Perfil de Pais", icon="🔍"),
            st.Page("pages/signals.py",
                    title="Senales por Linea", icon="📊"),
        ],
        "Exploracion": [
            st.Page("pages/simulator.py",
                    title="Simulador de Pesos", icon="⚖️"),
            st.Page("pages/trends.py",
                    title="Tendencias Historicas", icon="📈"),
        ],
    }
    selected = st.navigation(pages)
    selected.run()


if __name__ == "__main__":
    main()
