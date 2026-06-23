"""
Componentes graficos reutilizables del dashboard RADAR Cibest.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.components.common import CIBEST_PALETTE, SIGNAL_COLORS


def choropleth_americas(
    df: pd.DataFrame,
    value_col: str,
    location_col: str = "country_iso3",
    title: str = "",
    color_scale: Optional[List[str]] = None,
) -> go.Figure:
    """Mapa coropletico de America + Espana."""
    if color_scale is None:
        color_scale = [
            CIBEST_PALETTE["gray"],
            CIBEST_PALETTE["gray_light"],
            CIBEST_PALETTE["gold"],
        ]

    fig = px.choropleth(
        df,
        locations=location_col,
        color=value_col,
        hover_name=location_col,
        color_continuous_scale=color_scale,
        projection="natural earth",
        scope="world",
        title=title,
    )
    fig.update_geos(
        showcountries=True, countrycolor=CIBEST_PALETTE["gray_border"],
        showcoastlines=False, showframe=False,
        lonaxis_range=[-130, 10],
        lataxis_range=[-60, 75],
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar={"title": value_col, "thickness": 15},
        height=500,
    )
    return fig


def choropleth_signals(
    df: pd.DataFrame,
    signal_col: str,
    location_col: str = "country_iso3",
    title: str = "",
) -> go.Figure:
    """Mapa coropletico de senales categoricas (cuatro niveles)."""
    fig = px.choropleth(
        df,
        locations=location_col,
        color=signal_col,
        hover_name=location_col,
        color_discrete_map=SIGNAL_COLORS,
        projection="natural earth",
        category_orders={
            signal_col: ["ALTA_OPORTUNIDAD", "OPORTUNIDAD_MODERADA", "BAJA_OPORTUNIDAD", "RIESGO"],
        },
        title=title,
    )
    fig.update_geos(
        showcountries=True, countrycolor=CIBEST_PALETTE["gray_border"],
        showcoastlines=False, showframe=False,
        lonaxis_range=[-130, 10],
        lataxis_range=[-60, 75],
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=500,
    )
    return fig


def radar_chart(
    dimension_scores_selected: dict,
    dimension_scores_benchmark: Optional[dict] = None,
    country_label: str = "Pais seleccionado",
    benchmark_label: str = "Mediana regional",
) -> go.Figure:
    """Radar chart del pais vs benchmark."""
    dims = list(dimension_scores_selected.keys())
    values = [dimension_scores_selected[d] for d in dims]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=dims + [dims[0]],
        fill="toself",
        name=country_label,
        line_color=CIBEST_PALETTE["gray"],
        fillcolor="rgba(13, 27, 42, 0.35)",
    ))

    if dimension_scores_benchmark:
        bench_values = [dimension_scores_benchmark[d] for d in dims]
        fig.add_trace(go.Scatterpolar(
            r=bench_values + [bench_values[0]],
            theta=dims + [dims[0]],
            fill="toself",
            name=benchmark_label,
            line_color=CIBEST_PALETTE["gold"],
            fillcolor="rgba(232, 185, 49, 0.25)",
        ))

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
        height=450,
        margin={"r": 30, "t": 40, "l": 30, "b": 30},
    )
    return fig


def score_gauge(score: float, label: str, max_score: float = 1.0) -> go.Figure:
    """Gauge de un score [0, 1]."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label, "font": {"size": 14}},
        number={"valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, max_score]},
            "bar": {"color": CIBEST_PALETTE["gray"]},
            "steps": [
                {"range": [0, 0.33], "color": "#F4D4D4"},
                {"range": [0.33, 0.66], "color": "#FCEDC6"},
                {"range": [0.66, 1.0], "color": "#D8ECD5"},
            ],
            "threshold": {
                "line": {"color": CIBEST_PALETTE["gold_dark"], "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=200, margin={"r": 20, "t": 40, "l": 20, "b": 10})
    return fig


def ranking_bar_chart(
    df: pd.DataFrame,
    score_col: str,
    country_col: str = "country_iso3",
    top_n: int = 15,
    title: str = "",
) -> go.Figure:
    """Grafico de barras horizontal del top-N."""
    df_top = df.nlargest(top_n, score_col).sort_values(score_col)
    fig = go.Figure(go.Bar(
        x=df_top[score_col],
        y=df_top[country_col],
        orientation="h",
        marker_color=CIBEST_PALETTE["gray"],
        text=df_top[score_col].round(3),
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=score_col,
        yaxis_title="Pais",
        height=max(400, 25 * top_n),
        margin={"r": 30, "t": 40, "l": 80, "b": 40},
    )
    return fig
