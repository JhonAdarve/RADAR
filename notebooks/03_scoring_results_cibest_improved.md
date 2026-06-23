# 03 — Resultados ejecutivos de scoring | RADAR Cibest

**Fase ASUM-DM:** 4 — Modelado  
**Responsable:** Jhon Farley Adarve Díaz  
**Fecha de ejecución:** Junio 2026  
**Propósito:** ejecutar, auditar e interpretar el scoring híbrido RADAR Cibest, integrando TOPSIS, Índice de Proximidad con Colombia, Trend macroeconómico, rankings por línea, bandas de atractividad, gaps, drivers y restricciones.

---

## 0. El score RADAR debe interpretarse como una tesis de atractividad, no como un ranking aislado

Este notebook responde la pregunta central del modelo:

> **¿Qué países presentan mayor atractividad relativa para la internacionalización de Grupo Cibest, considerando fundamentos estructurales, proximidad con Colombia y momentum macroeconómico reciente?**

El scoring RADAR combina tres componentes:

```text
RADAR = alpha * TOPSIS + beta * IPC + gamma * Trend
```

Donde:

- **TOPSIS** mide atractivo estructural país-variable.
- **IPC** mide proximidad con Colombia.
- **Trend** captura momentum macro reciente a partir de `gdp_growth`.

### Mensaje ejecutivo preliminar

El output principal no debe leerse como una lista mecánica de posiciones. Debe interpretarse por:

1. score y ranking RADAR;
2. ranking TOPSIS estructural;
3. contribución de IPC y Trend;
4. bandas de atractividad;
5. gaps y empates prácticos;
6. drivers y restricciones por país/línea;
7. diferencias de tesis entre líneas de negocio.

---

## 1. Configuración del entorno y estilo Cibest

```python
# ---------------------------------------------------------------------------
# Configuración inicial del notebook
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
import importlib
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, display, Markdown

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent))

# ---------------------------------------------------------------------------
# Recarga controlada de módulos del proyecto
# ---------------------------------------------------------------------------
import src
import src.utils as utils
import src.data_preparation.feature_engineering as feature_engineering
import src.scoring.hybrid_scorer as hybrid_scorer
import src.scoring.ranking as ranking
import src.scoring.explainability as explain_scorer

importlib.invalidate_caches()
importlib.reload(utils)
importlib.reload(feature_engineering)
importlib.reload(ranking)
importlib.reload(hybrid_scorer)
importlib.reload(explain_scorer)

from src.utils import (
    load_all_configs,
    resolve_data_path,
    setup_logger,
    get_variable_catalog,
    get_world_bank_variable_catalog,
)

from src.scoring.hybrid_scorer import (
    run_full_scoring,
    prepare_decision_matrix,
    _build_business_line_weights,
)

from src.scoring.explainability import (
    compute_all_business_line_contributions,
    build_explainability_table_for_line,
    get_top_contributors,
    get_top_shortfalls,
    summarize_contributions_by_dimension,
    generate_country_line_explanation,
    compare_country_across_lines,
    compare_countries_in_line,
    compute_all_marginal_effects,
    combine_contribution_and_marginal,
    classify_driver_robustness,
    build_country_driver_table,
)

configs = load_all_configs()
setup_logger(configs["settings"].get("logging"))
variable_catalog = get_variable_catalog(configs["variables"])

# ---------------------------------------------------------------------------
# Identidad visual Cibest
# ---------------------------------------------------------------------------
CIBEST = {
    "gray": "#2C2A28",
    "gray_light": "#CCCAC7",
    "yellow": "#FDD923",
    "gold": "#D6B302",
    "gold_light": "#FFF7D3",
    "gold_dark": "#8F7701",
    "gray_bg": "#F5F5F5",
    "gray_border": "#D0D0D0",
    "white": "#FFFFFF",
    "green": "#0BA682",
    "amber": "#FF7E41",
    "red": "#C62828",
}

TIER_COLORS = {
    "Alta": CIBEST["green"],
    "Media-alta": CIBEST["gold"],
    "Media": CIBEST["amber"],
    "Baja": CIBEST["red"],
}

px.defaults.template = dict(
    layout=dict(
        font=dict(family="Arial, sans-serif", size=13, color=CIBEST["gray"]),
        title=dict(font=dict(size=17, color=CIBEST["gray"])),
        plot_bgcolor=CIBEST["white"],
        paper_bgcolor=CIBEST["white"],
        xaxis=dict(gridcolor=CIBEST["gray_border"], linecolor=CIBEST["gray"]),
        yaxis=dict(gridcolor=CIBEST["gray_border"], linecolor=CIBEST["gray"]),
        colorway=[CIBEST["gray"], CIBEST["gold"], CIBEST["green"], CIBEST["amber"]],
    )
)


def style_table(df, gradient_cols=None, gradient_cmap="YlGnBu", format_dict=None):
    """Aplica estilo Cibest a tablas pandas."""
    styler = df.style.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", CIBEST["gray"]),
            ("color", CIBEST["yellow"]),
            ("font-weight", "bold"),
            ("text-align", "center"),
            ("padding", "8px"),
            ("font-family", "Arial, sans-serif"),
        ]},
        {"selector": "td", "props": [
            ("padding", "6px 10px"),
            ("font-family", "Arial, sans-serif"),
            ("border-bottom", f"1px solid {CIBEST['gray_border']}"),
        ]},
    ])
    if gradient_cols:
        styler = styler.background_gradient(subset=gradient_cols, cmap=gradient_cmap)
    if format_dict:
        styler = styler.format(format_dict)
    return styler


def insight_box(title: str, text: str):
    display(HTML(f"""
    <div style="border-left:6px solid {CIBEST['gold']}; background-color:{CIBEST['gold_light']};
                padding:14px 18px; margin:12px 0; font-family:Arial, sans-serif; color:{CIBEST['gray']};">
        <b>{title}</b><br>{text}
    </div>
    """))


def risk_box(title: str, text: str):
    display(HTML(f"""
    <div style="border-left:6px solid {CIBEST['red']}; background-color:#FDECEC;
                padding:14px 18px; margin:12px 0; font-family:Arial, sans-serif; color:{CIBEST['gray']};">
        <b>{title}</b><br>{text}
    </div>
    """))


def success_box(title: str, text: str):
    display(HTML(f"""
    <div style="border-left:6px solid {CIBEST['green']}; background-color:#E8F7F3;
                padding:14px 18px; margin:12px 0; font-family:Arial, sans-serif; color:{CIBEST['gray']};">
        <b>{title}</b><br>{text}
    </div>
    """))

success_box("Entorno listo", "Módulos recargados, configuración activa y estilo Cibest disponible para visualización ejecutiva.")
```

---

## 2. Se carga el master vigente y se valida que sea apto para scoring

```python
raw_dir = resolve_data_path(configs["settings"]["data"]["raw_path"])
pattern = re.compile(r"^master_raw_\d{8}\.parquet$")

master_files = sorted(
    [path for path in raw_dir.glob("master_raw_*.parquet") if pattern.match(path.name)],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not master_files:
    raise FileNotFoundError("Falta master_raw_YYYYMMDD.parquet. Ejecute primero notebook 01.")

master_path = master_files[0]
master = pd.read_parquet(master_path)

required_cols = {"country_iso3", "year", "variable", "value", "source"}
missing_cols = required_cols - set(master.columns)

if missing_cols:
    raise ValueError(f"Master inválido. Faltan columnas: {missing_cols}")

master["country_iso3"] = master["country_iso3"].astype(str).str.strip()
master["variable"] = master["variable"].astype(str).str.strip()

if "gdp_growth" not in master["variable"].unique():
    raise ValueError("El master no contiene gdp_growth. Trend no podrá calcularse correctamente.")

min_expected_variables = int(configs["settings"].get("data_quality", {}).get("min_expected_variables", 35))
if master["variable"].nunique() < min_expected_variables:
    raise ValueError(
        f"Master parcial: solo {master['variable'].nunique()} variables. "
        f"Mínimo esperado: {min_expected_variables}."
    )

master = master[master["variable"] != "wgi_composite"].copy()

master_summary = pd.DataFrame({
    "métrica": ["Archivo", "Filas", "Países", "Variables", "Tiene gdp_growth"],
    "valor": [
        master_path.name,
        master.shape[0],
        master["country_iso3"].nunique(),
        master["variable"].nunique(),
        "Sí" if "gdp_growth" in master["variable"].unique() else "No",
    ],
})

display(style_table(master_summary))
```

**Lectura ejecutiva.** El scoring solo debe ejecutarse si el master contiene el universo de variables esperado, incluye `gdp_growth` para Trend y no corresponde a una extracción parcial.

---

## 3. La matriz de decisión excluye variables que no deben entrar a TOPSIS

```python
wide_raw, decision_matrix, excluded = prepare_decision_matrix(master, configs)

matrix_checks = pd.DataFrame({
    "control": [
        "wide_raw filas",
        "wide_raw variables",
        "decision_matrix filas",
        "decision_matrix variables",
        "gdp_growth en master",
        "gdp_growth en wide_raw",
        "gdp_growth en decision_matrix",
        "wgi_composite en decision_matrix",
    ],
    "resultado": [
        wide_raw.shape[0],
        wide_raw.shape[1],
        decision_matrix.shape[0],
        decision_matrix.shape[1],
        "Sí" if "gdp_growth" in master["variable"].unique() else "No",
        "Sí" if "gdp_growth" in wide_raw.columns else "No",
        "Sí" if "gdp_growth" in decision_matrix.columns else "No",
        "Sí" if "wgi_composite" in decision_matrix.columns else "No",
    ],
})

display(style_table(matrix_checks))

if "gdp_growth" in decision_matrix.columns:
    raise ValueError("gdp_growth está en decision_matrix. Debe excluirse de TOPSIS y usarse solo en Trend.")
```

**Interpretación.** `wide_raw` conserva variables necesarias para IPC, Trend, auditoría y dashboard. `decision_matrix` debe contener únicamente variables elegibles para TOPSIS. `gdp_growth` debe estar en `master` y `wide_raw`, pero no en `decision_matrix`.

---

## 4. El scoring completo integra TOPSIS, IPC, Trend y RADAR

```python
results = run_full_scoring(master, configs, persist=True)

scoring_summary = pd.DataFrame({
    "métrica": [
        "Países evaluados",
        "Países excluidos por cobertura",
        "País origen",
        "País origen excluido",
        "Alpha TOPSIS",
        "Beta IPC",
        "Gamma Trend",
    ],
    "valor": [
        len(results["radar_global"]),
        ", ".join(results["excluded_countries"]) if results["excluded_countries"] else "Ninguno",
        results["origin_country"],
        results["origin_country_excluded"],
        results["composite_weights"]["alpha"],
        results["composite_weights"]["beta"],
        results["composite_weights"]["gamma"],
    ],
})

display(style_table(scoring_summary))
```

**Lectura ejecutiva.** El score RADAR usa la combinación configurada de TOPSIS estructural, IPC y Trend. El país origen debe excluirse del universo evaluado para evitar comparaciones no válidas.

---

## 5. El ranking RADAR global muestra la prioridad final bajo la fórmula compuesta

```python
radar_global = results["radar_global"].copy()

if "country_iso3" not in radar_global.columns:
    radar_global = radar_global.reset_index().rename(columns={"index": "country_iso3"})

radar_global = radar_global.sort_values("radar_score", ascending=False).copy()
radar_global["rank"] = radar_global["radar_score"].rank(ascending=False, method="min").astype(int)

display(style_table(
    radar_global.head(15),
    gradient_cols=["radar_score"],
    gradient_cmap="RdYlGn",
    format_dict={"radar_score": "{:.3f}", "rank": "{:.0f}"},
))
```

```python
fig = px.bar(
    radar_global.head(15).sort_values("radar_score"),
    x="radar_score",
    y="country_iso3",
    orientation="h",
    title="Top 15 países por score RADAR global",
    color="radar_score",
    color_continuous_scale=[[0, CIBEST["gold_light"]], [1, CIBEST["green"]]],
)
fig.update_layout(xaxis_title="Score RADAR", yaxis_title="País")
fig.show()
```

**Interpretación.** Este ranking es la lectura final del modelo híbrido global. Sin embargo, no reemplaza la lectura por línea de negocio: cada línea tiene una tesis de atractividad distinta.

---

## 6. TOPSIS global representa el atractivo estructural antes de proximidad y tendencia

```python
global_ranking = results["global_ranking"].copy()

if "country_iso3" not in global_ranking.columns:
    global_ranking = global_ranking.reset_index().rename(columns={"index": "country_iso3"})

display(style_table(
    global_ranking.head(15),
    gradient_cols=["score"],
    gradient_cmap="RdYlGn",
    format_dict={"score": "{:.3f}", "rank": "{:.0f}"},
))
```

**Lectura ejecutiva.** TOPSIS responde quién tiene mejores fundamentos estructurales. RADAR puede alterar el orden al incorporar proximidad con Colombia y momentum macro.

---

## 7. IPC y Trend explican movimientos entre TOPSIS y RADAR

```python
ipc = results["ipc"].copy()
trend = results["trend"].copy()

if "country_iso3" not in ipc.columns:
    ipc = ipc.reset_index().rename(columns={"index": "country_iso3"})
if "country_iso3" not in trend.columns:
    trend = trend.reset_index().rename(columns={"index": "country_iso3"})

ipc_display = ipc[["country_iso3", "ipc"]].sort_values("ipc", ascending=False)
trend_display = trend[["country_iso3", "trend"]].sort_values("trend", ascending=False)

display(style_table(ipc_display.head(15), gradient_cols=["ipc"], gradient_cmap="RdYlGn", format_dict={"ipc": "{:.3f}"}))
display(style_table(trend_display.head(15), gradient_cols=["trend"], gradient_cmap="RdYlGn", format_dict={"trend": "{:.3f}"}))
```

**Interpretación.** IPC identifica países con mayor proximidad relativa a Colombia. Trend identifica países con mayor momentum macro reciente. Un país puede subir en RADAR aunque no sea top estructural si presenta alta proximidad o tendencia favorable.

---

## 8. La descomposición RADAR muestra qué componente mueve el ranking final

```python
# TOPSIS
base_topsis = results["global_ranking"].copy()
if "country_iso3" in base_topsis.columns:
    topsis = base_topsis.set_index("country_iso3")["score"].rename("topsis_score")
else:
    topsis = base_topsis["score"].rename("topsis_score")

# IPC
ipc_series = ipc.set_index("country_iso3")["ipc"].rename("ipc")

# Trend
trend_series = trend.set_index("country_iso3")["trend"].rename("trend")

component_df = pd.concat([topsis, ipc_series, trend_series], axis=1)
component_df["ipc"] = component_df["ipc"].fillna(component_df["ipc"].median())
component_df["trend"] = component_df["trend"].fillna(component_df["trend"].median())

alpha = results["composite_weights"]["alpha"]
beta = results["composite_weights"]["beta"]
gamma = results["composite_weights"]["gamma"]

component_df["aporte_topsis"] = alpha * component_df["topsis_score"]
component_df["aporte_ipc"] = beta * component_df["ipc"]
component_df["aporte_trend"] = gamma * component_df["trend"]
component_df["radar_score_recalc"] = component_df[["aporte_topsis", "aporte_ipc", "aporte_trend"]].sum(axis=1)
component_df["rank_topsis"] = component_df["topsis_score"].rank(ascending=False, method="min").astype(int)
component_df["rank_radar"] = component_df["radar_score_recalc"].rank(ascending=False, method="min").astype(int)
component_df["delta_rank"] = component_df["rank_topsis"] - component_df["rank_radar"]
component_df = component_df.sort_values("rank_radar")

component_display = component_df.reset_index().rename(columns={"index": "country_iso3"})

display(style_table(
    component_display.head(20),
    gradient_cols=["radar_score_recalc", "delta_rank"],
    gradient_cmap="RdYlGn",
    format_dict={
        "topsis_score": "{:.3f}",
        "ipc": "{:.3f}",
        "trend": "{:.3f}",
        "aporte_topsis": "{:.3f}",
        "aporte_ipc": "{:.3f}",
        "aporte_trend": "{:.3f}",
        "radar_score_recalc": "{:.3f}",
        "rank_topsis": "{:.0f}",
        "rank_radar": "{:.0f}",
        "delta_rank": "{:+.0f}",
    },
))
```

```python
component_means = component_df[["aporte_topsis", "aporte_ipc", "aporte_trend"]].mean().reset_index()
component_means.columns = ["componente", "aporte_promedio"]

fig = px.bar(
    component_means,
    x="componente",
    y="aporte_promedio",
    title="TOPSIS domina el score RADAR por diseño, pero IPC y Trend explican cambios marginales",
    color="componente",
    color_discrete_sequence=[CIBEST["gray"], CIBEST["gold"], CIBEST["green"]],
)
fig.update_layout(showlegend=False, yaxis_title="Aporte promedio")
fig.show()
```

**Implicación.** Esta descomposición permite explicar por qué países estructuralmente fuertes pueden bajar si tienen baja proximidad, o por qué países con fundamentos medios pueden subir por IPC o Trend.

---

## 9. Rankings por línea: cada línea expresa una tesis de atractividad distinta

```python
for business_line, ranking_df in results["business_line_rankings"].items():
    tmp = ranking_df.copy()
    if "country_iso3" not in tmp.columns:
        tmp = tmp.reset_index().rename(columns={"index": "country_iso3"})

    display(Markdown(f"### {business_line} — Top 15"))
    display(style_table(
        tmp.sort_values("rank").head(15),
        gradient_cols=["score"],
        gradient_cmap="RdYlGn",
        format_dict={"score": "{:.3f}", "rank": "{:.0f}"},
    ))
```

**Lectura ejecutiva.** El modelo no produce cinco rankings independientes sin relación; produce cinco lecturas del atractivo país según tesis de negocio. `IB` y `CIB` tienden a privilegiar profundidad financiera, escala e institucionalidad. `PF` y `BD` privilegian adopción digital, flujos, canales e inclusión. `AD` opera como tesis digital-institucional.

---

## 10. La auditoría de pesos verifica que cada línea usa pesos efectivos y overrides esperados

```python
from typing import Any, Dict


def audit_business_line_weights(
    configs: Dict[str, Dict[str, Any]],
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Audita pesos efectivos usados por TOPSIS para cada línea de negocio."""

    business_lines = configs["business_lines"]["business_lines"]
    global_dim_weights = configs["weights"]["dimension_weights"]
    global_variable_weights = configs["weights"]["variable_weights"]

    rows = []

    for bl_key, bl_cfg in business_lines.items():
        dim_weights_line, final_var_weights = _build_business_line_weights(
            business_line_cfg=bl_cfg,
            variable_weights_by_dim=global_variable_weights,
        )

        final_var_weights_filtered = {
            var: weight
            for var, weight in final_var_weights.items()
            if var in decision_matrix.columns
        }

        total_filtered = sum(final_var_weights_filtered.values())
        if total_filtered > 0:
            final_var_weights_filtered = {
                var: weight / total_filtered
                for var, weight in final_var_weights_filtered.items()
            }

        overrides = bl_cfg.get("variable_weight_overrides", {}) or {}

        for dim, vars_global in global_variable_weights.items():
            dim_weight_line = dim_weights_line.get(dim, 0.0)
            dim_weight_global = global_dim_weights.get(dim)

            for var, global_var_weight in vars_global.items():
                override_weight = None
                if dim in overrides and var in overrides[dim]:
                    override_weight = overrides[dim][var]

                rows.append({
                    "business_line": bl_key,
                    "dimension": dim,
                    "variable": var,
                    "in_decision_matrix": var in decision_matrix.columns,
                    "global_dimension_weight": dim_weight_global,
                    "line_dimension_weight": dim_weight_line,
                    "global_variable_weight_in_dim": global_var_weight,
                    "override_variable_weight_in_dim": override_weight,
                    "has_override": override_weight is not None,
                    "final_topsis_weight": final_var_weights_filtered.get(var, 0.0),
                })

    return (
        pd.DataFrame(rows)
        .sort_values(["business_line", "dimension", "final_topsis_weight"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

weights_audit = audit_business_line_weights(configs=configs, decision_matrix=decision_matrix)

weight_sum_check = (
    weights_audit[weights_audit["in_decision_matrix"]]
    .groupby("business_line")["final_topsis_weight"]
    .sum()
    .round(6)
    .reset_index()
)

display(style_table(weight_sum_check, format_dict={"final_topsis_weight": "{:.6f}"}))
```

```python
inactive_weighted_vars = (
    weights_audit[~weights_audit["in_decision_matrix"]]
    [["business_line", "dimension", "variable", "global_variable_weight_in_dim"]]
    .drop_duplicates()
)

display(style_table(inactive_weighted_vars))
```

**Interpretación.** Los pesos finales deben sumar 1 por línea después de filtrar variables presentes en `decision_matrix`. Si aparecen variables ponderadas pero ausentes, se requiere corregir configuración, extracción o matriz de decisión.

---

## 11. El spread de pesos muestra qué variables diferencian realmente las líneas

```python
weight_matrix = (
    weights_audit[weights_audit["in_decision_matrix"]]
    .pivot_table(
        index="variable",
        columns="business_line",
        values="final_topsis_weight",
        aggfunc="sum",
        fill_value=0.0,
    )
)

weight_matrix["spread"] = weight_matrix.max(axis=1) - weight_matrix.min(axis=1)

weight_spread = weight_matrix.sort_values("spread", ascending=False).reset_index()

display(style_table(
    weight_spread.head(25),
    gradient_cols=["spread"],
    gradient_cmap="YlOrRd",
    format_dict={"spread": "{:.4f}"},
))
```

**Implicación.** Las variables con mayor `spread` son las que explican la diferenciación metodológica entre líneas. Deben ser conceptualmente coherentes con la tesis de negocio de cada línea.

---

## 12. La correlación entre líneas identifica familias de tesis de negocio

```python
rankings = {}

for business_line, ranking_df in results["business_line_rankings"].items():
    tmp = ranking_df.copy()
    if "country_iso3" in tmp.columns:
        tmp = tmp.set_index("country_iso3")
    rankings[business_line] = tmp["rank"]

rank_df = pd.DataFrame(rankings)
rank_corr = rank_df.corr(method="spearman")

display(style_table(rank_corr, gradient_cols=rank_corr.columns.tolist(), gradient_cmap="YlOrRd", format_dict={col: "{:.3f}" for col in rank_corr.columns}))
```

```python
fig = px.imshow(
    rank_corr,
    text_auto=".2f",
    color_continuous_scale=[[0, CIBEST["green"]], [0.5, CIBEST["gold"]], [1, CIBEST["red"]]],
    title="Las correlaciones entre líneas revelan familias de atractividad",
)
fig.update_layout(height=550)
fig.show()
```

**Lectura ejecutiva.** Una correlación alta no implica error. Puede indicar que dos líneas comparten fundamentos país. La meta no es forzar rankings independientes, sino que cada línea responda a una tesis de negocio defendible.

---

## 13. Bandas, gaps y empates prácticos evitan sobreinterpretar rankings finos

```python
def classify_gap(gap: float) -> str:
    if pd.isna(gap):
        return "Sin siguiente país"
    if gap < 0.005:
        return "Empate práctico"
    if gap < 0.015:
        return "Diferencia débil"
    if gap < 0.030:
        return "Diferencia moderada"
    return "Diferencia material"


def assign_tier_by_score(score: float, q80: float, q60: float, q40: float) -> str:
    if score >= q80:
        return "Alta"
    if score >= q60:
        return "Media-alta"
    if score >= q40:
        return "Media"
    return "Baja"


def strategic_read(row: pd.Series) -> str:
    tier = row["attractiveness_tier"]
    gap = row["gap_interpretation"]

    if tier == "Alta" and gap == "Diferencia material":
        return "Liderazgo o posicionamiento claramente diferenciado"
    if tier == "Alta" and gap in ["Empate práctico", "Diferencia débil"]:
        return "Alta atractividad, pero no distinguible ordinalmente del país siguiente"
    if tier == "Media-alta" and gap == "Empate práctico":
        return "Banda competitiva media-alta; decisión requiere análisis de drivers"
    if tier == "Media":
        return "Atractividad intermedia; priorizar solo si hay racional estratégico específico"
    return "Lectura dependiente de drivers y restricciones de implementación"


tier_tables = {}

for business_line, ranking_df in results["business_line_rankings"].items():
    tmp = ranking_df.copy()
    if "country_iso3" in tmp.columns:
        tmp = tmp.set_index("country_iso3")

    tmp = tmp.sort_values("score", ascending=False).copy()

    q80 = tmp["score"].quantile(0.80)
    q60 = tmp["score"].quantile(0.60)
    q40 = tmp["score"].quantile(0.40)

    tmp["attractiveness_tier"] = tmp["score"].apply(lambda x: assign_tier_by_score(x, q80, q60, q40))
    tmp["score_gap_next"] = tmp["score"] - tmp["score"].shift(-1)
    tmp["gap_interpretation"] = tmp["score_gap_next"].apply(classify_gap)
    tmp["strategic_read"] = tmp.apply(strategic_read, axis=1)

    tier_tables[business_line] = tmp[["score", "rank", "attractiveness_tier", "score_gap_next", "gap_interpretation", "strategic_read"]]

for business_line, table in tier_tables.items():
    display(Markdown(f"### {business_line} — bandas y gaps"))
    display(style_table(
        table.head(15).reset_index(),
        gradient_cols=["score", "score_gap_next"],
        gradient_cmap="RdYlGn",
        format_dict={"score": "{:.3f}", "rank": "{:.0f}", "score_gap_next": "{:.3f}"},
    ))
```

**Implicación.** Cuando los gaps son bajos, las diferencias de posición deben comunicarse como empates prácticos o bandas, no como jerarquías finas.

---

## 14. La dispersión entre líneas muestra países sensibles a la tesis de negocio

```python
rank_cols = ["IB", "PF", "AD", "BD", "CIB"]
rank_df_clean = rank_df[rank_cols].copy()

rank_df_clean["rank_std_across_lines"] = rank_df_clean[rank_cols].std(axis=1)
rank_df_clean["rank_range_across_lines"] = rank_df_clean[rank_cols].max(axis=1) - rank_df_clean[rank_cols].min(axis=1)

rank_dispersion = rank_df_clean.sort_values("rank_range_across_lines", ascending=False)

display(style_table(
    rank_dispersion.head(20).reset_index(),
    gradient_cols=["rank_range_across_lines", "rank_std_across_lines"],
    gradient_cmap="YlOrRd",
    format_dict={"rank_std_across_lines": "{:.2f}", "rank_range_across_lines": "{:.0f}"},
))
```

**Lectura ejecutiva.** Países con alto rango entre líneas no son “inestables”; son países cuya atractividad depende fuertemente de la tesis de negocio. Estos casos requieren análisis comercial específico.

---

## 15. Contribuciones ponderadas explican drivers y restricciones de forma ejecutiva

> Nota metodológica: `contribution = normalized_value * final_topsis_weight` no es una descomposición exacta de TOPSIS, porque TOPSIS depende de distancias al ideal positivo y negativo. Es una capa de explicabilidad ejecutiva post-normalización.

```python
contrib_by_line = compute_all_business_line_contributions(
    decision_matrix=decision_matrix,
    weights_audit=weights_audit,
)

pf_explainability = build_explainability_table_for_line(
    ranking_df=results["business_line_rankings"]["PF"],
    contributions=contrib_by_line["PF"],
    top_n=3,
)

display(style_table(pf_explainability.head(15), format_dict={"score": "{:.3f}", "rank": "{:.0f}"}))
```

```python
country_focus = "ARG"
line_focus = "PF"

display(style_table(get_top_contributors(contrib_by_line[line_focus], country_focus, top_n=8), gradient_cols=["contribution"], gradient_cmap="RdYlGn", format_dict={"normalized_value": "{:.3f}", "final_topsis_weight": "{:.4f}", "contribution": "{:.4f}"}))
display(style_table(get_top_shortfalls(contrib_by_line[line_focus], country_focus, top_n=8), gradient_cols=["shortfall"], gradient_cmap="YlOrRd", format_dict={"normalized_value": "{:.3f}", "final_topsis_weight": "{:.4f}", "shortfall": "{:.4f}"}))

print(generate_country_line_explanation(contrib_by_line[line_focus], country_focus, top_n=3))
```

**Interpretación.** Los drivers positivos explican qué variables elevan el score estructural de un país. Las restricciones muestran dónde el país pierde atractivo relativo frente al ideal.

---

## 16. Comparar países y líneas permite convertir ranking en narrativa estratégica

```python
# País entre líneas: por qué un país cambia según tesis de negocio
compare_country = compare_country_across_lines(
    contrib_by_line=contrib_by_line,
    country_iso3="ARG",
    top_n=5,
)

display(style_table(compare_country, gradient_cols=["contribution"], gradient_cmap="RdYlGn", format_dict={"normalized_value": "{:.3f}", "final_topsis_weight": "{:.4f}", "contribution": "{:.4f}", "shortfall": "{:.4f}"}))
```

```python
# Países dentro de una línea: por qué un líder supera al bloque siguiente
compare_pf_top = compare_countries_in_line(
    contributions=contrib_by_line["PF"],
    countries=["ESP", "USA", "CAN", "CHL"],
    top_n=12,
)

display(style_table(compare_pf_top))
```

**Implicación.** Esta capa permite explicar diferencias como: un país puede ser competitivo en `PF` y `BD`, pero débil en `IB` o `CIB`, no por error del modelo, sino porque cada línea pondera tesis diferentes.

---

## 17. El análisis marginal valida si un driver es robusto o solo descriptivo

El análisis marginal leave-one-variable-out calcula:

```text
score_effect = score_full - score_without_variable
```

Interpretación:

- `score_effect > 0`: la variable ayuda al país.
- `score_effect < 0`: la variable penaliza al país.
- `abs(score_effect)` alto: variable material.
- `abs(score_effect)` bajo: variable poco decisiva.

```python
marginal_by_line = compute_all_marginal_effects(
    decision_matrix=decision_matrix,
    weights_audit=weights_audit,
    variable_catalog=variable_catalog,
    distance_metric=configs["settings"]["scoring"]["topsis"].get("distance_metric", "euclidean"),
)

pf_marginal = marginal_by_line["PF"]

line_variable_importance = (
    pf_marginal
    .groupby(["business_line", "dimension", "removed_variable"], as_index=False)
    .agg(
        mean_abs_effect=("abs_score_effect", "mean"),
        mean_effect=("score_effect", "mean"),
        max_abs_effect=("abs_score_effect", "max"),
    )
    .sort_values("mean_abs_effect", ascending=False)
)

display(style_table(
    line_variable_importance.head(20),
    gradient_cols=["mean_abs_effect"],
    gradient_cmap="YlOrRd",
    format_dict={"mean_abs_effect": "{:.4f}", "mean_effect": "{:.4f}", "max_abs_effect": "{:.4f}"},
))
```

**Advertencia metodológica.** No se deben sumar efectos marginales como una descomposición contable del score. TOPSIS es no lineal y depende de distancias al ideal y renormalización de pesos.

---

## 18. Combinar contribución y marginal permite clasificar drivers robustos

```python
pf_combined_explainability = combine_contribution_and_marginal(
    contributions=contrib_by_line["PF"],
    marginal_effects=marginal_by_line["PF"],
)

pf_combined_explainability["driver_class"] = pf_combined_explainability.apply(
    classify_driver_robustness,
    axis=1,
)

arg_driver_table = build_country_driver_table(
    explainability_df=pf_combined_explainability,
    country_iso3="ARG",
    top_n=5,
)

display(style_table(
    arg_driver_table,
    gradient_cols=["contribution", "shortfall", "score_effect"],
    gradient_cmap="RdYlGn",
    format_dict={"normalized_value": "{:.3f}", "contribution": "{:.4f}", "shortfall": "{:.4f}", "score_effect": "{:.4f}", "rank_effect": "{:+.0f}"},
))
```

**Lectura ejecutiva.** Un driver robusto combina alta contribución y efecto marginal positivo. Una restricción crítica puede tener baja contribución, alta brecha y efecto marginal negativo.

---

## 19. Hallazgos principales

1. RADAR debe leerse como score compuesto: TOPSIS aporta fundamentos estructurales, IPC proximidad y Trend momentum macro.
2. `gdp_growth` debe permanecer fuera de TOPSIS y usarse exclusivamente para Trend, evitando doble conteo de crecimiento.
3. Las líneas de negocio no son rankings independientes; son tesis diferenciadas de atractividad país.
4. La auditoría de pesos permite verificar que overrides y pesos finales cierren correctamente por línea.
5. Las correlaciones entre líneas identifican familias de negocio y similitudes estructurales.
6. Los gaps y bandas son necesarios para no sobrerrepresentar posiciones con diferencias mínimas.
7. La explicabilidad debe combinar contribución ponderada y análisis marginal para distinguir drivers robustos de drivers descriptivos.

---

## 20. Limitaciones

- TOPSIS es relativo al conjunto de países evaluados; agregar o quitar países puede alterar scores.
- Las contribuciones ponderadas no son una descomposición exacta de TOPSIS.
- El análisis marginal es sensibilidad leave-one-variable-out, no causalidad.
- Las bandas por percentiles dependen del universo evaluado.
- RADAR combina componentes con pesos configurados; la robustez de la decisión final debe evaluarse en el notebook 04 mediante Monte Carlo.

---

## 21. Recomendaciones y próximos pasos

1. Presentar rankings por bandas y no solo por posición ordinal.
2. Usar gaps para distinguir líderes robustos de empates prácticos.
3. Reportar drivers y restricciones para países prioritarios por línea.
4. Usar dispersión entre líneas para identificar países dependientes de tesis específica.
5. Validar robustez del ranking RADAR completo en `04_sensitivity_analysis.ipynb` mediante Monte Carlo.
6. Persistir tablas de pesos, bandas, drivers y restricciones como artefactos reproducibles.

---

## 22. Síntesis Ejecutiva

- El scoring RADAR integra atractivo estructural, proximidad y momentum macro en una métrica ejecutiva final.
- Los rankings por línea reflejan tesis de negocio distintas; no deben interpretarse como listas independientes sin contexto.
- Las bandas, gaps y empates prácticos son esenciales para evitar decisiones basadas en diferencias marginales.
- La explicabilidad debe cruzar contribuciones y efectos marginales para identificar drivers realmente robustos.
- El siguiente paso es validar robustez probabilística del RADAR completo en el notebook de sensibilidad.
