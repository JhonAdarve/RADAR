# 00 — Guía metodológica ejecutiva con datos reales | RADAR Cibest

**Proyecto:** RADAR Cibest — Ranking de Atractivo y Diagnóstico Analítico Regional  
**Autor:** Jhon Farley Adarve Díaz — Dirección de Estrategia  
**Audiencia:** Comité Ejecutivo, Junta Directiva, equipos de Estrategia, Analítica y líneas de negocio  
**Propósito:** explicar, auditar y ejecutar la metodología completa del modelo RADAR usando la **data real completa del pipeline**, no ejemplos sintéticos.

---

## 0. La metodología RADAR debe explicarse sobre el pipeline real, no sobre una miniatura sintética

La versión inicial del notebook metodológico cumple una función pedagógica: explica el problema de negocio, ASUM-DM, BWM, TOPSIS, IPC, RADAR, señales y sensibilidad. Sin embargo, para un entregable ejecutivo y auditable, los ejemplos sintéticos deben reemplazarse por ejecuciones sobre:

- el `master_raw_YYYYMMDD.parquet` más reciente;
- el catálogo real de variables en `variables.yaml`;
- los pesos reales en `weights.yaml` y `business_lines.yaml`;
- la matriz real `decision_matrix`;
- los resultados reales de TOPSIS, IPC, Trend y RADAR;
- las simulaciones Monte Carlo reales de robustez.

Este notebook mantiene la explicación conceptual, pero cada fórmula se conecta con una celda ejecutable sobre el dataset completo.

### Decisiones metodológicas corregidas frente a la versión pedagógica

1. **No se usan cinco países sintéticos.** Todas las tablas y rankings se generan con el universo real de países configurado.
2. **No se usa una matriz reducida de siete variables.** Se trabaja con el catálogo real vigente y con las variables efectivamente elegibles para TOPSIS.
3. **`gdp_growth` no entra a TOPSIS.** Se usa exclusivamente para el componente `Trend`, evitando doble conteo macroeconómico.
4. **La proximidad no debe duplicarse.** Si el diseño vigente usa IPC como componente explícito del RADAR, las variables de proximidad deben excluirse de la matriz TOPSIS estructural para evitar doble conteo.
5. **La cobertura se evalúa después del filtro de vigencia.** La regla de máximo 5 años convierte datos antiguos en faltantes efectivos antes de imputación.
6. **La robustez principal se evalúa sobre RADAR completo.** Monte Carlo TOPSIS es diagnóstico estructural; Monte Carlo RADAR es evidencia decisional.

---

## 1. Configuración del entorno y estilo Cibest

```python
# ---------------------------------------------------------------------------
# Configuración inicial del notebook
# ---------------------------------------------------------------------------
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import importlib
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML, display, Markdown

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent))

# ---------------------------------------------------------------------------
# Módulos RADAR
# ---------------------------------------------------------------------------
import src
import src.utils as utils
import src.data_preparation.cleaning as cleaning
import src.data_preparation.feature_engineering as feature_engineering
import src.scoring.hybrid_scorer as hybrid_scorer
import src.scoring.ranking as ranking
import src.scoring.explainability as explainability
import src.scoring.monte_carlo as monte_carlo

importlib.invalidate_caches()
importlib.reload(utils)
importlib.reload(cleaning)
importlib.reload(feature_engineering)
importlib.reload(hybrid_scorer)
importlib.reload(ranking)
importlib.reload(explainability)
importlib.reload(monte_carlo)

from src.utils import (
    load_all_configs,
    setup_logger,
    resolve_data_path,
    get_variable_catalog,
)

from src.data_preparation.cleaning import (
    pivot_latest_value_and_year,
    apply_freshness_filter,
    run_cleaning,
)

from src.scoring.hybrid_scorer import (
    prepare_decision_matrix,
    run_full_scoring,
    _build_business_line_weights,
)

from src.scoring.explainability import (
    compute_all_business_line_contributions,
    build_explainability_table_for_line,
    compute_all_marginal_effects,
    combine_contribution_and_marginal,
    classify_driver_robustness,
    build_country_driver_table,
)

from src.scoring.monte_carlo import (
    coerce_component_series,
    run_monte_carlo_topsis_robustness,
    run_monte_carlo_radar_robustness,
)

configs = load_all_configs()
setup_logger(configs["settings"].get("logging"))
variable_catalog = get_variable_catalog(configs["variables"])

# ---------------------------------------------------------------------------
# Paleta Cibest
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

success_box("Entorno listo", "La metodología se ejecutará sobre la data real completa del proyecto RADAR Cibest.")
```

---

## 2. El problema de negocio exige una metodología híbrida, no un ranking unidimensional

RADAR Cibest resuelve un problema de decisión multicriterio: priorizar mercados internacionales para distintas líneas de negocio financiero. La decisión no puede depender solo de tamaño de mercado, proximidad o riesgo país. Requiere combinar fundamentos estructurales, afinidad bilateral con Colombia y momentum reciente.

### Líneas de negocio evaluadas

```python
business_lines_cfg = configs["business_lines"]["business_lines"]

business_lines_df = pd.DataFrame([
    {
        "business_line": key,
        "descripcion": cfg.get("description", key),
        "n_dimensiones": len(cfg.get("weight_profile", {})),
    }
    for key, cfg in business_lines_cfg.items()
])

display(style_table(business_lines_df))
```

### Dimensiones configuradas

```python
dimensions_df = pd.DataFrame({
    "dimension": configs["settings"]["project"]["dimensions"]
})

display(style_table(dimensions_df))
```

**Interpretación ejecutiva.** Las líneas de negocio comparten el mismo universo país-variable, pero no la misma tesis de atractividad. Por eso RADAR ejecuta scoring global y scoring diferenciado por línea.

---

## 3. El catálogo real define el universo metodológico vigente

```python
catalog_df = pd.DataFrame.from_dict(variable_catalog, orient="index").reset_index()
catalog_df = catalog_df.rename(columns={"index": "variable"})

catalog_summary = pd.DataFrame({
    "métrica": ["Variables en catálogo", "Dimensiones", "Fuentes", "Variables include_in_topsis=False"],
    "valor": [
        catalog_df["variable"].nunique(),
        catalog_df["dimension"].nunique() if "dimension" in catalog_df.columns else None,
        catalog_df["source"].nunique() if "source" in catalog_df.columns else None,
        int((catalog_df.get("include_in_topsis", True) == False).sum()) if "include_in_topsis" in catalog_df.columns else 0,
    ],
})

display(style_table(catalog_summary))

by_dimension = (
    catalog_df.groupby("dimension", dropna=False)
    .size()
    .rename("n_variables")
    .reset_index()
    .sort_values("n_variables", ascending=False)
)

display(style_table(by_dimension, gradient_cols=["n_variables"], format_dict={"n_variables": "{:,.0f}"}))
```

```python
fig = px.bar(
    by_dimension.sort_values("n_variables"),
    x="n_variables",
    y="dimension",
    orientation="h",
    title="Variables reales por dimensión configurada",
    color="n_variables",
    color_continuous_scale=[[0, CIBEST["gray_light"]], [1, CIBEST["gold"]]],
)
fig.update_layout(xaxis_title="Número de variables", yaxis_title="Dimensión")
fig.show()
```

**Control metodológico.** Si el catálogo muestra variables marcadas como `include_in_topsis: false`, estas deben permanecer fuera de `decision_matrix`. El caso crítico es `gdp_growth`, que debe usarse para `Trend` y no para TOPSIS.

---

## 4. La data real se carga desde el master vigente

```python
raw_dir = resolve_data_path(configs["settings"]["data"]["raw_path"])
pattern = re.compile(r"^master_raw_\d{8}\.parquet$")

master_files = sorted(
    [path for path in raw_dir.glob("master_raw_*.parquet") if pattern.match(path.name)],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not master_files:
    raise FileNotFoundError("No se encontró master_raw_YYYYMMDD.parquet. Ejecute primero el notebook 01.")

master_path = master_files[0]
master = pd.read_parquet(master_path)

required_cols = {"country_iso3", "year", "variable", "value", "source"}
missing_cols = required_cols - set(master.columns)
if missing_cols:
    raise ValueError(f"Master inválido. Faltan columnas: {sorted(missing_cols)}")

master = master.copy()
master["country_iso3"] = master["country_iso3"].astype(str).str.strip()
master["variable"] = master["variable"].astype(str).str.strip()
master["year"] = pd.to_numeric(master["year"], errors="coerce")
master["value"] = pd.to_numeric(master["value"], errors="coerce")

master_summary = pd.DataFrame({
    "métrica": ["Archivo", "Filas", "Países", "Variables", "Fuentes", "Año mínimo", "Año máximo", "Tiene gdp_growth"],
    "valor": [
        master_path.name,
        master.shape[0],
        master["country_iso3"].nunique(),
        master["variable"].nunique(),
        master["source"].nunique(),
        int(master["year"].min()),
        int(master["year"].max()),
        "Sí" if "gdp_growth" in master["variable"].unique() else "No",
    ],
})

display(style_table(master_summary))
```

---

## 5. La calidad metodológica empieza con vigencia: datos antiguos no deben entrar como actuales

```python
wide_values, wide_years = pivot_latest_value_and_year(master)

wide_fresh, stale_report = apply_freshness_filter(
    wide_values=wide_values,
    wide_years=wide_years,
    variable_catalog=variable_catalog,
    settings=configs["settings"],
)

stale_mask = wide_values.notna() & wide_fresh.isna()
stale_cells = int(stale_mask.sum().sum())

freshness_cfg = configs["settings"].get("data_quality", {})
freshness_summary = pd.DataFrame({
    "métrica": ["freshness_reference_year", "max_data_age_years", "celdas stale", "países", "variables"],
    "valor": [
        freshness_cfg.get("freshness_reference_year"),
        freshness_cfg.get("max_data_age_years"),
        stale_cells,
        wide_values.shape[0],
        wide_values.shape[1],
    ],
})

display(style_table(freshness_summary))
```

```python
stale_by_variable = (
    stale_mask.sum(axis=0)
    .sort_values(ascending=False)
    .rename("n_stale")
    .reset_index()
    .rename(columns={"index": "variable"})
)
stale_by_variable["pct_stale_countries"] = stale_by_variable["n_stale"] / wide_values.shape[0]

display(style_table(
    stale_by_variable.head(20),
    gradient_cols=["pct_stale_countries"],
    gradient_cmap="YlOrRd",
    format_dict={"n_stale": "{:,.0f}", "pct_stale_countries": "{:.1%}"},
))
```

**Interpretación metodológica.** La regla de vigencia no elimina variables; convierte valores observados antiguos en faltantes efectivos antes de cobertura e imputación. Esto mejora la calidad temporal del scoring y evita que datos de hace más de cinco años contaminen la lectura actual.

---

## 6. La matriz de decisión real traduce datos crudos en insumo TOPSIS

```python
wide_raw, decision_matrix, excluded_countries = prepare_decision_matrix(master, configs)

matrix_summary = pd.DataFrame({
    "métrica": ["wide_raw", "decision_matrix", "países excluidos", "gdp_growth en wide_raw", "gdp_growth en decision_matrix"],
    "valor": [
        str(wide_raw.shape),
        str(decision_matrix.shape),
        ", ".join(excluded_countries) if excluded_countries else "Ninguno",
        "Sí" if "gdp_growth" in wide_raw.columns else "No",
        "Sí" if "gdp_growth" in decision_matrix.columns else "No",
    ],
})

display(style_table(matrix_summary))

if "gdp_growth" in decision_matrix.columns:
    raise ValueError("gdp_growth aparece en decision_matrix. Debe excluirse de TOPSIS para evitar doble conteo con Trend.")
```

**Interpretación.** `wide_raw` conserva variables necesarias para IPC, Trend y auditoría. `decision_matrix` conserva solo variables estructurales elegibles para TOPSIS, normalizadas y orientadas para que valores altos signifiquen mayor atractivo.

---

## 7. Los pesos efectivos se auditan con la configuración real, no con ejemplos

```python
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
            for var, global_var_weight in vars_global.items():
                override_weight = None
                if dim in overrides and var in overrides[dim]:
                    override_weight = overrides[dim][var]

                rows.append({
                    "business_line": bl_key,
                    "dimension": dim,
                    "variable": var,
                    "in_decision_matrix": var in decision_matrix.columns,
                    "global_dimension_weight": global_dim_weights.get(dim),
                    "line_dimension_weight": dim_weights_line.get(dim, 0.0),
                    "global_variable_weight_in_dim": global_var_weight,
                    "override_variable_weight_in_dim": override_weight,
                    "has_override": override_weight is not None,
                    "final_topsis_weight": final_var_weights_filtered.get(var, 0.0),
                })

    return pd.DataFrame(rows).sort_values(
        ["business_line", "dimension", "final_topsis_weight"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

weights_audit = audit_business_line_weights(configs, decision_matrix)

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

**Interpretación.** Los pesos efectivos muestran la tesis real por línea. Las variables con mayor `spread` son las que verdaderamente diferencian IB, PF, AD, BD y CIB.

---

## 8. TOPSIS real: ranking estructural global y por línea

```python
results = run_full_scoring(master, configs, persist=True)

global_topsis = results["global_ranking"].copy()
if "country_iso3" not in global_topsis.columns:
    global_topsis = global_topsis.reset_index().rename(columns={"index": "country_iso3"})

display(style_table(
    global_topsis.sort_values("rank").head(15),
    gradient_cols=["score"],
    gradient_cmap="RdYlGn",
    format_dict={"score": "{:.3f}", "rank": "{:.0f}"},
))
```

```python
for business_line, ranking_df in results["business_line_rankings"].items():
    tmp = ranking_df.copy()
    if "country_iso3" not in tmp.columns:
        tmp = tmp.reset_index().rename(columns={"index": "country_iso3"})

    display(Markdown(f"### {business_line} — TOPSIS estructural"))
    display(style_table(
        tmp.sort_values("rank").head(15),
        gradient_cols=["score"],
        gradient_cmap="RdYlGn",
        format_dict={"score": "{:.3f}", "rank": "{:.0f}"},
    ))
```

**Interpretación.** TOPSIS mide atractivo estructural. No incorpora por sí mismo el componente de proximidad bilateral ni el momentum macro reciente, salvo que esas variables hayan sido incluidas explícitamente en la matriz, lo cual el diseño vigente evita para prevenir doble conteo.

---

## 9. IPC real: proximidad bilateral con Colombia como componente separado

```python
ipc_df = results["ipc"].copy()
if "country_iso3" not in ipc_df.columns:
    ipc_df = ipc_df.reset_index().rename(columns={"index": "country_iso3"})

ipc_display = ipc_df[["country_iso3", "ipc"]].sort_values("ipc", ascending=False)

display(style_table(
    ipc_display.head(20),
    gradient_cols=["ipc"],
    gradient_cmap="RdYlGn",
    format_dict={"ipc": "{:.3f}"},
))
```

```python
fig = px.bar(
    ipc_display.head(20).sort_values("ipc"),
    x="ipc",
    y="country_iso3",
    orientation="h",
    title="IPC real — proximidad bilateral con Colombia",
    color="ipc",
    color_continuous_scale=[[0, CIBEST["gold_light"]], [1, CIBEST["green"]]],
)
fig.update_layout(xaxis_title="IPC", yaxis_title="País")
fig.show()
```

**Interpretación.** IPC responde una pregunta distinta a TOPSIS: no mide atractivo absoluto del país, sino afinidad relativa con Colombia. Por eso debe mantenerse como componente explícito en RADAR.

---

## 10. Trend real: momentum macroeconómico reciente

```python
trend_df = results["trend"].copy()
if "country_iso3" not in trend_df.columns:
    trend_df = trend_df.reset_index().rename(columns={"index": "country_iso3"})

trend_display = trend_df[["country_iso3", "trend"]].sort_values("trend", ascending=False)

display(style_table(
    trend_display.head(20),
    gradient_cols=["trend"],
    gradient_cmap="RdYlGn",
    format_dict={"trend": "{:.3f}"},
))
```

**Control metodológico.** `Trend` debe calcularse desde `gdp_growth`, pero `gdp_growth` no debe estar en `decision_matrix`. Esta separación mantiene independencia conceptual entre atractivo estructural y momentum macro.

---

## 11. RADAR real: integración de TOPSIS, IPC y Trend

```python
radar_global = results["radar_global"].copy()
if "country_iso3" not in radar_global.columns:
    radar_global = radar_global.reset_index().rename(columns={"index": "country_iso3"})

radar_global = radar_global.sort_values("radar_score", ascending=False).copy()
radar_global["rank"] = radar_global["radar_score"].rank(ascending=False, method="min").astype(int)

display(style_table(
    radar_global.head(20),
    gradient_cols=["radar_score"],
    gradient_cmap="RdYlGn",
    format_dict={"radar_score": "{:.3f}", "rank": "{:.0f}"},
))
```

```python
fig = px.bar(
    radar_global.head(20).sort_values("radar_score"),
    x="radar_score",
    y="country_iso3",
    orientation="h",
    title="RADAR global real — ranking compuesto de atractivo",
    color="radar_score",
    color_continuous_scale=[[0, CIBEST["gold_light"]], [1, CIBEST["green"]]],
)
fig.update_layout(xaxis_title="Score RADAR", yaxis_title="País")
fig.show()
```

---

## 12. Descomposición real del score RADAR

```python
base_topsis = results["global_ranking"].copy()
if "country_iso3" in base_topsis.columns:
    topsis_series = base_topsis.set_index("country_iso3")["score"].rename("topsis_score")
else:
    topsis_series = base_topsis["score"].rename("topsis_score")

ipc_series = ipc_df.set_index("country_iso3")["ipc"].rename("ipc")
trend_series = trend_df.set_index("country_iso3")["trend"].rename("trend")

component_df = pd.concat([topsis_series, ipc_series, trend_series], axis=1)
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

**Interpretación.** Esta tabla responde por qué un país sube o baja al pasar de TOPSIS a RADAR. Un país puede ganar posiciones por IPC o Trend aunque no sea el más fuerte estructuralmente.

---

## 13. Bandas y empates prácticos sobre la data real

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

    tier_tables[business_line] = tmp[["score", "rank", "attractiveness_tier", "score_gap_next", "gap_interpretation"]]

for business_line, table in tier_tables.items():
    display(Markdown(f"### {business_line} — bandas y gaps reales"))
    display(style_table(
        table.head(15).reset_index(),
        gradient_cols=["score", "score_gap_next"],
        gradient_cmap="RdYlGn",
        format_dict={"score": "{:.3f}", "rank": "{:.0f}", "score_gap_next": "{:.3f}"},
    ))
```

**Interpretación.** Las bandas evitan sobrerreaccionar ante diferencias marginales. Un ranking ejecutivo debe distinguir liderazgo material, diferencia moderada y empate práctico.

---

## 14. Explicabilidad real: drivers, restricciones y análisis marginal

```python
contrib_by_line = compute_all_business_line_contributions(
    decision_matrix=decision_matrix,
    weights_audit=weights_audit,
)

marginal_by_line = compute_all_marginal_effects(
    decision_matrix=decision_matrix,
    weights_audit=weights_audit,
    variable_catalog=variable_catalog,
    distance_metric=configs["settings"]["scoring"]["topsis"].get("distance_metric", "euclidean"),
)
```

```python
# Ejemplo real parametrizable: cambiar país y línea según interés ejecutivo.
country_focus = "ARG"
line_focus = "PF"

line_explainability = combine_contribution_and_marginal(
    contributions=contrib_by_line[line_focus],
    marginal_effects=marginal_by_line[line_focus],
)

line_explainability["driver_class"] = line_explainability.apply(
    classify_driver_robustness,
    axis=1,
)

country_driver_table = build_country_driver_table(
    explainability_df=line_explainability,
    country_iso3=country_focus,
    top_n=5,
)

display(style_table(
    country_driver_table,
    gradient_cols=["contribution", "shortfall", "score_effect"],
    gradient_cmap="RdYlGn",
    format_dict={
        "normalized_value": "{:.3f}",
        "contribution": "{:.4f}",
        "shortfall": "{:.4f}",
        "score_effect": "{:.4f}",
        "rank_effect": "{:+.0f}",
    },
))
```

**Interpretación.** La contribución ponderada explica perfil; el análisis marginal evalúa robustez. La combinación permite clasificar variables como drivers robustos, drivers descriptivos, restricciones críticas o efectos marginales bajos.

---

## 15. Robustez real: Monte Carlo TOPSIS y RADAR

```python
mc_cfg = configs["settings"].get("monte_carlo", {})

ipc_scores = coerce_component_series(results["ipc"], value_col=None, component_name="ipc")
trend_scores = coerce_component_series(results["trend"], value_col="trend", component_name="trend")

mc_topsis = run_monte_carlo_topsis_robustness(
    decision_matrix=decision_matrix,
    configs=configs,
    variable_catalog=variable_catalog,
    n_simulations=int(mc_cfg.get("n_simulations", 1000)),
    dimension_concentration=float(mc_cfg.get("topsis", {}).get("dimension_concentration", 150)),
    variable_concentration=float(mc_cfg.get("topsis", {}).get("variable_concentration", 100)),
    random_seed=int(mc_cfg.get("random_seed", 42)),
)

mc_radar = run_monte_carlo_radar_robustness(
    decision_matrix=decision_matrix,
    configs=configs,
    variable_catalog=variable_catalog,
    ipc_scores=ipc_scores,
    trend_scores=trend_scores,
    n_simulations=int(mc_cfg.get("n_simulations", 1000)),
    dimension_concentration=float(mc_cfg.get("topsis", {}).get("dimension_concentration", 150)),
    variable_concentration=float(mc_cfg.get("topsis", {}).get("variable_concentration", 100)),
    composite_concentration=float(mc_cfg.get("radar", {}).get("composite_concentration", 150)),
    perturb_composite_weights=bool(mc_cfg.get("radar", {}).get("perturb_composite_weights", True)),
    random_seed=int(mc_cfg.get("random_seed", 42)),
)

mc_summary = pd.DataFrame({
    "métrica": ["MC TOPSIS filas", "MC RADAR filas", "Simulaciones RADAR", "Países RADAR", "Líneas RADAR"],
    "valor": [
        mc_topsis["simulation_long"].shape[0],
        mc_radar["simulation_long"].shape[0],
        mc_radar["simulation_long"]["simulation_id"].nunique(),
        mc_radar["simulation_long"]["country_iso3"].nunique(),
        mc_radar["simulation_long"]["business_line"].nunique(),
    ],
})

display(style_table(mc_summary))
```

```python
radar_robustness = (
    mc_radar["rank_robustness"]
    .merge(mc_radar["topn_probabilities"], on=["business_line", "country_iso3"], how="left")
    .merge(mc_radar["tier_probabilities"], on=["business_line", "country_iso3"], how="left")
)

for business_line in sorted(radar_robustness["business_line"].unique()):
    display(Markdown(f"### {business_line} — Robustez Monte Carlo RADAR"))
    display(style_table(
        radar_robustness.query("business_line == @business_line").sort_values("mean_rank").head(15),
        gradient_cols=["mean_rank", "std_rank", "prob_top_5", "Alta"],
        gradient_cmap="RdYlGn",
        format_dict={
            "mean_rank": "{:.2f}",
            "std_rank": "{:.2f}",
            "p10_rank": "{:.1f}",
            "p90_rank": "{:.1f}",
            "prob_top_3": "{:.1%}",
            "prob_top_5": "{:.1%}",
            "prob_top_10": "{:.1%}",
            "prob_top_15": "{:.1%}",
            "Alta": "{:.1%}",
            "Media-alta": "{:.1%}",
            "Media": "{:.1%}",
            "Baja": "{:.1%}",
        },
    ))
```

**Interpretación.** Monte Carlo RADAR transforma el ranking determinístico en una lectura probabilística. `prob_top_5`, `prob_top_10` y probabilidad de banda son más útiles para decisión que el rank puntual.

---

## 16. Arquitectura metodológica completa con datos reales

```python
layers = [
    ("1. Configuración estratégica", "settings.yaml, variables.yaml, weights.yaml, business_lines.yaml"),
    ("2. Extracción", "master_raw_YYYYMMDD.parquet"),
    ("3. Calidad", "freshness filter, cobertura efectiva, imputación"),
    ("4. Matriz estructural", "decision_matrix sin gdp_growth ni proximidad duplicada"),
    ("5. TOPSIS", "ranking estructural global y por línea"),
    ("6. IPC", "proximidad bilateral con Colombia"),
    ("7. Trend", "momentum macro vía gdp_growth"),
    ("8. RADAR", "alpha·TOPSIS + beta·IPC + gamma·Trend"),
    ("9. Explicabilidad", "drivers, restricciones, contribuciones, marginal"),
    ("10. Robustez", "Monte Carlo TOPSIS y Monte Carlo RADAR"),
]

fig = go.Figure()
for idx, (name, desc) in enumerate(layers):
    y = len(layers) - idx
    color = CIBEST["gold"] if "RADAR" in name else CIBEST["gray"] if idx in [4, 8, 9] else CIBEST["gray_light"]
    font_color = CIBEST["gray"] if color in [CIBEST["gold"], CIBEST["gray_light"]] else CIBEST["yellow"]
    fig.add_shape(type="rect", x0=0, x1=10, y0=y-0.35, y1=y+0.35, fillcolor=color, line=dict(color=CIBEST["gray"], width=1))
    fig.add_annotation(x=5, y=y, text=f"<b>{name}</b><br><span style='font-size:11px'>{desc}</span>", showarrow=False, font=dict(color=font_color, size=12))

fig.update_layout(
    title="Arquitectura metodológica RADAR Cibest — ejecución sobre data real",
    xaxis=dict(visible=False, range=[-0.5, 10.5]),
    yaxis=dict(visible=False, range=[0.2, len(layers)+1]),
    height=760,
    width=1050,
    plot_bgcolor=CIBEST["white"],
    paper_bgcolor=CIBEST["white"],
)
fig.show()
```

---

## 17. Hallazgos metodológicos esperados del notebook real

1. La metodología RADAR es híbrida porque ninguna dimensión por sí sola captura el atractivo internacional de un negocio financiero.
2. TOPSIS mide atractivo estructural; IPC mide afinidad bilateral; Trend mide momentum reciente.
3. `gdp_growth` debe alimentar Trend y no TOPSIS.
4. La proximidad debe tratarse como componente explícito si IPC está activo, evitando doble conteo dentro de TOPSIS.
5. La regla de vigencia de 5 años es necesaria para evitar que `latest_available` use observaciones obsoletas.
6. Los rankings deben comunicarse por bandas, gaps y robustez, no solo por posición ordinal.
7. Monte Carlo RADAR es la validación decisional principal.

---

## 18. Limitaciones metodológicas

- TOPSIS y RADAR son modelos relativos al universo de países evaluado.
- La normalización min-max puede cambiar si se agregan o eliminan países.
- La imputación regional preserva cobertura, pero puede suavizar diferencias reales.
- Monte Carlo perturba pesos, no errores de medición en variables.
- La probabilidad de Top-N no sustituye due diligence regulatoria, competitiva o financiera.

---

## 19. Recomendaciones de gobierno metodológico

1. Mantener versiones fechadas de `settings.yaml`, `variables.yaml`, `weights.yaml` y `business_lines.yaml`.
2. Persistir `weights_audit`, `freshness_audit`, rankings, contribuciones y resultados Monte Carlo por corrida.
3. Exigir que todo ranking ejecutivo incluya bandas, gaps y robustez Monte Carlo.
4. Revisar anualmente variables con alta imputación o alta stale data.
5. No modificar pesos para “forzar” resultados; toda recalibración debe justificarse por tesis de negocio.
6. Mantener una bitácora de cambios metodológicos aprobada por Estrategia y analítica.

---

## 20. Síntesis Ejecutiva

- Esta versión del notebook 00 explica RADAR usando la data real completa, no ejemplos sintéticos.
- El flujo metodológico real separa datos, calidad, TOPSIS, IPC, Trend, RADAR, explicabilidad y robustez.
- La arquitectura evita doble conteo: `gdp_growth` va a Trend y proximidad va a IPC.
- La decisión ejecutiva debe basarse en RADAR robusto, no solo en TOPSIS determinístico.
- El modelo es auditable si cada corrida conserva configuración, matriz, pesos, scoring, explicabilidad y Monte Carlo.
