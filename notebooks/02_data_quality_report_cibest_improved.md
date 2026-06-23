# 02 — Calidad efectiva de datos | RADAR Cibest

**Fase ASUM-DM:** 2 — Entendimiento de datos  
**Responsable:** Jhon Farley Adarve Díaz  
**Fecha de ejecución:** Junio 2026  
**Propósito:** evaluar si el `master_raw` generado por extracción tiene cobertura, vigencia y calidad suficientes para alimentar la matriz de decisión del scoring RADAR.

---

## 0. La calidad del dato debe evaluarse como cobertura efectiva, no solo como cobertura bruta

Este notebook responde una pregunta metodológica crítica:

> **¿Qué tan confiable es el dataset después de considerar cobertura, antigüedad de datos, faltantes efectivos, imputación y exclusión de países?**

La extracción del notebook 01 confirma que el master existe y contiene variables suficientes. Este notebook determina si esos datos son **usables para scoring**, aplicando controles de:

- cobertura bruta por país y variable;
- vigencia del último dato disponible;
- stale data bajo regla de máximo 5 años;
- faltantes efectivos después del filtro de frescura;
- riesgo de imputación;
- países con cobertura insuficiente;
- variables que deben mantenerse, excluirse o buscar fuente alternativa.

### Mensaje ejecutivo preliminar

La cobertura bruta puede sobreestimar la calidad real si incluye valores antiguos. Por tanto, la métrica relevante para scoring no es solo:

```text
variable con dato disponible
```

sino:

```text
variable con dato disponible y vigente, o imputación aceptable después del filtro de frescura
```

---

## 1. Configuración del entorno y estilo Cibest

```python
# ---------------------------------------------------------------------------
# Configuración inicial del notebook
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
import re
import importlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, display, Markdown

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path.cwd().parent))

from src.utils import (
    load_all_configs,
    resolve_data_path,
    get_variable_catalog,
    setup_logger,
)

from src.data_preparation.cleaning import (
    pivot_latest_value_and_year,
    apply_freshness_filter,
    run_cleaning,
)

configs = load_all_configs()
setup_logger(configs["settings"].get("logging"))
catalog = get_variable_catalog(configs["variables"])

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
        {"selector": "tbody tr:hover", "props": [("background-color", CIBEST["gray_bg"])]},
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

success_box("Entorno listo", "Configuración, catálogo y funciones de calidad de datos cargadas correctamente.")
```

---

## 2. Se carga el master más reciente para evitar usar resultados obsoletos

```python
raw_dir = resolve_data_path(configs["settings"]["data"]["raw_path"])
pattern = re.compile(r"^master_raw_\d{8}\.parquet$")

candidates = sorted(
    [
        path for path in raw_dir.glob("master_raw_*.parquet")
        if pattern.match(path.name)
    ],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not candidates:
    raise FileNotFoundError("No se encontró master_raw_YYYYMMDD.parquet. Ejecute primero el notebook 01.")

master_path = candidates[0]
master = pd.read_parquet(master_path)

required_cols = {"country_iso3", "year", "variable", "value", "source"}
missing_cols = required_cols - set(master.columns)

if missing_cols:
    raise ValueError(f"Master inválido. Faltan columnas: {sorted(missing_cols)}")

master["country_iso3"] = master["country_iso3"].astype(str).str.strip()
master["variable"] = master["variable"].astype(str).str.strip()
master["year"] = pd.to_numeric(master["year"], errors="coerce")
master["value"] = pd.to_numeric(master["value"], errors="coerce")

master_summary = pd.DataFrame({
    "métrica": ["Archivo", "Filas", "Países", "Variables", "Fuentes", "Año mínimo", "Año máximo"],
    "valor": [
        master_path.name,
        master.shape[0],
        master["country_iso3"].nunique(),
        master["variable"].nunique(),
        master["source"].nunique(),
        int(master["year"].min()),
        int(master["year"].max()),
    ],
})

display(style_table(master_summary))
```

**Lectura ejecutiva.** Este control confirma que el notebook está trabajando con el último master persistido y que contiene la estructura mínima esperada: país, año, variable, valor y fuente.

---

## 3. La cobertura bruta mide disponibilidad, pero aún no descuenta antigüedad

La cobertura bruta responde:

> **¿Qué porcentaje de países tiene algún dato disponible para cada variable, sin importar la antigüedad?**

```python
n_countries = master["country_iso3"].nunique()

stats = (
    master.dropna(subset=["value"])
    .groupby("variable")["value"]
    .agg(n_obs="count", mean="mean", std="std", min="min", max="max")
    .reset_index()
)

coverage_var = (
    master.dropna(subset=["value"])
    .groupby("variable")["country_iso3"]
    .nunique()
    .reset_index(name="n_countries_with_data")
)

summary = stats.merge(coverage_var, on="variable", how="left")
summary["n_countries_expected"] = n_countries
summary["n_missing_countries"] = summary["n_countries_expected"] - summary["n_countries_with_data"]
summary["pct_coverage"] = summary["n_countries_with_data"] / summary["n_countries_expected"]
summary["pct_missing"] = 1 - summary["pct_coverage"]

summary = summary.sort_values("pct_missing", ascending=False)

summary_display = summary.copy()
summary_display["pct_coverage"] = summary_display["pct_coverage"] * 100
summary_display["pct_missing"] = summary_display["pct_missing"] * 100

display(style_table(
    summary_display[[
        "variable", "n_obs", "n_countries_with_data", "n_missing_countries",
        "pct_coverage", "pct_missing", "mean", "std", "min", "max"
    ]],
    gradient_cols=["pct_coverage"],
    gradient_cmap="RdYlGn",
    format_dict={
        "n_obs": "{:,.0f}",
        "n_countries_with_data": "{:,.0f}",
        "n_missing_countries": "{:,.0f}",
        "pct_coverage": "{:.1f}%",
        "pct_missing": "{:.1f}%",
        "mean": "{:,.2f}",
        "std": "{:,.2f}",
        "min": "{:,.2f}",
        "max": "{:,.2f}",
    },
))
```

**Interpretación.** Esta tabla identifica variables con menor presencia bruta en países. Sin embargo, todavía no distingue si el dato disponible es reciente o si proviene de años obsoletos.

**Implicación.** Las variables con baja cobertura bruta son candidatas a revisión de fuente, exclusión o imputación. Las variables con alta cobertura bruta todavía deben pasar la prueba de vigencia.

---

## 4. La matriz de cobertura bruta permite identificar patrones de faltantes por país y variable

```python
coverage_matrix = (
    master.dropna(subset=["value"])
    .groupby(["country_iso3", "variable"])
    .size()
    .unstack(fill_value=0)
)

coverage_matrix = (coverage_matrix > 0).astype(int)

coverage_mean = coverage_matrix.mean().mean() * 100
print(f"Cobertura bruta media: {coverage_mean:.1f}%")

fig = px.imshow(
    coverage_matrix.T,
    color_continuous_scale=["#FDECEC", CIBEST["green"]],
    aspect="auto",
    title="La cobertura bruta muestra disponibilidad, pero no vigencia del dato",
)
fig.update_layout(height=900, xaxis_title="País", yaxis_title="Variable")
fig.show()
```

**Lectura ejecutiva.** El heatmap permite detectar si los faltantes se concentran en algunos países, algunas variables o combinaciones específicas.

**Implicación.** Países con múltiples brechas requerirán mayor imputación o podrían ser excluidos si no cumplen la cobertura efectiva mínima después del filtro de frescura.

---

## 5. La regla de máximo 5 años convierte datos antiguos en faltantes efectivos

La cobertura efectiva se calcula después de aplicar la regla:

```text
stale = freshness_reference_year - latest_observed_year > max_data_age_years
```

Con la configuración recomendada:

```text
freshness_reference_year = 2026
max_data_age_years = 5
```

se aceptan datos desde 2021 en adelante y se marcan como stale los datos 2020 o anteriores, salvo variables estáticas o exentas.

```python
wide_values, wide_years = pivot_latest_value_and_year(master)

wide_fresh, stale_report = apply_freshness_filter(
    wide_values=wide_values,
    wide_years=wide_years,
    variable_catalog=catalog,
    settings=configs["settings"],
)

stale_mask = wide_values.notna() & wide_fresh.isna()
stale_cells = int(stale_mask.sum().sum())

freshness_cfg = configs["settings"].get("data_quality", {})
reference_year = freshness_cfg.get("freshness_reference_year")
max_age = freshness_cfg.get("max_data_age_years")

freshness_summary = pd.DataFrame({
    "métrica": ["Año de referencia", "Antigüedad máxima", "Celdas stale", "Variables evaluadas", "Países evaluados"],
    "valor": [reference_year, max_age, stale_cells, wide_values.shape[1], wide_values.shape[0]],
})

display(style_table(freshness_summary))
```

```python
if stale_cells > 0:
    risk_box(
        "Se detectaron datos obsoletos",
        f"{stale_cells} celdas con dato observado fueron marcadas como stale y tratadas como faltantes efectivos antes de imputación."
    )
else:
    success_box(
        "No se detectaron datos obsoletos",
        "Ninguna celda observada superó el umbral de antigüedad configurado."
    )
```

---

## 6. Variables con mayor stale data: mantener, excluir o buscar fuente alternativa

```python
stale_by_variable = (
    stale_mask.sum(axis=0)
    .sort_values(ascending=False)
    .rename("n_stale")
    .reset_index()
    .rename(columns={"index": "variable"})
)

stale_by_variable["pct_stale_countries"] = stale_by_variable["n_stale"] / wide_values.shape[0]
stale_by_variable = stale_by_variable.merge(
    pd.DataFrame.from_dict(catalog, orient="index")[["dimension", "source", "frequency"]].reset_index().rename(columns={"index": "variable"}),
    on="variable",
    how="left",
)

display(style_table(
    stale_by_variable,
    gradient_cols=["pct_stale_countries"],
    gradient_cmap="YlOrRd",
    format_dict={"pct_stale_countries": "{:.1%}", "n_stale": "{:,.0f}"},
))
```

```python
fig = px.bar(
    stale_by_variable.head(20).sort_values("n_stale"),
    x="n_stale",
    y="variable",
    orientation="h",
    color="pct_stale_countries",
    color_continuous_scale=[[0, CIBEST["gold_light"]], [1, CIBEST["red"]]],
    title="Las variables con más datos stale concentran el mayor riesgo de vigencia",
)
fig.update_layout(xaxis_title="Número de países con dato stale", yaxis_title="Variable")
fig.show()
```

**Criterio recomendado de decisión por variable:**

- `pct_stale_countries = 0%`: mantener sin acción.
- `0% < pct_stale_countries <= 20%`: mantener, pero documentar imputación.
- `20% < pct_stale_countries <= 40%`: revisar fuente alternativa o tratamiento específico.
- `> 40%`: candidata a exclusión, sustitución o búsqueda prioritaria de fuente.

```python
def variable_quality_recommendation(pct_stale: float) -> str:
    if pct_stale == 0:
        return "Mantener"
    if pct_stale <= 0.20:
        return "Mantener con documentación"
    if pct_stale <= 0.40:
        return "Revisar fuente alternativa"
    return "Candidata a exclusión o sustitución"

stale_by_variable["recomendacion"] = stale_by_variable["pct_stale_countries"].apply(variable_quality_recommendation)

display(style_table(
    stale_by_variable[["variable", "dimension", "source", "frequency", "n_stale", "pct_stale_countries", "recomendacion"]],
    gradient_cols=["pct_stale_countries"],
    gradient_cmap="YlOrRd",
    format_dict={"pct_stale_countries": "{:.1%}", "n_stale": "{:,.0f}"},
))
```

---

## 7. Países con mayor riesgo de calidad después del filtro de frescura

```python
stale_by_country = (
    stale_mask.sum(axis=1)
    .sort_values(ascending=False)
    .rename("n_stale")
    .reset_index()
    .rename(columns={"index": "country_iso3"})
)

stale_by_country["pct_stale_variables"] = stale_by_country["n_stale"] / wide_values.shape[1]

missing_after_freshness = wide_fresh.isna().sum(axis=1).rename("n_missing_effective").reset_index()
missing_after_freshness = missing_after_freshness.rename(columns={"index": "country_iso3"})
missing_after_freshness["pct_missing_effective"] = missing_after_freshness["n_missing_effective"] / wide_fresh.shape[1]

country_quality = stale_by_country.merge(missing_after_freshness, on="country_iso3", how="left")
country_quality = country_quality.sort_values("pct_missing_effective", ascending=False)

display(style_table(
    country_quality,
    gradient_cols=["pct_missing_effective"],
    gradient_cmap="YlOrRd",
    format_dict={"pct_stale_variables": "{:.1%}", "pct_missing_effective": "{:.1%}", "n_stale": "{:,.0f}", "n_missing_effective": "{:,.0f}"},
))
```

```python
fig = px.bar(
    country_quality.sort_values("pct_missing_effective"),
    x="pct_missing_effective",
    y="country_iso3",
    orientation="h",
    color="pct_missing_effective",
    color_continuous_scale=[[0, CIBEST["green"]], [0.5, CIBEST["gold"]], [1, CIBEST["red"]]],
    title="Países con mayor faltante efectivo después de controlar antigüedad",
)
fig.update_layout(xaxis_title="Faltante efectivo (%)", yaxis_title="País")
fig.show()
```

**Interpretación.** Esta tabla transforma el riesgo de calidad en una métrica accionable por país. Un país puede tener cobertura bruta aceptable, pero perder elegibilidad si muchas variables disponibles son antiguas.

---

## 8. La cobertura efectiva determina exclusión de países antes de imputar

```python
missing_threshold = float(
    configs["settings"].get("data_quality", {}).get(
        "max_missing_pct",
        configs["settings"]["scoring"].get("missing_data_threshold", 0.30),
    )
)

expected_vars = [variable for variable in catalog if variable in wide_fresh.columns]
missing_ratio_effective = wide_fresh[expected_vars].isna().mean(axis=1)

coverage_effective = pd.DataFrame({
    "country_iso3": missing_ratio_effective.index,
    "pct_missing_effective": missing_ratio_effective.values,
})
coverage_effective["pct_coverage_effective"] = 1 - coverage_effective["pct_missing_effective"]
coverage_effective["excluded_by_quality_rule"] = coverage_effective["pct_missing_effective"] > missing_threshold
coverage_effective = coverage_effective.sort_values("pct_coverage_effective")

display(style_table(
    coverage_effective,
    gradient_cols=["pct_coverage_effective"],
    gradient_cmap="RdYlGn",
    format_dict={"pct_missing_effective": "{:.1%}", "pct_coverage_effective": "{:.1%}"},
))
```

```python
excluded_quality = coverage_effective.loc[
    coverage_effective["excluded_by_quality_rule"],
    "country_iso3",
].tolist()

if excluded_quality:
    risk_box(
        "Países excluidos por cobertura efectiva",
        f"Los países {excluded_quality} superan el umbral de faltantes efectivos de {missing_threshold:.0%}."
    )
else:
    success_box(
        "Ningún país supera el umbral de faltantes efectivos",
        f"Todos los países cumplen el mínimo de cobertura efectiva bajo el umbral de {missing_threshold:.0%}."
    )
```

---

## 9. La imputación debe documentarse como riesgo, no ocultarse dentro del score

Este notebook no reemplaza el flujo oficial de limpieza, pero puede ejecutar `run_cleaning()` para auditar qué países quedan excluidos y qué matriz imputada pasaría a scoring.

```python
wide_imputed, excluded_from_cleaning = run_cleaning(master, configs)

imputation_summary = pd.DataFrame({
    "métrica": ["Países después de limpieza", "Variables después de limpieza", "Países excluidos"],
    "valor": [wide_imputed.shape[0], wide_imputed.shape[1], ", ".join(excluded_from_cleaning) if excluded_from_cleaning else "Ninguno"],
})

display(style_table(imputation_summary))
```

```python
# Celdas que eran faltantes efectivas antes de imputar y quedan presentes después.
imputed_candidates = wide_fresh.reindex(index=wide_imputed.index, columns=wide_imputed.columns).isna()
imputed_cells = int(imputed_candidates.sum().sum())

total_cells_scoring_pool = wide_imputed.shape[0] * wide_imputed.shape[1]
pct_imputed_candidates = imputed_cells / total_cells_scoring_pool if total_cells_scoring_pool else 0

imputation_risk = pd.DataFrame({
    "métrica": ["Celdas imputadas potenciales", "% de matriz imputada potencial"],
    "valor": [imputed_cells, pct_imputed_candidates],
})

display(style_table(
    imputation_risk,
    format_dict={"valor": "{}"},
))
```

**Interpretación.** Las celdas imputadas no son errores; son una decisión metodológica para preservar comparabilidad. Pero deben reportarse porque un país con alta imputación tiene menor confiabilidad que uno con datos observados y vigentes.

---

## 10. Auditoría final país-variable: trazabilidad de stale data

```python
age_matrix = int(configs["settings"]["data_quality"]["freshness_reference_year"]) - wide_years

freshness_audit = (
    wide_values.reset_index().melt(id_vars="country_iso3", var_name="variable", value_name="latest_observed_value")
    .merge(
        wide_years.reset_index().melt(id_vars="country_iso3", var_name="variable", value_name="latest_observed_year"),
        on=["country_iso3", "variable"],
        how="left",
    )
    .merge(
        age_matrix.reset_index().melt(id_vars="country_iso3", var_name="variable", value_name="data_age_years"),
        on=["country_iso3", "variable"],
        how="left",
    )
    .merge(
        stale_mask.reset_index().melt(id_vars="country_iso3", var_name="variable", value_name="is_stale_observed"),
        on=["country_iso3", "variable"],
        how="left",
    )
)

freshness_audit["is_missing_original"] = freshness_audit["latest_observed_value"].isna()
freshness_audit["freshness_status"] = "fresh"
freshness_audit.loc[freshness_audit["is_missing_original"], "freshness_status"] = "missing_original"
freshness_audit.loc[freshness_audit["is_stale_observed"], "freshness_status"] = "stale_replaced_by_missing"

output_dir = Path("data/reports")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "freshness_audit_scoring.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    freshness_audit.to_excel(writer, sheet_name="cell_level_audit", index=False)
    stale_by_variable.to_excel(writer, sheet_name="stale_by_variable", index=False)
    country_quality.to_excel(writer, sheet_name="quality_by_country", index=False)
    coverage_effective.to_excel(writer, sheet_name="effective_coverage", index=False)

print(f"Auditoría de frescura exportada en: {output_file}")
```

**Implicación.** Este archivo permite revisar celda por celda qué valor fue considerado vigente, faltante original o stale. Es el soporte metodológico para defender la regla de 5 años.

---

## 11. Hallazgos principales

1. La cobertura bruta no es suficiente para habilitar scoring; debe convertirse en cobertura efectiva después de controlar vigencia.
2. La regla de máximo 5 años evita que datos antiguos entren al score como si fueran actuales.
3. Las variables con mayor `pct_stale_countries` deben clasificarse entre mantener, documentar, buscar fuente alternativa o excluir.
4. Los países con mayor `pct_missing_effective` presentan mayor riesgo de imputación y menor confiabilidad analítica.
5. La imputación preserva comparabilidad, pero debe reportarse explícitamente como riesgo metodológico.
6. La auditoría celda a celda permite demostrar que el scoring no usa valores observados obsoletos de forma directa.
7. La salida de este notebook determina si el master está listo para construir `decision_matrix` en el notebook 03.

---

## 12. Limitaciones

- La regla de 5 años mejora actualidad, pero puede aumentar imputación en variables de baja frecuencia.
- Variables trienales como Findex pueden requerir interpretación distinta frente a variables anuales.
- La cobertura efectiva no mide calidad conceptual de la fuente, solo presencia y vigencia.
- La imputación regional reduce pérdida de países, pero puede suavizar diferencias reales entre mercados.
- Las variables estáticas y estructurales deben quedar explícitamente exentas para evitar castigos incorrectos.

---

## 13. Recomendaciones y próximos pasos

1. Mantener la regla de `max_data_age_years = 5` como estándar mínimo de vigencia.
2. Revisar variables con más de 20% de stale data para buscar fuente alternativa o justificar permanencia.
3. Documentar países con alto faltante efectivo antes de presentar resultados de scoring.
4. Exportar y conservar `freshness_audit_scoring.xlsx` como evidencia metodológica.
5. Pasar al notebook 03 solo si la matriz imputada conserva cobertura suficiente y no depende excesivamente de imputación.
6. Validar en scoring que `gdp_growth` permanezca fuera de TOPSIS y solo alimente `Trend`.

---

## 14. Síntesis Ejecutiva

- La calidad real del dataset debe evaluarse con cobertura efectiva, no solo cobertura bruta.
- El filtro de frescura convierte datos antiguos en faltantes efectivos antes de cobertura e imputación.
- Países y variables con alto stale data requieren documentación y posible revisión de fuente.
- La imputación es aceptable si se reporta; no debe ocultarse dentro del score.
- El master está listo para scoring solo si supera cobertura efectiva, controles de stale data y auditoría de imputación.
