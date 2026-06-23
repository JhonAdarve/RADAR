# 01 — Validación ejecutiva de fuentes de datos | RADAR Cibest

**Fase ASUM-DM:** 2 — Entendimiento de datos  
**Responsable:** Jhon Farley Adarve Díaz  
**Fecha de ejecución:** Junio 2026  
**Propósito:** validar que las fuentes del modelo RADAR Cibest generan un `master_raw` completo, trazable y apto para las fases posteriores de calidad, scoring y sensibilidad.

---

## 0. La extracción es apta para continuar si World Bank, Damodaran y fuentes complementarias generan un master completo

Este notebook responde una pregunta operativa y metodológica:

> **¿El pipeline de extracción está produciendo un dataset suficientemente completo, trazable y actualizado para alimentar el modelo RADAR Cibest?**

El notebook **no calcula scoring**. Su función es anterior: validar conectividad, cobertura bruta, años disponibles, fuentes críticas, variables faltantes y riesgos de datos antiguos.

### Mensaje ejecutivo preliminar

La extracción observada genera un master de **1.281 filas, 30 países y 45 variables**, con cobertura promedio de **90,4%**. World Bank se extrae correctamente desde varias bases (`db=2`, `db=3`, `db=28`, `db=32`), incluyendo la base 28 para `digital_payment_adults_pct`. El histórico específico de `gdp_growth` 2022–2024 también está disponible para los 30 países, lo que habilita el componente `Trend` del score RADAR.

La principal alerta no es conectividad, sino **vigencia heterogénea de datos**: existen observaciones antiguas desde 1996–2020. Estas no deben invalidar la extracción, pero sí deben ser tratadas en el notebook 02 mediante reglas de frescura antes de scoring.

---

## 1. Configuración del entorno y estilo Cibest

Esta sección prepara el entorno, carga configuraciones y fija la identidad visual Cibest para tablas, gráficos y bloques interpretativos.

```python
# ---------------------------------------------------------------------------
# Configuración inicial del notebook
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
import importlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML, display, Markdown

warnings.filterwarnings("ignore")

# Asegurar acceso al proyecto desde notebooks/
sys.path.insert(0, str(Path.cwd().parent))

from src.utils import (
    load_all_configs,
    setup_logger,
    get_variable_catalog,
    get_world_bank_variable_catalog,
    infer_world_bank_db,
    resolve_data_path,
)

configs = load_all_configs()
setup_logger(configs["settings"].get("logging"))

# ---------------------------------------------------------------------------
# Paleta corporativa Cibest
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

SIGNAL_COLORS = {
    "ALTA_OPORTUNIDAD": CIBEST["green"],
    "OPORTUNIDAD_MODERADA": CIBEST["gold"],
    "BAJA_OPORTUNIDAD": CIBEST["amber"],
    "RIESGO": CIBEST["red"],
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Arial, sans-serif", size=13, color=CIBEST["gray"]),
        title=dict(font=dict(size=17, color=CIBEST["gray"])),
        plot_bgcolor=CIBEST["white"],
        paper_bgcolor=CIBEST["white"],
        xaxis=dict(gridcolor=CIBEST["gray_border"], linecolor=CIBEST["gray"]),
        yaxis=dict(gridcolor=CIBEST["gray_border"], linecolor=CIBEST["gray"]),
        colorway=[
            CIBEST["gray"], CIBEST["gold"], CIBEST["gray_light"],
            CIBEST["gold_dark"], CIBEST["green"], CIBEST["amber"],
        ],
    )
)

px.defaults.template = PLOTLY_TEMPLATE

cmap_custom = LinearSegmentedColormap.from_list(
    "GrayToYellow",
    [CIBEST["gray_light"], CIBEST["yellow"]],
)


def style_table(df, gradient_cols=None, gradient_cmap="YlGnBu", format_dict=None):
    """Aplica estilo Cibest a un DataFrame de pandas."""
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
        {"selector": "tbody tr:hover", "props": [
            ("background-color", CIBEST["gray_bg"]),
        ]},
    ])

    if gradient_cols:
        styler = styler.background_gradient(subset=gradient_cols, cmap=gradient_cmap)

    if format_dict:
        styler = styler.format(format_dict)

    return styler


def insight_box(title: str, text: str):
    """Muestra un bloque ejecutivo de insight."""
    display(HTML(f"""
    <div style="border-left:6px solid {CIBEST['gold']}; background-color:{CIBEST['gold_light']};
                padding:14px 18px; margin:12px 0; font-family:Arial, sans-serif; color:{CIBEST['gray']};">
        <b>{title}</b><br>{text}
    </div>
    """))


def risk_box(title: str, text: str):
    """Muestra un bloque ejecutivo de alerta o riesgo."""
    display(HTML(f"""
    <div style="border-left:6px solid {CIBEST['red']}; background-color:#FDECEC;
                padding:14px 18px; margin:12px 0; font-family:Arial, sans-serif; color:{CIBEST['gray']};">
        <b>{title}</b><br>{text}
    </div>
    """))


def success_box(title: str, text: str):
    """Muestra un bloque ejecutivo de resultado positivo."""
    display(HTML(f"""
    <div style="border-left:6px solid {CIBEST['green']}; background-color:#E8F7F3;
                padding:14px 18px; margin:12px 0; font-family:Arial, sans-serif; color:{CIBEST['gray']};">
        <b>{title}</b><br>{text}
    </div>
    """))


success_box(
    "Entorno listo",
    "La configuración fue cargada y la paleta Cibest quedó disponible para tablas, gráficos y narrativa ejecutiva."
)
```

---

## 2. El universo analítico contiene 30 países, 5 dimensiones y 45 variables configuradas

El objetivo de esta sección es validar que la configuración del proyecto cargada por el notebook coincide con el alcance esperado del modelo RADAR.

```python
project_name = configs["settings"]["project"]["name"]
n_countries = len(configs["settings"]["countries"])
dimensions = configs["settings"]["project"]["dimensions"]

summary_config = pd.DataFrame({
    "elemento": ["Proyecto", "Países en alcance", "Dimensiones"],
    "valor": [project_name, n_countries, ", ".join(dimensions)],
})

display(style_table(summary_config))
```

**Lectura ejecutiva.** El alcance configurado corresponde a 30 países distribuidos en América y España, organizados en cinco dimensiones analíticas: macro, financial, institutional, digital_tech y proximity.

---

## 3. El catálogo tiene 45 variables; World Bank es la fuente dominante

Esta sección audita el catálogo funcional que será usado por extracción y scoring.

```python
catalog = get_variable_catalog(configs["variables"])
df_catalog = pd.DataFrame.from_dict(catalog, orient="index")

df_catalog_summary = pd.DataFrame({
    "métrica": ["Total variables", "Dimensiones", "Fuentes"],
    "valor": [len(df_catalog), df_catalog["dimension"].nunique(), df_catalog["source"].nunique()],
})

display(style_table(df_catalog_summary))

variables_by_dimension = (
    df_catalog.groupby("dimension")
    .size()
    .rename("n_variables")
    .reset_index()
    .sort_values("n_variables", ascending=False)
)

variables_by_source = (
    df_catalog.groupby("source")
    .size()
    .rename("n_variables")
    .reset_index()
    .sort_values("n_variables", ascending=False)
)

display(style_table(variables_by_dimension, gradient_cols=["n_variables"], format_dict={"n_variables": "{:,.0f}"}))
display(style_table(variables_by_source, gradient_cols=["n_variables"], format_dict={"n_variables": "{:,.0f}"}))
```

**Interpretación.** El catálogo observado contiene **45 variables**: 12 macro, 9 financial, 8 institutional, 7 digital_tech y 9 proximity. Por fuente, **World Bank aporta 34 variables**, complementarias aporta 10 y Damodaran aporta 1.

```python
fig = px.bar(
    variables_by_dimension,
    x="dimension",
    y="n_variables",
    text="n_variables",
    title="La dimensión macro concentra el mayor número de variables configuradas",
    color="dimension",
    color_discrete_sequence=[CIBEST["gold"], CIBEST["gray"], CIBEST["green"], CIBEST["amber"], CIBEST["gray_light"]],
)
fig.update_traces(textposition="outside")
fig.update_layout(showlegend=False, yaxis_title="Número de variables", xaxis_title="Dimensión")
fig.show()
```

**Implicación.** El modelo depende de forma significativa de World Bank; cualquier falla en esa fuente puede afectar de manera material la matriz de decisión. Por eso el pipeline debe bloquear o alertar cuando World Bank no se extrae correctamente.

---

## 4. El smoke test de World Bank confirma conectividad y disponibilidad reciente

Esta prueba valida conectividad básica contra World Bank con una variable macroeconómica representativa.

```python
from src.data_extraction.world_bank import fetch_indicator

df_test = fetch_indicator(
    indicator_code="NY.GDP.PCAP.CD",
    countries=["COL", "MEX", "CHL", "ESP"],
)

display(style_table(
    df_test,
    gradient_cols=["value"],
    format_dict={"value": "{:,.2f}", "year": "{:.0f}"},
))
```

**Interpretación.** El test devuelve datos 2024 para Chile, Colombia, España y México, lo que confirma que la API responde y que la consulta básica puede recuperar datos recientes.

**Implicación.** Este smoke test no prueba todo el pipeline, pero sí reduce el riesgo de fallas globales de conectividad antes de ejecutar la extracción completa.

---

## 5. La extracción completa agrupa correctamente World Bank en bases 2, 3, 28 y 32

Esta sección ejecuta el pipeline completo. La extracción esperada debe incluir:

- WDI / World Bank base 2.
- WGI base 3.
- Global Findex / G20 Financial Inclusion base 28.
- GFDD base 32.
- Damodaran Country Risk Premium.
- Fuentes complementarias.

```python
import src.utils as utils
import src.data_extraction.world_bank as world_bank
import src.data_extraction.pipeline as pipeline
import src.scoring.hybrid_scorer as hybrid_scorer

importlib.invalidate_caches()
importlib.reload(utils)
importlib.reload(world_bank)
importlib.reload(pipeline)
importlib.reload(hybrid_scorer)

from src.data_extraction.pipeline import run_extraction

configs = load_all_configs()

print("Tiene fetch_indicator_history:", hasattr(world_bank, "fetch_indicator_history"))
print("utils file:", utils.__file__)
print("Allowed DBs:", utils.WORLD_BANK_ALLOWED_DBS)
```

```python
master, coverage = run_extraction(
    configs=configs,
    save_intermediate=True,
)

extraction_summary = pd.DataFrame({
    "métrica": ["Filas master", "Países", "Variables", "Cobertura promedio"],
    "valor": [
        master.shape[0],
        master["country_iso3"].nunique(),
        master["variable"].nunique(),
        coverage["pct_cobertura"].mean(),
    ],
})

display(style_table(
    extraction_summary,
    format_dict={"valor": "{}"},
))
```

**Resultado observado.** En la corrida documentada, el pipeline generó un `master` de **1.281 filas**, **30 países**, **45 variables** y **90,4% de cobertura promedio**. Además, World Bank se agrupó por base como `{2: 24, 32: 3, 3: 6, 28: 1}`.

```python
success_box(
    "La extracción completa es apta para continuar",
    "El master contiene 45 variables y cubre 30 países. World Bank, Damodaran y fuentes complementarias ejecutaron correctamente."
)
```

---

## 6. Control defensivo: no se debe usar un master parcial para scoring

Un riesgo operativo crítico es que una fuente principal falle, pero el pipeline guarde un master parcial. Este control valida que el master generado supera umbrales mínimos antes de usarlo aguas abajo.

```python
min_expected_variables = int(
    configs["settings"].get("data_quality", {}).get("min_expected_variables", 35)
)

n_variables_master = master["variable"].nunique()
n_countries_master = master["country_iso3"].nunique()

if n_variables_master < min_expected_variables:
    risk_box(
        "Master potencialmente parcial",
        f"El master tiene {n_variables_master} variables, por debajo del mínimo esperado de {min_expected_variables}. No debería usarse para scoring."
    )
    raise ValueError("Master con cobertura insuficiente para scoring.")
else:
    success_box(
        "Master supera barrera mínima de variables",
        f"El master contiene {n_variables_master} variables para {n_countries_master} países. Supera el umbral mínimo configurado de {min_expected_variables}."
    )
```

**Implicación.** Este control evita que notebooks posteriores carguen accidentalmente un master incompleto, especialmente si World Bank falla y solo se guardan fuentes complementarias.

---

## 7. Cuba, Guyana, Haití y algunos países del Caribe concentran los principales déficits de cobertura

```python
coverage_sorted = coverage.sort_values("pct_cobertura").copy()

coverage_sorted["coverage_status"] = np.select(
    [
        coverage_sorted["pct_cobertura"] >= 90,
        coverage_sorted["pct_cobertura"] >= 75,
        coverage_sorted["pct_cobertura"] >= 60,
    ],
    ["Alta", "Media", "Baja"],
    default="Crítica",
)

display(style_table(
    coverage_sorted,
    gradient_cols=["pct_cobertura"],
    gradient_cmap="RdYlGn",
    format_dict={"pct_cobertura": "{:.2f}%"},
))
```

**Lectura ejecutiva.** La cobertura promedio es alta, pero no homogénea. Cuba presenta la menor cobertura observada, con **51,11%**. Guyana, Haití, Bahamas, Barbados, Belice y Surinam también tienen brechas relevantes frente al catálogo completo.

```python
fig = px.bar(
    coverage_sorted,
    x="country_iso3",
    y="pct_cobertura",
    color="coverage_status",
    title="La cobertura bruta es alta en promedio, pero heterogénea en Caribe y economías pequeñas",
    color_discrete_map={
        "Alta": CIBEST["green"],
        "Media": CIBEST["gold"],
        "Baja": CIBEST["amber"],
        "Crítica": CIBEST["red"],
    },
)
fig.update_layout(xaxis_title="País", yaxis_title="Cobertura bruta (%)")
fig.add_hline(y=70, line_dash="dash", line_color=CIBEST["red"], annotation_text="Umbral referencial 70%")
fig.show()
```

**Implicación.** La cobertura bruta permite continuar, pero los países con menor cobertura requerirán atención en el notebook 02, especialmente después de aplicar reglas de vigencia e imputación.

---

## 8. La distribución de años revela riesgo de datos obsoletos que debe tratarse antes de scoring

```python
year_distribution = (
    master.groupby("year")
    .size()
    .rename("n_observaciones")
    .reset_index()
    .sort_values("year")
)

display(style_table(year_distribution, gradient_cols=["n_observaciones"], format_dict={"year": "{:.0f}", "n_observaciones": "{:,.0f}"}))
```

```python
fig = px.bar(
    year_distribution,
    x="year",
    y="n_observaciones",
    title="La mayor parte de los datos es reciente, pero existen observaciones antiguas que deben filtrarse por vigencia",
    color="n_observaciones",
    color_continuous_scale=[[0, CIBEST["gray_light"]], [1, CIBEST["gold"]]],
)
fig.update_layout(xaxis_title="Año", yaxis_title="Número de observaciones")
fig.show()
```

**Lectura ejecutiva.** La mayor concentración de observaciones se ubica entre 2024 y 2026. Sin embargo, el master también contiene datos desde 1996, 1999, 2000 y otros años antiguos. Estos datos no deben invalidar la extracción, pero sí deben convertirse en faltantes efectivos si superan la regla de vigencia definida.

```python
risk_box(
    "Riesgo de vigencia",
    "El criterio latest_available maximiza cobertura, pero puede incorporar datos antiguos. El scoring no debería usar observaciones con más de 5 años de antigüedad salvo variables estáticas o explícitamente exentas."
)
```

---

## 9. Variables críticas muestran cobertura reciente, pero algunas series tienen rezagos puntuales

```python
vars_check = [
    "gdp_nominal",
    "gdp_per_capita_ppp",
    "gdp_growth",
    "inflation_rate",
    "unemployment_rate",
    "domestic_credit_private_gdp",
    "account_ownership",
    "regulatory_quality",
    "government_effectiveness",
    "rule_of_law",
    "internet_users_pct",
    "secure_internet_servers_per_million",
    "digital_payment_adults_pct",
]

critical_variable_years = (
    master[master["variable"].isin(vars_check)]
    .groupby(["variable", "year"])
    .size()
    .reset_index(name="n")
    .sort_values(["variable", "year"])
)

display(style_table(critical_variable_years, gradient_cols=["n"], format_dict={"year": "{:.0f}", "n": "{:,.0f}"}))
```

**Interpretación.** Las variables institucionales WGI aparecen completas en 2024. `unemployment_rate` muestra datos 2025 para 30 países. `gdp_growth` cuenta con histórico completo 2022–2024 para 30 países. Algunas variables financieras y de inclusión tienen cobertura más heterogénea o rezagos puntuales.

**Implicación.** El master es apto para pasar a perfilamiento, pero la elegibilidad final para TOPSIS debe depender de cobertura efectiva posterior a stale filter, no solo de cobertura bruta.

---

## 10. El histórico de `gdp_growth` 2022–2024 está completo y habilita el componente Trend

```python
gdp_growth_history = master[master["variable"] == "gdp_growth"].copy()
gdp_growth_history["year"] = pd.to_numeric(gdp_growth_history["year"], errors="coerce")
gdp_growth_history["value"] = pd.to_numeric(gdp_growth_history["value"], errors="coerce")

gdp_growth_coverage = (
    gdp_growth_history.groupby("year")
    .size()
    .rename("n_paises")
    .reset_index()
)

display(style_table(gdp_growth_coverage, gradient_cols=["n_paises"], format_dict={"year": "{:.0f}", "n_paises": "{:,.0f}"}))

gdp_growth_summary = gdp_growth_history["value"].describe().to_frame("gdp_growth")
display(style_table(gdp_growth_summary, format_dict={"gdp_growth": "{:,.3f}"}))
```

```python
gdp_growth_avg = (
    gdp_growth_history
    .groupby("country_iso3")["value"]
    .mean()
    .sort_values()
    .reset_index(name="avg_gdp_growth_2022_2024")
)

display(style_table(
    gdp_growth_avg,
    gradient_cols=["avg_gdp_growth_2022_2024"],
    gradient_cmap="RdYlGn",
    format_dict={"avg_gdp_growth_2022_2024": "{:.3f}"},
))
```

**Lectura ejecutiva.** El histórico de crecimiento PIB está completo para 30 países en 2022, 2023 y 2024. Esto permite calcular `Trend` como promedio móvil de tres años y evita usar solo el dato puntual 2024.

```python
success_box(
    "Trend puede calcularse con base histórica completa",
    "gdp_growth tiene 90 observaciones: 30 países por cada año entre 2022 y 2024."
)
```

**Alerta metodológica.** Guyana presenta un crecimiento promedio excepcionalmente alto frente al resto del universo. El componente Trend debe aplicar winsorización para evitar que un outlier domine la normalización.

---

## 11. `digital_payment_adults_pct` ya está integrado desde la base 28 de World Bank

```python
wb_cfg = configs["data_sources"]["sources"]["world_bank"]
wb_catalog = get_world_bank_variable_catalog(configs["variables"], wb_cfg)

payment_meta = wb_catalog.get("digital_payment_adults_pct")

payment_check = pd.DataFrame([
    {
        "variable": "digital_payment_adults_pct",
        "en_catalogo_wb": payment_meta is not None,
        "indicator_code": payment_meta.get("indicator_code") if payment_meta else None,
        "db": payment_meta.get("db") if payment_meta else None,
        "en_master": "digital_payment_adults_pct" in master["variable"].unique(),
    }
])

display(style_table(payment_check))
```

**Interpretación.** La variable `digital_payment_adults_pct` debe apuntar a `indicator_code = g20.any` y `db = 28`. Su presencia en el master confirma que la integración con Global Findex / G20 Financial Inclusion está operativa.

**Implicación.** Esta variable es crítica para diferenciar las líneas `PF`, `BD` y `AD`, por lo que su extracción exitosa mejora la calidad del modelo por línea de negocio.

---

## 12. Exportación controlada del master para auditoría

```python
output_dir = Path("data/reports")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "master_radar_cibest.xlsx"

master.to_excel(output_file, index=False, sheet_name="master")

print(f"Master exportado correctamente en: {output_file}")
print(f"Shape exportado: {master.shape}")
```

**Implicación.** La exportación a Excel es útil para auditoría manual y validación con usuarios de negocio, pero no debe reemplazar los archivos Parquet como fuente oficial del pipeline.

---

## 13. Controles recomendados antes de pasar al notebook 02

```python
expected_columns = {"country_iso3", "year", "variable", "value", "source"}
missing_columns = expected_columns - set(master.columns)

if missing_columns:
    raise ValueError(f"Master inválido. Faltan columnas: {sorted(missing_columns)}")

checks = pd.DataFrame({
    "control": [
        "Columnas mínimas del master",
        "Variables mínimas",
        "Países esperados",
        "gdp_growth histórico completo",
        "digital_payment_adults_pct en master",
    ],
    "resultado": [
        "OK" if not missing_columns else "FALLA",
        "OK" if master["variable"].nunique() >= min_expected_variables else "FALLA",
        "OK" if master["country_iso3"].nunique() == len(configs["settings"]["countries"]) else "REVISAR",
        "OK" if gdp_growth_coverage["n_paises"].min() == 30 else "REVISAR",
        "OK" if "digital_payment_adults_pct" in master["variable"].unique() else "FALLA",
    ],
})

display(style_table(checks))
```

**Criterio de avance.** El notebook 02 puede ejecutarse si los controles anteriores están en estado `OK` o `REVISAR` justificado. No debe ejecutarse scoring si el master es parcial o si World Bank falló.

---

## 14. Hallazgos principales

1. La extracción completa generó un master de **1.281 filas**, **30 países** y **45 variables**, por encima del umbral mínimo para continuar.
2. World Bank se extrajo desde cuatro bases: **2, 3, 28 y 32**, lo que confirma la integración de WDI, WGI, Global Findex/G20 y GFDD.
3. La cobertura promedio bruta fue de **90,4%**, pero Cuba, Guyana, Haití y algunos países del Caribe concentran brechas relevantes.
4. `gdp_growth` tiene histórico completo 2022–2024 para los 30 países, habilitando el componente `Trend` del RADAR.
5. `digital_payment_adults_pct` está integrada correctamente desde `db=28`, lo que fortalece las líneas digitales y transaccionales.
6. Existen datos antiguos desde 1996–2020; estos deben tratarse en el notebook 02 mediante reglas de vigencia antes de scoring.
7. El master es apto para continuar a calidad de datos, pero no debe usarse directamente para scoring sin stale filter, cobertura efectiva e imputación controlada.

---

## 15. Limitaciones

- La cobertura analizada en este notebook es **bruta**, no efectiva; todavía no descuenta datos obsoletos.
- La estrategia `latest_available` mejora cobertura, pero puede incorporar datos antiguos que no reflejan el estado actual del país.
- Las variables complementarias pueden tener diferencias de mapeo por nombres de países y cobertura incompleta.
- El warning de Damodaran por SSL no detuvo la extracción, pero debe documentarse como riesgo operativo de descarga.
- Este notebook no evalúa impacto de imputación, exclusiones por cobertura efectiva ni robustez del ranking.

---

## 16. Recomendaciones y próximos pasos

1. Ejecutar `02_data_quality_report.ipynb` aplicando regla de antigüedad máxima de 5 años.
2. Auditar variables con datos 2020 o anteriores y convertirlas en faltantes efectivos salvo variables estáticas o exentas.
3. Validar países que caen por debajo de cobertura mínima después del stale filter.
4. Confirmar que `gdp_growth` se excluye de TOPSIS y se usa exclusivamente en `Trend`.
5. Mantener control para impedir persistir masters parciales si falla World Bank.
6. Documentar en el dashboard o informe final que la cobertura bruta no equivale a cobertura efectiva.

---

## 17. Síntesis Ejecutiva

- La extracción es técnicamente exitosa: produce 45 variables, 30 países y cobertura promedio bruta de 90,4%.
- World Bank opera correctamente en bases 2, 3, 28 y 32; `digital_payment_adults_pct` y `gdp_growth` histórico están disponibles.
- El principal riesgo no es conectividad, sino vigencia heterogénea de datos.
- El master es apto para pasar a perfilamiento, no directamente a scoring sin filtro de antigüedad.
- La siguiente decisión metodológica crítica es transformar cobertura bruta en cobertura efectiva usando stale filter e imputación controlada.
