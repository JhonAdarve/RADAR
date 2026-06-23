# 04 — Robustez y sensibilidad del score RADAR | RADAR Cibest

**Fase ASUM-DM:** 5 — Evaluación  
**Responsable:** Jhon Farley Adarve Díaz  
**Fecha de ejecución:** Junio 2026  
**Propósito:** evaluar la robustez de rankings y prioridades país/línea bajo perturbaciones razonables de pesos, comparando sensibilidad determinística, TOPSIS vs VIKOR, Monte Carlo TOPSIS y Monte Carlo RADAR.

---

## 0. La decisión final debe validarse sobre RADAR, no solo sobre TOPSIS

Este notebook responde la pregunta de robustez ejecutiva:

> **¿Qué países mantienen su prioridad cuando se perturban razonablemente los pesos del modelo?**

El análisis distingue dos niveles:

1. **Robustez TOPSIS:** estabilidad del componente estructural ante incertidumbre en pesos de dimensiones y variables.
2. **Robustez RADAR:** estabilidad del score final ante incertidumbre en TOPSIS y en la mezcla `alpha`, `beta`, `gamma`.

La lectura ejecutiva debe privilegiar Monte Carlo RADAR, porque esa es la métrica de decisión final:

```text
RADAR = alpha * TOPSIS + beta * IPC + gamma * Trend
```

### Mensaje ejecutivo preliminar

Un país no debe considerarse prioritario solo por ocupar un lugar alto en el ranking base. Debe clasificarse según:

- ranking base;
- ranking medio Monte Carlo;
- volatilidad del ranking;
- probabilidad de pertenecer al Top-N;
- probabilidad de quedar en banda alta o media-alta;
- estabilidad de correlaciones entre líneas;
- sensibilidad de la decisión frente a cambios de pesos.

---

## 1. Configuración del entorno y estilo Cibest

```python
# ---------------------------------------------------------------------------
# Configuración inicial del notebook
# ---------------------------------------------------------------------------
import sys
import re
from pathlib import Path
from typing import Optional, Union, Dict, List
import importlib
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, display, Markdown

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent))

import src
import src.scoring.hybrid_scorer
import src.scoring.sensitivity
import src.scoring.monte_carlo as monte_carlo

importlib.invalidate_caches()
importlib.reload(src.scoring.hybrid_scorer)
importlib.reload(src.scoring.sensitivity)
importlib.reload(monte_carlo)

from src.utils import (
    load_all_configs,
    get_variable_catalog,
    resolve_data_path,
    setup_logger,
)

from src.scoring.sensitivity import (
    run_sensitivity_analysis,
    compare_topsis_vikor,
)

from src.scoring.hybrid_scorer import (
    prepare_decision_matrix,
    run_full_scoring,
)

from src.scoring.weighting import compute_hierarchical_weights

from src.scoring.monte_carlo import (
    coerce_component_series,
    run_monte_carlo_topsis_robustness,
    run_monte_carlo_radar_robustness,
)

configs = load_all_configs()
setup_logger(configs["settings"].get("logging"))
catalog = get_variable_catalog(configs["variables"])

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

success_box("Entorno listo", "Configuración, módulos de sensibilidad y Monte Carlo cargados correctamente.")
```

---

## 2. Se carga el master vigente y se prepara la matriz de decisión

```python
raw_dir = resolve_data_path(configs["settings"]["data"]["raw_path"])
pattern = re.compile(r"^master_raw_\d{8}\.parquet$")

master_files = sorted(
    [
        path for path in raw_dir.glob("master_raw_*.parquet")
        if pattern.match(path.name)
    ],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not master_files:
    raise FileNotFoundError("Falta master_raw_YYYYMMDD.parquet. Ejecute primero notebook 01.")

master_path = master_files[0]
master = pd.read_parquet(master_path)

master = master.copy()
master["country_iso3"] = master["country_iso3"].astype(str).str.strip()
master["variable"] = master["variable"].astype(str).str.strip()
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

wide_raw, decision_matrix, excluded = prepare_decision_matrix(master, configs)

matrix_summary = pd.DataFrame({
    "métrica": ["wide_raw", "decision_matrix", "Países excluidos", "gdp_growth en decision_matrix"],
    "valor": [
        str(wide_raw.shape),
        str(decision_matrix.shape),
        ", ".join(excluded) if excluded else "Ninguno",
        "Sí" if "gdp_growth" in decision_matrix.columns else "No",
    ],
})

display(style_table(matrix_summary))

if "gdp_growth" in decision_matrix.columns:
    raise ValueError("gdp_growth no debe estar en decision_matrix. Debe alimentar solo el componente Trend.")
```

**Lectura ejecutiva.** La sensibilidad debe ejecutarse sobre la misma matriz que alimenta scoring. `decision_matrix` debe excluir variables de IPC, auxiliares y `gdp_growth`, mientras `wide_raw` conserva insumos para auditoría, proximidad y tendencia.

---

## 3. Sensibilidad determinística de pesos: primera lectura de fragilidad estructural

Este análisis evalúa cómo cambia el ranking TOPSIS bajo perturbaciones de pesos configuradas en `src.scoring.sensitivity`.

```python
sens = run_sensitivity_analysis(
    decision_matrix=decision_matrix,
    dimension_weights=configs["weights"]["dimension_weights"],
    variable_weights_by_dim=configs["weights"]["variable_weights"],
    variable_catalog=catalog,
)

display(style_table(sens if isinstance(sens, pd.DataFrame) else pd.DataFrame(sens)))
```

**Interpretación.** Esta sensibilidad es útil como diagnóstico inicial, pero no reemplaza Monte Carlo. Sirve para identificar si el ranking estructural es muy sensible a cambios puntuales de pesos.

---

## 4. TOPSIS vs VIKOR: comparación de métodos multicriterio

```python
var_weights = compute_hierarchical_weights(
    configs["weights"]["dimension_weights"],
    configs["weights"]["variable_weights"],
)

var_weights = {
    variable: weight
    for variable, weight in var_weights.items()
    if variable in decision_matrix.columns
}

comparison = compare_topsis_vikor(decision_matrix, var_weights, catalog)

display(style_table(
    comparison.head(20),
    gradient_cols=[col for col in comparison.columns if "rank" in col.lower() or "score" in col.lower()],
    gradient_cmap="RdYlGn",
))
```

**Lectura ejecutiva.** TOPSIS y VIKOR no optimizan exactamente la misma lógica. Si los rankings son similares, hay mayor robustez metodológica. Si divergen, los países afectados deben analizarse como decisiones sensibles al método multicriterio.

---

## 5. El scoring base genera IPC, Trend y ranking RADAR para usar como escenario central

```python
results = run_full_scoring(master, configs, persist=True)

ipc_scores = coerce_component_series(
    results["ipc"],
    value_col=None,
    component_name="ipc",
)

trend_scores = coerce_component_series(
    results["trend"],
    value_col="trend",
    component_name="trend",
)

base_summary = pd.DataFrame({
    "métrica": ["Países RADAR", "IPC disponible", "Trend disponible", "Alpha", "Beta", "Gamma"],
    "valor": [
        len(results["radar_global"]),
        ipc_scores.notna().sum(),
        trend_scores.notna().sum(),
        results["composite_weights"]["alpha"],
        results["composite_weights"]["beta"],
        results["composite_weights"]["gamma"],
    ],
})

display(style_table(base_summary))
```

**Implicación.** El ranking base es el escenario central. Monte Carlo no lo reemplaza; evalúa si sus conclusiones son estables ante incertidumbre razonable en pesos.

---

## 6. Monte Carlo TOPSIS mide robustez estructural por línea

```python
mc_cfg = configs["settings"].get("monte_carlo", {})

mc_topsis = run_monte_carlo_topsis_robustness(
    decision_matrix=decision_matrix,
    configs=configs,
    variable_catalog=catalog,
    n_simulations=int(mc_cfg.get("n_simulations", 1000)),
    dimension_concentration=float(mc_cfg.get("topsis", {}).get("dimension_concentration", 150)),
    variable_concentration=float(mc_cfg.get("topsis", {}).get("variable_concentration", 100)),
    random_seed=int(mc_cfg.get("random_seed", 42)),
)

mc_topsis_summary = pd.DataFrame({
    "métrica": ["Filas simuladas", "Líneas", "Países", "Simulaciones"],
    "valor": [
        mc_topsis["simulation_long"].shape[0],
        mc_topsis["simulation_long"]["business_line"].nunique(),
        mc_topsis["simulation_long"]["country_iso3"].nunique(),
        mc_topsis["simulation_long"]["simulation_id"].nunique(),
    ],
})

display(style_table(mc_topsis_summary))
```

**Interpretación.** Monte Carlo TOPSIS responde si el atractivo estructural por línea es estable cuando se perturban pesos de dimensiones y variables. No incorpora IPC ni Trend.

---

## 7. Monte Carlo RADAR mide robustez de la decisión final

```python
mc_radar = run_monte_carlo_radar_robustness(
    decision_matrix=decision_matrix,
    configs=configs,
    variable_catalog=catalog,
    ipc_scores=ipc_scores,
    trend_scores=trend_scores,
    n_simulations=int(mc_cfg.get("n_simulations", 1000)),
    dimension_concentration=float(mc_cfg.get("topsis", {}).get("dimension_concentration", 150)),
    variable_concentration=float(mc_cfg.get("topsis", {}).get("variable_concentration", 100)),
    composite_concentration=float(mc_cfg.get("radar", {}).get("composite_concentration", 150)),
    perturb_composite_weights=bool(mc_cfg.get("radar", {}).get("perturb_composite_weights", True)),
    random_seed=int(mc_cfg.get("random_seed", 42)),
)

mc_radar_summary = pd.DataFrame({
    "métrica": ["Filas simuladas", "Líneas", "Países", "Simulaciones"],
    "valor": [
        mc_radar["simulation_long"].shape[0],
        mc_radar["simulation_long"]["business_line"].nunique(),
        mc_radar["simulation_long"]["country_iso3"].nunique(),
        mc_radar["simulation_long"]["simulation_id"].nunique(),
    ],
})

display(style_table(mc_radar_summary))
```

```python
expected_rows = (
    int(mc_cfg.get("n_simulations", 1000))
    * mc_radar["simulation_long"]["business_line"].nunique()
    * mc_radar["simulation_long"]["country_iso3"].nunique()
)

if mc_radar["simulation_long"].shape[0] != expected_rows:
    risk_box(
        "Revisar tamaño de simulación",
        f"Se esperaban {expected_rows:,} filas y se obtuvieron {mc_radar['simulation_long'].shape[0]:,}."
    )
else:
    success_box(
        "Monte Carlo RADAR completado",
        f"La simulación produjo {expected_rows:,} observaciones país-línea-escenario."
    )
```

**Lectura ejecutiva.** Monte Carlo RADAR es la prueba principal de robustez porque evalúa la métrica final usada para decisión: TOPSIS + IPC + Trend.

---

## 8. Ranking medio y volatilidad: países robustos vs sensibles

```python
radar_rank_robustness = mc_radar["rank_robustness"]

def classify_rank_robustness(row: pd.Series) -> str:
    if row.get("prob_top_5", 0) >= 0.80 and row.get("Alta", 0) >= 0.70:
        return "Prioridad robusta"
    if row.get("prob_top_10", 0) >= 0.70 and (row.get("Alta", 0) + row.get("Media-alta", 0)) >= 0.70:
        return "Prioridad condicional"
    if row.get("std_rank", 999) >= 4:
        return "Alta sensibilidad"
    return "Prioridad no robusta"

radar_topn = mc_radar["topn_probabilities"]
radar_tiers = mc_radar["tier_probabilities"]

radar_robustness_exec = (
    radar_rank_robustness
    .merge(radar_topn, on=["business_line", "country_iso3"], how="left")
    .merge(radar_tiers, on=["business_line", "country_iso3"], how="left")
)

radar_robustness_exec["robustness_class"] = radar_robustness_exec.apply(classify_rank_robustness, axis=1)

display(style_table(
    radar_robustness_exec.query("business_line == 'PF'").sort_values("mean_rank").head(15),
    gradient_cols=["mean_rank", "std_rank", "prob_top_5", "Alta"],
    gradient_cmap="RdYlGn",
    format_dict={
        "mean_rank": "{:.2f}",
        "median_rank": "{:.2f}",
        "std_rank": "{:.2f}",
        "p10_rank": "{:.1f}",
        "p90_rank": "{:.1f}",
        "prob_top_3": "{:.1%}",
        "prob_top_5": "{:.1%}",
        "prob_top_10": "{:.1%}",
        "Alta": "{:.1%}",
        "Media-alta": "{:.1%}",
        "Media": "{:.1%}",
        "Baja": "{:.1%}",
    },
))
```

**Interpretación.** Un país robusto no es solo un país con buen rank base. Es un país con alta probabilidad de conservar posiciones superiores bajo cambios razonables de pesos.

---

## 9. Probabilidad Top-N: métrica ejecutiva superior al ranking puntual

```python
for business_line in sorted(radar_topn["business_line"].unique()):
    display(Markdown(f"### {business_line} — Probabilidad Top-N RADAR"))
    display(style_table(
        radar_topn.query("business_line == @business_line").sort_values("prob_top_5", ascending=False).head(15),
        gradient_cols=["prob_top_3", "prob_top_5", "prob_top_10", "prob_top_15"],
        gradient_cmap="RdYlGn",
        format_dict={
            "prob_top_3": "{:.1%}",
            "prob_top_5": "{:.1%}",
            "prob_top_10": "{:.1%}",
            "prob_top_15": "{:.1%}",
        },
    ))
```

**Implicación.** Para comité ejecutivo, `prob_top_5` y `prob_top_10` son más defendibles que una posición puntual, especialmente cuando existen empates prácticos.

---

## 10. Probabilidad por banda: estabilidad de atractividad, no solo de posición

```python
for business_line in sorted(radar_tiers["business_line"].unique()):
    display(Markdown(f"### {business_line} — Probabilidad por banda RADAR"))
    display(style_table(
        radar_tiers.query("business_line == @business_line").sort_values(["Alta", "Media-alta"], ascending=False).head(15),
        gradient_cols=["Alta", "Media-alta", "Media", "Baja"],
        gradient_cmap="RdYlGn",
        format_dict={"Alta": "{:.1%}", "Media-alta": "{:.1%}", "Media": "{:.1%}", "Baja": "{:.1%}"},
    ))
```

**Lectura ejecutiva.** Un país puede no tener alta probabilidad de Top 5, pero sí alta probabilidad de permanecer en banda alta o media-alta. Esa distinción es clave para priorización exploratoria.

---

## 11. Estabilidad de correlaciones entre líneas: la diferenciación debe sobrevivir al Monte Carlo

```python
line_corr_radar = mc_radar["line_correlation_robustness"]

display(style_table(
    line_corr_radar,
    gradient_cols=["mean_spearman", "p10_spearman", "p90_spearman"],
    gradient_cmap="YlOrRd",
    format_dict={
        "mean_spearman": "{:.3f}",
        "median_spearman": "{:.3f}",
        "std_spearman": "{:.3f}",
        "p10_spearman": "{:.3f}",
        "p90_spearman": "{:.3f}",
        "min_spearman": "{:.3f}",
        "max_spearman": "{:.3f}",
    },
))
```

```python
pf_bd_corr = line_corr_radar[
    ((line_corr_radar["line_a"] == "PF") & (line_corr_radar["line_b"] == "BD"))
    | ((line_corr_radar["line_a"] == "BD") & (line_corr_radar["line_b"] == "PF"))
]

display(style_table(pf_bd_corr, format_dict={col: "{:.3f}" for col in pf_bd_corr.select_dtypes(include="number").columns}))
```

**Interpretación.** Si PF–BD mantiene correlación alta en Monte Carlo, la cercanía entre pagos y banca digital es estructural, no un artefacto de pesos. Si la correlación se vuelve inestable, la diferenciación depende demasiado de la parametrización.

---

## 12. Distribución de alpha, beta y gamma: la mezcla RADAR también es un supuesto

```python
composite_distribution = mc_radar["composite_weight_distribution"]
display(style_table(composite_distribution))
```

**Lectura ejecutiva.** La simulación perturba `alpha`, `beta` y `gamma` alrededor de la mezcla base. Esto permite evaluar si la decisión final depende excesivamente de la ponderación entre estructura, proximidad y tendencia.

---

## 13. Tabla ejecutiva final por línea: ranking base + robustez Monte Carlo

```python
def build_base_radar_by_line(
    radar_by_line,
    countries_eval: Optional[List[str]] = None,
    business_lines: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Convierte results['radar_by_line'] en DataFrames por línea."""

    out: Dict[str, pd.DataFrame] = {}

    if isinstance(radar_by_line, pd.DataFrame):
        df = radar_by_line.copy()
        if "country_iso3" not in df.columns:
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "country_iso3"})
        if "country_iso3" not in df.columns:
            raise ValueError("radar_by_line no contiene country_iso3.")

        df["country_iso3"] = df["country_iso3"].astype(str).str.strip()

        if business_lines is None:
            business_lines = [col for col in df.columns if col != "country_iso3"]

        for business_line in business_lines:
            if business_line not in df.columns:
                continue
            tmp = df[["country_iso3", business_line]].copy()
            tmp = tmp.rename(columns={business_line: "radar_score"})
            tmp["radar_score"] = pd.to_numeric(tmp["radar_score"], errors="coerce")
            tmp = tmp.dropna(subset=["radar_score"])
            tmp["rank"] = tmp["radar_score"].rank(ascending=False, method="min").astype(int)
            out[business_line] = tmp.sort_values("rank").reset_index(drop=True)

        return out

    if isinstance(radar_by_line, dict):
        for business_line, obj in radar_by_line.items():
            if business_line == "country_iso3":
                continue

            if isinstance(obj, pd.DataFrame):
                tmp = obj.copy()
                if "country_iso3" not in tmp.columns:
                    tmp = tmp.reset_index()
                    if "index" in tmp.columns:
                        tmp = tmp.rename(columns={"index": "country_iso3"})
                score_col = next((col for col in ["radar_score", "score", "final_score", business_line, "value"] if col in tmp.columns), None)
                if score_col is None:
                    raise ValueError(f"No se pudo inferir score para {business_line}.")
                tmp = tmp[["country_iso3", score_col]].rename(columns={score_col: "radar_score"})

            elif isinstance(obj, pd.Series):
                if not pd.api.types.is_integer_dtype(obj.index):
                    tmp = obj.rename("radar_score").reset_index()
                    if tmp.columns[0] != "country_iso3":
                        tmp = tmp.rename(columns={tmp.columns[0]: "country_iso3"})
                else:
                    if countries_eval is None:
                        raise ValueError(f"{business_line} es Series con índice numérico; pase countries_eval o corrija results['radar_by_line'].")
                    if len(obj) != len(countries_eval):
                        raise ValueError(f"Longitud incompatible para {business_line}.")
                    tmp = pd.DataFrame({"country_iso3": countries_eval, "radar_score": obj.values})
            else:
                raise TypeError(f"Tipo no soportado para {business_line}: {type(obj)}")

            tmp["country_iso3"] = tmp["country_iso3"].astype(str).str.strip()
            tmp["radar_score"] = pd.to_numeric(tmp["radar_score"], errors="coerce")
            tmp = tmp.dropna(subset=["radar_score"])
            tmp["rank"] = tmp["radar_score"].rank(ascending=False, method="min").astype(int)
            out[business_line] = tmp.sort_values("rank").reset_index(drop=True)

        return out

    raise TypeError(f"Tipo no soportado para radar_by_line: {type(radar_by_line)}")


def build_mc_executive_table(
    base_ranking: Union[pd.DataFrame, pd.Series],
    mc_rank_robustness: pd.DataFrame,
    mc_topn: pd.DataFrame,
    mc_tiers: pd.DataFrame,
    business_line: str,
    score_col: Optional[str] = "radar_score",
) -> pd.DataFrame:
    """Construye tabla ejecutiva de robustez Monte Carlo para una línea."""

    base = base_ranking.copy()

    if isinstance(base, pd.Series):
        if pd.api.types.is_integer_dtype(base.index):
            raise ValueError("base_ranking es Series con índice numérico. Use DataFrame con country_iso3.")
        base = base.rename("radar_score").reset_index()
        if base.columns[0] != "country_iso3":
            base = base.rename(columns={base.columns[0]: "country_iso3"})

    if "country_iso3" not in base.columns:
        base = base.reset_index()
        if "index" in base.columns:
            base = base.rename(columns={"index": "country_iso3"})

    if score_col not in base.columns:
        candidate = next((col for col in ["radar_score", "score", "final_score", "value"] if col in base.columns), None)
        if candidate is None:
            raise ValueError(f"No se pudo inferir score. Columnas: {list(base.columns)}")
        score_col = candidate

    base = base[["country_iso3", score_col]].rename(columns={score_col: "base_radar_score"})
    base["country_iso3"] = base["country_iso3"].astype(str).str.strip()
    base["base_rank"] = base["base_radar_score"].rank(ascending=False, method="min").astype(int)

    robustness = mc_rank_robustness.query("business_line == @business_line").copy()
    topn = mc_topn.query("business_line == @business_line").copy()
    tiers = mc_tiers.query("business_line == @business_line").copy()

    for df in [robustness, topn, tiers]:
        df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
        df["business_line"] = df["business_line"].astype(str).str.strip()

    out = (
        base
        .merge(robustness, on="country_iso3", how="left")
        .merge(topn, on=["business_line", "country_iso3"], how="left")
        .merge(tiers, on=["business_line", "country_iso3"], how="left")
    )

    return out.sort_values("base_rank").reset_index(drop=True)

base_radar_by_line = build_base_radar_by_line(results["radar_by_line"])

mc_exec_by_line = {}
for business_line, base_df in base_radar_by_line.items():
    mc_exec_by_line[business_line] = build_mc_executive_table(
        base_ranking=base_df,
        mc_rank_robustness=mc_radar["rank_robustness"],
        mc_topn=mc_radar["topn_probabilities"],
        mc_tiers=mc_radar["tier_probabilities"],
        business_line=business_line,
        score_col="radar_score",
    )

for business_line, exec_table in mc_exec_by_line.items():
    display(Markdown(f"### {business_line} — Tabla ejecutiva Monte Carlo RADAR"))
    display(style_table(
        exec_table.head(15),
        gradient_cols=["base_radar_score", "mean_rank", "std_rank", "prob_top_5", "Alta"],
        gradient_cmap="RdYlGn",
        format_dict={
            "base_radar_score": "{:.3f}",
            "base_rank": "{:.0f}",
            "mean_rank": "{:.2f}",
            "median_rank": "{:.2f}",
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

**Interpretación.** Esta tabla conecta el ranking base con su robustez probabilística. Es el output más útil para comité porque traduce posiciones puntuales en probabilidades de mantenerse en bandas o Top-N.

---

## 14. Países robustos, condicionales y sensibles

```python
robustness_counts = (
    radar_robustness_exec
    .groupby(["business_line", "robustness_class"])
    .size()
    .rename("n_countries")
    .reset_index()
)

display(style_table(robustness_counts, gradient_cols=["n_countries"], format_dict={"n_countries": "{:,.0f}"}))

fig = px.bar(
    robustness_counts,
    x="business_line",
    y="n_countries",
    color="robustness_class",
    title="Clasificación de países según robustez Monte Carlo RADAR",
    color_discrete_map={
        "Prioridad robusta": CIBEST["green"],
        "Prioridad condicional": CIBEST["gold"],
        "Alta sensibilidad": CIBEST["amber"],
        "Prioridad no robusta": CIBEST["red"],
    },
)
fig.update_layout(xaxis_title="Línea de negocio", yaxis_title="Número de países")
fig.show()
```

**Lectura ejecutiva.** Esta clasificación evita tratar todos los países top como equivalentes. Una prioridad robusta puede avanzar a análisis comercial; una prioridad condicional requiere validación adicional de drivers, riesgos y supuestos.

---

## 15. Persistencia de resultados para trazabilidad

```python
scores_dir = resolve_data_path(configs["settings"]["data"]["scores_path"])
scores_dir.mkdir(parents=True, exist_ok=True)

mc_radar["rank_robustness"].to_parquet(scores_dir / "mc_radar_rank_robustness_latest.parquet", index=False)
mc_radar["topn_probabilities"].to_parquet(scores_dir / "mc_radar_topn_probabilities_latest.parquet", index=False)
mc_radar["tier_probabilities"].to_parquet(scores_dir / "mc_radar_tier_probabilities_latest.parquet", index=False)
mc_radar["line_correlation_robustness"].to_parquet(scores_dir / "mc_radar_line_correlation_robustness_latest.parquet", index=False)
mc_topsis["rank_robustness"].to_parquet(scores_dir / "mc_topsis_rank_robustness_latest.parquet", index=False)

for business_line, exec_table in mc_exec_by_line.items():
    exec_table.to_parquet(scores_dir / f"mc_exec_{business_line}_latest.parquet", index=False)

print(f"Resultados Monte Carlo persistidos en: {scores_dir}")
```

**Implicación.** Los resultados de sensibilidad deben persistirse para reproducibilidad, comparación entre corridas y construcción posterior de visualizaciones ejecutivas.

---

## 16. Hallazgos principales

1. La robustez ejecutiva debe evaluarse sobre RADAR completo, no solo sobre TOPSIS.
2. Monte Carlo TOPSIS mide estabilidad estructural; Monte Carlo RADAR mide estabilidad de la decisión final.
3. La probabilidad Top-N y la probabilidad por banda son más útiles que el ranking puntual para decisiones de comité.
4. Países con alta volatilidad de ranking deben tratarse como prioridades condicionales, aunque aparezcan altos en el escenario base.
5. La estabilidad de correlaciones entre líneas permite saber si la diferenciación de tesis de negocio sobrevive a cambios de pesos.
6. `alpha`, `beta` y `gamma` son supuestos del modelo; simularlos evita sobreconfiar en una mezcla fija.
7. Las tablas ejecutivas por línea son el insumo principal para priorización robusta de mercados.

---

## 17. Limitaciones

- Monte Carlo perturba pesos, no datos. No captura error de medición en variables, IPC o Trend.
- Las distribuciones Dirichlet dependen de parámetros de concentración; estos deben documentarse como supuestos.
- Una probabilidad alta de Top-N no implica factibilidad comercial o regulatoria inmediata.
- La simulación evalúa robustez interna del modelo, no escenarios macroeconómicos futuros.
- TOPSIS y RADAR siguen siendo modelos relativos al conjunto de países evaluados.

---

## 18. Recomendaciones y próximos pasos

1. Presentar al comité el ranking base acompañado de `prob_top_5`, `prob_top_10` y probabilidad de banda alta.
2. Clasificar países como prioridad robusta, prioridad condicional, alta sensibilidad o no robusta.
3. Para países condicionales, revisar drivers, restricciones y riesgos cualitativos antes de recomendación final.
4. Usar estabilidad de correlaciones para justificar similitudes entre líneas como PF–BD o IB–CIB.
5. Mantener Monte Carlo RADAR como prueba estándar antes de publicar rankings ejecutivos.
6. En iteraciones futuras, evaluar perturbación de IPC y Trend solo si se cuenta con supuestos defendibles de error de medición.

---

## 19. Síntesis Ejecutiva

- El análisis de sensibilidad convierte el ranking RADAR en una lectura probabilística de robustez.
- La simulación principal debe ser Monte Carlo RADAR, porque el RADAR compuesto es la métrica de decisión.
- TOPSIS Monte Carlo es diagnóstico estructural; RADAR Monte Carlo es evidencia ejecutiva.
- Países prioritarios deben clasificarse por probabilidad de mantenerse en Top-N y banda alta, no solo por rank base.
- Las decisiones finales deben distinguir prioridades robustas de oportunidades condicionales sensibles a pesos.
