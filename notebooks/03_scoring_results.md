# 03 - Resultados de scoring | RADAR Cibest

**Fase ASUM-DM:** 4 - Modelado  
**Objetivo:** Ejecutar el scoring hibrido completo y revisar resultados


```python
import sys
from pathlib import Path
import importlib
import re

# Asegura que Python encuentre el proyecto
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd

# Importar paquete raíz y módulos que quieres recargar
import src
import src.utils as utils
import src.data_preparation.feature_engineering as feature_engineering
import src.scoring.hybrid_scorer as hybrid_scorer
import src.scoring.ranking as ranking
import src.scoring.explainability as explain_scorer


# Invalidar cachés de importación
importlib.invalidate_caches()

# Recargar módulos modificados
importlib.reload(utils)
importlib.reload(feature_engineering)
importlib.reload(ranking)
importlib.reload(hybrid_scorer)
importlib.reload(explain_scorer)

# Reimportar funciones después del reload
from src.utils import load_all_configs, resolve_data_path, setup_logger, get_variable_catalog
from src.scoring.hybrid_scorer import run_full_scoring, prepare_decision_matrix
from src.scoring.explainability import compute_all_business_line_contributions, build_explainability_table_for_line, get_top_contributors, get_top_shortfalls, summarize_contributions_by_dimension, generate_country_line_explanation, compare_country_across_lines, compare_countries_in_line, compute_all_marginal_effects, combine_contribution_and_marginal, classify_driver_robustness, build_country_driver_table

# Cargar configuración
configs = load_all_configs()
setup_logger(configs["settings"].get("logging"))

# ---------------------------------------------------------------------
# Cargar master latest_available correcto
# ---------------------------------------------------------------------
raw_dir = resolve_data_path(configs["settings"]["data"]["raw_path"])

pattern = re.compile(r"^master_raw_\d{8}\.parquet$")

master_files = sorted(
    [
        p for p in raw_dir.glob("master_raw_*.parquet")
        if pattern.match(p.name)
    ],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not master_files:
    raise FileNotFoundError(
        "Falta master_raw_YYYYMMDD.parquet. Ejecute primero notebook 01."
    )

master_path = master_files[0]
master = pd.read_parquet(master_path)

print(f"Archivo master cargado: {master_path.name}")
print(f"Master cargado: {master.shape}")
print(f"Variables únicas: {master['variable'].nunique()}")
print("Tiene gdp_growth:", "gdp_growth" in master["variable"].unique())

# ---------------------------------------------------------------------
# Validaciones defensivas
# ---------------------------------------------------------------------
required_cols = {"country_iso3", "year", "variable", "value", "source"}
missing_cols = required_cols - set(master.columns)

if missing_cols:
    raise ValueError(
        f"Master inválido. Faltan columnas: {missing_cols}. "
        f"Columnas actuales: {master.columns.tolist()}"
    )

master["variable"] = master["variable"].astype(str).str.strip()

if "gdp_growth" not in master["variable"].unique():
    raise ValueError(
        "El master cargado no contiene gdp_growth. "
        "Probablemente se cargó un master histórico/parcial equivocado."
    )

if master["variable"].nunique() < 35:
    raise ValueError(
        f"El master cargado tiene solo {master['variable'].nunique()} variables. "
        "Para el scoring actual se esperaba un master completo con cerca de 44 variables."
    )

# Limpieza defensiva por si algún master anterior quedó con wgi_composite
master = master[master["variable"] != "wgi_composite"].copy()

# ---------------------------------------------------------------------
# Validar matriz de decisión
# ---------------------------------------------------------------------
wide_raw, decision_matrix, excluded = prepare_decision_matrix(master, configs)

print(f"wide_raw shape: {wide_raw.shape}")
print(f"decision_matrix shape: {decision_matrix.shape}")
print("wgi_composite en decision_matrix:", "wgi_composite" in decision_matrix.columns)
print("gdp_growth en master:", "gdp_growth" in master["variable"].unique())

# ---------------------------------------------------------------------
# Ejecutar scoring
# ---------------------------------------------------------------------
results = run_full_scoring(master, configs, persist=True)

print(f"Paises evaluados: {len(results['radar_global'])}")
print(f"Excluidos por cobertura: {results['excluded_countries']}")

print(f"País origen: {results['origin_country']}")
print(f"País origen excluido: {results['origin_country_excluded']}")

# print("\nResumen tendencia:")
# print(results["trend"]["trend"].describe())
# print(results["trend"].sort_values("trend", ascending=False).head(10))
```

    2026-06-12 21:40:52 | INFO     | src.data_preparation.cleaning:pivot_long_to_wide:70 | Pivoteo ancho: 30 paises x 45 variables (estrategia=latest_available)
    2026-06-12 21:40:52 | WARNING  | src.data_preparation.cleaning:validate_country_coverage:92 | Paises excluidos por >30% variables faltantes: ['CUB']
    

    Archivo master cargado: master_raw_20260612.parquet
    Master cargado: (1281, 5)
    Variables únicas: 45
    Tiene gdp_growth: True
    

    2026-06-12 21:40:53 | INFO     | src.data_preparation.cleaning:impute_missing:145 | Imputacion (regional_median): 107 -> 0 celdas faltantes
    2026-06-12 21:40:53 | INFO     | src.data_preparation.cleaning:run_cleaning:183 | Limpieza completada: 29 paises x 45 variables
    2026-06-12 21:40:53 | INFO     | src.data_preparation.feature_engineering:apply_log_transform:73 | Log-transformacion (natural) aplicada a: ['gdp_nominal', 'population_total', 'geographic_distance_km', 'colombian_diaspora_stock', 'stock_market_cap_gdp', 'financial_system_deposits_gdp', 'domestic_credit_private_gdp', 'fdi_net_inflows_gdp', 'personal_remittances_gdp', 'secure_internet_servers_per_million', 'atms_per_100k_adults', 'commercial_bank_branches_per_100k_adults']
    2026-06-12 21:40:53 | INFO     | src.data_preparation.feature_engineering:run_feature_engineering:136 | Feature engineering completado: 29 paises x 46 variables
    2026-06-12 21:40:53 | INFO     | src.scoring.hybrid_scorer:prepare_decision_matrix:109 | Variables excluidas de TOPSIS: ['colombian_diaspora_stock', 'common_language_spanish', 'cultural_distance_hofstede', 'gdp_growth', 'geographic_distance_km', 'hofstede_idv', 'hofstede_ivr', 'hofstede_lto', 'hofstede_mas', 'hofstede_pdi', 'hofstede_uai']
    2026-06-12 21:40:53 | INFO     | src.data_preparation.normalization:normalize:107 | Normalizacion min_max: 29 paises x 35 variables
    2026-06-12 21:40:53 | INFO     | src.data_preparation.normalization:apply_direction:74 | Direccion aplicada: 8 variables invertidas -> ['inflation_rate', 'unemployment_rate', 'public_debt_gdp', 'weighted_mean_applied_tariff_all_products', 'interest_rate_spread', 'bank_npl_ratio', 'bank_concentration_5', 'country_risk_premium']
    2026-06-12 21:40:54 | INFO     | src.data_preparation.cleaning:pivot_long_to_wide:70 | Pivoteo ancho: 30 paises x 45 variables (estrategia=latest_available)
    2026-06-12 21:40:54 | WARNING  | src.data_preparation.cleaning:validate_country_coverage:92 | Paises excluidos por >30% variables faltantes: ['CUB']
    

    wide_raw shape: (29, 46)
    decision_matrix shape: (29, 35)
    wgi_composite en decision_matrix: False
    gdp_growth en master: True
    

    2026-06-12 21:40:55 | INFO     | src.data_preparation.cleaning:impute_missing:145 | Imputacion (regional_median): 107 -> 0 celdas faltantes
    2026-06-12 21:40:55 | INFO     | src.data_preparation.cleaning:run_cleaning:183 | Limpieza completada: 29 paises x 45 variables
    2026-06-12 21:40:55 | INFO     | src.data_preparation.feature_engineering:apply_log_transform:73 | Log-transformacion (natural) aplicada a: ['gdp_nominal', 'population_total', 'geographic_distance_km', 'colombian_diaspora_stock', 'stock_market_cap_gdp', 'financial_system_deposits_gdp', 'domestic_credit_private_gdp', 'fdi_net_inflows_gdp', 'personal_remittances_gdp', 'secure_internet_servers_per_million', 'atms_per_100k_adults', 'commercial_bank_branches_per_100k_adults']
    2026-06-12 21:40:55 | INFO     | src.data_preparation.feature_engineering:run_feature_engineering:136 | Feature engineering completado: 29 paises x 46 variables
    2026-06-12 21:40:55 | INFO     | src.scoring.hybrid_scorer:prepare_decision_matrix:109 | Variables excluidas de TOPSIS: ['colombian_diaspora_stock', 'common_language_spanish', 'cultural_distance_hofstede', 'gdp_growth', 'geographic_distance_km', 'hofstede_idv', 'hofstede_ivr', 'hofstede_lto', 'hofstede_mas', 'hofstede_pdi', 'hofstede_uai']
    2026-06-12 21:40:55 | INFO     | src.data_preparation.normalization:normalize:107 | Normalizacion min_max: 29 paises x 35 variables
    2026-06-12 21:40:55 | INFO     | src.data_preparation.normalization:apply_direction:74 | Direccion aplicada: 8 variables invertidas -> ['inflation_rate', 'unemployment_rate', 'public_debt_gdp', 'weighted_mean_applied_tariff_all_products', 'interest_rate_spread', 'bank_npl_ratio', 'bank_concentration_5', 'country_risk_premium']
    2026-06-12 21:40:55 | INFO     | src.scoring.hybrid_scorer:run_full_scoring:846 | País origen excluido del scoring (post-feature engineering): COL. Países 29 -> 28
    2026-06-12 21:40:55 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 28 paises | score max=0.719 (CAN)
    2026-06-12 21:40:55 | INFO     | src.scoring.hybrid_scorer:score_all_business_lines:240 | --- TOPSIS linea IB ---
    2026-06-12 21:40:55 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 28 paises | score max=0.784 (CAN)
    2026-06-12 21:40:55 | INFO     | src.scoring.hybrid_scorer:score_all_business_lines:240 | --- TOPSIS linea PF ---
    2026-06-12 21:40:56 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 28 paises | score max=0.669 (ESP)
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:score_all_business_lines:240 | --- TOPSIS linea AD ---
    2026-06-12 21:40:56 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 28 paises | score max=0.798 (USA)
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:score_all_business_lines:240 | --- TOPSIS linea BD ---
    2026-06-12 21:40:56 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 28 paises | score max=0.796 (USA)
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:score_all_business_lines:240 | --- TOPSIS linea CIB ---
    2026-06-12 21:40:56 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 28 paises | score max=0.753 (CAN)
    2026-06-12 21:40:56 | INFO     | src.scoring.gravity:compute_ipc:65 | IPC se calculara con 4 componentes: ['geographic_distance_km', 'cultural_distance_hofstede', 'common_language_spanish', 'colombian_diaspora_stock']
    2026-06-12 21:40:56 | INFO     | src.scoring.gravity:compute_ipc:92 | IPC calculado: top-3 afinidad -> {'ECU': 0.997, 'PAN': 0.981, 'VEN': 0.892}
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:run_full_scoring:885 | Método de tendencia configurado: gdp_growth | enabled=True | variable=gdp_growth
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:compute_trend_factor:574 | Calculando tendencia con variable=gdp_growth. Filas encontradas=90
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:compute_trend_factor:619 | Ventana tendencia gdp_growth: 2022-2024
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:compute_trend_factor:631 | Filas en ventana de tendencia: 90
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:compute_trend_factor:694 | Winsorización aplicada a tendencia gdp_growth: q_low=0.05, q_high=0.95, lower=1.1757, upper=7.3462
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:compute_trend_factor:711 | Resumen trend_raw gdp_growth: min=1.1757, max=7.3462, rango=6.1704
    2026-06-12 21:40:56 | INFO     | src.scoring.hybrid_scorer:compute_trend_factor:737 | Factor de tendencia calculado con gdp_growth, ventana=3 años, países=28
    2026-06-12 21:40:57 | INFO     | src.scoring.hybrid_scorer:_persist_results:1070 | Resultados persistidos en C:\Users\jadarve\OneDrive - Grupo Bancolombia\Bancolombia\GFEC-VEF\2026\Internacionalización\radar_cibest_v2\data\scores con timestamp 20260612
    

    Paises evaluados: 28
    Excluidos por cobertura: ['CUB']
    País origen: COL
    País origen excluido: True
    


```python

print("gdp_growth en master:", "gdp_growth" in master["variable"].unique())
print("gdp_growth en wide_raw:", "gdp_growth" in wide_raw.columns)
print("gdp_growth en decision_matrix:", "gdp_growth" in decision_matrix.columns)
```

    gdp_growth en master: True
    gdp_growth en wide_raw: True
    gdp_growth en decision_matrix: False
    


```python
"gdp_growth" in configs["weights"]["variable_weights"]["macro"]
```




    False



## Top 10 mercados (RADAR global)


```python
#radar_global representa el score RADAR compuesto para cada país, ordenado de mayor a menor. 
# Este score es el resultado final del modelo híbrido que combina los diferentes factores evaluados en el análisis. 
# Al mostrar las primeras filas de este DataFrame, puedes ver cuáles son los países mejor posicionados según el modelo 
# y sus respectivos scores.
results['radar_global'].round(3) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_iso3</th>
      <th>radar_score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PAN</td>
      <td>0.723</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CRI</td>
      <td>0.660</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DOM</td>
      <td>0.634</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ESP</td>
      <td>0.627</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHL</td>
      <td>0.592</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ECU</td>
      <td>0.583</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PER</td>
      <td>0.559</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MEX</td>
      <td>0.559</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GTM</td>
      <td>0.548</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VEN</td>
      <td>0.547</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>USA</td>
      <td>0.545</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SLV</td>
      <td>0.542</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>URY</td>
      <td>0.540</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HND</td>
      <td>0.534</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CAN</td>
      <td>0.524</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>NIC</td>
      <td>0.506</td>
      <td>16</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BOL</td>
      <td>0.505</td>
      <td>17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ARG</td>
      <td>0.502</td>
      <td>18</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PRY</td>
      <td>0.500</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BHS</td>
      <td>0.491</td>
      <td>20</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BRB</td>
      <td>0.489</td>
      <td>21</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GUY</td>
      <td>0.480</td>
      <td>22</td>
    </tr>
    <tr>
      <th>22</th>
      <td>JAM</td>
      <td>0.455</td>
      <td>23</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BRA</td>
      <td>0.438</td>
      <td>24</td>
    </tr>
    <tr>
      <th>24</th>
      <td>TTO</td>
      <td>0.435</td>
      <td>25</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BLZ</td>
      <td>0.425</td>
      <td>26</td>
    </tr>
    <tr>
      <th>26</th>
      <td>SUR</td>
      <td>0.404</td>
      <td>27</td>
    </tr>
    <tr>
      <th>27</th>
      <td>HTI</td>
      <td>0.300</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
# global_ranking representa el ranking TOPSIS puro.
results['global_ranking']#.head(10).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>d_pos</th>
      <th>d_neg</th>
      <th>score_macro</th>
      <th>score_financial</th>
      <th>score_institutional</th>
      <th>score_digital_tech</th>
      <th>rank</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CAN</th>
      <td>0.719156</td>
      <td>0.060493</td>
      <td>0.154904</td>
      <td>0.663832</td>
      <td>0.649818</td>
      <td>0.928552</td>
      <td>0.645724</td>
      <td>1</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>0.717864</td>
      <td>0.061544</td>
      <td>0.156592</td>
      <td>0.692189</td>
      <td>0.661877</td>
      <td>0.829612</td>
      <td>0.725722</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>0.665248</td>
      <td>0.068178</td>
      <td>0.135489</td>
      <td>0.608580</td>
      <td>0.611979</td>
      <td>0.782554</td>
      <td>0.723244</td>
      <td>3</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>0.636569</td>
      <td>0.075143</td>
      <td>0.131617</td>
      <td>0.541781</td>
      <td>0.577081</td>
      <td>0.813094</td>
      <td>0.667066</td>
      <td>4</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>0.589495</td>
      <td>0.086857</td>
      <td>0.124730</td>
      <td>0.458270</td>
      <td>0.486805</td>
      <td>0.855351</td>
      <td>0.631845</td>
      <td>5</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>0.581929</td>
      <td>0.084705</td>
      <td>0.117904</td>
      <td>0.478381</td>
      <td>0.505199</td>
      <td>0.775447</td>
      <td>0.579880</td>
      <td>6</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0.557503</td>
      <td>0.090649</td>
      <td>0.114209</td>
      <td>0.391290</td>
      <td>0.564119</td>
      <td>0.737779</td>
      <td>0.526366</td>
      <td>7</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>0.557322</td>
      <td>0.086909</td>
      <td>0.109417</td>
      <td>0.482605</td>
      <td>0.549518</td>
      <td>0.623587</td>
      <td>0.583590</td>
      <td>8</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>0.549547</td>
      <td>0.090138</td>
      <td>0.109967</td>
      <td>0.482665</td>
      <td>0.534937</td>
      <td>0.643661</td>
      <td>0.468258</td>
      <td>9</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>0.548609</td>
      <td>0.093334</td>
      <td>0.113436</td>
      <td>0.360532</td>
      <td>0.652289</td>
      <td>0.657059</td>
      <td>0.486120</td>
      <td>10</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>0.537553</td>
      <td>0.094691</td>
      <td>0.110070</td>
      <td>0.538070</td>
      <td>0.517693</td>
      <td>0.552911</td>
      <td>0.568993</td>
      <td>11</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>0.529872</td>
      <td>0.094147</td>
      <td>0.106111</td>
      <td>0.574572</td>
      <td>0.508523</td>
      <td>0.512910</td>
      <td>0.491977</td>
      <td>12</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0.528224</td>
      <td>0.102132</td>
      <td>0.114352</td>
      <td>0.313689</td>
      <td>0.525216</td>
      <td>0.786008</td>
      <td>0.423569</td>
      <td>13</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>0.528185</td>
      <td>0.095654</td>
      <td>0.107082</td>
      <td>0.439458</td>
      <td>0.535878</td>
      <td>0.613648</td>
      <td>0.508597</td>
      <td>14</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0.512394</td>
      <td>0.100972</td>
      <td>0.106105</td>
      <td>0.535745</td>
      <td>0.409405</td>
      <td>0.583914</td>
      <td>0.478549</td>
      <td>15</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>0.509698</td>
      <td>0.097842</td>
      <td>0.101712</td>
      <td>0.483074</td>
      <td>0.477663</td>
      <td>0.553473</td>
      <td>0.633223</td>
      <td>16</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>0.508048</td>
      <td>0.098910</td>
      <td>0.102147</td>
      <td>0.499167</td>
      <td>0.448262</td>
      <td>0.568780</td>
      <td>0.481710</td>
      <td>17</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>0.496062</td>
      <td>0.101318</td>
      <td>0.099735</td>
      <td>0.408892</td>
      <td>0.514727</td>
      <td>0.553638</td>
      <td>0.500717</td>
      <td>18</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>0.480442</td>
      <td>0.106480</td>
      <td>0.098464</td>
      <td>0.455753</td>
      <td>0.504646</td>
      <td>0.507861</td>
      <td>0.282180</td>
      <td>19</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.475898</td>
      <td>0.106376</td>
      <td>0.096592</td>
      <td>0.396711</td>
      <td>0.525032</td>
      <td>0.512579</td>
      <td>0.441178</td>
      <td>20</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>0.457679</td>
      <td>0.113102</td>
      <td>0.095449</td>
      <td>0.299040</td>
      <td>0.499468</td>
      <td>0.573326</td>
      <td>0.468837</td>
      <td>21</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>0.451550</td>
      <td>0.107273</td>
      <td>0.088320</td>
      <td>0.443785</td>
      <td>0.496490</td>
      <td>0.424079</td>
      <td>0.392510</td>
      <td>22</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>0.451407</td>
      <td>0.113926</td>
      <td>0.093744</td>
      <td>0.393719</td>
      <td>0.537009</td>
      <td>0.451477</td>
      <td>0.205949</td>
      <td>23</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0.447913</td>
      <td>0.111153</td>
      <td>0.090179</td>
      <td>0.326497</td>
      <td>0.479692</td>
      <td>0.514353</td>
      <td>0.580108</td>
      <td>24</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>0.426433</td>
      <td>0.116388</td>
      <td>0.086532</td>
      <td>0.408196</td>
      <td>0.527616</td>
      <td>0.327006</td>
      <td>0.437822</td>
      <td>25</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>0.400223</td>
      <td>0.126998</td>
      <td>0.084744</td>
      <td>0.398648</td>
      <td>0.425361</td>
      <td>0.395959</td>
      <td>0.231429</td>
      <td>26</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>0.340911</td>
      <td>0.141489</td>
      <td>0.073185</td>
      <td>0.400286</td>
      <td>0.468771</td>
      <td>0.050909</td>
      <td>0.445139</td>
      <td>27</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>0.338575</td>
      <td>0.142426</td>
      <td>0.072906</td>
      <td>0.301485</td>
      <td>0.382423</td>
      <td>0.350557</td>
      <td>0.068242</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



## Ranking por linea de negocio


```python
results.keys()
```




    dict_keys(['decision_matrix', 'wide_raw', 'global_ranking', 'business_line_rankings', 'ipc', 'trend', 'radar_global', 'radar_by_line', 'excluded_countries', 'origin_country', 'origin_country_excluded', 'composite_weights'])




```python
results["business_line_rankings"].keys()
```




    dict_keys(['IB', 'PF', 'AD', 'BD', 'CIB'])




```python
results["business_line_rankings"]["IB"]#.head(10).round(3) #ver una línea específica
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>d_pos</th>
      <th>d_neg</th>
      <th>score_macro</th>
      <th>score_financial</th>
      <th>score_institutional</th>
      <th>score_digital_tech</th>
      <th>rank</th>
      <th>business_line</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CAN</th>
      <td>0.784153</td>
      <td>0.054338</td>
      <td>0.197404</td>
      <td>0.662985</td>
      <td>0.774630</td>
      <td>0.936147</td>
      <td>0.679722</td>
      <td>1</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>0.735974</td>
      <td>0.064208</td>
      <td>0.178980</td>
      <td>0.610650</td>
      <td>0.739453</td>
      <td>0.836186</td>
      <td>0.759616</td>
      <td>2</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>0.709253</td>
      <td>0.069856</td>
      <td>0.170408</td>
      <td>0.572680</td>
      <td>0.715769</td>
      <td>0.776457</td>
      <td>0.754096</td>
      <td>3</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>0.677494</td>
      <td>0.075654</td>
      <td>0.158929</td>
      <td>0.621747</td>
      <td>0.652477</td>
      <td>0.789307</td>
      <td>0.698967</td>
      <td>4</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0.625508</td>
      <td>0.085281</td>
      <td>0.142443</td>
      <td>0.482449</td>
      <td>0.635704</td>
      <td>0.692603</td>
      <td>0.580584</td>
      <td>5</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>0.624730</td>
      <td>0.087105</td>
      <td>0.145008</td>
      <td>0.523585</td>
      <td>0.573514</td>
      <td>0.825400</td>
      <td>0.670617</td>
      <td>6</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>0.608649</td>
      <td>0.090426</td>
      <td>0.140636</td>
      <td>0.579099</td>
      <td>0.567710</td>
      <td>0.740949</td>
      <td>0.601370</td>
      <td>7</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>0.604456</td>
      <td>0.091757</td>
      <td>0.140220</td>
      <td>0.439036</td>
      <td>0.656483</td>
      <td>0.594806</td>
      <td>0.497629</td>
      <td>8</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0.596921</td>
      <td>0.102965</td>
      <td>0.152481</td>
      <td>0.397711</td>
      <td>0.593200</td>
      <td>0.774293</td>
      <td>0.436684</td>
      <td>9</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>0.596437</td>
      <td>0.091757</td>
      <td>0.135610</td>
      <td>0.580065</td>
      <td>0.621008</td>
      <td>0.546934</td>
      <td>0.567333</td>
      <td>10</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>0.589210</td>
      <td>0.094823</td>
      <td>0.136007</td>
      <td>0.573160</td>
      <td>0.618839</td>
      <td>0.533095</td>
      <td>0.507455</td>
      <td>11</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>0.580708</td>
      <td>0.098677</td>
      <td>0.136666</td>
      <td>0.522373</td>
      <td>0.632583</td>
      <td>0.476842</td>
      <td>0.595716</td>
      <td>12</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>0.537614</td>
      <td>0.112085</td>
      <td>0.130320</td>
      <td>0.440113</td>
      <td>0.565634</td>
      <td>0.513859</td>
      <td>0.552583</td>
      <td>13</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.526782</td>
      <td>0.110726</td>
      <td>0.123259</td>
      <td>0.461973</td>
      <td>0.568764</td>
      <td>0.467179</td>
      <td>0.443694</td>
      <td>14</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>0.524485</td>
      <td>0.113094</td>
      <td>0.124741</td>
      <td>0.585241</td>
      <td>0.492388</td>
      <td>0.585185</td>
      <td>0.478436</td>
      <td>15</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>0.520560</td>
      <td>0.117074</td>
      <td>0.127115</td>
      <td>0.486014</td>
      <td>0.619173</td>
      <td>0.279232</td>
      <td>0.445966</td>
      <td>16</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>0.518804</td>
      <td>0.111311</td>
      <td>0.120010</td>
      <td>0.501036</td>
      <td>0.545268</td>
      <td>0.463306</td>
      <td>0.512764</td>
      <td>17</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>0.516601</td>
      <td>0.117251</td>
      <td>0.125305</td>
      <td>0.505249</td>
      <td>0.594528</td>
      <td>0.345076</td>
      <td>0.214464</td>
      <td>18</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0.496233</td>
      <td>0.116220</td>
      <td>0.114481</td>
      <td>0.452780</td>
      <td>0.515498</td>
      <td>0.461284</td>
      <td>0.597945</td>
      <td>19</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>0.492456</td>
      <td>0.118273</td>
      <td>0.114757</td>
      <td>0.556417</td>
      <td>0.523357</td>
      <td>0.397679</td>
      <td>0.283608</td>
      <td>20</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>0.491902</td>
      <td>0.117063</td>
      <td>0.113331</td>
      <td>0.510376</td>
      <td>0.535139</td>
      <td>0.374993</td>
      <td>0.417308</td>
      <td>21</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>0.489151</td>
      <td>0.117457</td>
      <td>0.112468</td>
      <td>0.566957</td>
      <td>0.470139</td>
      <td>0.488700</td>
      <td>0.511044</td>
      <td>22</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>0.477534</td>
      <td>0.119948</td>
      <td>0.109633</td>
      <td>0.599844</td>
      <td>0.477951</td>
      <td>0.409288</td>
      <td>0.495962</td>
      <td>23</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0.445067</td>
      <td>0.129073</td>
      <td>0.103519</td>
      <td>0.634023</td>
      <td>0.371244</td>
      <td>0.504602</td>
      <td>0.467998</td>
      <td>24</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>0.440249</td>
      <td>0.134841</td>
      <td>0.106054</td>
      <td>0.412123</td>
      <td>0.411024</td>
      <td>0.520628</td>
      <td>0.662942</td>
      <td>25</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>0.405284</td>
      <td>0.146672</td>
      <td>0.099953</td>
      <td>0.346271</td>
      <td>0.530336</td>
      <td>0.059170</td>
      <td>0.423902</td>
      <td>26</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>0.398305</td>
      <td>0.146834</td>
      <td>0.097200</td>
      <td>0.516610</td>
      <td>0.418535</td>
      <td>0.281226</td>
      <td>0.245293</td>
      <td>27</td>
      <td>IB</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>0.259167</td>
      <td>0.182713</td>
      <td>0.063919</td>
      <td>0.395633</td>
      <td>0.254282</td>
      <td>0.192298</td>
      <td>0.051024</td>
      <td>28</td>
      <td>IB</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_view = results["radar_by_line"].copy() #ver una línea específica.copy()

score_candidates = ["score", "topsis_score", "closeness", "radar_score"]
rank_candidates = ["rank", "ranking", "position"]

score_col = next((c for c in score_candidates if c in df_view.columns), None)
rank_col = next((c for c in rank_candidates if c in df_view.columns), None)

format_dict = {}

if score_col:
    format_dict[score_col] = "{:.3f}"

if rank_col:
    format_dict[rank_col] = "{:.0f}"

styler = df_view.style.format(format_dict)

if score_col:
    styler = styler.background_gradient(
        subset=[score_col],
        cmap="RdYlGn"
    )

if rank_col:
    styler = styler.background_gradient(
        subset=[rank_col],
        cmap="RdYlGn_r"
    )

styler.format(precision=3)
```




<style type="text/css">
</style>
<table id="T_92c31">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_92c31_level0_col0" class="col_heading level0 col0" >country_iso3</th>
      <th id="T_92c31_level0_col1" class="col_heading level0 col1" >IB</th>
      <th id="T_92c31_level0_col2" class="col_heading level0 col2" >PF</th>
      <th id="T_92c31_level0_col3" class="col_heading level0 col3" >AD</th>
      <th id="T_92c31_level0_col4" class="col_heading level0 col4" >BD</th>
      <th id="T_92c31_level0_col5" class="col_heading level0 col5" >CIB</th>
      <th id="T_92c31_level0_col6" class="col_heading level0 col6" >GLOBAL</th>
      <th id="T_92c31_level0_col7" class="col_heading level0 col7" >rank_global</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_92c31_level0_row0" class="row_heading level0 row0" >19</th>
      <td id="T_92c31_row0_col0" class="data row0 col0" >PAN</td>
      <td id="T_92c31_row0_col1" class="data row0 col1" >0.746</td>
      <td id="T_92c31_row0_col2" class="data row0 col2" >0.698</td>
      <td id="T_92c31_row0_col3" class="data row0 col3" >0.717</td>
      <td id="T_92c31_row0_col4" class="data row0 col4" >0.723</td>
      <td id="T_92c31_row0_col5" class="data row0 col5" >0.703</td>
      <td id="T_92c31_row0_col6" class="data row0 col6" >0.723</td>
      <td id="T_92c31_row0_col7" class="data row0 col7" >1</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row1" class="row_heading level0 row1" >8</th>
      <td id="T_92c31_row1_col0" class="data row1 col0" >CRI</td>
      <td id="T_92c31_row1_col1" class="data row1 col1" >0.676</td>
      <td id="T_92c31_row1_col2" class="data row1 col2" >0.639</td>
      <td id="T_92c31_row1_col3" class="data row1 col3" >0.675</td>
      <td id="T_92c31_row1_col4" class="data row1 col4" >0.681</td>
      <td id="T_92c31_row1_col5" class="data row1 col5" >0.593</td>
      <td id="T_92c31_row1_col6" class="data row1 col6" >0.660</td>
      <td id="T_92c31_row1_col7" class="data row1 col7" >2</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row2" class="row_heading level0 row2" >9</th>
      <td id="T_92c31_row2_col0" class="data row2 col0" >DOM</td>
      <td id="T_92c31_row2_col1" class="data row2 col1" >0.619</td>
      <td id="T_92c31_row2_col2" class="data row2 col2" >0.604</td>
      <td id="T_92c31_row2_col3" class="data row2 col3" >0.606</td>
      <td id="T_92c31_row2_col4" class="data row2 col4" >0.630</td>
      <td id="T_92c31_row2_col5" class="data row2 col5" >0.629</td>
      <td id="T_92c31_row2_col6" class="data row2 col6" >0.634</td>
      <td id="T_92c31_row2_col7" class="data row2 col7" >3</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row3" class="row_heading level0 row3" >11</th>
      <td id="T_92c31_row3_col0" class="data row3 col0" >ESP</td>
      <td id="T_92c31_row3_col1" class="data row3 col1" >0.653</td>
      <td id="T_92c31_row3_col2" class="data row3 col2" >0.629</td>
      <td id="T_92c31_row3_col3" class="data row3 col3" >0.655</td>
      <td id="T_92c31_row3_col4" class="data row3 col4" >0.678</td>
      <td id="T_92c31_row3_col5" class="data row3 col5" >0.622</td>
      <td id="T_92c31_row3_col6" class="data row3 col6" >0.627</td>
      <td id="T_92c31_row3_col7" class="data row3 col7" >4</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row4" class="row_heading level0 row4" >7</th>
      <td id="T_92c31_row4_col0" class="data row4 col0" >CHL</td>
      <td id="T_92c31_row4_col1" class="data row4 col1" >0.616</td>
      <td id="T_92c31_row4_col2" class="data row4 col2" >0.578</td>
      <td id="T_92c31_row4_col3" class="data row4 col3" >0.625</td>
      <td id="T_92c31_row4_col4" class="data row4 col4" >0.641</td>
      <td id="T_92c31_row4_col5" class="data row4 col5" >0.602</td>
      <td id="T_92c31_row4_col6" class="data row4 col6" >0.592</td>
      <td id="T_92c31_row4_col7" class="data row4 col7" >5</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row5" class="row_heading level0 row5" >10</th>
      <td id="T_92c31_row5_col0" class="data row5 col0" >ECU</td>
      <td id="T_92c31_row5_col1" class="data row5 col1" >0.607</td>
      <td id="T_92c31_row5_col2" class="data row5 col2" >0.569</td>
      <td id="T_92c31_row5_col3" class="data row5 col3" >0.536</td>
      <td id="T_92c31_row5_col4" class="data row5 col4" >0.594</td>
      <td id="T_92c31_row5_col5" class="data row5 col5" >0.539</td>
      <td id="T_92c31_row5_col6" class="data row5 col6" >0.583</td>
      <td id="T_92c31_row5_col7" class="data row5 col7" >6</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row6" class="row_heading level0 row6" >20</th>
      <td id="T_92c31_row6_col0" class="data row6 col0" >PER</td>
      <td id="T_92c31_row6_col1" class="data row6 col1" >0.548</td>
      <td id="T_92c31_row6_col2" class="data row6 col2" >0.541</td>
      <td id="T_92c31_row6_col3" class="data row6 col3" >0.537</td>
      <td id="T_92c31_row6_col4" class="data row6 col4" >0.576</td>
      <td id="T_92c31_row6_col5" class="data row6 col5" >0.549</td>
      <td id="T_92c31_row6_col6" class="data row6 col6" >0.559</td>
      <td id="T_92c31_row6_col7" class="data row6 col7" >7</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row7" class="row_heading level0 row7" >17</th>
      <td id="T_92c31_row7_col0" class="data row7 col0" >MEX</td>
      <td id="T_92c31_row7_col1" class="data row7 col1" >0.527</td>
      <td id="T_92c31_row7_col2" class="data row7 col2" >0.510</td>
      <td id="T_92c31_row7_col3" class="data row7 col3" >0.515</td>
      <td id="T_92c31_row7_col4" class="data row7 col4" >0.563</td>
      <td id="T_92c31_row7_col5" class="data row7 col5" >0.540</td>
      <td id="T_92c31_row7_col6" class="data row7 col6" >0.559</td>
      <td id="T_92c31_row7_col7" class="data row7 col7" >8</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row8" class="row_heading level0 row8" >12</th>
      <td id="T_92c31_row8_col0" class="data row8 col0" >GTM</td>
      <td id="T_92c31_row8_col1" class="data row8 col1" >0.556</td>
      <td id="T_92c31_row8_col2" class="data row8 col2" >0.494</td>
      <td id="T_92c31_row8_col3" class="data row8 col3" >0.456</td>
      <td id="T_92c31_row8_col4" class="data row8 col4" >0.475</td>
      <td id="T_92c31_row8_col5" class="data row8 col5" >0.518</td>
      <td id="T_92c31_row8_col6" class="data row8 col6" >0.548</td>
      <td id="T_92c31_row8_col7" class="data row8 col7" >9</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row9" class="row_heading level0 row9" >27</th>
      <td id="T_92c31_row9_col0" class="data row9 col0" >VEN</td>
      <td id="T_92c31_row9_col1" class="data row9 col1" >0.585</td>
      <td id="T_92c31_row9_col2" class="data row9 col2" >0.616</td>
      <td id="T_92c31_row9_col3" class="data row9 col3" >0.523</td>
      <td id="T_92c31_row9_col4" class="data row9 col4" >0.626</td>
      <td id="T_92c31_row9_col5" class="data row9 col5" >0.511</td>
      <td id="T_92c31_row9_col6" class="data row9 col6" >0.547</td>
      <td id="T_92c31_row9_col7" class="data row9 col7" >10</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row10" class="row_heading level0 row10" >26</th>
      <td id="T_92c31_row10_col0" class="data row10 col0" >USA</td>
      <td id="T_92c31_row10_col1" class="data row10 col1" >0.556</td>
      <td id="T_92c31_row10_col2" class="data row10 col2" >0.494</td>
      <td id="T_92c31_row10_col3" class="data row10 col3" >0.593</td>
      <td id="T_92c31_row10_col4" class="data row10 col4" >0.592</td>
      <td id="T_92c31_row10_col5" class="data row10 col5" >0.542</td>
      <td id="T_92c31_row10_col6" class="data row10 col6" >0.545</td>
      <td id="T_92c31_row10_col7" class="data row10 col7" >11</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row11" class="row_heading level0 row11" >22</th>
      <td id="T_92c31_row11_col0" class="data row11 col0" >SLV</td>
      <td id="T_92c31_row11_col1" class="data row11 col1" >0.572</td>
      <td id="T_92c31_row11_col2" class="data row11 col2" >0.552</td>
      <td id="T_92c31_row11_col3" class="data row11 col3" >0.489</td>
      <td id="T_92c31_row11_col4" class="data row11 col4" >0.521</td>
      <td id="T_92c31_row11_col5" class="data row11 col5" >0.519</td>
      <td id="T_92c31_row11_col6" class="data row11 col6" >0.542</td>
      <td id="T_92c31_row11_col7" class="data row11 col7" >12</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row12" class="row_heading level0 row12" >25</th>
      <td id="T_92c31_row12_col0" class="data row12 col0" >URY</td>
      <td id="T_92c31_row12_col1" class="data row12 col1" >0.561</td>
      <td id="T_92c31_row12_col2" class="data row12 col2" >0.535</td>
      <td id="T_92c31_row12_col3" class="data row12 col3" >0.580</td>
      <td id="T_92c31_row12_col4" class="data row12 col4" >0.577</td>
      <td id="T_92c31_row12_col5" class="data row12 col5" >0.463</td>
      <td id="T_92c31_row12_col6" class="data row12 col6" >0.540</td>
      <td id="T_92c31_row12_col7" class="data row12 col7" >13</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row13" class="row_heading level0 row13" >14</th>
      <td id="T_92c31_row13_col0" class="data row13 col0" >HND</td>
      <td id="T_92c31_row13_col1" class="data row13 col1" >0.573</td>
      <td id="T_92c31_row13_col2" class="data row13 col2" >0.492</td>
      <td id="T_92c31_row13_col3" class="data row13 col3" >0.439</td>
      <td id="T_92c31_row13_col4" class="data row13 col4" >0.461</td>
      <td id="T_92c31_row13_col5" class="data row13 col5" >0.520</td>
      <td id="T_92c31_row13_col6" class="data row13 col6" >0.534</td>
      <td id="T_92c31_row13_col7" class="data row13 col7" >14</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row14" class="row_heading level0 row14" >6</th>
      <td id="T_92c31_row14_col0" class="data row14 col0" >CAN</td>
      <td id="T_92c31_row14_col1" class="data row14 col1" >0.563</td>
      <td id="T_92c31_row14_col2" class="data row14 col2" >0.469</td>
      <td id="T_92c31_row14_col3" class="data row14 col3" >0.548</td>
      <td id="T_92c31_row14_col4" class="data row14 col4" >0.533</td>
      <td id="T_92c31_row14_col5" class="data row14 col5" >0.545</td>
      <td id="T_92c31_row14_col6" class="data row14 col6" >0.524</td>
      <td id="T_92c31_row14_col7" class="data row14 col7" >15</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row15" class="row_heading level0 row15" >18</th>
      <td id="T_92c31_row15_col0" class="data row15 col0" >NIC</td>
      <td id="T_92c31_row15_col1" class="data row15 col1" >0.505</td>
      <td id="T_92c31_row15_col2" class="data row15 col2" >0.485</td>
      <td id="T_92c31_row15_col3" class="data row15 col3" >0.415</td>
      <td id="T_92c31_row15_col4" class="data row15 col4" >0.447</td>
      <td id="T_92c31_row15_col5" class="data row15 col5" >0.488</td>
      <td id="T_92c31_row15_col6" class="data row15 col6" >0.506</td>
      <td id="T_92c31_row15_col7" class="data row15 col7" >16</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row16" class="row_heading level0 row16" >3</th>
      <td id="T_92c31_row16_col0" class="data row16 col0" >BOL</td>
      <td id="T_92c31_row16_col1" class="data row16 col1" >0.561</td>
      <td id="T_92c31_row16_col2" class="data row16 col2" >0.500</td>
      <td id="T_92c31_row16_col3" class="data row16 col3" >0.450</td>
      <td id="T_92c31_row16_col4" class="data row16 col4" >0.533</td>
      <td id="T_92c31_row16_col5" class="data row16 col5" >0.483</td>
      <td id="T_92c31_row16_col6" class="data row16 col6" >0.505</td>
      <td id="T_92c31_row16_col7" class="data row16 col7" >17</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row17" class="row_heading level0 row17" >0</th>
      <td id="T_92c31_row17_col0" class="data row17 col0" >ARG</td>
      <td id="T_92c31_row17_col1" class="data row17 col1" >0.461</td>
      <td id="T_92c31_row17_col2" class="data row17 col2" >0.525</td>
      <td id="T_92c31_row17_col3" class="data row17 col3" >0.517</td>
      <td id="T_92c31_row17_col4" class="data row17 col4" >0.577</td>
      <td id="T_92c31_row17_col5" class="data row17 col5" >0.444</td>
      <td id="T_92c31_row17_col6" class="data row17 col6" >0.502</td>
      <td id="T_92c31_row17_col7" class="data row17 col7" >18</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row18" class="row_heading level0 row18" >21</th>
      <td id="T_92c31_row18_col0" class="data row18 col0" >PRY</td>
      <td id="T_92c31_row18_col1" class="data row18 col1" >0.513</td>
      <td id="T_92c31_row18_col2" class="data row18 col2" >0.500</td>
      <td id="T_92c31_row18_col3" class="data row18 col3" >0.474</td>
      <td id="T_92c31_row18_col4" class="data row18 col4" >0.525</td>
      <td id="T_92c31_row18_col5" class="data row18 col5" >0.437</td>
      <td id="T_92c31_row18_col6" class="data row18 col6" >0.500</td>
      <td id="T_92c31_row18_col7" class="data row18 col7" >19</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row19" class="row_heading level0 row19" >1</th>
      <td id="T_92c31_row19_col0" class="data row19 col0" >BHS</td>
      <td id="T_92c31_row19_col1" class="data row19 col1" >0.532</td>
      <td id="T_92c31_row19_col2" class="data row19 col2" >0.441</td>
      <td id="T_92c31_row19_col3" class="data row19 col3" >0.508</td>
      <td id="T_92c31_row19_col4" class="data row19 col4" >0.495</td>
      <td id="T_92c31_row19_col5" class="data row19 col5" >0.501</td>
      <td id="T_92c31_row19_col6" class="data row19 col6" >0.491</td>
      <td id="T_92c31_row19_col7" class="data row19 col7" >20</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row20" class="row_heading level0 row20" >5</th>
      <td id="T_92c31_row20_col0" class="data row20 col0" >BRB</td>
      <td id="T_92c31_row20_col1" class="data row20 col1" >0.530</td>
      <td id="T_92c31_row20_col2" class="data row20 col2" >0.446</td>
      <td id="T_92c31_row20_col3" class="data row20 col3" >0.510</td>
      <td id="T_92c31_row20_col4" class="data row20 col4" >0.449</td>
      <td id="T_92c31_row20_col5" class="data row20 col5" >0.511</td>
      <td id="T_92c31_row20_col6" class="data row20 col6" >0.489</td>
      <td id="T_92c31_row20_col7" class="data row20 col7" >21</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row21" class="row_heading level0 row21" >13</th>
      <td id="T_92c31_row21_col0" class="data row21 col0" >GUY</td>
      <td id="T_92c31_row21_col1" class="data row21 col1" >0.439</td>
      <td id="T_92c31_row21_col2" class="data row21 col2" >0.489</td>
      <td id="T_92c31_row21_col3" class="data row21 col3" >0.448</td>
      <td id="T_92c31_row21_col4" class="data row21 col4" >0.474</td>
      <td id="T_92c31_row21_col5" class="data row21 col5" >0.443</td>
      <td id="T_92c31_row21_col6" class="data row21 col6" >0.480</td>
      <td id="T_92c31_row21_col7" class="data row21 col7" >22</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row22" class="row_heading level0 row22" >16</th>
      <td id="T_92c31_row22_col0" class="data row22 col0" >JAM</td>
      <td id="T_92c31_row22_col1" class="data row22 col1" >0.489</td>
      <td id="T_92c31_row22_col2" class="data row22 col2" >0.449</td>
      <td id="T_92c31_row22_col3" class="data row22 col3" >0.427</td>
      <td id="T_92c31_row22_col4" class="data row22 col4" >0.448</td>
      <td id="T_92c31_row22_col5" class="data row22 col5" >0.463</td>
      <td id="T_92c31_row22_col6" class="data row22 col6" >0.455</td>
      <td id="T_92c31_row22_col7" class="data row22 col7" >23</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row23" class="row_heading level0 row23" >4</th>
      <td id="T_92c31_row23_col0" class="data row23 col0" >BRA</td>
      <td id="T_92c31_row23_col1" class="data row23 col1" >0.464</td>
      <td id="T_92c31_row23_col2" class="data row23 col2" >0.441</td>
      <td id="T_92c31_row23_col3" class="data row23 col3" >0.431</td>
      <td id="T_92c31_row23_col4" class="data row23 col4" >0.492</td>
      <td id="T_92c31_row23_col5" class="data row23 col5" >0.433</td>
      <td id="T_92c31_row23_col6" class="data row23 col6" >0.438</td>
      <td id="T_92c31_row23_col7" class="data row23 col7" >24</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_92c31_row24_col0" class="data row24 col0" >TTO</td>
      <td id="T_92c31_row24_col1" class="data row24 col1" >0.472</td>
      <td id="T_92c31_row24_col2" class="data row24 col2" >0.428</td>
      <td id="T_92c31_row24_col3" class="data row24 col3" >0.414</td>
      <td id="T_92c31_row24_col4" class="data row24 col4" >0.440</td>
      <td id="T_92c31_row24_col5" class="data row24 col5" >0.440</td>
      <td id="T_92c31_row24_col6" class="data row24 col6" >0.435</td>
      <td id="T_92c31_row24_col7" class="data row24 col7" >25</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row25" class="row_heading level0 row25" >2</th>
      <td id="T_92c31_row25_col0" class="data row25 col0" >BLZ</td>
      <td id="T_92c31_row25_col1" class="data row25 col1" >0.473</td>
      <td id="T_92c31_row25_col2" class="data row25 col2" >0.421</td>
      <td id="T_92c31_row25_col3" class="data row25 col3" >0.467</td>
      <td id="T_92c31_row25_col4" class="data row25 col4" >0.436</td>
      <td id="T_92c31_row25_col5" class="data row25 col5" >0.418</td>
      <td id="T_92c31_row25_col6" class="data row25 col6" >0.425</td>
      <td id="T_92c31_row25_col7" class="data row25 col7" >26</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row26" class="row_heading level0 row26" >23</th>
      <td id="T_92c31_row26_col0" class="data row26 col0" >SUR</td>
      <td id="T_92c31_row26_col1" class="data row26 col1" >0.433</td>
      <td id="T_92c31_row26_col2" class="data row26 col2" >0.471</td>
      <td id="T_92c31_row26_col3" class="data row26 col3" >0.419</td>
      <td id="T_92c31_row26_col4" class="data row26 col4" >0.459</td>
      <td id="T_92c31_row26_col5" class="data row26 col5" >0.370</td>
      <td id="T_92c31_row26_col6" class="data row26 col6" >0.404</td>
      <td id="T_92c31_row26_col7" class="data row26 col7" >27</td>
    </tr>
    <tr>
      <th id="T_92c31_level0_row27" class="row_heading level0 row27" >15</th>
      <td id="T_92c31_row27_col0" class="data row27 col0" >HTI</td>
      <td id="T_92c31_row27_col1" class="data row27 col1" >0.253</td>
      <td id="T_92c31_row27_col2" class="data row27 col2" >0.269</td>
      <td id="T_92c31_row27_col3" class="data row27 col3" >0.201</td>
      <td id="T_92c31_row27_col4" class="data row27 col4" >0.228</td>
      <td id="T_92c31_row27_col5" class="data row27 col5" >0.314</td>
      <td id="T_92c31_row27_col6" class="data row27 col6" >0.300</td>
      <td id="T_92c31_row27_col7" class="data row27 col7" >28</td>
    </tr>
  </tbody>
</table>




## Indice de Proximidad Compuesto


```python
results['ipc'][['ipc']].round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ipc</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ECU</th>
      <td>0.997</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>0.981</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>0.892</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>0.854</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>0.847</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>0.810</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>0.801</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.753</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>0.742</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>0.739</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>0.726</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>0.714</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>0.666</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>0.655</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>0.601</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>0.568</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>0.535</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0.396</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>0.371</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>0.328</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>0.327</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>0.324</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>0.298</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>0.276</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0.274</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>0.242</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0.241</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0.240</td>
    </tr>
  </tbody>
</table>
</div>



## Tendencia


```python
#validar fuente de calculo:
configs["settings"]["scoring"]["trend"]
```




    {'enabled': True,
     'method': 'gdp_growth',
     'variable': 'gdp_growth',
     'end_year': 2024,
     'n_years': 3,
     'neutral_value': 0.5,
     'winsor_lower_quantile': 0.05,
     'winsor_upper_quantile': 0.95,
     'risk_adjusted': False,
     'risk_variable': 'country_risk_premium',
     'risk_floor': 0.5}




```python
results["trend"].sort_values("trend", ascending=False).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_iso3</th>
      <th>trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>BRB</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GUY</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PAN</td>
      <td>0.941</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BHS</td>
      <td>0.744</td>
    </tr>
    <tr>
      <th>27</th>
      <td>VEN</td>
      <td>0.744</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CRI</td>
      <td>0.565</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BLZ</td>
      <td>0.526</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DOM</td>
      <td>0.478</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ESP</td>
      <td>0.473</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NIC</td>
      <td>0.434</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GTM</td>
      <td>0.424</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HND</td>
      <td>0.418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BRA</td>
      <td>0.332</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PRY</td>
      <td>0.319</td>
    </tr>
    <tr>
      <th>22</th>
      <td>SLV</td>
      <td>0.301</td>
    </tr>
    <tr>
      <th>16</th>
      <td>JAM</td>
      <td>0.277</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MEX</td>
      <td>0.268</td>
    </tr>
    <tr>
      <th>25</th>
      <td>URY</td>
      <td>0.260</td>
    </tr>
    <tr>
      <th>26</th>
      <td>USA</td>
      <td>0.252</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CAN</td>
      <td>0.202</td>
    </tr>
    <tr>
      <th>23</th>
      <td>SUR</td>
      <td>0.163</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ECU</td>
      <td>0.126</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PER</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CHL</td>
      <td>0.097</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BOL</td>
      <td>0.087</td>
    </tr>
    <tr>
      <th>24</th>
      <td>TTO</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>15</th>
      <td>HTI</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ARG</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
results["trend"].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.379298</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.289187</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.153660</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.309718</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.490212</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Esto mostrará qué países están recibiendo mayor impulso por momentum macroeconómico.
audit = (
    results["radar_global"]
    .merge(results["trend"], on="country_iso3", how="left")
    .sort_values("trend", ascending=False)
)

audit#.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_iso3</th>
      <th>radar_score</th>
      <th>rank</th>
      <th>trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>BRB</td>
      <td>0.489188</td>
      <td>21</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GUY</td>
      <td>0.479533</td>
      <td>22</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>PAN</td>
      <td>0.722897</td>
      <td>1</td>
      <td>0.941349</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BHS</td>
      <td>0.491199</td>
      <td>20</td>
      <td>0.744282</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VEN</td>
      <td>0.546586</td>
      <td>10</td>
      <td>0.744162</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CRI</td>
      <td>0.659813</td>
      <td>2</td>
      <td>0.564922</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BLZ</td>
      <td>0.425406</td>
      <td>26</td>
      <td>0.525522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DOM</td>
      <td>0.633875</td>
      <td>3</td>
      <td>0.478441</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ESP</td>
      <td>0.626675</td>
      <td>4</td>
      <td>0.473182</td>
    </tr>
    <tr>
      <th>15</th>
      <td>NIC</td>
      <td>0.506170</td>
      <td>16</td>
      <td>0.434257</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GTM</td>
      <td>0.548403</td>
      <td>9</td>
      <td>0.423651</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HND</td>
      <td>0.534283</td>
      <td>14</td>
      <td>0.418456</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BRA</td>
      <td>0.438470</td>
      <td>24</td>
      <td>0.332254</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PRY</td>
      <td>0.499803</td>
      <td>19</td>
      <td>0.318617</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SLV</td>
      <td>0.541649</td>
      <td>12</td>
      <td>0.300820</td>
    </tr>
    <tr>
      <th>22</th>
      <td>JAM</td>
      <td>0.455292</td>
      <td>23</td>
      <td>0.276771</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MEX</td>
      <td>0.558872</td>
      <td>8</td>
      <td>0.268134</td>
    </tr>
    <tr>
      <th>12</th>
      <td>URY</td>
      <td>0.540315</td>
      <td>13</td>
      <td>0.259808</td>
    </tr>
    <tr>
      <th>10</th>
      <td>USA</td>
      <td>0.545222</td>
      <td>11</td>
      <td>0.252047</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CAN</td>
      <td>0.524463</td>
      <td>15</td>
      <td>0.202328</td>
    </tr>
    <tr>
      <th>26</th>
      <td>SUR</td>
      <td>0.403891</td>
      <td>27</td>
      <td>0.162957</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ECU</td>
      <td>0.582632</td>
      <td>6</td>
      <td>0.125770</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PER</td>
      <td>0.559473</td>
      <td>7</td>
      <td>0.117901</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHL</td>
      <td>0.591509</td>
      <td>5</td>
      <td>0.096829</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BOL</td>
      <td>0.504764</td>
      <td>17</td>
      <td>0.087146</td>
    </tr>
    <tr>
      <th>24</th>
      <td>TTO</td>
      <td>0.435269</td>
      <td>25</td>
      <td>0.070751</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ARG</td>
      <td>0.502442</td>
      <td>18</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>HTI</td>
      <td>0.300393</td>
      <td>28</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
results["composite_weights"]
```




    {'alpha': 0.6, 'beta': 0.3, 'gamma': 0.1}




```python
# 1. TOPSIS (score base)
global_rank = results["global_ranking"].copy()

if "country_iso3" in global_rank.columns:
    global_rank = global_rank.set_index("country_iso3")

topsis = global_rank["score"].rename("topsis_score")


# 2. IPC
ipc_df = results["ipc"].copy()

if "country_iso3" in ipc_df.columns:
    ipc = ipc_df.set_index("country_iso3")["ipc"]
else:
    ipc = ipc_df["ipc"]


# 3. Trend
trend_df = results["trend"].copy()

if "country_iso3" in trend_df.columns:
    trend = trend_df.set_index("country_iso3")["trend"]
else:
    trend = trend_df["trend"]


# 4. Unir todo
df = pd.concat([topsis, ipc, trend], axis=1)


# 5. Manejo de faltantes (igual que el modelo)
df["ipc"] = df["ipc"].fillna(df["ipc"].median())
df["trend"] = df["trend"].fillna(df["trend"].median())


# 6. Pesos
alpha = results["composite_weights"]["alpha"]
beta = results["composite_weights"]["beta"]
gamma = results["composite_weights"]["gamma"]



# 7. Aportes
df["aporte_topsis"] = alpha * df["topsis_score"]
df["aporte_ipc"] = beta * df["ipc"]
df["aporte_trend"] = gamma * df["trend"]


# 8. Score final
df["radar_score"] = (
    df["aporte_topsis"] +
    df["aporte_ipc"] +
    df["aporte_trend"]
)


# 9. Rankings comparativos
df["rank_topsis"] = df["topsis_score"].rank(ascending=False, method="min").astype(int)
df["rank_radar"] = df["radar_score"].rank(ascending=False, method="min").astype(int)

df["delta_rank"] = df["rank_topsis"] - df["rank_radar"]


# 10. Orden final
df = df.sort_values("rank_radar")

df#.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>topsis_score</th>
      <th>ipc</th>
      <th>trend</th>
      <th>aporte_topsis</th>
      <th>aporte_ipc</th>
      <th>aporte_trend</th>
      <th>radar_score</th>
      <th>rank_topsis</th>
      <th>rank_radar</th>
      <th>delta_rank</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PAN</th>
      <td>0.557322</td>
      <td>0.981231</td>
      <td>0.941349</td>
      <td>0.334393</td>
      <td>0.294369</td>
      <td>0.094135</td>
      <td>0.722897</td>
      <td>8</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>0.581929</td>
      <td>0.847213</td>
      <td>0.564922</td>
      <td>0.349157</td>
      <td>0.254164</td>
      <td>0.056492</td>
      <td>0.659813</td>
      <td>6</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>0.549547</td>
      <td>0.854342</td>
      <td>0.478441</td>
      <td>0.329728</td>
      <td>0.256303</td>
      <td>0.047844</td>
      <td>0.633875</td>
      <td>9</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>0.665248</td>
      <td>0.600693</td>
      <td>0.473182</td>
      <td>0.399149</td>
      <td>0.180208</td>
      <td>0.047318</td>
      <td>0.626675</td>
      <td>3</td>
      <td>4</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>0.636569</td>
      <td>0.666281</td>
      <td>0.096829</td>
      <td>0.381941</td>
      <td>0.199884</td>
      <td>0.009683</td>
      <td>0.591509</td>
      <td>4</td>
      <td>5</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>0.451550</td>
      <td>0.997082</td>
      <td>0.125770</td>
      <td>0.270930</td>
      <td>0.299125</td>
      <td>0.012577</td>
      <td>0.582632</td>
      <td>22</td>
      <td>6</td>
      <td>16</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>0.508048</td>
      <td>0.809513</td>
      <td>0.117901</td>
      <td>0.304829</td>
      <td>0.242854</td>
      <td>0.011790</td>
      <td>0.559473</td>
      <td>17</td>
      <td>7</td>
      <td>10</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>0.529872</td>
      <td>0.713784</td>
      <td>0.268134</td>
      <td>0.317923</td>
      <td>0.214135</td>
      <td>0.026813</td>
      <td>0.558872</td>
      <td>12</td>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>0.480442</td>
      <td>0.725909</td>
      <td>0.423651</td>
      <td>0.288265</td>
      <td>0.217773</td>
      <td>0.042365</td>
      <td>0.548403</td>
      <td>19</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>0.340911</td>
      <td>0.892077</td>
      <td>0.744162</td>
      <td>0.204547</td>
      <td>0.267623</td>
      <td>0.074416</td>
      <td>0.546586</td>
      <td>27</td>
      <td>10</td>
      <td>17</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>0.717864</td>
      <td>0.297664</td>
      <td>0.252047</td>
      <td>0.430718</td>
      <td>0.089299</td>
      <td>0.025205</td>
      <td>0.545222</td>
      <td>2</td>
      <td>11</td>
      <td>-9</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.475898</td>
      <td>0.753428</td>
      <td>0.300820</td>
      <td>0.285539</td>
      <td>0.226028</td>
      <td>0.030082</td>
      <td>0.541649</td>
      <td>20</td>
      <td>12</td>
      <td>8</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>0.589495</td>
      <td>0.535456</td>
      <td>0.259808</td>
      <td>0.353697</td>
      <td>0.160637</td>
      <td>0.025981</td>
      <td>0.540315</td>
      <td>5</td>
      <td>13</td>
      <td>-8</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>0.451407</td>
      <td>0.738644</td>
      <td>0.418456</td>
      <td>0.270844</td>
      <td>0.221593</td>
      <td>0.041846</td>
      <td>0.534283</td>
      <td>23</td>
      <td>14</td>
      <td>9</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>0.719156</td>
      <td>0.242455</td>
      <td>0.202328</td>
      <td>0.431494</td>
      <td>0.072736</td>
      <td>0.020233</td>
      <td>0.524463</td>
      <td>1</td>
      <td>15</td>
      <td>-14</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>0.400223</td>
      <td>0.742037</td>
      <td>0.434257</td>
      <td>0.240134</td>
      <td>0.222611</td>
      <td>0.043426</td>
      <td>0.506170</td>
      <td>26</td>
      <td>16</td>
      <td>10</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>0.426433</td>
      <td>0.800632</td>
      <td>0.087146</td>
      <td>0.255860</td>
      <td>0.240190</td>
      <td>0.008715</td>
      <td>0.504764</td>
      <td>25</td>
      <td>17</td>
      <td>8</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>0.509698</td>
      <td>0.655409</td>
      <td>0.000000</td>
      <td>0.305819</td>
      <td>0.196623</td>
      <td>0.000000</td>
      <td>0.502442</td>
      <td>16</td>
      <td>18</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>0.496062</td>
      <td>0.567679</td>
      <td>0.318617</td>
      <td>0.297637</td>
      <td>0.170304</td>
      <td>0.031862</td>
      <td>0.499803</td>
      <td>18</td>
      <td>19</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0.557503</td>
      <td>0.274230</td>
      <td>0.744282</td>
      <td>0.334502</td>
      <td>0.082269</td>
      <td>0.074428</td>
      <td>0.491199</td>
      <td>7</td>
      <td>20</td>
      <td>-13</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0.528224</td>
      <td>0.240845</td>
      <td>1.000000</td>
      <td>0.316935</td>
      <td>0.072254</td>
      <td>0.100000</td>
      <td>0.489188</td>
      <td>13</td>
      <td>21</td>
      <td>-8</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0.512394</td>
      <td>0.240324</td>
      <td>1.000000</td>
      <td>0.307436</td>
      <td>0.072097</td>
      <td>0.100000</td>
      <td>0.479533</td>
      <td>15</td>
      <td>22</td>
      <td>-7</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>0.548609</td>
      <td>0.328165</td>
      <td>0.276771</td>
      <td>0.329166</td>
      <td>0.098450</td>
      <td>0.027677</td>
      <td>0.455292</td>
      <td>10</td>
      <td>23</td>
      <td>-13</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>0.537553</td>
      <td>0.275712</td>
      <td>0.332254</td>
      <td>0.322532</td>
      <td>0.082714</td>
      <td>0.033225</td>
      <td>0.438470</td>
      <td>11</td>
      <td>24</td>
      <td>-13</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>0.528185</td>
      <td>0.370941</td>
      <td>0.070751</td>
      <td>0.316911</td>
      <td>0.111282</td>
      <td>0.007075</td>
      <td>0.435269</td>
      <td>14</td>
      <td>25</td>
      <td>-11</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>0.457679</td>
      <td>0.327486</td>
      <td>0.525522</td>
      <td>0.274608</td>
      <td>0.098246</td>
      <td>0.052552</td>
      <td>0.425406</td>
      <td>21</td>
      <td>26</td>
      <td>-5</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0.447913</td>
      <td>0.396159</td>
      <td>0.162957</td>
      <td>0.268748</td>
      <td>0.118848</td>
      <td>0.016296</td>
      <td>0.403891</td>
      <td>24</td>
      <td>27</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>0.338575</td>
      <td>0.324160</td>
      <td>0.000000</td>
      <td>0.203145</td>
      <td>0.097248</td>
      <td>0.000000</td>
      <td>0.300393</td>
      <td>28</td>
      <td>28</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 1. ¿Por qué sube un país?
Ejemplo típico (Panamá):

Tiene buen ipc
Tiene buen trend
Aunque su topsis_score no sea top

--> Resultado: sube en rank_radar

## 2. ¿Quién “pierde” cuando se mete IPC y Trend?
Países como:

USA
CAN

Muchas veces bajan porque:

Aunque son fuertes en TOPSIS
No necesariamente tienen la mejor proximidad (IPC)

##  ¿Cuál componente está dominando?


```python
df[["aporte_topsis", "aporte_ipc", "aporte_trend"]].mean()
```




    aporte_topsis    0.311664
    aporte_ipc       0.173567
    aporte_trend     0.037930
    dtype: float64




```python
def generate_explanations(df):
    explanations = []

    for country, row in df.iterrows():
        # Componentes
        t = row["aporte_topsis"]
        i = row["aporte_ipc"]
        tr = row["aporte_trend"]
        total = row["radar_score"]

        # Participación porcentual
        t_pct = t / total
        i_pct = i / total
        tr_pct = tr / total

        # Ranking change
        delta = row["delta_rank"]

        # Identificar driver principal
        components = {
            "TOPSIS": t_pct,
            "IPC": i_pct,
            "Tendencia": tr_pct,
        }

        main_driver = max(components, key=components.get)
        
        # Segundo driver
        sorted_comp = sorted(components.items(), key=lambda x: x[1], reverse=True)
        second_driver = sorted_comp[1][0]

        # Etiquetas de intensidad
        def label(p):
            if p > 0.5:
                return "dominante"
            elif p > 0.3:
                return "fuerte"
            elif p > 0.2:
                return "moderado"
            else:
                return "bajo"

        # Construcción del mensaje
        if delta > 0:
            movement = f"sube {delta} posiciones"
        elif delta < 0:
            movement = f"pierde {abs(delta)} posiciones"
        else:
            movement = "mantiene su posición"

        explanation = (
            f"{country} {movement} en el ranking RADAR. "
            f"El principal driver es {main_driver} ({components[main_driver]:.0%}, impacto {label(components[main_driver])}), "
            f"seguido por {second_driver}. "
            f"La contribución de TOPSIS es {t_pct:.0%}, IPC {i_pct:.0%} y Tendencia {tr_pct:.0%}."
        )

        explanations.append(explanation)

    df["explain"] = explanations
    return df
```


```python
df_explain = generate_explanations(df)

df_explain[["radar_score", "rank_radar", "rank_topsis", "delta_rank", "explain"]]#.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>radar_score</th>
      <th>rank_radar</th>
      <th>rank_topsis</th>
      <th>delta_rank</th>
      <th>explain</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PAN</th>
      <td>0.722897</td>
      <td>1</td>
      <td>8</td>
      <td>7</td>
      <td>PAN sube 7.0 posiciones en el ranking RADAR. E...</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>0.659813</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>CRI sube 4.0 posiciones en el ranking RADAR. E...</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>0.633875</td>
      <td>3</td>
      <td>9</td>
      <td>6</td>
      <td>DOM sube 6.0 posiciones en el ranking RADAR. E...</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>0.626675</td>
      <td>4</td>
      <td>3</td>
      <td>-1</td>
      <td>ESP pierde 1.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>0.591509</td>
      <td>5</td>
      <td>4</td>
      <td>-1</td>
      <td>CHL pierde 1.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>0.582632</td>
      <td>6</td>
      <td>22</td>
      <td>16</td>
      <td>ECU sube 16.0 posiciones en el ranking RADAR. ...</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>0.559473</td>
      <td>7</td>
      <td>17</td>
      <td>10</td>
      <td>PER sube 10.0 posiciones en el ranking RADAR. ...</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>0.558872</td>
      <td>8</td>
      <td>12</td>
      <td>4</td>
      <td>MEX sube 4.0 posiciones en el ranking RADAR. E...</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>0.548403</td>
      <td>9</td>
      <td>19</td>
      <td>10</td>
      <td>GTM sube 10.0 posiciones en el ranking RADAR. ...</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>0.546586</td>
      <td>10</td>
      <td>27</td>
      <td>17</td>
      <td>VEN sube 17.0 posiciones en el ranking RADAR. ...</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>0.545222</td>
      <td>11</td>
      <td>2</td>
      <td>-9</td>
      <td>USA pierde 9.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.541649</td>
      <td>12</td>
      <td>20</td>
      <td>8</td>
      <td>SLV sube 8.0 posiciones en el ranking RADAR. E...</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>0.540315</td>
      <td>13</td>
      <td>5</td>
      <td>-8</td>
      <td>URY pierde 8.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>0.534283</td>
      <td>14</td>
      <td>23</td>
      <td>9</td>
      <td>HND sube 9.0 posiciones en el ranking RADAR. E...</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>0.524463</td>
      <td>15</td>
      <td>1</td>
      <td>-14</td>
      <td>CAN pierde 14.0 posiciones en el ranking RADAR...</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>0.506170</td>
      <td>16</td>
      <td>26</td>
      <td>10</td>
      <td>NIC sube 10.0 posiciones en el ranking RADAR. ...</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>0.504764</td>
      <td>17</td>
      <td>25</td>
      <td>8</td>
      <td>BOL sube 8.0 posiciones en el ranking RADAR. E...</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>0.502442</td>
      <td>18</td>
      <td>16</td>
      <td>-2</td>
      <td>ARG pierde 2.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>0.499803</td>
      <td>19</td>
      <td>18</td>
      <td>-1</td>
      <td>PRY pierde 1.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0.491199</td>
      <td>20</td>
      <td>7</td>
      <td>-13</td>
      <td>BHS pierde 13.0 posiciones en el ranking RADAR...</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0.489188</td>
      <td>21</td>
      <td>13</td>
      <td>-8</td>
      <td>BRB pierde 8.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0.479533</td>
      <td>22</td>
      <td>15</td>
      <td>-7</td>
      <td>GUY pierde 7.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>0.455292</td>
      <td>23</td>
      <td>10</td>
      <td>-13</td>
      <td>JAM pierde 13.0 posiciones en el ranking RADAR...</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>0.438470</td>
      <td>24</td>
      <td>11</td>
      <td>-13</td>
      <td>BRA pierde 13.0 posiciones en el ranking RADAR...</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>0.435269</td>
      <td>25</td>
      <td>14</td>
      <td>-11</td>
      <td>TTO pierde 11.0 posiciones en el ranking RADAR...</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>0.425406</td>
      <td>26</td>
      <td>21</td>
      <td>-5</td>
      <td>BLZ pierde 5.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0.403891</td>
      <td>27</td>
      <td>24</td>
      <td>-3</td>
      <td>SUR pierde 3.0 posiciones en el ranking RADAR....</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>0.300393</td>
      <td>28</td>
      <td>28</td>
      <td>0</td>
      <td>HTI mantiene su posición en el ranking RADAR. ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
check = df[["radar_score"]].merge(
    results["radar_global"][["country_iso3", "radar_score"]].set_index("country_iso3"),
    left_index=True,
    right_index=True,
    suffixes=("_composite", "_global"),
)

check["diff"] = check["radar_score_composite"] - check["radar_score_global"]

check.sort_values("diff", ascending=False)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>radar_score_composite</th>
      <th>radar_score_global</th>
      <th>diff</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PAN</th>
      <td>0.722897</td>
      <td>0.722897</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>0.659813</td>
      <td>0.659813</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0.403891</td>
      <td>0.403891</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>0.425406</td>
      <td>0.425406</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>0.435269</td>
      <td>0.435269</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>0.438470</td>
      <td>0.438470</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>0.455292</td>
      <td>0.455292</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0.479533</td>
      <td>0.479533</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0.489188</td>
      <td>0.489188</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0.491199</td>
      <td>0.491199</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>0.499803</td>
      <td>0.499803</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>0.502442</td>
      <td>0.502442</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>0.504764</td>
      <td>0.504764</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>0.506170</td>
      <td>0.506170</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>0.524463</td>
      <td>0.524463</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>0.534283</td>
      <td>0.534283</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>0.540315</td>
      <td>0.540315</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.541649</td>
      <td>0.541649</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>0.545222</td>
      <td>0.545222</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>0.546586</td>
      <td>0.546586</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>0.548403</td>
      <td>0.548403</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>0.558872</td>
      <td>0.558872</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>0.559473</td>
      <td>0.559473</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>0.582632</td>
      <td>0.582632</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>0.591509</td>
      <td>0.591509</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>0.626675</td>
      <td>0.626675</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>0.633875</td>
      <td>0.633875</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>0.300393</td>
      <td>0.300393</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
results["decision_matrix"]#.head(10).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>variable</th>
      <th>gdp_nominal</th>
      <th>gdp_per_capita_ppp</th>
      <th>inflation_rate</th>
      <th>population_total</th>
      <th>urban_population_pct</th>
      <th>unemployment_rate</th>
      <th>current_account_gdp</th>
      <th>public_debt_gdp</th>
      <th>trade_openness</th>
      <th>fdi_net_inflows_gdp</th>
      <th>...</th>
      <th>control_of_corruption</th>
      <th>country_risk_premium</th>
      <th>heritage_efi</th>
      <th>internet_users_pct</th>
      <th>mobile_subscriptions</th>
      <th>digital_payment_adults_pct</th>
      <th>secure_internet_servers_per_million</th>
      <th>commercial_bank_branches_per_100k_adults</th>
      <th>atms_per_100k_adults</th>
      <th>ict_goods_exports_pct_total_goods_exports</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARG</th>
      <td>0.581690</td>
      <td>0.329687</td>
      <td>0.137314</td>
      <td>0.717027</td>
      <td>0.951911</td>
      <td>0.632010</td>
      <td>0.366979</td>
      <td>0.515770</td>
      <td>0.033014</td>
      <td>0.328804</td>
      <td>...</td>
      <td>0.391865</td>
      <td>0.635858</td>
      <td>0.623188</td>
      <td>0.872843</td>
      <td>0.674096</td>
      <td>0.681100</td>
      <td>0.578476</td>
      <td>0.456250</td>
      <td>0.658154</td>
      <td>0.001684</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0.175541</td>
      <td>0.460010</td>
      <td>0.996781</td>
      <td>0.049497</td>
      <td>0.793502</td>
      <td>0.464911</td>
      <td>0.000000</td>
      <td>0.479288</td>
      <td>0.331166</td>
      <td>0.292636</td>
      <td>...</td>
      <td>0.875796</td>
      <td>0.781401</td>
      <td>0.782609</td>
      <td>0.931419</td>
      <td>0.282865</td>
      <td>0.423450</td>
      <td>0.593809</td>
      <td>0.640412</td>
      <td>0.763867</td>
      <td>0.039843</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>0.000000</td>
      <td>0.134996</td>
      <td>0.985501</td>
      <td>0.054937</td>
      <td>0.220998</td>
      <td>0.493355</td>
      <td>0.245046</td>
      <td>0.572398</td>
      <td>0.503911</td>
      <td>0.510352</td>
      <td>...</td>
      <td>0.443214</td>
      <td>0.635858</td>
      <td>0.774327</td>
      <td>0.671129</td>
      <td>0.017840</td>
      <td>0.424201</td>
      <td>1.000000</td>
      <td>0.512201</td>
      <td>0.542293</td>
      <td>0.007856</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>0.312112</td>
      <td>0.117216</td>
      <td>0.978412</td>
      <td>0.533302</td>
      <td>0.647542</td>
      <td>0.970583</td>
      <td>0.207814</td>
      <td>0.507131</td>
      <td>0.143693</td>
      <td>0.159006</td>
      <td>...</td>
      <td>0.183146</td>
      <td>0.417259</td>
      <td>0.312629</td>
      <td>0.664842</td>
      <td>0.294982</td>
      <td>0.332414</td>
      <td>0.307436</td>
      <td>1.000000</td>
      <td>0.512925</td>
      <td>0.001684</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>0.716915</td>
      <td>0.231732</td>
      <td>0.981280</td>
      <td>0.933362</td>
      <td>0.888566</td>
      <td>0.727229</td>
      <td>0.176219</td>
      <td>0.386440</td>
      <td>0.077492</td>
      <td>0.468489</td>
      <td>...</td>
      <td>0.366024</td>
      <td>0.878429</td>
      <td>0.519669</td>
      <td>0.764198</td>
      <td>0.329996</td>
      <td>0.743942</td>
      <td>0.602660</td>
      <td>0.540167</td>
      <td>0.752254</td>
      <td>0.015152</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0.093424</td>
      <td>0.261799</td>
      <td>0.992719</td>
      <td>0.000000</td>
      <td>0.478324</td>
      <td>0.682334</td>
      <td>0.069742</td>
      <td>0.000000</td>
      <td>0.171616</td>
      <td>0.512364</td>
      <td>...</td>
      <td>0.873833</td>
      <td>0.732887</td>
      <td>0.892340</td>
      <td>0.469836</td>
      <td>0.443658</td>
      <td>0.423450</td>
      <td>0.424357</td>
      <td>0.476255</td>
      <td>0.444719</td>
      <td>0.061728</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>0.719783</td>
      <td>0.743398</td>
      <td>0.989057</td>
      <td>0.702728</td>
      <td>0.813399</td>
      <td>0.651297</td>
      <td>0.299501</td>
      <td>0.537962</td>
      <td>0.249281</td>
      <td>0.423664</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.970683</td>
      <td>0.259745</td>
      <td>1.000000</td>
      <td>0.777542</td>
      <td>0.596512</td>
      <td>0.878013</td>
      <td>0.076880</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>0.509289</td>
      <td>0.399286</td>
      <td>0.981554</td>
      <td>0.598874</td>
      <td>0.904470</td>
      <td>0.483793</td>
      <td>0.252023</td>
      <td>1.000000</td>
      <td>0.241785</td>
      <td>0.496220</td>
      <td>...</td>
      <td>0.836563</td>
      <td>0.958906</td>
      <td>0.973085</td>
      <td>0.996509</td>
      <td>0.606058</td>
      <td>0.828429</td>
      <td>0.656813</td>
      <td>0.370697</td>
      <td>0.552881</td>
      <td>0.016274</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>0.372799</td>
      <td>0.337865</td>
      <td>1.000000</td>
      <td>0.408725</td>
      <td>0.764362</td>
      <td>0.656483</td>
      <td>0.277466</td>
      <td>0.766135</td>
      <td>0.285196</td>
      <td>0.595573</td>
      <td>...</td>
      <td>0.704052</td>
      <td>0.853887</td>
      <td>0.865424</td>
      <td>0.820711</td>
      <td>0.636275</td>
      <td>0.537679</td>
      <td>0.477917</td>
      <td>0.536153</td>
      <td>0.618757</td>
      <td>0.046577</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>0.401913</td>
      <td>0.294712</td>
      <td>0.985452</td>
      <td>0.521638</td>
      <td>0.664557</td>
      <td>0.798379</td>
      <td>0.160363</td>
      <td>0.970789</td>
      <td>0.171616</td>
      <td>0.483413</td>
      <td>...</td>
      <td>0.410571</td>
      <td>0.853887</td>
      <td>0.755694</td>
      <td>0.900718</td>
      <td>0.256185</td>
      <td>0.440881</td>
      <td>0.237468</td>
      <td>0.413232</td>
      <td>0.512510</td>
      <td>0.031425</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>0.402261</td>
      <td>0.153076</td>
      <td>0.992324</td>
      <td>0.586746</td>
      <td>0.531825</td>
      <td>0.942950</td>
      <td>0.598469</td>
      <td>0.515770</td>
      <td>0.203107</td>
      <td>0.096302</td>
      <td>...</td>
      <td>0.252081</td>
      <td>0.514288</td>
      <td>0.585921</td>
      <td>0.611978</td>
      <td>0.327044</td>
      <td>0.328283</td>
      <td>0.337224</td>
      <td>0.348521</td>
      <td>0.490901</td>
      <td>0.002245</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>0.690946</td>
      <td>0.662964</td>
      <td>0.987520</td>
      <td>0.726432</td>
      <td>0.778893</td>
      <td>0.370178</td>
      <td>0.478418</td>
      <td>0.174059</td>
      <td>0.277174</td>
      <td>0.394853</td>
      <td>...</td>
      <td>0.715035</td>
      <td>0.941783</td>
      <td>0.738095</td>
      <td>1.000000</td>
      <td>0.584505</td>
      <td>0.990013</td>
      <td>0.749374</td>
      <td>0.775418</td>
      <td>0.694999</td>
      <td>0.076880</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>0.391652</td>
      <td>0.135264</td>
      <td>0.987145</td>
      <td>0.588836</td>
      <td>0.426458</td>
      <td>1.000000</td>
      <td>0.464009</td>
      <td>0.892269</td>
      <td>0.145822</td>
      <td>0.306556</td>
      <td>...</td>
      <td>0.198467</td>
      <td>0.878429</td>
      <td>0.749482</td>
      <td>0.258059</td>
      <td>0.425156</td>
      <td>0.075467</td>
      <td>0.187179</td>
      <td>0.657361</td>
      <td>0.489550</td>
      <td>0.014590</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0.224234</td>
      <td>0.931553</td>
      <td>0.987011</td>
      <td>0.152136</td>
      <td>0.000000</td>
      <td>0.241653</td>
      <td>1.000000</td>
      <td>0.515770</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.385286</td>
      <td>0.878429</td>
      <td>0.650104</td>
      <td>0.733660</td>
      <td>0.425552</td>
      <td>0.553165</td>
      <td>0.234556</td>
      <td>0.316245</td>
      <td>0.366690</td>
      <td>0.000561</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>0.269075</td>
      <td>0.051955</td>
      <td>0.980345</td>
      <td>0.514010</td>
      <td>0.467406</td>
      <td>0.812318</td>
      <td>0.106819</td>
      <td>0.669267</td>
      <td>0.400123</td>
      <td>0.478392</td>
      <td>...</td>
      <td>0.106382</td>
      <td>0.781401</td>
      <td>0.658385</td>
      <td>0.224680</td>
      <td>0.051656</td>
      <td>0.166583</td>
      <td>0.218020</td>
      <td>0.530777</td>
      <td>0.406678</td>
      <td>0.018519</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>0.226707</td>
      <td>0.000000</td>
      <td>0.892850</td>
      <td>0.525831</td>
      <td>0.413185</td>
      <td>0.000000</td>
      <td>0.295522</td>
      <td>0.479288</td>
      <td>0.000000</td>
      <td>0.024167</td>
      <td>...</td>
      <td>0.014536</td>
      <td>0.825349</td>
      <td>0.389234</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.135783</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031425</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>0.211754</td>
      <td>0.117369</td>
      <td>0.977190</td>
      <td>0.325328</td>
      <td>0.469505</td>
      <td>0.944733</td>
      <td>0.473561</td>
      <td>0.243467</td>
      <td>0.171616</td>
      <td>0.275392</td>
      <td>...</td>
      <td>0.463139</td>
      <td>0.825349</td>
      <td>0.846791</td>
      <td>0.882062</td>
      <td>0.471622</td>
      <td>0.406019</td>
      <td>0.243397</td>
      <td>0.261985</td>
      <td>0.500446</td>
      <td>0.015713</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>0.698966</td>
      <td>0.278295</td>
      <td>0.979891</td>
      <td>0.865350</td>
      <td>0.770772</td>
      <td>0.994408</td>
      <td>0.279756</td>
      <td>0.674738</td>
      <td>0.304126</td>
      <td>0.392174</td>
      <td>...</td>
      <td>0.200762</td>
      <td>0.907538</td>
      <td>0.672878</td>
      <td>0.736197</td>
      <td>0.460779</td>
      <td>0.305058</td>
      <td>0.345213</td>
      <td>0.450593</td>
      <td>0.638903</td>
      <td>0.769921</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>0.199517</td>
      <td>0.066756</td>
      <td>0.980273</td>
      <td>0.450844</td>
      <td>0.474319</td>
      <td>0.802593</td>
      <td>0.525537</td>
      <td>0.669267</td>
      <td>0.443211</td>
      <td>0.653252</td>
      <td>...</td>
      <td>0.047979</td>
      <td>0.732887</td>
      <td>0.544513</td>
      <td>0.282572</td>
      <td>0.366233</td>
      <td>0.000000</td>
      <td>0.172327</td>
      <td>0.306209</td>
      <td>0.380970</td>
      <td>0.001684</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>0.362128</td>
      <td>0.462085</td>
      <td>0.995669</td>
      <td>0.390743</td>
      <td>0.571728</td>
      <td>0.533549</td>
      <td>0.417563</td>
      <td>0.669267</td>
      <td>0.356914</td>
      <td>0.493152</td>
      <td>...</td>
      <td>0.320909</td>
      <td>0.893269</td>
      <td>0.778468</td>
      <td>0.520087</td>
      <td>0.820972</td>
      <td>0.435470</td>
      <td>0.489037</td>
      <td>0.588880</td>
      <td>0.651977</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>0.494709</td>
      <td>0.176827</td>
      <td>0.990521</td>
      <td>0.676248</td>
      <td>0.849438</td>
      <td>0.796353</td>
      <td>0.431027</td>
      <td>0.802621</td>
      <td>0.169829</td>
      <td>0.382977</td>
      <td>...</td>
      <td>0.269257</td>
      <td>0.922377</td>
      <td>0.807453</td>
      <td>0.711868</td>
      <td>0.533867</td>
      <td>0.430979</td>
      <td>0.389422</td>
      <td>0.161932</td>
      <td>0.786304</td>
      <td>0.003367</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>0.288972</td>
      <td>0.185557</td>
      <td>0.983364</td>
      <td>0.451109</td>
      <td>0.627666</td>
      <td>0.821799</td>
      <td>0.156971</td>
      <td>0.515770</td>
      <td>0.317155</td>
      <td>0.393893</td>
      <td>...</td>
      <td>0.174665</td>
      <td>0.893269</td>
      <td>0.809524</td>
      <td>0.703955</td>
      <td>0.551670</td>
      <td>0.476813</td>
      <td>0.382979</td>
      <td>0.387974</td>
      <td>0.435720</td>
      <td>0.004489</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.263832</td>
      <td>0.121893</td>
      <td>0.995040</td>
      <td>0.438542</td>
      <td>0.702972</td>
      <td>0.943355</td>
      <td>0.236486</td>
      <td>0.172580</td>
      <td>0.362642</td>
      <td>0.406774</td>
      <td>...</td>
      <td>0.313929</td>
      <td>0.684373</td>
      <td>0.629400</td>
      <td>0.388681</td>
      <td>1.000000</td>
      <td>0.141188</td>
      <td>0.243014</td>
      <td>0.364019</td>
      <td>0.545537</td>
      <td>0.022447</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0.035280</td>
      <td>0.225220</td>
      <td>0.934828</td>
      <td>0.114072</td>
      <td>0.568377</td>
      <td>0.576175</td>
      <td>0.333773</td>
      <td>0.515770</td>
      <td>0.399258</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.375970</td>
      <td>0.635858</td>
      <td>0.532091</td>
      <td>0.825509</td>
      <td>0.691051</td>
      <td>0.553165</td>
      <td>0.480893</td>
      <td>0.400274</td>
      <td>0.346726</td>
      <td>0.004489</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>0.228476</td>
      <td>0.401069</td>
      <td>0.996320</td>
      <td>0.222428</td>
      <td>0.397615</td>
      <td>0.941248</td>
      <td>0.446051</td>
      <td>0.975154</td>
      <td>0.171616</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.387481</td>
      <td>0.853887</td>
      <td>0.726708</td>
      <td>0.716103</td>
      <td>0.481438</td>
      <td>0.582274</td>
      <td>0.335015</td>
      <td>0.364662</td>
      <td>0.514584</td>
      <td>0.003367</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>0.354827</td>
      <td>0.402151</td>
      <td>0.979394</td>
      <td>0.350183</td>
      <td>1.000000</td>
      <td>0.601945</td>
      <td>0.285303</td>
      <td>0.524409</td>
      <td>0.175575</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.967698</td>
      <td>0.922377</td>
      <td>0.879917</td>
      <td>0.921364</td>
      <td>0.721404</td>
      <td>0.629517</td>
      <td>0.562296</td>
      <td>0.352590</td>
      <td>1.000000</td>
      <td>0.003928</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.986833</td>
      <td>1.000000</td>
      <td>0.776138</td>
      <td>0.870827</td>
      <td>0.122914</td>
      <td>0.063918</td>
      <td>0.018191</td>
      <td>0.224742</td>
      <td>...</td>
      <td>0.830690</td>
      <td>0.991246</td>
      <td>0.942029</td>
      <td>0.977793</td>
      <td>0.431170</td>
      <td>0.934333</td>
      <td>0.937901</td>
      <td>0.708966</td>
      <td>0.857850</td>
      <td>0.440516</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>0.397880</td>
      <td>0.218453</td>
      <td>0.000000</td>
      <td>0.650003</td>
      <td>0.908676</td>
      <td>0.780956</td>
      <td>0.156746</td>
      <td>0.515770</td>
      <td>0.022273</td>
      <td>0.272311</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.601668</td>
      <td>0.052266</td>
      <td>0.721512</td>
      <td>0.268430</td>
      <td>0.550211</td>
      <td>0.556731</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>28 rows × 35 columns</p>
</div>




```python
results["wide_raw"]#.head(10).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>variable</th>
      <th>gdp_nominal</th>
      <th>gdp_per_capita_ppp</th>
      <th>gdp_growth</th>
      <th>inflation_rate</th>
      <th>population_total</th>
      <th>urban_population_pct</th>
      <th>unemployment_rate</th>
      <th>current_account_gdp</th>
      <th>public_debt_gdp</th>
      <th>trade_openness</th>
      <th>...</th>
      <th>geographic_distance_km</th>
      <th>common_language_spanish</th>
      <th>hofstede_pdi</th>
      <th>hofstede_idv</th>
      <th>hofstede_mas</th>
      <th>hofstede_uai</th>
      <th>hofstede_lto</th>
      <th>hofstede_ivr</th>
      <th>colombian_diaspora_stock</th>
      <th>cultural_distance_hofstede</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARG</th>
      <td>27.182177</td>
      <td>30431.193122</td>
      <td>-1.342931</td>
      <td>219.883929</td>
      <td>17.637525</td>
      <td>92.274232</td>
      <td>7.145</td>
      <td>0.893118</td>
      <td>67.372401</td>
      <td>27.929761</td>
      <td>...</td>
      <td>8.457655</td>
      <td>1.0</td>
      <td>49.0</td>
      <td>51.0</td>
      <td>56.0</td>
      <td>86.0</td>
      <td>29.0</td>
      <td>62.0</td>
      <td>9.486987</td>
      <td>43.335897</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>23.485350</td>
      <td>41197.934255</td>
      <td>3.378666</td>
      <td>0.409162</td>
      <td>12.902425</td>
      <td>81.324926</td>
      <td>9.207</td>
      <td>-6.648608</td>
      <td>71.457821</td>
      <td>79.242459</td>
      <td>...</td>
      <td>7.779467</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>65.0</td>
      <td>45.0</td>
      <td>14.0</td>
      <td>67.0</td>
      <td>5.926926</td>
      <td>45.022217</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>21.887551</td>
      <td>14346.518805</td>
      <td>3.504664</td>
      <td>3.289560</td>
      <td>12.941017</td>
      <td>41.753211</td>
      <td>8.856</td>
      <td>-1.612700</td>
      <td>61.030832</td>
      <td>108.972447</td>
      <td>...</td>
      <td>7.754053</td>
      <td>0.0</td>
      <td>80.0</td>
      <td>19.0</td>
      <td>40.0</td>
      <td>86.0</td>
      <td>22.5</td>
      <td>89.0</td>
      <td>7.598650</td>
      <td>34.485504</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>24.728439</td>
      <td>12877.635118</td>
      <td>-1.123356</td>
      <td>5.099766</td>
      <td>16.334280</td>
      <td>71.236145</td>
      <td>2.967</td>
      <td>-2.377848</td>
      <td>68.339853</td>
      <td>46.977872</td>
      <td>...</td>
      <td>7.501634</td>
      <td>1.0</td>
      <td>78.0</td>
      <td>23.0</td>
      <td>42.0</td>
      <td>87.0</td>
      <td>21.0</td>
      <td>46.0</td>
      <td>9.238199</td>
      <td>47.791213</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>28.413013</td>
      <td>22338.476564</td>
      <td>3.419315</td>
      <td>4.367464</td>
      <td>19.172090</td>
      <td>87.895855</td>
      <td>5.970</td>
      <td>-3.027160</td>
      <td>81.855443</td>
      <td>35.584524</td>
      <td>...</td>
      <td>8.368925</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>36.0</td>
      <td>49.0</td>
      <td>76.0</td>
      <td>28.0</td>
      <td>59.0</td>
      <td>9.173054</td>
      <td>36.796739</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>22.737909</td>
      <td>24822.534331</td>
      <td>2.482262</td>
      <td>1.446437</td>
      <td>12.551321</td>
      <td>59.539703</td>
      <td>6.524</td>
      <td>-5.215344</td>
      <td>125.131136</td>
      <td>51.783514</td>
      <td>...</td>
      <td>7.528869</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>65.0</td>
      <td>45.0</td>
      <td>14.0</td>
      <td>67.0</td>
      <td>3.433987</td>
      <td>45.022217</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>28.439119</td>
      <td>64610.379517</td>
      <td>1.554795</td>
      <td>2.381584</td>
      <td>17.536097</td>
      <td>82.700257</td>
      <td>6.907</td>
      <td>-0.493602</td>
      <td>64.887158</td>
      <td>65.149921</td>
      <td>...</td>
      <td>8.571871</td>
      <td>0.0</td>
      <td>39.0</td>
      <td>72.0</td>
      <td>52.0</td>
      <td>48.0</td>
      <td>54.0</td>
      <td>68.0</td>
      <td>9.143346</td>
      <td>79.561297</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>26.523168</td>
      <td>36181.156617</td>
      <td>2.644312</td>
      <td>4.297639</td>
      <td>16.799412</td>
      <td>88.995091</td>
      <td>8.974</td>
      <td>-1.469331</td>
      <td>13.145640</td>
      <td>63.859857</td>
      <td>...</td>
      <td>8.354910</td>
      <td>1.0</td>
      <td>63.0</td>
      <td>49.0</td>
      <td>28.0</td>
      <td>86.0</td>
      <td>12.0</td>
      <td>68.0</td>
      <td>9.299358</td>
      <td>44.821870</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>25.280825</td>
      <td>31106.764356</td>
      <td>4.321224</td>
      <td>-0.412853</td>
      <td>15.450599</td>
      <td>79.310767</td>
      <td>6.843</td>
      <td>-0.946456</td>
      <td>39.335112</td>
      <td>71.331014</td>
      <td>...</td>
      <td>7.074117</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>86.0</td>
      <td>22.5</td>
      <td>89.0</td>
      <td>8.503094</td>
      <td>58.423026</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>25.545821</td>
      <td>27541.662528</td>
      <td>4.953522</td>
      <td>3.302233</td>
      <td>16.251538</td>
      <td>72.412187</td>
      <td>5.092</td>
      <td>-3.353013</td>
      <td>16.416804</td>
      <td>51.783514</td>
      <td>...</td>
      <td>7.082549</td>
      <td>1.0</td>
      <td>65.0</td>
      <td>38.0</td>
      <td>65.0</td>
      <td>45.0</td>
      <td>11.0</td>
      <td>54.0</td>
      <td>8.791030</td>
      <td>46.658333</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>25.548985</td>
      <td>15840.266857</td>
      <td>-2.001255</td>
      <td>1.547325</td>
      <td>16.713381</td>
      <td>63.237693</td>
      <td>3.308</td>
      <td>5.650429</td>
      <td>67.372401</td>
      <td>57.203140</td>
      <td>...</td>
      <td>6.580639</td>
      <td>1.0</td>
      <td>78.0</td>
      <td>24.0</td>
      <td>63.0</td>
      <td>67.0</td>
      <td>24.0</td>
      <td>59.0</td>
      <td>10.919841</td>
      <td>34.871192</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>28.176637</td>
      <td>57965.294814</td>
      <td>3.455254</td>
      <td>2.774178</td>
      <td>17.704241</td>
      <td>80.315200</td>
      <td>10.376</td>
      <td>3.183274</td>
      <td>105.639075</td>
      <td>69.950261</td>
      <td>...</td>
      <td>8.991064</td>
      <td>1.0</td>
      <td>57.0</td>
      <td>67.0</td>
      <td>42.0</td>
      <td>86.0</td>
      <td>47.0</td>
      <td>44.0</td>
      <td>10.519808</td>
      <td>72.567210</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>25.452418</td>
      <td>14368.660568</td>
      <td>3.651864</td>
      <td>2.869928</td>
      <td>16.728207</td>
      <td>55.954703</td>
      <td>2.604</td>
      <td>2.887155</td>
      <td>25.209962</td>
      <td>47.344197</td>
      <td>...</td>
      <td>7.626083</td>
      <td>1.0</td>
      <td>95.0</td>
      <td>36.0</td>
      <td>37.0</td>
      <td>98.0</td>
      <td>25.0</td>
      <td>89.0</td>
      <td>7.374002</td>
      <td>47.780749</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>23.928558</td>
      <td>80155.039393</td>
      <td>43.819311</td>
      <td>2.903952</td>
      <td>13.630491</td>
      <td>26.477718</td>
      <td>11.962</td>
      <td>13.902229</td>
      <td>67.372401</td>
      <td>194.350757</td>
      <td>...</td>
      <td>7.484930</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>29.0</td>
      <td>42.0</td>
      <td>86.0</td>
      <td>20.5</td>
      <td>59.0</td>
      <td>3.178054</td>
      <td>36.197376</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>24.336709</td>
      <td>7486.018696</td>
      <td>3.553970</td>
      <td>4.606211</td>
      <td>16.197434</td>
      <td>58.785073</td>
      <td>4.920</td>
      <td>-4.453380</td>
      <td>50.182972</td>
      <td>91.110159</td>
      <td>...</td>
      <td>7.421178</td>
      <td>1.0</td>
      <td>80.0</td>
      <td>20.0</td>
      <td>40.0</td>
      <td>50.0</td>
      <td>22.5</td>
      <td>89.0</td>
      <td>6.695799</td>
      <td>45.102661</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>23.951068</td>
      <td>3193.671364</td>
      <td>-4.169634</td>
      <td>26.949056</td>
      <td>16.281282</td>
      <td>55.037265</td>
      <td>14.944</td>
      <td>-0.575393</td>
      <td>71.457821</td>
      <td>22.247885</td>
      <td>...</td>
      <td>7.186144</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>65.0</td>
      <td>45.0</td>
      <td>14.0</td>
      <td>67.0</td>
      <td>4.406719</td>
      <td>45.022217</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>23.814962</td>
      <td>12890.266005</td>
      <td>-0.535039</td>
      <td>5.411944</td>
      <td>14.859024</td>
      <td>58.930159</td>
      <td>3.286</td>
      <td>3.083475</td>
      <td>97.866400</td>
      <td>51.783514</td>
      <td>...</td>
      <td>7.438972</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>39.0</td>
      <td>68.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>67.0</td>
      <td>5.913503</td>
      <td>73.545904</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>28.249642</td>
      <td>26185.356471</td>
      <td>1.427428</td>
      <td>4.722256</td>
      <td>18.689646</td>
      <td>79.753881</td>
      <td>2.673</td>
      <td>-0.899378</td>
      <td>49.570212</td>
      <td>74.588908</td>
      <td>...</td>
      <td>8.157944</td>
      <td>1.0</td>
      <td>81.0</td>
      <td>34.0</td>
      <td>69.0</td>
      <td>82.0</td>
      <td>23.0</td>
      <td>97.0</td>
      <td>9.845541</td>
      <td>27.110883</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>23.703579</td>
      <td>8708.799831</td>
      <td>3.587830</td>
      <td>4.624738</td>
      <td>15.749369</td>
      <td>59.262840</td>
      <td>5.040</td>
      <td>4.151615</td>
      <td>50.182972</td>
      <td>98.525838</td>
      <td>...</td>
      <td>7.230563</td>
      <td>1.0</td>
      <td>80.0</td>
      <td>19.0</td>
      <td>40.0</td>
      <td>86.0</td>
      <td>22.5</td>
      <td>89.0</td>
      <td>5.777652</td>
      <td>34.485504</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>25.183687</td>
      <td>41369.422541</td>
      <td>2.747885</td>
      <td>0.693226</td>
      <td>15.323044</td>
      <td>65.995845</td>
      <td>8.360</td>
      <td>1.932669</td>
      <td>50.182972</td>
      <td>83.673779</td>
      <td>...</td>
      <td>6.698268</td>
      <td>1.0</td>
      <td>95.0</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>86.0</td>
      <td>22.5</td>
      <td>89.0</td>
      <td>11.018973</td>
      <td>43.037774</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>26.390460</td>
      <td>17802.418250</td>
      <td>3.303941</td>
      <td>2.007707</td>
      <td>17.348258</td>
      <td>85.191273</td>
      <td>5.117</td>
      <td>2.209367</td>
      <td>35.249227</td>
      <td>51.475888</td>
      <td>...</td>
      <td>7.539559</td>
      <td>1.0</td>
      <td>64.0</td>
      <td>20.0</td>
      <td>42.0</td>
      <td>87.0</td>
      <td>5.0</td>
      <td>46.0</td>
      <td>9.745546</td>
      <td>44.643029</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>24.517813</td>
      <td>18523.681015</td>
      <td>4.249960</td>
      <td>3.835403</td>
      <td>15.751248</td>
      <td>69.862259</td>
      <td>4.803</td>
      <td>-3.422725</td>
      <td>67.372401</td>
      <td>76.831157</td>
      <td>...</td>
      <td>8.286773</td>
      <td>1.0</td>
      <td>70.0</td>
      <td>12.0</td>
      <td>40.0</td>
      <td>85.0</td>
      <td>20.0</td>
      <td>56.0</td>
      <td>5.579730</td>
      <td>42.708313</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>24.288987</td>
      <td>13264.027091</td>
      <td>2.602020</td>
      <td>0.853782</td>
      <td>15.662104</td>
      <td>75.067491</td>
      <td>3.303</td>
      <td>-1.788631</td>
      <td>105.804729</td>
      <td>84.659646</td>
      <td>...</td>
      <td>7.528869</td>
      <td>1.0</td>
      <td>66.0</td>
      <td>19.0</td>
      <td>40.0</td>
      <td>94.0</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>7.781973</td>
      <td>33.241540</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>22.208676</td>
      <td>21800.526599</td>
      <td>1.720957</td>
      <td>16.229616</td>
      <td>13.360485</td>
      <td>65.764186</td>
      <td>7.834</td>
      <td>0.210701</td>
      <td>67.372401</td>
      <td>90.961394</td>
      <td>...</td>
      <td>7.626083</td>
      <td>0.0</td>
      <td>85.0</td>
      <td>47.0</td>
      <td>37.0</td>
      <td>92.0</td>
      <td>20.5</td>
      <td>59.0</td>
      <td>9.238199</td>
      <td>48.033842</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>23.967168</td>
      <td>36328.513007</td>
      <td>2.509208</td>
      <td>0.526885</td>
      <td>14.129104</td>
      <td>53.961057</td>
      <td>3.329</td>
      <td>2.518108</td>
      <td>15.928004</td>
      <td>51.783514</td>
      <td>...</td>
      <td>7.170888</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>25.0</td>
      <td>58.0</td>
      <td>55.0</td>
      <td>17.0</td>
      <td>80.0</td>
      <td>5.913503</td>
      <td>34.741906</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>25.117240</td>
      <td>36417.874162</td>
      <td>3.108255</td>
      <td>4.849144</td>
      <td>15.035334</td>
      <td>95.598195</td>
      <td>7.516</td>
      <td>-0.785397</td>
      <td>66.404950</td>
      <td>52.464917</td>
      <td>...</td>
      <td>8.535230</td>
      <td>1.0</td>
      <td>61.0</td>
      <td>60.0</td>
      <td>38.0</td>
      <td>98.0</td>
      <td>28.0</td>
      <td>53.0</td>
      <td>5.831882</td>
      <td>58.146367</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>30.989692</td>
      <td>85809.900385</td>
      <td>2.793001</td>
      <td>2.949525</td>
      <td>19.644783</td>
      <td>80.124780</td>
      <td>4.198</td>
      <td>-4.122625</td>
      <td>117.973235</td>
      <td>25.378634</td>
      <td>...</td>
      <td>8.301770</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>62.0</td>
      <td>46.0</td>
      <td>50.0</td>
      <td>68.0</td>
      <td>9.554852</td>
      <td>70.788417</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>25.509114</td>
      <td>21241.435073</td>
      <td>5.300000</td>
      <td>254.948535</td>
      <td>17.162095</td>
      <td>89.285860</td>
      <td>5.307</td>
      <td>-3.427356</td>
      <td>67.372401</td>
      <td>26.081166</td>
      <td>...</td>
      <td>6.928538</td>
      <td>1.0</td>
      <td>81.0</td>
      <td>26.0</td>
      <td>73.0</td>
      <td>76.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>9.238199</td>
      <td>25.039968</td>
    </tr>
  </tbody>
</table>
<p>28 rows × 46 columns</p>
</div>




```python
results["wide_raw"].columns
```




    Index(['gdp_nominal', 'gdp_per_capita_ppp', 'gdp_growth', 'inflation_rate',
           'population_total', 'urban_population_pct', 'unemployment_rate',
           'current_account_gdp', 'public_debt_gdp', 'trade_openness',
           'fdi_net_inflows_gdp', 'weighted_mean_applied_tariff_all_products',
           'domestic_credit_private_gdp', 'account_ownership',
           'interest_rate_spread', 'bank_npl_ratio', 'stock_market_cap_gdp',
           'personal_remittances_gdp', 'bank_concentration_5',
           'financial_system_deposits_gdp', 'bank_capital_rwa',
           'regulatory_quality', 'government_effectiveness', 'rule_of_law',
           'political_stability', 'voice_accountability', 'control_of_corruption',
           'country_risk_premium', 'heritage_efi', 'internet_users_pct',
           'mobile_subscriptions', 'digital_payment_adults_pct',
           'secure_internet_servers_per_million',
           'commercial_bank_branches_per_100k_adults', 'atms_per_100k_adults',
           'ict_goods_exports_pct_total_goods_exports', 'geographic_distance_km',
           'common_language_spanish', 'hofstede_pdi', 'hofstede_idv',
           'hofstede_mas', 'hofstede_uai', 'hofstede_lto', 'hofstede_ivr',
           'colombian_diaspora_stock', 'cultural_distance_hofstede'],
          dtype='object', name='variable')




```python
from typing import Any, Dict

import pandas as pd

from src.scoring.hybrid_scorer import _build_business_line_weights



def audit_business_line_weights(
    configs: Dict[str, Dict[str, Any]],
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Audita los pesos efectivos usados por TOPSIS para cada línea de negocio.

    Permite verificar:
    - peso global por dimensión;
    - peso de dimensión por línea;
    - peso global de variable dentro de dimensión;
    - override por variable, si existe;
    - peso final TOPSIS usado después de filtrar por decision_matrix.
    """

    business_lines = configs["business_lines"]["business_lines"]
    global_dim_weights = configs["weights"]["dimension_weights"]
    global_variable_weights = configs["weights"]["variable_weights"]

    rows = []

    for bl_key, bl_cfg in business_lines.items():
        dim_weights_line, final_var_weights = _build_business_line_weights(
            business_line_cfg=bl_cfg,
            variable_weights_by_dim=global_variable_weights,
        )

        # Replicar la lógica usada antes de TOPSIS:
        # 1. filtrar variables presentes en decision_matrix;
        # 2. renormalizar pesos restantes para que sumen 1.
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

                rows.append(
                    {
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
                    }
                )

    audit = pd.DataFrame(rows)

    return (
        audit
        .sort_values(
            ["business_line", "dimension", "final_topsis_weight"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )

```


```python
weights_audit = audit_business_line_weights(
    configs=configs,
    decision_matrix=decision_matrix,
)

weights_audit#.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>dimension</th>
      <th>variable</th>
      <th>in_decision_matrix</th>
      <th>global_dimension_weight</th>
      <th>line_dimension_weight</th>
      <th>global_variable_weight_in_dim</th>
      <th>override_variable_weight_in_dim</th>
      <th>has_override</th>
      <th>final_topsis_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AD</td>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>True</td>
      <td>0.1</td>
      <td>0.37</td>
      <td>0.15</td>
      <td>0.25</td>
      <td>True</td>
      <td>0.092500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AD</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>True</td>
      <td>0.1</td>
      <td>0.37</td>
      <td>0.21</td>
      <td>0.23</td>
      <td>True</td>
      <td>0.085100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AD</td>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>True</td>
      <td>0.1</td>
      <td>0.37</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>True</td>
      <td>0.074000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AD</td>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>True</td>
      <td>0.1</td>
      <td>0.37</td>
      <td>0.10</td>
      <td>0.16</td>
      <td>True</td>
      <td>0.059200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AD</td>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>True</td>
      <td>0.1</td>
      <td>0.37</td>
      <td>0.19</td>
      <td>0.10</td>
      <td>True</td>
      <td>0.037000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>170</th>
      <td>PF</td>
      <td>macro</td>
      <td>current_account_gdp</td>
      <td>True</td>
      <td>0.3</td>
      <td>0.18</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>True</td>
      <td>0.012727</td>
    </tr>
    <tr>
      <th>171</th>
      <td>PF</td>
      <td>macro</td>
      <td>gdp_nominal</td>
      <td>True</td>
      <td>0.3</td>
      <td>0.18</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>True</td>
      <td>0.010909</td>
    </tr>
    <tr>
      <th>172</th>
      <td>PF</td>
      <td>macro</td>
      <td>fdi_net_inflows_gdp</td>
      <td>True</td>
      <td>0.3</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>True</td>
      <td>0.010909</td>
    </tr>
    <tr>
      <th>173</th>
      <td>PF</td>
      <td>macro</td>
      <td>population_total</td>
      <td>True</td>
      <td>0.3</td>
      <td>0.18</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>True</td>
      <td>0.007273</td>
    </tr>
    <tr>
      <th>174</th>
      <td>PF</td>
      <td>macro</td>
      <td>urban_population_pct</td>
      <td>True</td>
      <td>0.3</td>
      <td>0.18</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>True</td>
      <td>0.007273</td>
    </tr>
  </tbody>
</table>
<p>175 rows × 10 columns</p>
</div>




```python
weights_audit[
    ~weights_audit["in_decision_matrix"]
][
    ["business_line", "dimension", "variable", "global_variable_weight_in_dim"]
].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>dimension</th>
      <th>variable</th>
      <th>global_variable_weight_in_dim</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
weights_audit[
    ~weights_audit["in_decision_matrix"]
]["variable"].unique()
```




    array([], dtype=object)




```python
from src.utils import get_variable_catalog, get_world_bank_variable_catalog

var = "digital_payment_adults_pct"

catalog = get_variable_catalog(configs["variables"])

wb_catalog = get_world_bank_variable_catalog(
    configs["variables"],
    configs["data_sources"]["sources"]["world_bank"],
)

print("1. En catalog general:", var in catalog)
print("2. En catalog WB:", var in wb_catalog)
print("3. En weights.yaml:",
      any(var in dim_vars for dim_vars in configs["weights"]["variable_weights"].values()))
print("4. En master:",
      var in master["variable"].astype(str).str.strip().unique())
print("5. En wide_raw:",
      var in wide_raw.columns)
print("6. En decision_matrix:",
      var in decision_matrix.columns)

if var in wb_catalog:
    print("\nMetadata WB:")
    print(wb_catalog[var])
```

    1. En catalog general: True
    2. En catalog WB: True
    3. En weights.yaml: True
    4. En master: True
    5. En wide_raw: True
    6. En decision_matrix: True
    
    Metadata WB:
    {'source': 'world_bank', 'db': 28, 'indicator_code': 'g20.any', 'direction': 'positive', 'frequency': 'triennial', 'description': 'Adultos que hicieron o recibieron pagos digitales en el ultimo ano', 'name': 'digital_payment_adults_pct', 'dimension': 'digital_tech'}
    


```python
master["variable"].unique()
```




    array(['account_ownership', 'atms_per_100k_adults', 'bank_capital_rwa',
           'bank_concentration_5', 'bank_npl_ratio',
           'colombian_diaspora_stock',
           'commercial_bank_branches_per_100k_adults',
           'common_language_spanish', 'control_of_corruption',
           'country_risk_premium', 'current_account_gdp',
           'digital_payment_adults_pct', 'domestic_credit_private_gdp',
           'fdi_net_inflows_gdp', 'financial_system_deposits_gdp',
           'gdp_growth', 'gdp_nominal', 'gdp_per_capita_ppp',
           'geographic_distance_km', 'government_effectiveness',
           'heritage_efi', 'hofstede_idv', 'hofstede_ivr', 'hofstede_lto',
           'hofstede_mas', 'hofstede_pdi', 'hofstede_uai',
           'ict_goods_exports_pct_total_goods_exports', 'inflation_rate',
           'interest_rate_spread', 'internet_users_pct',
           'mobile_subscriptions', 'personal_remittances_gdp',
           'political_stability', 'population_total', 'regulatory_quality',
           'rule_of_law', 'secure_internet_servers_per_million',
           'stock_market_cap_gdp', 'trade_openness', 'unemployment_rate',
           'urban_population_pct', 'voice_accountability',
           'weighted_mean_applied_tariff_all_products', 'public_debt_gdp'],
          dtype=object)




```python
from src.utils import get_world_bank_variable_catalog

wb_catalog = get_world_bank_variable_catalog(
    configs["variables"],
    configs["data_sources"]["sources"]["world_bank"],
)

"digital_payment_adults_pct" in wb_catalog
```




    True




```python
from src.utils import get_variable_catalog, get_world_bank_variable_catalog

catalog = get_variable_catalog(configs["variables"])

print("En catalog general:", "digital_payment_adults_pct" in catalog)

wb_catalog = get_world_bank_variable_catalog(
    configs["variables"],
    configs["data_sources"]["sources"]["world_bank"],
)

print("En catalog WB:", "digital_payment_adults_pct" in wb_catalog)
print(wb_catalog.get("digital_payment_adults_pct"))
```

    En catalog general: True
    En catalog WB: True
    {'source': 'world_bank', 'db': 28, 'indicator_code': 'g20.any', 'direction': 'positive', 'frequency': 'triennial', 'description': 'Adultos que hicieron o recibieron pagos digitales en el ultimo ano', 'name': 'digital_payment_adults_pct', 'dimension': 'digital_tech'}
    


```python
#Validar que los pesos finales sumen 1 por línea
(
    weights_audit[weights_audit["in_decision_matrix"]]
    .groupby("business_line")["final_topsis_weight"]
    .sum()
    .round(6)
)
```




    business_line
    AD     1.0
    BD     1.0
    CIB    1.0
    IB     1.0
    PF     1.0
    Name: final_topsis_weight, dtype: float64




```python
#Esto te dirá qué variables realmente diferencian líneas.
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

weight_matrix["spread"] = (
    weight_matrix.max(axis=1) - weight_matrix.min(axis=1)
)

weight_matrix.sort_values("spread", ascending=False)#.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>business_line</th>
      <th>AD</th>
      <th>BD</th>
      <th>CIB</th>
      <th>IB</th>
      <th>PF</th>
      <th>spread</th>
    </tr>
    <tr>
      <th>variable</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>digital_payment_adults_pct</th>
      <td>0.0851</td>
      <td>0.0805</td>
      <td>0.0036</td>
      <td>0.0128</td>
      <td>0.134400</td>
      <td>0.130800</td>
    </tr>
    <tr>
      <th>stock_market_cap_gdp</th>
      <td>0.0260</td>
      <td>0.0023</td>
      <td>0.1140</td>
      <td>0.0129</td>
      <td>0.008526</td>
      <td>0.111700</td>
    </tr>
    <tr>
      <th>internet_users_pct</th>
      <td>0.0740</td>
      <td>0.1015</td>
      <td>0.0054</td>
      <td>0.0176</td>
      <td>0.067200</td>
      <td>0.096100</td>
    </tr>
    <tr>
      <th>mobile_subscriptions</th>
      <td>0.0370</td>
      <td>0.0665</td>
      <td>0.0030</td>
      <td>0.0144</td>
      <td>0.092400</td>
      <td>0.089400</td>
    </tr>
    <tr>
      <th>secure_internet_servers_per_million</th>
      <td>0.0925</td>
      <td>0.0595</td>
      <td>0.0084</td>
      <td>0.0176</td>
      <td>0.029400</td>
      <td>0.084100</td>
    </tr>
    <tr>
      <th>domestic_credit_private_gdp</th>
      <td>0.0182</td>
      <td>0.0345</td>
      <td>0.0646</td>
      <td>0.0946</td>
      <td>0.028421</td>
      <td>0.076400</td>
    </tr>
    <tr>
      <th>personal_remittances_gdp</th>
      <td>0.0052</td>
      <td>0.0046</td>
      <td>0.0076</td>
      <td>0.0043</td>
      <td>0.079579</td>
      <td>0.075279</td>
    </tr>
    <tr>
      <th>regulatory_quality</th>
      <td>0.1026</td>
      <td>0.0374</td>
      <td>0.0594</td>
      <td>0.0540</td>
      <td>0.028600</td>
      <td>0.074000</td>
    </tr>
    <tr>
      <th>financial_system_deposits_gdp</th>
      <td>0.0156</td>
      <td>0.0253</td>
      <td>0.0570</td>
      <td>0.0860</td>
      <td>0.028421</td>
      <td>0.070400</td>
    </tr>
    <tr>
      <th>rule_of_law</th>
      <td>0.0836</td>
      <td>0.0306</td>
      <td>0.0675</td>
      <td>0.0594</td>
      <td>0.022100</td>
      <td>0.061500</td>
    </tr>
    <tr>
      <th>gdp_nominal</th>
      <td>0.0072</td>
      <td>0.0275</td>
      <td>0.0672</td>
      <td>0.0220</td>
      <td>0.010909</td>
      <td>0.060000</td>
    </tr>
    <tr>
      <th>ict_goods_exports_pct_total_goods_exports</th>
      <td>0.0592</td>
      <td>0.0210</td>
      <td>0.0075</td>
      <td>0.0048</td>
      <td>0.016800</td>
      <td>0.054400</td>
    </tr>
    <tr>
      <th>account_ownership</th>
      <td>0.0143</td>
      <td>0.0644</td>
      <td>0.0114</td>
      <td>0.0516</td>
      <td>0.062526</td>
      <td>0.053000</td>
    </tr>
    <tr>
      <th>population_total</th>
      <td>0.0048</td>
      <td>0.0550</td>
      <td>0.0160</td>
      <td>0.0154</td>
      <td>0.007273</td>
      <td>0.050200</td>
    </tr>
    <tr>
      <th>control_of_corruption</th>
      <td>0.0646</td>
      <td>0.0221</td>
      <td>0.0405</td>
      <td>0.0405</td>
      <td>0.016900</td>
      <td>0.047700</td>
    </tr>
    <tr>
      <th>atms_per_100k_adults</th>
      <td>0.0111</td>
      <td>0.0105</td>
      <td>0.0009</td>
      <td>0.0064</td>
      <td>0.046200</td>
      <td>0.045300</td>
    </tr>
    <tr>
      <th>bank_npl_ratio</th>
      <td>0.0156</td>
      <td>0.0253</td>
      <td>0.0342</td>
      <td>0.0602</td>
      <td>0.022737</td>
      <td>0.044600</td>
    </tr>
    <tr>
      <th>bank_capital_rwa</th>
      <td>0.0169</td>
      <td>0.0161</td>
      <td>0.0494</td>
      <td>0.0516</td>
      <td>0.011368</td>
      <td>0.040232</td>
    </tr>
    <tr>
      <th>trade_openness</th>
      <td>0.0096</td>
      <td>0.0050</td>
      <td>0.0448</td>
      <td>0.0176</td>
      <td>0.040000</td>
      <td>0.039800</td>
    </tr>
    <tr>
      <th>commercial_bank_branches_per_100k_adults</th>
      <td>0.0111</td>
      <td>0.0105</td>
      <td>0.0012</td>
      <td>0.0064</td>
      <td>0.033600</td>
      <td>0.032400</td>
    </tr>
    <tr>
      <th>bank_concentration_5</th>
      <td>0.0078</td>
      <td>0.0391</td>
      <td>0.0152</td>
      <td>0.0344</td>
      <td>0.008526</td>
      <td>0.031300</td>
    </tr>
    <tr>
      <th>urban_population_pct</th>
      <td>0.0048</td>
      <td>0.0350</td>
      <td>0.0064</td>
      <td>0.0088</td>
      <td>0.007273</td>
      <td>0.030200</td>
    </tr>
    <tr>
      <th>government_effectiveness</th>
      <td>0.0494</td>
      <td>0.0255</td>
      <td>0.0486</td>
      <td>0.0459</td>
      <td>0.019500</td>
      <td>0.029900</td>
    </tr>
    <tr>
      <th>fdi_net_inflows_gdp</th>
      <td>0.0096</td>
      <td>0.0050</td>
      <td>0.0320</td>
      <td>0.0132</td>
      <td>0.010909</td>
      <td>0.027000</td>
    </tr>
    <tr>
      <th>interest_rate_spread</th>
      <td>0.0104</td>
      <td>0.0184</td>
      <td>0.0266</td>
      <td>0.0344</td>
      <td>0.019895</td>
      <td>0.024000</td>
    </tr>
    <tr>
      <th>gdp_per_capita_ppp</th>
      <td>0.0252</td>
      <td>0.0450</td>
      <td>0.0448</td>
      <td>0.0352</td>
      <td>0.021818</td>
      <td>0.023182</td>
    </tr>
    <tr>
      <th>public_debt_gdp</th>
      <td>0.0180</td>
      <td>0.0200</td>
      <td>0.0352</td>
      <td>0.0308</td>
      <td>0.012727</td>
      <td>0.022473</td>
    </tr>
    <tr>
      <th>political_stability</th>
      <td>0.0380</td>
      <td>0.0204</td>
      <td>0.0270</td>
      <td>0.0324</td>
      <td>0.016900</td>
      <td>0.021100</td>
    </tr>
    <tr>
      <th>current_account_gdp</th>
      <td>0.0060</td>
      <td>0.0075</td>
      <td>0.0256</td>
      <td>0.0132</td>
      <td>0.012727</td>
      <td>0.019600</td>
    </tr>
    <tr>
      <th>weighted_mean_applied_tariff_all_products</th>
      <td>0.0048</td>
      <td>0.0000</td>
      <td>0.0064</td>
      <td>0.0154</td>
      <td>0.014545</td>
      <td>0.015400</td>
    </tr>
    <tr>
      <th>unemployment_rate</th>
      <td>0.0060</td>
      <td>0.0175</td>
      <td>0.0096</td>
      <td>0.0176</td>
      <td>0.012727</td>
      <td>0.011600</td>
    </tr>
    <tr>
      <th>country_risk_premium</th>
      <td>0.0228</td>
      <td>0.0170</td>
      <td>0.0162</td>
      <td>0.0216</td>
      <td>0.013000</td>
      <td>0.009800</td>
    </tr>
    <tr>
      <th>inflation_rate</th>
      <td>0.0240</td>
      <td>0.0325</td>
      <td>0.0320</td>
      <td>0.0308</td>
      <td>0.029091</td>
      <td>0.008500</td>
    </tr>
    <tr>
      <th>voice_accountability</th>
      <td>0.0076</td>
      <td>0.0068</td>
      <td>0.0027</td>
      <td>0.0054</td>
      <td>0.005200</td>
      <td>0.004900</td>
    </tr>
    <tr>
      <th>heritage_efi</th>
      <td>0.0114</td>
      <td>0.0102</td>
      <td>0.0081</td>
      <td>0.0108</td>
      <td>0.007800</td>
      <td>0.003600</td>
    </tr>
  </tbody>
</table>
</div>




```python
for bl in weights_audit["business_line"].unique():
    print(f"\n--- {bl} ---")
    display(
        weights_audit[
            (weights_audit["business_line"] == bl)
            & (weights_audit["in_decision_matrix"])
        ][
            [
                "dimension",
                "variable",
                "has_override",
                "line_dimension_weight",
                "global_variable_weight_in_dim",
                "override_variable_weight_in_dim",
                "final_topsis_weight",
            ]
        ]
        .sort_values("final_topsis_weight", ascending=False)
        #.head(15)
    )
```

    
    --- AD ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dimension</th>
      <th>variable</th>
      <th>has_override</th>
      <th>line_dimension_weight</th>
      <th>global_variable_weight_in_dim</th>
      <th>override_variable_weight_in_dim</th>
      <th>final_topsis_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.13</td>
      <td>0.27</td>
      <td>0.1026</td>
    </tr>
    <tr>
      <th>0</th>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>True</td>
      <td>0.37</td>
      <td>0.15</td>
      <td>0.25</td>
      <td>0.0925</td>
    </tr>
    <tr>
      <th>1</th>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>True</td>
      <td>0.37</td>
      <td>0.21</td>
      <td>0.23</td>
      <td>0.0851</td>
    </tr>
    <tr>
      <th>17</th>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.13</td>
      <td>0.22</td>
      <td>0.0836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>True</td>
      <td>0.37</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>0.0740</td>
    </tr>
    <tr>
      <th>18</th>
      <td>institutional</td>
      <td>control_of_corruption</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.13</td>
      <td>0.17</td>
      <td>0.0646</td>
    </tr>
    <tr>
      <th>3</th>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>True</td>
      <td>0.37</td>
      <td>0.10</td>
      <td>0.16</td>
      <td>0.0592</td>
    </tr>
    <tr>
      <th>19</th>
      <td>institutional</td>
      <td>government_effectiveness</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.0494</td>
    </tr>
    <tr>
      <th>20</th>
      <td>institutional</td>
      <td>political_stability</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.12</td>
      <td>0.10</td>
      <td>0.0380</td>
    </tr>
    <tr>
      <th>4</th>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>True</td>
      <td>0.37</td>
      <td>0.19</td>
      <td>0.10</td>
      <td>0.0370</td>
    </tr>
    <tr>
      <th>7</th>
      <td>financial</td>
      <td>stock_market_cap_gdp</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.10</td>
      <td>0.20</td>
      <td>0.0260</td>
    </tr>
    <tr>
      <th>24</th>
      <td>macro</td>
      <td>gdp_per_capita_ppp</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.17</td>
      <td>0.21</td>
      <td>0.0252</td>
    </tr>
    <tr>
      <th>25</th>
      <td>macro</td>
      <td>inflation_rate</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.08</td>
      <td>0.20</td>
      <td>0.0240</td>
    </tr>
    <tr>
      <th>21</th>
      <td>institutional</td>
      <td>country_risk_premium</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>0.0228</td>
    </tr>
    <tr>
      <th>8</th>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.14</td>
      <td>0.0182</td>
    </tr>
    <tr>
      <th>26</th>
      <td>macro</td>
      <td>public_debt_gdp</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.08</td>
      <td>0.15</td>
      <td>0.0180</td>
    </tr>
    <tr>
      <th>9</th>
      <td>financial</td>
      <td>bank_capital_rwa</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.13</td>
      <td>0.0169</td>
    </tr>
    <tr>
      <th>10</th>
      <td>financial</td>
      <td>bank_npl_ratio</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.10</td>
      <td>0.12</td>
      <td>0.0156</td>
    </tr>
    <tr>
      <th>11</th>
      <td>financial</td>
      <td>financial_system_deposits_gdp</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.12</td>
      <td>0.0156</td>
    </tr>
    <tr>
      <th>12</th>
      <td>financial</td>
      <td>account_ownership</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.11</td>
      <td>0.0143</td>
    </tr>
    <tr>
      <th>22</th>
      <td>institutional</td>
      <td>heritage_efi</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.10</td>
      <td>0.03</td>
      <td>0.0114</td>
    </tr>
    <tr>
      <th>6</th>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>True</td>
      <td>0.37</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.0111</td>
    </tr>
    <tr>
      <th>5</th>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>True</td>
      <td>0.37</td>
      <td>0.10</td>
      <td>0.03</td>
      <td>0.0111</td>
    </tr>
    <tr>
      <th>13</th>
      <td>financial</td>
      <td>interest_rate_spread</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.0104</td>
    </tr>
    <tr>
      <th>27</th>
      <td>macro</td>
      <td>trade_openness</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.0096</td>
    </tr>
    <tr>
      <th>28</th>
      <td>macro</td>
      <td>fdi_net_inflows_gdp</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.08</td>
      <td>0.0096</td>
    </tr>
    <tr>
      <th>14</th>
      <td>financial</td>
      <td>bank_concentration_5</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.0078</td>
    </tr>
    <tr>
      <th>23</th>
      <td>institutional</td>
      <td>voice_accountability</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.09</td>
      <td>0.02</td>
      <td>0.0076</td>
    </tr>
    <tr>
      <th>29</th>
      <td>macro</td>
      <td>gdp_nominal</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>0.0072</td>
    </tr>
    <tr>
      <th>30</th>
      <td>macro</td>
      <td>unemployment_rate</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.0060</td>
    </tr>
    <tr>
      <th>31</th>
      <td>macro</td>
      <td>current_account_gdp</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.07</td>
      <td>0.05</td>
      <td>0.0060</td>
    </tr>
    <tr>
      <th>15</th>
      <td>financial</td>
      <td>personal_remittances_gdp</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>0.0052</td>
    </tr>
    <tr>
      <th>32</th>
      <td>macro</td>
      <td>population_total</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>0.0048</td>
    </tr>
    <tr>
      <th>33</th>
      <td>macro</td>
      <td>urban_population_pct</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.0048</td>
    </tr>
    <tr>
      <th>34</th>
      <td>macro</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>True</td>
      <td>0.12</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.0048</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- BD ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dimension</th>
      <th>variable</th>
      <th>has_override</th>
      <th>line_dimension_weight</th>
      <th>global_variable_weight_in_dim</th>
      <th>override_variable_weight_in_dim</th>
      <th>final_topsis_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>True</td>
      <td>0.35</td>
      <td>0.20</td>
      <td>0.29</td>
      <td>0.1015</td>
    </tr>
    <tr>
      <th>36</th>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>True</td>
      <td>0.35</td>
      <td>0.21</td>
      <td>0.23</td>
      <td>0.0805</td>
    </tr>
    <tr>
      <th>37</th>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>True</td>
      <td>0.35</td>
      <td>0.19</td>
      <td>0.19</td>
      <td>0.0665</td>
    </tr>
    <tr>
      <th>42</th>
      <td>financial</td>
      <td>account_ownership</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.13</td>
      <td>0.28</td>
      <td>0.0644</td>
    </tr>
    <tr>
      <th>38</th>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>True</td>
      <td>0.35</td>
      <td>0.15</td>
      <td>0.17</td>
      <td>0.0595</td>
    </tr>
    <tr>
      <th>59</th>
      <td>macro</td>
      <td>population_total</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.10</td>
      <td>0.22</td>
      <td>0.0550</td>
    </tr>
    <tr>
      <th>60</th>
      <td>macro</td>
      <td>gdp_per_capita_ppp</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.17</td>
      <td>0.18</td>
      <td>0.0450</td>
    </tr>
    <tr>
      <th>43</th>
      <td>financial</td>
      <td>bank_concentration_5</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.10</td>
      <td>0.17</td>
      <td>0.0391</td>
    </tr>
    <tr>
      <th>51</th>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>True</td>
      <td>0.17</td>
      <td>0.13</td>
      <td>0.22</td>
      <td>0.0374</td>
    </tr>
    <tr>
      <th>61</th>
      <td>macro</td>
      <td>urban_population_pct</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.08</td>
      <td>0.14</td>
      <td>0.0350</td>
    </tr>
    <tr>
      <th>44</th>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.13</td>
      <td>0.15</td>
      <td>0.0345</td>
    </tr>
    <tr>
      <th>62</th>
      <td>macro</td>
      <td>inflation_rate</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.08</td>
      <td>0.13</td>
      <td>0.0325</td>
    </tr>
    <tr>
      <th>52</th>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>True</td>
      <td>0.17</td>
      <td>0.13</td>
      <td>0.18</td>
      <td>0.0306</td>
    </tr>
    <tr>
      <th>63</th>
      <td>macro</td>
      <td>gdp_nominal</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.17</td>
      <td>0.11</td>
      <td>0.0275</td>
    </tr>
    <tr>
      <th>53</th>
      <td>institutional</td>
      <td>government_effectiveness</td>
      <td>True</td>
      <td>0.17</td>
      <td>0.13</td>
      <td>0.15</td>
      <td>0.0255</td>
    </tr>
    <tr>
      <th>45</th>
      <td>financial</td>
      <td>bank_npl_ratio</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.10</td>
      <td>0.11</td>
      <td>0.0253</td>
    </tr>
    <tr>
      <th>46</th>
      <td>financial</td>
      <td>financial_system_deposits_gdp</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.12</td>
      <td>0.11</td>
      <td>0.0253</td>
    </tr>
    <tr>
      <th>54</th>
      <td>institutional</td>
      <td>control_of_corruption</td>
      <td>True</td>
      <td>0.17</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.0221</td>
    </tr>
    <tr>
      <th>39</th>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>True</td>
      <td>0.35</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.0210</td>
    </tr>
    <tr>
      <th>55</th>
      <td>institutional</td>
      <td>political_stability</td>
      <td>True</td>
      <td>0.17</td>
      <td>0.12</td>
      <td>0.12</td>
      <td>0.0204</td>
    </tr>
    <tr>
      <th>64</th>
      <td>macro</td>
      <td>public_debt_gdp</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>0.0200</td>
    </tr>
    <tr>
      <th>47</th>
      <td>financial</td>
      <td>interest_rate_spread</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.0184</td>
    </tr>
    <tr>
      <th>65</th>
      <td>macro</td>
      <td>unemployment_rate</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.0175</td>
    </tr>
    <tr>
      <th>56</th>
      <td>institutional</td>
      <td>country_risk_premium</td>
      <td>True</td>
      <td>0.17</td>
      <td>0.17</td>
      <td>0.10</td>
      <td>0.0170</td>
    </tr>
    <tr>
      <th>48</th>
      <td>financial</td>
      <td>bank_capital_rwa</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.12</td>
      <td>0.07</td>
      <td>0.0161</td>
    </tr>
    <tr>
      <th>41</th>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>True</td>
      <td>0.35</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.0105</td>
    </tr>
    <tr>
      <th>40</th>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>True</td>
      <td>0.35</td>
      <td>0.10</td>
      <td>0.03</td>
      <td>0.0105</td>
    </tr>
    <tr>
      <th>57</th>
      <td>institutional</td>
      <td>heritage_efi</td>
      <td>True</td>
      <td>0.17</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.0102</td>
    </tr>
    <tr>
      <th>66</th>
      <td>macro</td>
      <td>current_account_gdp</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.07</td>
      <td>0.03</td>
      <td>0.0075</td>
    </tr>
    <tr>
      <th>58</th>
      <td>institutional</td>
      <td>voice_accountability</td>
      <td>True</td>
      <td>0.17</td>
      <td>0.09</td>
      <td>0.04</td>
      <td>0.0068</td>
    </tr>
    <tr>
      <th>67</th>
      <td>macro</td>
      <td>trade_openness</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.07</td>
      <td>0.02</td>
      <td>0.0050</td>
    </tr>
    <tr>
      <th>68</th>
      <td>macro</td>
      <td>fdi_net_inflows_gdp</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.06</td>
      <td>0.02</td>
      <td>0.0050</td>
    </tr>
    <tr>
      <th>49</th>
      <td>financial</td>
      <td>personal_remittances_gdp</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.10</td>
      <td>0.02</td>
      <td>0.0046</td>
    </tr>
    <tr>
      <th>50</th>
      <td>financial</td>
      <td>stock_market_cap_gdp</td>
      <td>True</td>
      <td>0.23</td>
      <td>0.10</td>
      <td>0.01</td>
      <td>0.0023</td>
    </tr>
    <tr>
      <th>69</th>
      <td>macro</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>True</td>
      <td>0.25</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- CIB ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dimension</th>
      <th>variable</th>
      <th>has_override</th>
      <th>line_dimension_weight</th>
      <th>global_variable_weight_in_dim</th>
      <th>override_variable_weight_in_dim</th>
      <th>final_topsis_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>77</th>
      <td>financial</td>
      <td>stock_market_cap_gdp</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.10</td>
      <td>0.30</td>
      <td>0.1140</td>
    </tr>
    <tr>
      <th>86</th>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.25</td>
      <td>0.0675</td>
    </tr>
    <tr>
      <th>94</th>
      <td>macro</td>
      <td>gdp_nominal</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.17</td>
      <td>0.21</td>
      <td>0.0672</td>
    </tr>
    <tr>
      <th>78</th>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.13</td>
      <td>0.17</td>
      <td>0.0646</td>
    </tr>
    <tr>
      <th>87</th>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.22</td>
      <td>0.0594</td>
    </tr>
    <tr>
      <th>79</th>
      <td>financial</td>
      <td>financial_system_deposits_gdp</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.12</td>
      <td>0.15</td>
      <td>0.0570</td>
    </tr>
    <tr>
      <th>80</th>
      <td>financial</td>
      <td>bank_capital_rwa</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.12</td>
      <td>0.13</td>
      <td>0.0494</td>
    </tr>
    <tr>
      <th>88</th>
      <td>institutional</td>
      <td>government_effectiveness</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.18</td>
      <td>0.0486</td>
    </tr>
    <tr>
      <th>96</th>
      <td>macro</td>
      <td>trade_openness</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.07</td>
      <td>0.14</td>
      <td>0.0448</td>
    </tr>
    <tr>
      <th>95</th>
      <td>macro</td>
      <td>gdp_per_capita_ppp</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.17</td>
      <td>0.14</td>
      <td>0.0448</td>
    </tr>
    <tr>
      <th>89</th>
      <td>institutional</td>
      <td>control_of_corruption</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.15</td>
      <td>0.0405</td>
    </tr>
    <tr>
      <th>97</th>
      <td>macro</td>
      <td>public_debt_gdp</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.08</td>
      <td>0.11</td>
      <td>0.0352</td>
    </tr>
    <tr>
      <th>81</th>
      <td>financial</td>
      <td>bank_npl_ratio</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.10</td>
      <td>0.09</td>
      <td>0.0342</td>
    </tr>
    <tr>
      <th>98</th>
      <td>macro</td>
      <td>inflation_rate</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.08</td>
      <td>0.10</td>
      <td>0.0320</td>
    </tr>
    <tr>
      <th>99</th>
      <td>macro</td>
      <td>fdi_net_inflows_gdp</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.06</td>
      <td>0.10</td>
      <td>0.0320</td>
    </tr>
    <tr>
      <th>90</th>
      <td>institutional</td>
      <td>political_stability</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.12</td>
      <td>0.10</td>
      <td>0.0270</td>
    </tr>
    <tr>
      <th>82</th>
      <td>financial</td>
      <td>interest_rate_spread</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.10</td>
      <td>0.07</td>
      <td>0.0266</td>
    </tr>
    <tr>
      <th>100</th>
      <td>macro</td>
      <td>current_account_gdp</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.0256</td>
    </tr>
    <tr>
      <th>91</th>
      <td>institutional</td>
      <td>country_risk_premium</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>0.0162</td>
    </tr>
    <tr>
      <th>101</th>
      <td>macro</td>
      <td>population_total</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.10</td>
      <td>0.05</td>
      <td>0.0160</td>
    </tr>
    <tr>
      <th>83</th>
      <td>financial</td>
      <td>bank_concentration_5</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>0.0152</td>
    </tr>
    <tr>
      <th>84</th>
      <td>financial</td>
      <td>account_ownership</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.13</td>
      <td>0.03</td>
      <td>0.0114</td>
    </tr>
    <tr>
      <th>102</th>
      <td>macro</td>
      <td>unemployment_rate</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.08</td>
      <td>0.03</td>
      <td>0.0096</td>
    </tr>
    <tr>
      <th>70</th>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>True</td>
      <td>0.03</td>
      <td>0.15</td>
      <td>0.28</td>
      <td>0.0084</td>
    </tr>
    <tr>
      <th>92</th>
      <td>institutional</td>
      <td>heritage_efi</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.10</td>
      <td>0.03</td>
      <td>0.0081</td>
    </tr>
    <tr>
      <th>85</th>
      <td>financial</td>
      <td>personal_remittances_gdp</td>
      <td>True</td>
      <td>0.38</td>
      <td>0.10</td>
      <td>0.02</td>
      <td>0.0076</td>
    </tr>
    <tr>
      <th>71</th>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>True</td>
      <td>0.03</td>
      <td>0.10</td>
      <td>0.25</td>
      <td>0.0075</td>
    </tr>
    <tr>
      <th>103</th>
      <td>macro</td>
      <td>urban_population_pct</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.08</td>
      <td>0.02</td>
      <td>0.0064</td>
    </tr>
    <tr>
      <th>104</th>
      <td>macro</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>True</td>
      <td>0.32</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.0064</td>
    </tr>
    <tr>
      <th>72</th>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>True</td>
      <td>0.03</td>
      <td>0.20</td>
      <td>0.18</td>
      <td>0.0054</td>
    </tr>
    <tr>
      <th>73</th>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>True</td>
      <td>0.03</td>
      <td>0.21</td>
      <td>0.12</td>
      <td>0.0036</td>
    </tr>
    <tr>
      <th>74</th>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>True</td>
      <td>0.03</td>
      <td>0.19</td>
      <td>0.10</td>
      <td>0.0030</td>
    </tr>
    <tr>
      <th>93</th>
      <td>institutional</td>
      <td>voice_accountability</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.09</td>
      <td>0.01</td>
      <td>0.0027</td>
    </tr>
    <tr>
      <th>75</th>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>True</td>
      <td>0.03</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <th>76</th>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>True</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.0009</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- IB ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dimension</th>
      <th>variable</th>
      <th>has_override</th>
      <th>line_dimension_weight</th>
      <th>global_variable_weight_in_dim</th>
      <th>override_variable_weight_in_dim</th>
      <th>final_topsis_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>112</th>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.13</td>
      <td>0.22</td>
      <td>0.0946</td>
    </tr>
    <tr>
      <th>113</th>
      <td>financial</td>
      <td>financial_system_deposits_gdp</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.12</td>
      <td>0.20</td>
      <td>0.0860</td>
    </tr>
    <tr>
      <th>114</th>
      <td>financial</td>
      <td>bank_npl_ratio</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.10</td>
      <td>0.14</td>
      <td>0.0602</td>
    </tr>
    <tr>
      <th>121</th>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.22</td>
      <td>0.0594</td>
    </tr>
    <tr>
      <th>122</th>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.20</td>
      <td>0.0540</td>
    </tr>
    <tr>
      <th>115</th>
      <td>financial</td>
      <td>account_ownership</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.0516</td>
    </tr>
    <tr>
      <th>116</th>
      <td>financial</td>
      <td>bank_capital_rwa</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.12</td>
      <td>0.12</td>
      <td>0.0516</td>
    </tr>
    <tr>
      <th>123</th>
      <td>institutional</td>
      <td>government_effectiveness</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.17</td>
      <td>0.0459</td>
    </tr>
    <tr>
      <th>124</th>
      <td>institutional</td>
      <td>control_of_corruption</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.15</td>
      <td>0.0405</td>
    </tr>
    <tr>
      <th>129</th>
      <td>macro</td>
      <td>gdp_per_capita_ppp</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.17</td>
      <td>0.16</td>
      <td>0.0352</td>
    </tr>
    <tr>
      <th>117</th>
      <td>financial</td>
      <td>interest_rate_spread</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.0344</td>
    </tr>
    <tr>
      <th>118</th>
      <td>financial</td>
      <td>bank_concentration_5</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.0344</td>
    </tr>
    <tr>
      <th>125</th>
      <td>institutional</td>
      <td>political_stability</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.12</td>
      <td>0.12</td>
      <td>0.0324</td>
    </tr>
    <tr>
      <th>131</th>
      <td>macro</td>
      <td>public_debt_gdp</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.14</td>
      <td>0.0308</td>
    </tr>
    <tr>
      <th>130</th>
      <td>macro</td>
      <td>inflation_rate</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.14</td>
      <td>0.0308</td>
    </tr>
    <tr>
      <th>132</th>
      <td>macro</td>
      <td>gdp_nominal</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.17</td>
      <td>0.10</td>
      <td>0.0220</td>
    </tr>
    <tr>
      <th>126</th>
      <td>institutional</td>
      <td>country_risk_premium</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.17</td>
      <td>0.08</td>
      <td>0.0216</td>
    </tr>
    <tr>
      <th>133</th>
      <td>macro</td>
      <td>unemployment_rate</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>0.0176</td>
    </tr>
    <tr>
      <th>134</th>
      <td>macro</td>
      <td>trade_openness</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.0176</td>
    </tr>
    <tr>
      <th>105</th>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>True</td>
      <td>0.08</td>
      <td>0.20</td>
      <td>0.22</td>
      <td>0.0176</td>
    </tr>
    <tr>
      <th>106</th>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>True</td>
      <td>0.08</td>
      <td>0.15</td>
      <td>0.22</td>
      <td>0.0176</td>
    </tr>
    <tr>
      <th>135</th>
      <td>macro</td>
      <td>population_total</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.10</td>
      <td>0.07</td>
      <td>0.0154</td>
    </tr>
    <tr>
      <th>136</th>
      <td>macro</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.04</td>
      <td>0.07</td>
      <td>0.0154</td>
    </tr>
    <tr>
      <th>107</th>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>True</td>
      <td>0.08</td>
      <td>0.19</td>
      <td>0.18</td>
      <td>0.0144</td>
    </tr>
    <tr>
      <th>137</th>
      <td>macro</td>
      <td>current_account_gdp</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.0132</td>
    </tr>
    <tr>
      <th>138</th>
      <td>macro</td>
      <td>fdi_net_inflows_gdp</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.0132</td>
    </tr>
    <tr>
      <th>119</th>
      <td>financial</td>
      <td>stock_market_cap_gdp</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.10</td>
      <td>0.03</td>
      <td>0.0129</td>
    </tr>
    <tr>
      <th>108</th>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>True</td>
      <td>0.08</td>
      <td>0.21</td>
      <td>0.16</td>
      <td>0.0128</td>
    </tr>
    <tr>
      <th>127</th>
      <td>institutional</td>
      <td>heritage_efi</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>0.0108</td>
    </tr>
    <tr>
      <th>139</th>
      <td>macro</td>
      <td>urban_population_pct</td>
      <td>True</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.0088</td>
    </tr>
    <tr>
      <th>110</th>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>True</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.08</td>
      <td>0.0064</td>
    </tr>
    <tr>
      <th>109</th>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>True</td>
      <td>0.08</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.0064</td>
    </tr>
    <tr>
      <th>128</th>
      <td>institutional</td>
      <td>voice_accountability</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.09</td>
      <td>0.02</td>
      <td>0.0054</td>
    </tr>
    <tr>
      <th>111</th>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>True</td>
      <td>0.08</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.0048</td>
    </tr>
    <tr>
      <th>120</th>
      <td>financial</td>
      <td>personal_remittances_gdp</td>
      <td>True</td>
      <td>0.43</td>
      <td>0.10</td>
      <td>0.01</td>
      <td>0.0043</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- PF ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dimension</th>
      <th>variable</th>
      <th>has_override</th>
      <th>line_dimension_weight</th>
      <th>global_variable_weight_in_dim</th>
      <th>override_variable_weight_in_dim</th>
      <th>final_topsis_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>140</th>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>True</td>
      <td>0.42</td>
      <td>0.21</td>
      <td>0.32</td>
      <td>0.134400</td>
    </tr>
    <tr>
      <th>141</th>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>True</td>
      <td>0.42</td>
      <td>0.19</td>
      <td>0.22</td>
      <td>0.092400</td>
    </tr>
    <tr>
      <th>147</th>
      <td>financial</td>
      <td>personal_remittances_gdp</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.10</td>
      <td>0.28</td>
      <td>0.079579</td>
    </tr>
    <tr>
      <th>142</th>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>True</td>
      <td>0.42</td>
      <td>0.20</td>
      <td>0.16</td>
      <td>0.067200</td>
    </tr>
    <tr>
      <th>148</th>
      <td>financial</td>
      <td>account_ownership</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.22</td>
      <td>0.062526</td>
    </tr>
    <tr>
      <th>143</th>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>True</td>
      <td>0.42</td>
      <td>0.05</td>
      <td>0.11</td>
      <td>0.046200</td>
    </tr>
    <tr>
      <th>164</th>
      <td>macro</td>
      <td>trade_openness</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.07</td>
      <td>0.22</td>
      <td>0.040000</td>
    </tr>
    <tr>
      <th>144</th>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>True</td>
      <td>0.42</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.033600</td>
    </tr>
    <tr>
      <th>145</th>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>True</td>
      <td>0.42</td>
      <td>0.15</td>
      <td>0.07</td>
      <td>0.029400</td>
    </tr>
    <tr>
      <th>165</th>
      <td>macro</td>
      <td>inflation_rate</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.08</td>
      <td>0.16</td>
      <td>0.029091</td>
    </tr>
    <tr>
      <th>156</th>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.22</td>
      <td>0.028600</td>
    </tr>
    <tr>
      <th>149</th>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.13</td>
      <td>0.10</td>
      <td>0.028421</td>
    </tr>
    <tr>
      <th>150</th>
      <td>financial</td>
      <td>financial_system_deposits_gdp</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.12</td>
      <td>0.10</td>
      <td>0.028421</td>
    </tr>
    <tr>
      <th>151</th>
      <td>financial</td>
      <td>bank_npl_ratio</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.022737</td>
    </tr>
    <tr>
      <th>157</th>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.17</td>
      <td>0.022100</td>
    </tr>
    <tr>
      <th>166</th>
      <td>macro</td>
      <td>gdp_per_capita_ppp</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.17</td>
      <td>0.12</td>
      <td>0.021818</td>
    </tr>
    <tr>
      <th>152</th>
      <td>financial</td>
      <td>interest_rate_spread</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.10</td>
      <td>0.07</td>
      <td>0.019895</td>
    </tr>
    <tr>
      <th>158</th>
      <td>institutional</td>
      <td>government_effectiveness</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.15</td>
      <td>0.019500</td>
    </tr>
    <tr>
      <th>159</th>
      <td>institutional</td>
      <td>political_stability</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.13</td>
      <td>0.016900</td>
    </tr>
    <tr>
      <th>160</th>
      <td>institutional</td>
      <td>control_of_corruption</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.13</td>
      <td>0.016900</td>
    </tr>
    <tr>
      <th>146</th>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>True</td>
      <td>0.42</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>0.016800</td>
    </tr>
    <tr>
      <th>167</th>
      <td>macro</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.04</td>
      <td>0.08</td>
      <td>0.014545</td>
    </tr>
    <tr>
      <th>161</th>
      <td>institutional</td>
      <td>country_risk_premium</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.17</td>
      <td>0.10</td>
      <td>0.013000</td>
    </tr>
    <tr>
      <th>169</th>
      <td>macro</td>
      <td>public_debt_gdp</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.012727</td>
    </tr>
    <tr>
      <th>170</th>
      <td>macro</td>
      <td>current_account_gdp</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.012727</td>
    </tr>
    <tr>
      <th>168</th>
      <td>macro</td>
      <td>unemployment_rate</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.012727</td>
    </tr>
    <tr>
      <th>153</th>
      <td>financial</td>
      <td>bank_capital_rwa</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.12</td>
      <td>0.04</td>
      <td>0.011368</td>
    </tr>
    <tr>
      <th>172</th>
      <td>macro</td>
      <td>fdi_net_inflows_gdp</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.010909</td>
    </tr>
    <tr>
      <th>171</th>
      <td>macro</td>
      <td>gdp_nominal</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>0.010909</td>
    </tr>
    <tr>
      <th>155</th>
      <td>financial</td>
      <td>bank_concentration_5</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.10</td>
      <td>0.03</td>
      <td>0.008526</td>
    </tr>
    <tr>
      <th>154</th>
      <td>financial</td>
      <td>stock_market_cap_gdp</td>
      <td>True</td>
      <td>0.27</td>
      <td>0.10</td>
      <td>0.03</td>
      <td>0.008526</td>
    </tr>
    <tr>
      <th>162</th>
      <td>institutional</td>
      <td>heritage_efi</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.007800</td>
    </tr>
    <tr>
      <th>173</th>
      <td>macro</td>
      <td>population_total</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>0.007273</td>
    </tr>
    <tr>
      <th>174</th>
      <td>macro</td>
      <td>urban_population_pct</td>
      <td>True</td>
      <td>0.18</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.007273</td>
    </tr>
    <tr>
      <th>163</th>
      <td>institutional</td>
      <td>voice_accountability</td>
      <td>True</td>
      <td>0.13</td>
      <td>0.09</td>
      <td>0.04</td>
      <td>0.005200</td>
    </tr>
  </tbody>
</table>
</div>



```python
configs["business_lines"]["business_lines"]["AD"].keys()
```




    dict_keys(['label', 'description', 'weight_profile', 'variable_weight_overrides', 'key_variables', 'signal_logic'])




```python
configs["business_lines"]["business_lines"]["AD"].get("variable_weight_overrides")
```




    {'macro': {'gdp_nominal': 0.06,
      'gdp_per_capita_ppp': 0.21,
      'inflation_rate': 0.2,
      'population_total': 0.04,
      'urban_population_pct': 0.04,
      'unemployment_rate': 0.05,
      'current_account_gdp': 0.05,
      'public_debt_gdp': 0.15,
      'trade_openness': 0.08,
      'fdi_net_inflows_gdp': 0.08,
      'weighted_mean_applied_tariff_all_products': 0.04},
     'financial': {'stock_market_cap_gdp': 0.2,
      'domestic_credit_private_gdp': 0.14,
      'financial_system_deposits_gdp': 0.12,
      'account_ownership': 0.11,
      'bank_capital_rwa': 0.13,
      'bank_npl_ratio': 0.12,
      'interest_rate_spread': 0.08,
      'bank_concentration_5': 0.06,
      'personal_remittances_gdp': 0.04},
     'institutional': {'regulatory_quality': 0.27,
      'rule_of_law': 0.22,
      'control_of_corruption': 0.17,
      'government_effectiveness': 0.13,
      'political_stability': 0.1,
      'country_risk_premium': 0.06,
      'heritage_efi': 0.03,
      'voice_accountability': 0.02},
     'digital_tech': {'secure_internet_servers_per_million': 0.25,
      'digital_payment_adults_pct': 0.23,
      'internet_users_pct': 0.2,
      'ict_goods_exports_pct_total_goods_exports': 0.16,
      'mobile_subscriptions': 0.1,
      'atms_per_100k_adults': 0.03,
      'commercial_bank_branches_per_100k_adults': 0.03}}




```python
rankings = {}

for bl, df_bl in results["business_line_rankings"].items():
    tmp = df_bl.copy()

    if "country_iso3" in tmp.columns:
        tmp = tmp.set_index("country_iso3")

    rankings[bl] = tmp["rank"]

rank_df = pd.DataFrame(rankings)

rank_df.corr(method="spearman")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IB</th>
      <th>PF</th>
      <th>AD</th>
      <th>BD</th>
      <th>CIB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IB</th>
      <td>1.000000</td>
      <td>0.646962</td>
      <td>0.844007</td>
      <td>0.684729</td>
      <td>0.823755</td>
    </tr>
    <tr>
      <th>PF</th>
      <td>0.646962</td>
      <td>1.000000</td>
      <td>0.813355</td>
      <td>0.895457</td>
      <td>0.610837</td>
    </tr>
    <tr>
      <th>AD</th>
      <td>0.844007</td>
      <td>0.813355</td>
      <td>1.000000</td>
      <td>0.872469</td>
      <td>0.818829</td>
    </tr>
    <tr>
      <th>BD</th>
      <td>0.684729</td>
      <td>0.895457</td>
      <td>0.872469</td>
      <td>1.000000</td>
      <td>0.667214</td>
    </tr>
    <tr>
      <th>CIB</th>
      <td>0.823755</td>
      <td>0.610837</td>
      <td>0.818829</td>
      <td>0.667214</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#validación: gaps de score

# creterio:
# gap < 0.005       empate práctico
# 0.005 - 0.015     diferencia débil
# 0.015 - 0.030     diferencia moderada
# > 0.030           diferencia material

# Si varios países tienen gaps pequeños, el output ejecutivo debe presentarse por bandas de atractividad, no solo ranking ordinal.

def assign_tier_by_score(score: float, q80: float, q60: float, q40: float) -> str:
    if score >= q80:
        return "Alta"
    if score >= q60:
        return "Media-alta"
    if score >= q40:
        return "Media"
    return "Baja"


tier_tables = {}

for bl, df_bl in results["business_line_rankings"].items():
    tmp = df_bl.copy()

    if "country_iso3" in tmp.columns:
        tmp = tmp.set_index("country_iso3")

    tmp = tmp.sort_values("score", ascending=False).copy()

    q80 = tmp["score"].quantile(0.80)
    q60 = tmp["score"].quantile(0.60)
    q40 = tmp["score"].quantile(0.40)

    tmp["attractiveness_tier"] = tmp["score"].apply(
        lambda x: assign_tier_by_score(x, q80, q60, q40)
    )

    tmp["score_gap_next"] = tmp["score"] - tmp["score"].shift(-1)
    tmp["gap_interpretation"] = tmp["score_gap_next"].apply(classify_gap)

    tier_tables[bl] = tmp[
        ["score", "rank", "attractiveness_tier", "score_gap_next", "gap_interpretation"]
    ]


```


```python
def strategic_read(row):
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


for bl, table in tier_tables.items():
    table = table.copy()
    table["strategic_read"] = table.apply(strategic_read, axis=1)
    tier_tables[bl] = table
```


```python
tier_tables["PF"]#.head(15).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>rank</th>
      <th>attractiveness_tier</th>
      <th>score_gap_next</th>
      <th>gap_interpretation</th>
      <th>strategic_read</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ESP</th>
      <td>0.668780</td>
      <td>1</td>
      <td>Alta</td>
      <td>0.036149</td>
      <td>Diferencia material</td>
      <td>Liderazgo o posicionamiento claramente diferen...</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>0.632632</td>
      <td>2</td>
      <td>Alta</td>
      <td>0.005514</td>
      <td>Diferencia débil</td>
      <td>Alta atractividad, pero no distinguible ordina...</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>0.627117</td>
      <td>3</td>
      <td>Alta</td>
      <td>0.012880</td>
      <td>Diferencia débil</td>
      <td>Alta atractividad, pero no distinguible ordina...</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>0.614237</td>
      <td>4</td>
      <td>Alta</td>
      <td>0.033931</td>
      <td>Diferencia material</td>
      <td>Liderazgo o posicionamiento claramente diferen...</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>0.580306</td>
      <td>5</td>
      <td>Alta</td>
      <td>0.019760</td>
      <td>Diferencia moderada</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0.560546</td>
      <td>6</td>
      <td>Alta</td>
      <td>0.012751</td>
      <td>Diferencia débil</td>
      <td>Alta atractividad, pero no distinguible ordina...</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>0.547795</td>
      <td>7</td>
      <td>Media-alta</td>
      <td>0.001263</td>
      <td>Empate práctico</td>
      <td>Banda competitiva media-alta; decisión requier...</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>0.546532</td>
      <td>8</td>
      <td>Media-alta</td>
      <td>0.004164</td>
      <td>Empate práctico</td>
      <td>Banda competitiva media-alta; decisión requier...</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>0.542368</td>
      <td>9</td>
      <td>Media-alta</td>
      <td>0.003683</td>
      <td>Empate práctico</td>
      <td>Banda competitiva media-alta; decisión requier...</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>0.538685</td>
      <td>10</td>
      <td>Media-alta</td>
      <td>0.011232</td>
      <td>Diferencia débil</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0.527453</td>
      <td>11</td>
      <td>Media-alta</td>
      <td>0.011197</td>
      <td>Diferencia débil</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>0.516256</td>
      <td>12</td>
      <td>Media</td>
      <td>0.001223</td>
      <td>Empate práctico</td>
      <td>Atractividad intermedia; priorizar solo si hay...</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>0.515033</td>
      <td>13</td>
      <td>Media</td>
      <td>0.015093</td>
      <td>Diferencia moderada</td>
      <td>Atractividad intermedia; priorizar solo si hay...</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>0.499940</td>
      <td>14</td>
      <td>Media</td>
      <td>0.003632</td>
      <td>Empate práctico</td>
      <td>Atractividad intermedia; priorizar solo si hay...</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>0.496309</td>
      <td>15</td>
      <td>Media</td>
      <td>0.003593</td>
      <td>Empate práctico</td>
      <td>Atractividad intermedia; priorizar solo si hay...</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.492715</td>
      <td>16</td>
      <td>Media</td>
      <td>0.015885</td>
      <td>Diferencia moderada</td>
      <td>Atractividad intermedia; priorizar solo si hay...</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>0.476830</td>
      <td>17</td>
      <td>Media</td>
      <td>0.003388</td>
      <td>Empate práctico</td>
      <td>Atractividad intermedia; priorizar solo si hay...</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0.473441</td>
      <td>18</td>
      <td>Baja</td>
      <td>0.016918</td>
      <td>Diferencia moderada</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>0.456524</td>
      <td>19</td>
      <td>Baja</td>
      <td>0.000699</td>
      <td>Empate práctico</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0.455825</td>
      <td>20</td>
      <td>Baja</td>
      <td>0.005968</td>
      <td>Diferencia débil</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>0.449856</td>
      <td>21</td>
      <td>Baja</td>
      <td>0.001578</td>
      <td>Empate práctico</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>0.448279</td>
      <td>22</td>
      <td>Baja</td>
      <td>0.020120</td>
      <td>Diferencia moderada</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>0.428158</td>
      <td>23</td>
      <td>Baja</td>
      <td>0.009900</td>
      <td>Diferencia débil</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>0.418259</td>
      <td>24</td>
      <td>Baja</td>
      <td>0.029072</td>
      <td>Diferencia moderada</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>0.389187</td>
      <td>25</td>
      <td>Baja</td>
      <td>0.008369</td>
      <td>Diferencia débil</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>0.380818</td>
      <td>26</td>
      <td>Baja</td>
      <td>0.015191</td>
      <td>Diferencia moderada</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>0.365627</td>
      <td>27</td>
      <td>Baja</td>
      <td>0.079228</td>
      <td>Diferencia material</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>0.286399</td>
      <td>28</td>
      <td>Baja</td>
      <td>NaN</td>
      <td>Sin siguiente país</td>
      <td>Lectura dependiente de drivers y restricciones...</td>
    </tr>
  </tbody>
</table>
</div>



El modelo ya no debe interpretarse como cinco rankings independientes, sino como cinco tesis de atractividad. Las líneas muestran diferencias relevantes: CIB e IB privilegian profundidad financiera, escala e institucionalidad; PF privilegia transaccionalidad y flujos; BD privilegia escala retail digital; AD privilegia madurez digital e institucional. Sin embargo, dentro de cada línea existen bandas donde las diferencias de score son pequeñas y no deben sobrerrepresentarse como posiciones ordinales fuertes.


```python
#dispersión de ranking entre líneas

# qué países cambian realmente de atractividad según la línea:
# Países con rank_range_across_lines alto son los más sensibles a la tesis de negocio. Esos son estratégicamente interesantes porque muestran donde la línea de negocio cambia la decisión.

rank_cols = ["IB", "PF", "AD", "BD", "CIB"]

rank_df_clean = rank_df[rank_cols].copy()

rank_df_clean["rank_std_across_lines"] = rank_df_clean[rank_cols].std(axis=1)
rank_df_clean["rank_range_across_lines"] = (
    rank_df_clean[rank_cols].max(axis=1)
    - rank_df_clean[rank_cols].min(axis=1)
)

rank_df_clean.sort_values(
    "rank_range_across_lines",
    ascending=False
)#.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IB</th>
      <th>PF</th>
      <th>AD</th>
      <th>BD</th>
      <th>CIB</th>
      <th>rank_std_across_lines</th>
      <th>rank_range_across_lines</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARG</th>
      <td>25</td>
      <td>7</td>
      <td>10</td>
      <td>6</td>
      <td>21</td>
      <td>8.642916</td>
      <td>19</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>19</td>
      <td>6</td>
      <td>16</td>
      <td>12</td>
      <td>23</td>
      <td>6.534524</td>
      <td>17</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>9</td>
      <td>20</td>
      <td>8</td>
      <td>23</td>
      <td>6</td>
      <td>7.726578</td>
      <td>17</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>24</td>
      <td>11</td>
      <td>18</td>
      <td>18</td>
      <td>16</td>
      <td>4.669047</td>
      <td>13</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>5</td>
      <td>18</td>
      <td>7</td>
      <td>9</td>
      <td>5</td>
      <td>5.403702</td>
      <td>13</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>23</td>
      <td>22</td>
      <td>19</td>
      <td>15</td>
      <td>12</td>
      <td>4.658326</td>
      <td>11</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>13</td>
      <td>21</td>
      <td>11</td>
      <td>19</td>
      <td>17</td>
      <td>4.147288</td>
      <td>10</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>15</td>
      <td>4.381780</td>
      <td>10</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>14</td>
      <td>16</td>
      <td>21</td>
      <td>24</td>
      <td>18</td>
      <td>3.974921</td>
      <td>10</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>17</td>
      <td>15</td>
      <td>20</td>
      <td>13</td>
      <td>22</td>
      <td>3.646917</td>
      <td>9</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>22</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>13</td>
      <td>3.193744</td>
      <td>9</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>8</td>
      <td>10</td>
      <td>14</td>
      <td>16</td>
      <td>7</td>
      <td>3.872983</td>
      <td>9</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>26</td>
      <td>19</td>
      <td>25</td>
      <td>21</td>
      <td>28</td>
      <td>3.701351</td>
      <td>9</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>7</td>
      <td>8</td>
      <td>6</td>
      <td>8</td>
      <td>14</td>
      <td>3.130495</td>
      <td>8</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>16</td>
      <td>24</td>
      <td>23</td>
      <td>20</td>
      <td>24</td>
      <td>3.435113</td>
      <td>8</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>18</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>20</td>
      <td>3.898718</td>
      <td>8</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>15</td>
      <td>14</td>
      <td>13</td>
      <td>11</td>
      <td>8</td>
      <td>2.774887</td>
      <td>7</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>20</td>
      <td>25</td>
      <td>24</td>
      <td>25</td>
      <td>19</td>
      <td>2.880972</td>
      <td>6</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>11</td>
      <td>12</td>
      <td>15</td>
      <td>14</td>
      <td>9</td>
      <td>2.387467</td>
      <td>6</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>12</td>
      <td>9</td>
      <td>12</td>
      <td>7</td>
      <td>10</td>
      <td>2.121320</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>21</td>
      <td>23</td>
      <td>22</td>
      <td>22</td>
      <td>25</td>
      <td>1.516575</td>
      <td>4</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>10</td>
      <td>13</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>1.516575</td>
      <td>4</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0.894427</td>
      <td>2</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>27</td>
      <td>0.447214</td>
      <td>1</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>26</td>
      <td>0.447214</td>
      <td>1</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.547723</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### contribuciones para todas las líneas

Importante: esto no es la descomposición matemática exacta del score TOPSIS, porque TOPSIS calcula distancia al ideal positivo/negativo. Pero sí es una explicabilidad ejecutiva robusta.

NO es “contribución exacta al TOPSIS” es Contribución ponderada post-normalización


```python
contrib_by_line = compute_all_business_line_contributions(
    decision_matrix=decision_matrix,
    weights_audit=weights_audit,
)
if isinstance(contrib_by_line, dict):
    try:
        contrib_df = pd.concat(contrib_by_line, axis=1)
    except Exception:
        contrib_df = pd.DataFrame.from_dict(contrib_by_line)
else:
    contrib_df = pd.DataFrame(contrib_by_line)

contrib_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">AD</th>
      <th>...</th>
      <th colspan="10" halign="left">PF</th>
    </tr>
    <tr>
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>dimension</th>
      <th>variable</th>
      <th>normalized_value</th>
      <th>final_topsis_weight</th>
      <th>contribution</th>
      <th>shortfall</th>
      <th>has_override</th>
      <th>override_variable_weight_in_dim</th>
      <th>...</th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>dimension</th>
      <th>variable</th>
      <th>normalized_value</th>
      <th>final_topsis_weight</th>
      <th>contribution</th>
      <th>shortfall</th>
      <th>has_override</th>
      <th>override_variable_weight_in_dim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>812</th>
      <td>AD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>0.872843</td>
      <td>0.0740</td>
      <td>0.064590</td>
      <td>0.009410</td>
      <td>True</td>
      <td>0.20</td>
      <td>...</td>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>0.872843</td>
      <td>0.0672</td>
      <td>0.058655</td>
      <td>0.008545</td>
      <td>True</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>870</th>
      <td>AD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.681100</td>
      <td>0.0851</td>
      <td>0.057962</td>
      <td>0.027138</td>
      <td>True</td>
      <td>0.23</td>
      <td>...</td>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.681100</td>
      <td>0.1344</td>
      <td>0.091540</td>
      <td>0.042860</td>
      <td>True</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>899</th>
      <td>AD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>0.578476</td>
      <td>0.0925</td>
      <td>0.053509</td>
      <td>0.038991</td>
      <td>True</td>
      <td>0.25</td>
      <td>...</td>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>0.578476</td>
      <td>0.0294</td>
      <td>0.017007</td>
      <td>0.012393</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>580</th>
      <td>AD</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>0.510843</td>
      <td>0.1026</td>
      <td>0.052412</td>
      <td>0.050188</td>
      <td>True</td>
      <td>0.27</td>
      <td>...</td>
      <td>PF</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>0.510843</td>
      <td>0.0286</td>
      <td>0.014610</td>
      <td>0.013990</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>638</th>
      <td>AD</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>0.532835</td>
      <td>0.0836</td>
      <td>0.044545</td>
      <td>0.039055</td>
      <td>True</td>
      <td>0.22</td>
      <td>...</td>
      <td>PF</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>0.532835</td>
      <td>0.0221</td>
      <td>0.011776</td>
      <td>0.010324</td>
      <td>True</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>724</th>
      <td>AD</td>
      <td>VEN</td>
      <td>institutional</td>
      <td>voice_accountability</td>
      <td>0.000000</td>
      <td>0.0076</td>
      <td>0.000000</td>
      <td>0.007600</td>
      <td>True</td>
      <td>0.02</td>
      <td>...</td>
      <td>PF</td>
      <td>VEN</td>
      <td>institutional</td>
      <td>voice_accountability</td>
      <td>0.000000</td>
      <td>0.0052</td>
      <td>0.000000</td>
      <td>0.005200</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>753</th>
      <td>AD</td>
      <td>VEN</td>
      <td>institutional</td>
      <td>control_of_corruption</td>
      <td>0.000000</td>
      <td>0.0646</td>
      <td>0.000000</td>
      <td>0.064600</td>
      <td>True</td>
      <td>0.17</td>
      <td>...</td>
      <td>PF</td>
      <td>VEN</td>
      <td>institutional</td>
      <td>control_of_corruption</td>
      <td>0.000000</td>
      <td>0.0169</td>
      <td>0.000000</td>
      <td>0.016900</td>
      <td>True</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>782</th>
      <td>AD</td>
      <td>VEN</td>
      <td>institutional</td>
      <td>country_risk_premium</td>
      <td>0.000000</td>
      <td>0.0228</td>
      <td>0.000000</td>
      <td>0.022800</td>
      <td>True</td>
      <td>0.06</td>
      <td>...</td>
      <td>PF</td>
      <td>VEN</td>
      <td>institutional</td>
      <td>country_risk_premium</td>
      <td>0.000000</td>
      <td>0.0130</td>
      <td>0.000000</td>
      <td>0.013000</td>
      <td>True</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>811</th>
      <td>AD</td>
      <td>VEN</td>
      <td>institutional</td>
      <td>heritage_efi</td>
      <td>0.000000</td>
      <td>0.0114</td>
      <td>0.000000</td>
      <td>0.011400</td>
      <td>True</td>
      <td>0.03</td>
      <td>...</td>
      <td>PF</td>
      <td>VEN</td>
      <td>institutional</td>
      <td>heritage_efi</td>
      <td>0.000000</td>
      <td>0.0078</td>
      <td>0.000000</td>
      <td>0.007800</td>
      <td>True</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>1014</th>
      <td>AD</td>
      <td>VEN</td>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>0.000000</td>
      <td>0.0592</td>
      <td>0.000000</td>
      <td>0.059200</td>
      <td>True</td>
      <td>0.16</td>
      <td>...</td>
      <td>PF</td>
      <td>VEN</td>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>0.000000</td>
      <td>0.0168</td>
      <td>0.000000</td>
      <td>0.016800</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
<p>1015 rows × 50 columns</p>
</div>



### principales drivers positivos por país y línea


```python
get_top_contributors(
    contributions=contrib_by_line["PF"],
    country_iso3="ARG",
    top_n=8,
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>dimension</th>
      <th>variable</th>
      <th>normalized_value</th>
      <th>final_topsis_weight</th>
      <th>contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>870</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.681100</td>
      <td>0.134400</td>
      <td>0.091540</td>
    </tr>
    <tr>
      <th>841</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>0.674096</td>
      <td>0.092400</td>
      <td>0.062286</td>
    </tr>
    <tr>
      <th>812</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>0.872843</td>
      <td>0.067200</td>
      <td>0.058655</td>
    </tr>
    <tr>
      <th>348</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial</td>
      <td>account_ownership</td>
      <td>0.777665</td>
      <td>0.062526</td>
      <td>0.048625</td>
    </tr>
    <tr>
      <th>957</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>0.658154</td>
      <td>0.046200</td>
      <td>0.030407</td>
    </tr>
    <tr>
      <th>899</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>0.578476</td>
      <td>0.029400</td>
      <td>0.017007</td>
    </tr>
    <tr>
      <th>377</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial</td>
      <td>interest_rate_spread</td>
      <td>0.807242</td>
      <td>0.019895</td>
      <td>0.016060</td>
    </tr>
    <tr>
      <th>928</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>0.456250</td>
      <td>0.033600</td>
      <td>0.015330</td>
    </tr>
  </tbody>
</table>
</div>



### Principales restricciones o brechas


```python
get_top_shortfalls(
    contributions=contrib_by_line["PF"],
    country_iso3="ARG",
    top_n=8,
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>dimension</th>
      <th>variable</th>
      <th>normalized_value</th>
      <th>final_topsis_weight</th>
      <th>shortfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>464</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial</td>
      <td>personal_remittances_gdp</td>
      <td>0.036989</td>
      <td>0.079579</td>
      <td>0.076635</td>
    </tr>
    <tr>
      <th>870</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.681100</td>
      <td>0.134400</td>
      <td>0.042860</td>
    </tr>
    <tr>
      <th>232</th>
      <td>PF</td>
      <td>ARG</td>
      <td>macro</td>
      <td>trade_openness</td>
      <td>0.033014</td>
      <td>0.040000</td>
      <td>0.038679</td>
    </tr>
    <tr>
      <th>841</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>0.674096</td>
      <td>0.092400</td>
      <td>0.030114</td>
    </tr>
    <tr>
      <th>522</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial</td>
      <td>financial_system_deposits_gdp</td>
      <td>0.000000</td>
      <td>0.028421</td>
      <td>0.028421</td>
    </tr>
    <tr>
      <th>58</th>
      <td>PF</td>
      <td>ARG</td>
      <td>macro</td>
      <td>inflation_rate</td>
      <td>0.137314</td>
      <td>0.029091</td>
      <td>0.025096</td>
    </tr>
    <tr>
      <th>319</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>0.323269</td>
      <td>0.028421</td>
      <td>0.019233</td>
    </tr>
    <tr>
      <th>928</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>0.456250</td>
      <td>0.033600</td>
      <td>0.018270</td>
    </tr>
  </tbody>
</table>
</div>



### Resumen por dimensión

Esto es útil para explicar de forma ejecutiva:

Ejemplo: gentina en PF destaca por digital_tech y financial, pero se limita por institutional/macro.


```python
dimension_summary_pf = summarize_contributions_by_dimension(
    contrib_by_line["PF"]
)

dimension_summary_pf[
    dimension_summary_pf["country_iso3"] == "ARG"
]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>dimension</th>
      <th>contribution</th>
      <th>shortfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>0.275254</td>
      <td>0.144746</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial</td>
      <td>0.104048</td>
      <td>0.165952</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PF</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>0.069741</td>
      <td>0.060259</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PF</td>
      <td>ARG</td>
      <td>macro</td>
      <td>0.063450</td>
      <td>0.116550</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(
    generate_country_line_explanation(
        contributions=contrib_by_line["PF"],
        country_iso3="ARG",
        top_n=3,
    )
)
```

    ARG en PF está impulsado principalmente por digital_payment_adults_pct (0.092), mobile_subscriptions (0.062), internet_users_pct (0.059). Sus principales restricciones relativas son personal_remittances_gdp (0.077), digital_payment_adults_pct (0.043), trade_openness (0.039).
    

### Comparar un país entre líneas

Esto es muy útil para casos como Argentina, Surinam, Uruguay o Bahamas. Esto te permite ver por qué Argentina sube en PF y BD, pero cae en IB y CIB.


```python
compare_country_across_lines(
    contrib_by_line=contrib_by_line,
    country_iso3="ARG",
    top_n=5,
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>dimension</th>
      <th>variable</th>
      <th>normalized_value</th>
      <th>final_topsis_weight</th>
      <th>contribution</th>
      <th>shortfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>0.872843</td>
      <td>0.074000</td>
      <td>0.064590</td>
      <td>0.009410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.681100</td>
      <td>0.085100</td>
      <td>0.057962</td>
      <td>0.027138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>0.578476</td>
      <td>0.092500</td>
      <td>0.053509</td>
      <td>0.038991</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AD</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>0.510843</td>
      <td>0.102600</td>
      <td>0.052412</td>
      <td>0.050188</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AD</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>0.532835</td>
      <td>0.083600</td>
      <td>0.044545</td>
      <td>0.039055</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>0.872843</td>
      <td>0.101500</td>
      <td>0.088594</td>
      <td>0.012906</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.681100</td>
      <td>0.080500</td>
      <td>0.054829</td>
      <td>0.025671</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BD</td>
      <td>ARG</td>
      <td>financial</td>
      <td>account_ownership</td>
      <td>0.777665</td>
      <td>0.064400</td>
      <td>0.050082</td>
      <td>0.014318</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BD</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>0.674096</td>
      <td>0.066500</td>
      <td>0.044827</td>
      <td>0.021673</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BD</td>
      <td>ARG</td>
      <td>macro</td>
      <td>population_total</td>
      <td>0.717027</td>
      <td>0.055000</td>
      <td>0.039436</td>
      <td>0.015564</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CIB</td>
      <td>ARG</td>
      <td>financial</td>
      <td>bank_capital_rwa</td>
      <td>0.858999</td>
      <td>0.049400</td>
      <td>0.042435</td>
      <td>0.006965</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CIB</td>
      <td>ARG</td>
      <td>macro</td>
      <td>gdp_nominal</td>
      <td>0.581690</td>
      <td>0.067200</td>
      <td>0.039090</td>
      <td>0.028110</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CIB</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>0.532835</td>
      <td>0.067500</td>
      <td>0.035966</td>
      <td>0.031534</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CIB</td>
      <td>ARG</td>
      <td>financial</td>
      <td>stock_market_cap_gdp</td>
      <td>0.283868</td>
      <td>0.114000</td>
      <td>0.032361</td>
      <td>0.081639</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CIB</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>0.510843</td>
      <td>0.059400</td>
      <td>0.030344</td>
      <td>0.029056</td>
    </tr>
    <tr>
      <th>15</th>
      <td>IB</td>
      <td>ARG</td>
      <td>financial</td>
      <td>bank_capital_rwa</td>
      <td>0.858999</td>
      <td>0.051600</td>
      <td>0.044324</td>
      <td>0.007276</td>
    </tr>
    <tr>
      <th>16</th>
      <td>IB</td>
      <td>ARG</td>
      <td>financial</td>
      <td>account_ownership</td>
      <td>0.777665</td>
      <td>0.051600</td>
      <td>0.040128</td>
      <td>0.011472</td>
    </tr>
    <tr>
      <th>17</th>
      <td>IB</td>
      <td>ARG</td>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>0.532835</td>
      <td>0.059400</td>
      <td>0.031650</td>
      <td>0.027750</td>
    </tr>
    <tr>
      <th>18</th>
      <td>IB</td>
      <td>ARG</td>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>0.323269</td>
      <td>0.094600</td>
      <td>0.030581</td>
      <td>0.064019</td>
    </tr>
    <tr>
      <th>19</th>
      <td>IB</td>
      <td>ARG</td>
      <td>financial</td>
      <td>interest_rate_spread</td>
      <td>0.807242</td>
      <td>0.034400</td>
      <td>0.027769</td>
      <td>0.006631</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.681100</td>
      <td>0.134400</td>
      <td>0.091540</td>
      <td>0.042860</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>0.674096</td>
      <td>0.092400</td>
      <td>0.062286</td>
      <td>0.030114</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>0.872843</td>
      <td>0.067200</td>
      <td>0.058655</td>
      <td>0.008545</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial</td>
      <td>account_ownership</td>
      <td>0.777665</td>
      <td>0.062526</td>
      <td>0.048625</td>
      <td>0.013902</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>0.658154</td>
      <td>0.046200</td>
      <td>0.030407</td>
      <td>0.015793</td>
    </tr>
  </tbody>
</table>
</div>



### Comparar países dentro de una misma línea

Ejemplo: explicar por qué España lidera PF frente a USA, CAN o CHL.


```python
compare_countries_in_line(
    contributions=contrib_by_line["PF"],
    countries=["ESP", "USA", "CAN", "CHL"],
    top_n=12,
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>country_iso3</th>
      <th>dimension</th>
      <th>variable</th>
      <th>CAN</th>
      <th>CHL</th>
      <th>ESP</th>
      <th>USA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>0.040564</td>
      <td>0.025543</td>
      <td>0.032109</td>
      <td>0.039633</td>
    </tr>
    <tr>
      <th>1</th>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>0.020043</td>
      <td>0.012455</td>
      <td>0.026054</td>
      <td>0.023821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.134400</td>
      <td>0.111341</td>
      <td>0.133058</td>
      <td>0.125574</td>
    </tr>
    <tr>
      <th>3</th>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>0.065230</td>
      <td>0.066965</td>
      <td>0.067200</td>
      <td>0.065708</td>
    </tr>
    <tr>
      <th>4</th>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>0.024000</td>
      <td>0.056000</td>
      <td>0.054008</td>
      <td>0.039840</td>
    </tr>
    <tr>
      <th>5</th>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>0.022860</td>
      <td>0.019310</td>
      <td>0.022032</td>
      <td>0.027574</td>
    </tr>
    <tr>
      <th>6</th>
      <td>financial</td>
      <td>account_ownership</td>
      <td>0.062526</td>
      <td>0.051399</td>
      <td>0.062506</td>
      <td>0.061373</td>
    </tr>
    <tr>
      <th>7</th>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>0.028421</td>
      <td>0.024026</td>
      <td>0.023727</td>
      <td>0.019961</td>
    </tr>
    <tr>
      <th>8</th>
      <td>financial</td>
      <td>financial_system_deposits_gdp</td>
      <td>0.028312</td>
      <td>0.017461</td>
      <td>0.028421</td>
      <td>0.025575</td>
    </tr>
    <tr>
      <th>9</th>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>0.028600</td>
      <td>0.023507</td>
      <td>0.022106</td>
      <td>0.028515</td>
    </tr>
    <tr>
      <th>10</th>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>0.022100</td>
      <td>0.017458</td>
      <td>0.018910</td>
      <td>0.019091</td>
    </tr>
    <tr>
      <th>11</th>
      <td>macro</td>
      <td>inflation_rate</td>
      <td>0.028773</td>
      <td>0.028554</td>
      <td>0.028728</td>
      <td>0.028708</td>
    </tr>
  </tbody>
</table>
</div>



Para tener una tabla ejecutiva con score, rank y drivers:


```python
pf_explainability = build_explainability_table_for_line(
    ranking_df=results["business_line_rankings"]["PF"],
    contributions=contrib_by_line["PF"],
    top_n=3,
)

pf_explainability#.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_iso3</th>
      <th>score</th>
      <th>rank</th>
      <th>top_drivers</th>
      <th>top_constraints</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ESP</td>
      <td>0.668780</td>
      <td>1</td>
      <td>digital_payment_adults_pct; internet_users_pct...</td>
      <td>personal_remittances_gdp; mobile_subscriptions...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>USA</td>
      <td>0.632632</td>
      <td>2</td>
      <td>digital_payment_adults_pct; internet_users_pct...</td>
      <td>personal_remittances_gdp; mobile_subscriptions...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CAN</td>
      <td>0.627117</td>
      <td>3</td>
      <td>digital_payment_adults_pct; internet_users_pct...</td>
      <td>personal_remittances_gdp; mobile_subscriptions...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CHL</td>
      <td>0.614237</td>
      <td>4</td>
      <td>digital_payment_adults_pct; internet_users_pct...</td>
      <td>personal_remittances_gdp; mobile_subscriptions...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>URY</td>
      <td>0.580306</td>
      <td>5</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
      <td>personal_remittances_gdp; digital_payment_adul...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SUR</td>
      <td>0.560546</td>
      <td>6</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
      <td>digital_payment_adults_pct; personal_remittanc...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ARG</td>
      <td>0.547795</td>
      <td>7</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
      <td>personal_remittances_gdp; digital_payment_adul...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CRI</td>
      <td>0.546532</td>
      <td>8</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
      <td>personal_remittances_gdp; digital_payment_adul...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BRA</td>
      <td>0.542368</td>
      <td>9</td>
      <td>digital_payment_adults_pct; account_ownership;...</td>
      <td>personal_remittances_gdp; mobile_subscriptions...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JAM</td>
      <td>0.538685</td>
      <td>10</td>
      <td>personal_remittances_gdp; internet_users_pct; ...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GUY</td>
      <td>0.527453</td>
      <td>11</td>
      <td>digital_payment_adults_pct; internet_users_pct...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TTO</td>
      <td>0.516256</td>
      <td>12</td>
      <td>digital_payment_adults_pct; internet_users_pct...</td>
      <td>personal_remittances_gdp; digital_payment_adul...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PAN</td>
      <td>0.515033</td>
      <td>13</td>
      <td>mobile_subscriptions; digital_payment_adults_p...</td>
      <td>digital_payment_adults_pct; personal_remittanc...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>DOM</td>
      <td>0.499940</td>
      <td>14</td>
      <td>internet_users_pct; digital_payment_adults_pct...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PRY</td>
      <td>0.496309</td>
      <td>15</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
      <td>digital_payment_adults_pct; personal_remittanc...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SLV</td>
      <td>0.492715</td>
      <td>16</td>
      <td>mobile_subscriptions; personal_remittances_gdp...</td>
      <td>digital_payment_adults_pct; account_ownership;...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PER</td>
      <td>0.476830</td>
      <td>17</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
      <td>digital_payment_adults_pct; personal_remittanc...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>BHS</td>
      <td>0.473441</td>
      <td>18</td>
      <td>internet_users_pct; digital_payment_adults_pct...</td>
      <td>digital_payment_adults_pct; personal_remittanc...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>VEN</td>
      <td>0.456524</td>
      <td>19</td>
      <td>digital_payment_adults_pct; account_ownership;...</td>
      <td>mobile_subscriptions; personal_remittances_gdp...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BRB</td>
      <td>0.455825</td>
      <td>20</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
      <td>digital_payment_adults_pct; personal_remittanc...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BLZ</td>
      <td>0.449856</td>
      <td>21</td>
      <td>digital_payment_adults_pct; internet_users_pct...</td>
      <td>mobile_subscriptions; digital_payment_adults_p...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>MEX</td>
      <td>0.448279</td>
      <td>22</td>
      <td>internet_users_pct; mobile_subscriptions; digi...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ECU</td>
      <td>0.428158</td>
      <td>23</td>
      <td>digital_payment_adults_pct; personal_remittanc...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BOL</td>
      <td>0.418259</td>
      <td>24</td>
      <td>internet_users_pct; digital_payment_adults_pct...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GTM</td>
      <td>0.389187</td>
      <td>25</td>
      <td>personal_remittances_gdp; mobile_subscriptions...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>HND</td>
      <td>0.380818</td>
      <td>26</td>
      <td>personal_remittances_gdp; inflation_rate; dome...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NIC</td>
      <td>0.365627</td>
      <td>27</td>
      <td>personal_remittances_gdp; mobile_subscriptions...</td>
      <td>digital_payment_adults_pct; account_ownership;...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>HTI</td>
      <td>0.286399</td>
      <td>28</td>
      <td>personal_remittances_gdp; inflation_rate; digi...</td>
      <td>digital_payment_adults_pct; mobile_subscriptio...</td>
    </tr>
  </tbody>
</table>
</div>



### Validación de consistencia

Para cada país-línea, la suma de contribuciones debería ser aproximadamente:
contribution_sum = normalized_values * weights
No necesariamente igual al score TOPSIS.


```python
check = (
    contrib_by_line["PF"]
    .groupby("country_iso3")["contribution"]
    .sum()
    .sort_values(ascending=False)
)

check#.head()
```




    country_iso3
    CAN    0.720717
    USA    0.713072
    ESP    0.703176
    CHL    0.643459
    URY    0.606650
    CRI    0.578843
    PAN    0.556241
    JAM    0.553253
    BRA    0.531795
    DOM    0.527727
    BHS    0.523927
    GUY    0.523496
    SUR    0.522206
    TTO    0.518906
    COL    0.517367
    ARG    0.512492
    SLV    0.510865
    MEX    0.500077
    PRY    0.495672
    PER    0.491994
    BRB    0.482536
    BLZ    0.473189
    ECU    0.446576
    BOL    0.444772
    GTM    0.429659
    HND    0.405650
    NIC    0.368952
    VEN    0.367237
    HTI    0.214732
    Name: contribution, dtype: float64



La contribución aditiva responde:
¿Qué variables explican positivamente el score dado el valor normalizado y el peso?


```python
#Esto genera un “score aditivo aproximado”. compararlo con TOPSIS:
# Si la correlación es alta, la explicación aditiva es razonablemente alineada con el ranking TOPSIS.
pf_ranking = results["business_line_rankings"]["PF"].copy()

if "country_iso3" in pf_ranking.columns:
    pf_ranking = pf_ranking.set_index("country_iso3")

comparison = pd.DataFrame({
    "topsis_score": pf_ranking["score"],
    "additive_contribution_score": check,
})

comparison.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>topsis_score</th>
      <th>additive_contribution_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>topsis_score</th>
      <td>1.000000</td>
      <td>0.941669</td>
    </tr>
    <tr>
      <th>additive_contribution_score</th>
      <td>0.941669</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Analisis marginal

La lógica es:

marginal_effect(variable, país, línea) = score_TOPSIS_completo(país, línea) - score_TOPSIS_sin_variable(país, línea)

Interpretación:

- marginal_effect > 0: la variable ayuda al país en esa línea.
- marginal_effect < 0: la variable lo está penalizando; al quitarla, el score mejora.
- abs(marginal_effect) alto: la variable es materialmente relevante para ese país/línea.
- abs(marginal_effect) cercano a cero: la variable es poco determinante.

No es causalidad. Es una prueba de sensibilidad tipo leave-one-variable-out sobre TOPSIS.

El marginal responde:

Qué tanto cambiaría el score TOPSIS si esta variable no existiera en el modelo?

ejemplo:

- digital_payment_adults_pct puede tener alta contribución en PF, pero si al removerla el ranking casi no cambia, no es un driver robusto.

- country_risk_premium puede tener contribución baja, pero si al removerla mejora mucho el score de un país, es una restricción crítica.



```python

variable_catalog = get_variable_catalog(configs["variables"])

marginal_by_line = compute_all_marginal_effects(
    decision_matrix=decision_matrix,
    weights_audit=weights_audit,
    variable_catalog=variable_catalog,
    distance_metric=configs["settings"]["scoring"]["topsis"].get(
        "distance_metric",
        "euclidean",
    ),
)

```

    2026-06-12 21:41:05 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:05 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:05 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.797 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.797 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.799 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.806 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.801 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.800 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.799 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.797 (USA)
    2026-06-12 21:41:06 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.799 (USA)
    2026-06-12 21:41:07 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:07 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:07 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.802 (USA)
    2026-06-12 21:41:07 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.776 (USA)
    2026-06-12 21:41:07 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.795 (USA)
    2026-06-12 21:41:07 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.792 (USA)
    2026-06-12 21:41:07 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.806 (USA)
    2026-06-12 21:41:08 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:08 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:08 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.797 (USA)
    2026-06-12 21:41:08 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:09 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.788 (USA)
    2026-06-12 21:41:09 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.811 (USA)
    2026-06-12 21:41:09 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.786 (USA)
    2026-06-12 21:41:10 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (USA)
    2026-06-12 21:41:10 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:10 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:10 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.839 (CAN)
    2026-06-12 21:41:11 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:11 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.794 (USA)
    2026-06-12 21:41:12 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.791 (USA)
    2026-06-12 21:41:12 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.794 (USA)
    2026-06-12 21:41:12 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.789 (USA)
    2026-06-12 21:41:12 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:13 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.795 (USA)
    2026-06-12 21:41:13 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.797 (USA)
    2026-06-12 21:41:13 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.808 (USA)
    2026-06-12 21:41:14 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.797 (USA)
    2026-06-12 21:41:14 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:14 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:14 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:15 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.787 (USA)
    2026-06-12 21:41:15 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.795 (USA)
    2026-06-12 21:41:15 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.795 (USA)
    2026-06-12 21:41:16 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:16 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:16 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.792 (USA)
    2026-06-12 21:41:16 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.795 (USA)
    2026-06-12 21:41:17 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.800 (USA)
    2026-06-12 21:41:17 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.793 (USA)
    2026-06-12 21:41:17 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.795 (USA)
    2026-06-12 21:41:18 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.795 (USA)
    2026-06-12 21:41:18 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.798 (USA)
    2026-06-12 21:41:18 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:18 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:19 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.795 (USA)
    2026-06-12 21:41:19 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:19 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.771 (USA)
    2026-06-12 21:41:20 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.854 (USA)
    2026-06-12 21:41:20 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (USA)
    2026-06-12 21:41:20 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.789 (USA)
    2026-06-12 21:41:20 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:21 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.796 (USA)
    2026-06-12 21:41:21 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.800 (USA)
    2026-06-12 21:41:21 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:22 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.756 (CAN)
    2026-06-12 21:41:22 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.754 (CAN)
    2026-06-12 21:41:22 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.751 (CAN)
    2026-06-12 21:41:22 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.754 (CAN)
    2026-06-12 21:41:22 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:23 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.754 (CAN)
    2026-06-12 21:41:23 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.761 (CAN)
    2026-06-12 21:41:23 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.758 (CAN)
    2026-06-12 21:41:23 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.781 (CAN)
    2026-06-12 21:41:23 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.761 (CAN)
    2026-06-12 21:41:23 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:23 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.743 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.752 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.751 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.721 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.755 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.756 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.745 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.786 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.745 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.748 (CAN)
    2026-06-12 21:41:24 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.742 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.749 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:25 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:26 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:26 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.753 (CAN)
    2026-06-12 21:41:26 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.754 (CAN)
    2026-06-12 21:41:26 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:26 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.785 (CAN)
    2026-06-12 21:41:26 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.785 (CAN)
    2026-06-12 21:41:26 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.782 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.785 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.785 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.787 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.790 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.789 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.786 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.761 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.778 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.782 (CAN)
    2026-06-12 21:41:27 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.776 (CAN)
    2026-06-12 21:41:28 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:28 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.785 (CAN)
    2026-06-12 21:41:28 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.802 (CAN)
    2026-06-12 21:41:28 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.766 (CAN)
    2026-06-12 21:41:28 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.835 (CAN)
    2026-06-12 21:41:28 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.777 (CAN)
    2026-06-12 21:41:28 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.779 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.776 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.780 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.783 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.787 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:29 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:30 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:30 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.784 (CAN)
    2026-06-12 21:41:30 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.785 (CAN)
    2026-06-12 21:41:30 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:30 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:30 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:30 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.666 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.670 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.670 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.679 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.668 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.667 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.656 (ESP)
    2026-06-12 21:41:31 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.668 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.670 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.760 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.666 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.668 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.668 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.668 (ESP)
    2026-06-12 21:41:32 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.668 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.669 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.654 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.679 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.612 (SLV)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.668 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.667 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.668 (ESP)
    2026-06-12 21:41:33 | INFO     | src.scoring.ranking:rank:133 | TOPSIS completado: 29 paises | score max=0.672 (ESP)
    

#### Identificar drivers robustos de un país en una línea

Variables con score_effect positivo y alto son drivers robustos.


```python
pf_marginal = marginal_by_line["PF"]

(
    pf_marginal[
        pf_marginal["country_iso3"] == "ARG"
    ]
    .sort_values("score_effect", ascending=False)
    #.head(10)
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>removed_variable</th>
      <th>dimension</th>
      <th>final_topsis_weight</th>
      <th>score_full</th>
      <th>score_without_variable</th>
      <th>score_effect</th>
      <th>abs_score_effect</th>
      <th>rank_full</th>
      <th>rank_without_variable</th>
      <th>rank_effect</th>
      <th>effect_type</th>
      <th>has_override</th>
      <th>override_variable_weight_in_dim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>876</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_payment_adults_pct</td>
      <td>digital_tech</td>
      <td>0.134400</td>
      <td>0.547795</td>
      <td>0.502110</td>
      <td>0.045685</td>
      <td>0.045685</td>
      <td>7</td>
      <td>16</td>
      <td>9</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>818</th>
      <td>PF</td>
      <td>ARG</td>
      <td>internet_users_pct</td>
      <td>digital_tech</td>
      <td>0.067200</td>
      <td>0.547795</td>
      <td>0.526430</td>
      <td>0.021365</td>
      <td>0.021365</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>847</th>
      <td>PF</td>
      <td>ARG</td>
      <td>mobile_subscriptions</td>
      <td>digital_tech</td>
      <td>0.092400</td>
      <td>0.547795</td>
      <td>0.530753</td>
      <td>0.017042</td>
      <td>0.017042</td>
      <td>7</td>
      <td>12</td>
      <td>5</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>354</th>
      <td>PF</td>
      <td>ARG</td>
      <td>account_ownership</td>
      <td>financial</td>
      <td>0.062526</td>
      <td>0.547795</td>
      <td>0.534796</td>
      <td>0.012999</td>
      <td>0.012999</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>963</th>
      <td>PF</td>
      <td>ARG</td>
      <td>atms_per_100k_adults</td>
      <td>digital_tech</td>
      <td>0.046200</td>
      <td>0.547795</td>
      <td>0.544429</td>
      <td>0.003366</td>
      <td>0.003366</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>383</th>
      <td>PF</td>
      <td>ARG</td>
      <td>interest_rate_spread</td>
      <td>financial</td>
      <td>0.019895</td>
      <td>0.547795</td>
      <td>0.546405</td>
      <td>0.001389</td>
      <td>0.001389</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>557</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_capital_rwa</td>
      <td>financial</td>
      <td>0.011368</td>
      <td>0.547795</td>
      <td>0.547259</td>
      <td>0.000536</td>
      <td>0.000536</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>905</th>
      <td>PF</td>
      <td>ARG</td>
      <td>secure_internet_servers_per_million</td>
      <td>digital_tech</td>
      <td>0.029400</td>
      <td>0.547795</td>
      <td>0.547417</td>
      <td>0.000378</td>
      <td>0.000378</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>296</th>
      <td>PF</td>
      <td>ARG</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>macro</td>
      <td>0.014545</td>
      <td>0.547795</td>
      <td>0.547467</td>
      <td>0.000328</td>
      <td>0.000328</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>122</th>
      <td>PF</td>
      <td>ARG</td>
      <td>urban_population_pct</td>
      <td>macro</td>
      <td>0.007273</td>
      <td>0.547795</td>
      <td>0.547516</td>
      <td>0.000279</td>
      <td>0.000279</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>760</th>
      <td>PF</td>
      <td>ARG</td>
      <td>country_risk_premium</td>
      <td>institutional</td>
      <td>0.013000</td>
      <td>0.547795</td>
      <td>0.547587</td>
      <td>0.000207</td>
      <td>0.000207</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>151</th>
      <td>PF</td>
      <td>ARG</td>
      <td>unemployment_rate</td>
      <td>macro</td>
      <td>0.012727</td>
      <td>0.547795</td>
      <td>0.547605</td>
      <td>0.000190</td>
      <td>0.000190</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>615</th>
      <td>PF</td>
      <td>ARG</td>
      <td>government_effectiveness</td>
      <td>institutional</td>
      <td>0.019500</td>
      <td>0.547795</td>
      <td>0.547608</td>
      <td>0.000187</td>
      <td>0.000187</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>93</th>
      <td>PF</td>
      <td>ARG</td>
      <td>population_total</td>
      <td>macro</td>
      <td>0.007273</td>
      <td>0.547795</td>
      <td>0.547672</td>
      <td>0.000123</td>
      <td>0.000123</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>499</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_concentration_5</td>
      <td>financial</td>
      <td>0.008526</td>
      <td>0.547795</td>
      <td>0.547713</td>
      <td>0.000082</td>
      <td>0.000082</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>789</th>
      <td>PF</td>
      <td>ARG</td>
      <td>heritage_efi</td>
      <td>institutional</td>
      <td>0.007800</td>
      <td>0.547795</td>
      <td>0.547731</td>
      <td>0.000064</td>
      <td>0.000064</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PF</td>
      <td>ARG</td>
      <td>gdp_nominal</td>
      <td>macro</td>
      <td>0.010909</td>
      <td>0.547795</td>
      <td>0.547738</td>
      <td>0.000057</td>
      <td>0.000057</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>702</th>
      <td>PF</td>
      <td>ARG</td>
      <td>voice_accountability</td>
      <td>institutional</td>
      <td>0.005200</td>
      <td>0.547795</td>
      <td>0.547752</td>
      <td>0.000043</td>
      <td>0.000043</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>209</th>
      <td>PF</td>
      <td>ARG</td>
      <td>public_debt_gdp</td>
      <td>macro</td>
      <td>0.012727</td>
      <td>0.547795</td>
      <td>0.547869</td>
      <td>-0.000074</td>
      <td>0.000074</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>644</th>
      <td>PF</td>
      <td>ARG</td>
      <td>rule_of_law</td>
      <td>institutional</td>
      <td>0.022100</td>
      <td>0.547795</td>
      <td>0.547899</td>
      <td>-0.000104</td>
      <td>0.000104</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>673</th>
      <td>PF</td>
      <td>ARG</td>
      <td>political_stability</td>
      <td>institutional</td>
      <td>0.016900</td>
      <td>0.547795</td>
      <td>0.547907</td>
      <td>-0.000112</td>
      <td>0.000112</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>441</th>
      <td>PF</td>
      <td>ARG</td>
      <td>stock_market_cap_gdp</td>
      <td>financial</td>
      <td>0.008526</td>
      <td>0.547795</td>
      <td>0.548080</td>
      <td>-0.000286</td>
      <td>0.000286</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>267</th>
      <td>PF</td>
      <td>ARG</td>
      <td>fdi_net_inflows_gdp</td>
      <td>macro</td>
      <td>0.010909</td>
      <td>0.547795</td>
      <td>0.548180</td>
      <td>-0.000385</td>
      <td>0.000385</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>180</th>
      <td>PF</td>
      <td>ARG</td>
      <td>current_account_gdp</td>
      <td>macro</td>
      <td>0.012727</td>
      <td>0.547795</td>
      <td>0.548225</td>
      <td>-0.000430</td>
      <td>0.000430</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>586</th>
      <td>PF</td>
      <td>ARG</td>
      <td>regulatory_quality</td>
      <td>institutional</td>
      <td>0.028600</td>
      <td>0.547795</td>
      <td>0.548231</td>
      <td>-0.000436</td>
      <td>0.000436</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>731</th>
      <td>PF</td>
      <td>ARG</td>
      <td>control_of_corruption</td>
      <td>institutional</td>
      <td>0.016900</td>
      <td>0.547795</td>
      <td>0.548447</td>
      <td>-0.000652</td>
      <td>0.000652</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>412</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_npl_ratio</td>
      <td>financial</td>
      <td>0.022737</td>
      <td>0.547795</td>
      <td>0.548714</td>
      <td>-0.000919</td>
      <td>0.000919</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>934</th>
      <td>PF</td>
      <td>ARG</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>digital_tech</td>
      <td>0.033600</td>
      <td>0.547795</td>
      <td>0.549308</td>
      <td>-0.001513</td>
      <td>0.001513</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PF</td>
      <td>ARG</td>
      <td>gdp_per_capita_ppp</td>
      <td>macro</td>
      <td>0.021818</td>
      <td>0.547795</td>
      <td>0.549338</td>
      <td>-0.001543</td>
      <td>0.001543</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>992</th>
      <td>PF</td>
      <td>ARG</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>digital_tech</td>
      <td>0.016800</td>
      <td>0.547795</td>
      <td>0.550228</td>
      <td>-0.002433</td>
      <td>0.002433</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>325</th>
      <td>PF</td>
      <td>ARG</td>
      <td>domestic_credit_private_gdp</td>
      <td>financial</td>
      <td>0.028421</td>
      <td>0.547795</td>
      <td>0.550510</td>
      <td>-0.002716</td>
      <td>0.002716</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>64</th>
      <td>PF</td>
      <td>ARG</td>
      <td>inflation_rate</td>
      <td>macro</td>
      <td>0.029091</td>
      <td>0.547795</td>
      <td>0.553213</td>
      <td>-0.005418</td>
      <td>0.005418</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>528</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial_system_deposits_gdp</td>
      <td>financial</td>
      <td>0.028421</td>
      <td>0.547795</td>
      <td>0.554906</td>
      <td>-0.007112</td>
      <td>0.007112</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>238</th>
      <td>PF</td>
      <td>ARG</td>
      <td>trade_openness</td>
      <td>macro</td>
      <td>0.040000</td>
      <td>0.547795</td>
      <td>0.561278</td>
      <td>-0.013483</td>
      <td>0.013483</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>470</th>
      <td>PF</td>
      <td>ARG</td>
      <td>personal_remittances_gdp</td>
      <td>financial</td>
      <td>0.079579</td>
      <td>0.547795</td>
      <td>0.611169</td>
      <td>-0.063374</td>
      <td>0.063374</td>
      <td>7</td>
      <td>6</td>
      <td>-1</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.28</td>
    </tr>
  </tbody>
</table>
</div>



#### Identificar restricciones robustas


Variables con score_effect negativo son variables que penalizan.
Al removerlas, el score TOPSIS mejora.



```python
(
    pf_marginal[
        pf_marginal["country_iso3"] == "ARG"
    ]
    .sort_values("score_effect", ascending=True)
    #.head(10)
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>removed_variable</th>
      <th>dimension</th>
      <th>final_topsis_weight</th>
      <th>score_full</th>
      <th>score_without_variable</th>
      <th>score_effect</th>
      <th>abs_score_effect</th>
      <th>rank_full</th>
      <th>rank_without_variable</th>
      <th>rank_effect</th>
      <th>effect_type</th>
      <th>has_override</th>
      <th>override_variable_weight_in_dim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>470</th>
      <td>PF</td>
      <td>ARG</td>
      <td>personal_remittances_gdp</td>
      <td>financial</td>
      <td>0.079579</td>
      <td>0.547795</td>
      <td>0.611169</td>
      <td>-0.063374</td>
      <td>0.063374</td>
      <td>7</td>
      <td>6</td>
      <td>-1</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>238</th>
      <td>PF</td>
      <td>ARG</td>
      <td>trade_openness</td>
      <td>macro</td>
      <td>0.040000</td>
      <td>0.547795</td>
      <td>0.561278</td>
      <td>-0.013483</td>
      <td>0.013483</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>528</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial_system_deposits_gdp</td>
      <td>financial</td>
      <td>0.028421</td>
      <td>0.547795</td>
      <td>0.554906</td>
      <td>-0.007112</td>
      <td>0.007112</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>64</th>
      <td>PF</td>
      <td>ARG</td>
      <td>inflation_rate</td>
      <td>macro</td>
      <td>0.029091</td>
      <td>0.547795</td>
      <td>0.553213</td>
      <td>-0.005418</td>
      <td>0.005418</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>325</th>
      <td>PF</td>
      <td>ARG</td>
      <td>domestic_credit_private_gdp</td>
      <td>financial</td>
      <td>0.028421</td>
      <td>0.547795</td>
      <td>0.550510</td>
      <td>-0.002716</td>
      <td>0.002716</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>992</th>
      <td>PF</td>
      <td>ARG</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>digital_tech</td>
      <td>0.016800</td>
      <td>0.547795</td>
      <td>0.550228</td>
      <td>-0.002433</td>
      <td>0.002433</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PF</td>
      <td>ARG</td>
      <td>gdp_per_capita_ppp</td>
      <td>macro</td>
      <td>0.021818</td>
      <td>0.547795</td>
      <td>0.549338</td>
      <td>-0.001543</td>
      <td>0.001543</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>934</th>
      <td>PF</td>
      <td>ARG</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>digital_tech</td>
      <td>0.033600</td>
      <td>0.547795</td>
      <td>0.549308</td>
      <td>-0.001513</td>
      <td>0.001513</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>412</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_npl_ratio</td>
      <td>financial</td>
      <td>0.022737</td>
      <td>0.547795</td>
      <td>0.548714</td>
      <td>-0.000919</td>
      <td>0.000919</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>731</th>
      <td>PF</td>
      <td>ARG</td>
      <td>control_of_corruption</td>
      <td>institutional</td>
      <td>0.016900</td>
      <td>0.547795</td>
      <td>0.548447</td>
      <td>-0.000652</td>
      <td>0.000652</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>586</th>
      <td>PF</td>
      <td>ARG</td>
      <td>regulatory_quality</td>
      <td>institutional</td>
      <td>0.028600</td>
      <td>0.547795</td>
      <td>0.548231</td>
      <td>-0.000436</td>
      <td>0.000436</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>180</th>
      <td>PF</td>
      <td>ARG</td>
      <td>current_account_gdp</td>
      <td>macro</td>
      <td>0.012727</td>
      <td>0.547795</td>
      <td>0.548225</td>
      <td>-0.000430</td>
      <td>0.000430</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>267</th>
      <td>PF</td>
      <td>ARG</td>
      <td>fdi_net_inflows_gdp</td>
      <td>macro</td>
      <td>0.010909</td>
      <td>0.547795</td>
      <td>0.548180</td>
      <td>-0.000385</td>
      <td>0.000385</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>441</th>
      <td>PF</td>
      <td>ARG</td>
      <td>stock_market_cap_gdp</td>
      <td>financial</td>
      <td>0.008526</td>
      <td>0.547795</td>
      <td>0.548080</td>
      <td>-0.000286</td>
      <td>0.000286</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>673</th>
      <td>PF</td>
      <td>ARG</td>
      <td>political_stability</td>
      <td>institutional</td>
      <td>0.016900</td>
      <td>0.547795</td>
      <td>0.547907</td>
      <td>-0.000112</td>
      <td>0.000112</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>644</th>
      <td>PF</td>
      <td>ARG</td>
      <td>rule_of_law</td>
      <td>institutional</td>
      <td>0.022100</td>
      <td>0.547795</td>
      <td>0.547899</td>
      <td>-0.000104</td>
      <td>0.000104</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>209</th>
      <td>PF</td>
      <td>ARG</td>
      <td>public_debt_gdp</td>
      <td>macro</td>
      <td>0.012727</td>
      <td>0.547795</td>
      <td>0.547869</td>
      <td>-0.000074</td>
      <td>0.000074</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>702</th>
      <td>PF</td>
      <td>ARG</td>
      <td>voice_accountability</td>
      <td>institutional</td>
      <td>0.005200</td>
      <td>0.547795</td>
      <td>0.547752</td>
      <td>0.000043</td>
      <td>0.000043</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PF</td>
      <td>ARG</td>
      <td>gdp_nominal</td>
      <td>macro</td>
      <td>0.010909</td>
      <td>0.547795</td>
      <td>0.547738</td>
      <td>0.000057</td>
      <td>0.000057</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>789</th>
      <td>PF</td>
      <td>ARG</td>
      <td>heritage_efi</td>
      <td>institutional</td>
      <td>0.007800</td>
      <td>0.547795</td>
      <td>0.547731</td>
      <td>0.000064</td>
      <td>0.000064</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>499</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_concentration_5</td>
      <td>financial</td>
      <td>0.008526</td>
      <td>0.547795</td>
      <td>0.547713</td>
      <td>0.000082</td>
      <td>0.000082</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>93</th>
      <td>PF</td>
      <td>ARG</td>
      <td>population_total</td>
      <td>macro</td>
      <td>0.007273</td>
      <td>0.547795</td>
      <td>0.547672</td>
      <td>0.000123</td>
      <td>0.000123</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>615</th>
      <td>PF</td>
      <td>ARG</td>
      <td>government_effectiveness</td>
      <td>institutional</td>
      <td>0.019500</td>
      <td>0.547795</td>
      <td>0.547608</td>
      <td>0.000187</td>
      <td>0.000187</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>151</th>
      <td>PF</td>
      <td>ARG</td>
      <td>unemployment_rate</td>
      <td>macro</td>
      <td>0.012727</td>
      <td>0.547795</td>
      <td>0.547605</td>
      <td>0.000190</td>
      <td>0.000190</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>760</th>
      <td>PF</td>
      <td>ARG</td>
      <td>country_risk_premium</td>
      <td>institutional</td>
      <td>0.013000</td>
      <td>0.547795</td>
      <td>0.547587</td>
      <td>0.000207</td>
      <td>0.000207</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>122</th>
      <td>PF</td>
      <td>ARG</td>
      <td>urban_population_pct</td>
      <td>macro</td>
      <td>0.007273</td>
      <td>0.547795</td>
      <td>0.547516</td>
      <td>0.000279</td>
      <td>0.000279</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>296</th>
      <td>PF</td>
      <td>ARG</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>macro</td>
      <td>0.014545</td>
      <td>0.547795</td>
      <td>0.547467</td>
      <td>0.000328</td>
      <td>0.000328</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>905</th>
      <td>PF</td>
      <td>ARG</td>
      <td>secure_internet_servers_per_million</td>
      <td>digital_tech</td>
      <td>0.029400</td>
      <td>0.547795</td>
      <td>0.547417</td>
      <td>0.000378</td>
      <td>0.000378</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>557</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_capital_rwa</td>
      <td>financial</td>
      <td>0.011368</td>
      <td>0.547795</td>
      <td>0.547259</td>
      <td>0.000536</td>
      <td>0.000536</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>383</th>
      <td>PF</td>
      <td>ARG</td>
      <td>interest_rate_spread</td>
      <td>financial</td>
      <td>0.019895</td>
      <td>0.547795</td>
      <td>0.546405</td>
      <td>0.001389</td>
      <td>0.001389</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>963</th>
      <td>PF</td>
      <td>ARG</td>
      <td>atms_per_100k_adults</td>
      <td>digital_tech</td>
      <td>0.046200</td>
      <td>0.547795</td>
      <td>0.544429</td>
      <td>0.003366</td>
      <td>0.003366</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>354</th>
      <td>PF</td>
      <td>ARG</td>
      <td>account_ownership</td>
      <td>financial</td>
      <td>0.062526</td>
      <td>0.547795</td>
      <td>0.534796</td>
      <td>0.012999</td>
      <td>0.012999</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>847</th>
      <td>PF</td>
      <td>ARG</td>
      <td>mobile_subscriptions</td>
      <td>digital_tech</td>
      <td>0.092400</td>
      <td>0.547795</td>
      <td>0.530753</td>
      <td>0.017042</td>
      <td>0.017042</td>
      <td>7</td>
      <td>12</td>
      <td>5</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>818</th>
      <td>PF</td>
      <td>ARG</td>
      <td>internet_users_pct</td>
      <td>digital_tech</td>
      <td>0.067200</td>
      <td>0.547795</td>
      <td>0.526430</td>
      <td>0.021365</td>
      <td>0.021365</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>876</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_payment_adults_pct</td>
      <td>digital_tech</td>
      <td>0.134400</td>
      <td>0.547795</td>
      <td>0.502110</td>
      <td>0.045685</td>
      <td>0.045685</td>
      <td>7</td>
      <td>16</td>
      <td>9</td>
      <td>Driver</td>
      <td>True</td>
      <td>0.32</td>
    </tr>
  </tbody>
</table>
</div>



#### Variables más importantes para una línea completa

Esto muestra qué variables son más materialmente relevantes para la línea, en promedio absoluto.

Interpretación:

- mean_abs_effect: relevancia promedio.
- mean_effect > 0: variable tiende a ayudar en promedio.
- mean_effect < 0: variable tiende a penalizar en promedio.
- max_abs_effect: variable muy importante para algunos países específicos.


```python
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

line_variable_importance#.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>dimension</th>
      <th>removed_variable</th>
      <th>mean_abs_effect</th>
      <th>mean_effect</th>
      <th>max_abs_effect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>PF</td>
      <td>digital_tech</td>
      <td>digital_payment_adults_pct</td>
      <td>0.046269</td>
      <td>-0.007342</td>
      <td>0.119396</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PF</td>
      <td>financial</td>
      <td>personal_remittances_gdp</td>
      <td>0.046125</td>
      <td>-0.012755</td>
      <td>0.110274</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PF</td>
      <td>digital_tech</td>
      <td>mobile_subscriptions</td>
      <td>0.024660</td>
      <td>-0.004929</td>
      <td>0.072625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PF</td>
      <td>digital_tech</td>
      <td>internet_users_pct</td>
      <td>0.016022</td>
      <td>0.013281</td>
      <td>0.036565</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PF</td>
      <td>financial</td>
      <td>account_ownership</td>
      <td>0.008385</td>
      <td>0.004429</td>
      <td>0.023296</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PF</td>
      <td>macro</td>
      <td>trade_openness</td>
      <td>0.007269</td>
      <td>-0.006191</td>
      <td>0.016521</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PF</td>
      <td>macro</td>
      <td>inflation_rate</td>
      <td>0.006403</td>
      <td>0.005708</td>
      <td>0.011053</td>
    </tr>
    <tr>
      <th>0</th>
      <td>PF</td>
      <td>digital_tech</td>
      <td>atms_per_100k_adults</td>
      <td>0.003854</td>
      <td>0.002039</td>
      <td>0.011625</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PF</td>
      <td>financial</td>
      <td>financial_system_deposits_gdp</td>
      <td>0.002593</td>
      <td>0.000489</td>
      <td>0.007337</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PF</td>
      <td>financial</td>
      <td>domestic_credit_private_gdp</td>
      <td>0.002581</td>
      <td>0.001836</td>
      <td>0.006238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PF</td>
      <td>digital_tech</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>0.002493</td>
      <td>-0.000455</td>
      <td>0.013071</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PF</td>
      <td>financial</td>
      <td>interest_rate_spread</td>
      <td>0.002169</td>
      <td>0.001942</td>
      <td>0.003398</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PF</td>
      <td>digital_tech</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>0.001971</td>
      <td>-0.001718</td>
      <td>0.003027</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PF</td>
      <td>digital_tech</td>
      <td>secure_internet_servers_per_million</td>
      <td>0.001890</td>
      <td>-0.000789</td>
      <td>0.008099</td>
    </tr>
    <tr>
      <th>27</th>
      <td>PF</td>
      <td>macro</td>
      <td>gdp_per_capita_ppp</td>
      <td>0.001646</td>
      <td>-0.001284</td>
      <td>0.003322</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PF</td>
      <td>institutional</td>
      <td>regulatory_quality</td>
      <td>0.001488</td>
      <td>0.000390</td>
      <td>0.004500</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PF</td>
      <td>financial</td>
      <td>bank_npl_ratio</td>
      <td>0.001429</td>
      <td>0.000637</td>
      <td>0.003567</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PF</td>
      <td>macro</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>0.000921</td>
      <td>0.000650</td>
      <td>0.001866</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PF</td>
      <td>institutional</td>
      <td>control_of_corruption</td>
      <td>0.000855</td>
      <td>-0.000279</td>
      <td>0.002099</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PF</td>
      <td>institutional</td>
      <td>rule_of_law</td>
      <td>0.000854</td>
      <td>0.000091</td>
      <td>0.002931</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PF</td>
      <td>institutional</td>
      <td>political_stability</td>
      <td>0.000787</td>
      <td>0.000376</td>
      <td>0.002786</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PF</td>
      <td>institutional</td>
      <td>country_risk_premium</td>
      <td>0.000787</td>
      <td>0.000723</td>
      <td>0.001811</td>
    </tr>
    <tr>
      <th>32</th>
      <td>PF</td>
      <td>macro</td>
      <td>unemployment_rate</td>
      <td>0.000669</td>
      <td>0.000514</td>
      <td>0.001708</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PF</td>
      <td>macro</td>
      <td>current_account_gdp</td>
      <td>0.000590</td>
      <td>-0.000438</td>
      <td>0.001254</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PF</td>
      <td>macro</td>
      <td>public_debt_gdp</td>
      <td>0.000550</td>
      <td>0.000115</td>
      <td>0.001436</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PF</td>
      <td>institutional</td>
      <td>government_effectiveness</td>
      <td>0.000542</td>
      <td>0.000153</td>
      <td>0.001545</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PF</td>
      <td>financial</td>
      <td>bank_capital_rwa</td>
      <td>0.000418</td>
      <td>-0.000304</td>
      <td>0.000966</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PF</td>
      <td>macro</td>
      <td>fdi_net_inflows_gdp</td>
      <td>0.000372</td>
      <td>-0.000246</td>
      <td>0.001267</td>
    </tr>
    <tr>
      <th>26</th>
      <td>PF</td>
      <td>macro</td>
      <td>gdp_nominal</td>
      <td>0.000319</td>
      <td>-0.000218</td>
      <td>0.001178</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PF</td>
      <td>financial</td>
      <td>bank_concentration_5</td>
      <td>0.000295</td>
      <td>-0.000183</td>
      <td>0.000773</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PF</td>
      <td>financial</td>
      <td>stock_market_cap_gdp</td>
      <td>0.000216</td>
      <td>-0.000044</td>
      <td>0.000706</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PF</td>
      <td>institutional</td>
      <td>heritage_efi</td>
      <td>0.000207</td>
      <td>0.000174</td>
      <td>0.000465</td>
    </tr>
    <tr>
      <th>29</th>
      <td>PF</td>
      <td>macro</td>
      <td>population_total</td>
      <td>0.000171</td>
      <td>-0.000013</td>
      <td>0.000436</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PF</td>
      <td>macro</td>
      <td>urban_population_pct</td>
      <td>0.000159</td>
      <td>0.000103</td>
      <td>0.000482</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PF</td>
      <td>institutional</td>
      <td>voice_accountability</td>
      <td>0.000056</td>
      <td>0.000032</td>
      <td>0.000184</td>
    </tr>
  </tbody>
</table>
</div>



### Combinar contribución aditiva y efecto marginal

La lectura más robusta aparece cuando cruzas ambas métricas:
- Alta contribución + alto marginal positivo = driver robusto.
- Alta contribución + bajo marginal = driver descriptivo, pero no decisivo.
- Baja contribución + marginal negativo fuerte = restricción crítica.



```python

pf_explainability = combine_contribution_and_marginal(
    contributions=contrib_by_line["PF"],
    marginal_effects=marginal_by_line["PF"],
)

```

ejemplo para ARG


```python
(
    pf_explainability[
        pf_explainability["country_iso3"] == "ARG"
    ][
        [
            "business_line",
            "country_iso3",
            "removed_variable",
            "dimension_contribution",
            "normalized_value",
            "final_topsis_weight_contribution",
            "contribution",
            "shortfall",
            "score_effect",
            "rank_effect",
            "effect_type",
        ]
    ]
    .sort_values("score_effect", ascending=False)
    #.head(10)
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>removed_variable</th>
      <th>dimension_contribution</th>
      <th>normalized_value</th>
      <th>final_topsis_weight_contribution</th>
      <th>contribution</th>
      <th>shortfall</th>
      <th>score_effect</th>
      <th>rank_effect</th>
      <th>effect_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_payment_adults_pct</td>
      <td>digital_tech</td>
      <td>0.681100</td>
      <td>0.134400</td>
      <td>0.091540</td>
      <td>0.042860</td>
      <td>0.045685</td>
      <td>9</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PF</td>
      <td>ARG</td>
      <td>internet_users_pct</td>
      <td>digital_tech</td>
      <td>0.872843</td>
      <td>0.067200</td>
      <td>0.058655</td>
      <td>0.008545</td>
      <td>0.021365</td>
      <td>1</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PF</td>
      <td>ARG</td>
      <td>mobile_subscriptions</td>
      <td>digital_tech</td>
      <td>0.674096</td>
      <td>0.092400</td>
      <td>0.062286</td>
      <td>0.030114</td>
      <td>0.017042</td>
      <td>5</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PF</td>
      <td>ARG</td>
      <td>account_ownership</td>
      <td>financial</td>
      <td>0.777665</td>
      <td>0.062526</td>
      <td>0.048625</td>
      <td>0.013902</td>
      <td>0.012999</td>
      <td>1</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PF</td>
      <td>ARG</td>
      <td>atms_per_100k_adults</td>
      <td>digital_tech</td>
      <td>0.658154</td>
      <td>0.046200</td>
      <td>0.030407</td>
      <td>0.015793</td>
      <td>0.003366</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PF</td>
      <td>ARG</td>
      <td>interest_rate_spread</td>
      <td>financial</td>
      <td>0.807242</td>
      <td>0.019895</td>
      <td>0.016060</td>
      <td>0.003835</td>
      <td>0.001389</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_capital_rwa</td>
      <td>financial</td>
      <td>0.858999</td>
      <td>0.011368</td>
      <td>0.009765</td>
      <td>0.001603</td>
      <td>0.000536</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PF</td>
      <td>ARG</td>
      <td>secure_internet_servers_per_million</td>
      <td>digital_tech</td>
      <td>0.578476</td>
      <td>0.029400</td>
      <td>0.017007</td>
      <td>0.012393</td>
      <td>0.000378</td>
      <td>1</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PF</td>
      <td>ARG</td>
      <td>weighted_mean_applied_tariff_all_products</td>
      <td>macro</td>
      <td>0.659466</td>
      <td>0.014545</td>
      <td>0.009592</td>
      <td>0.004953</td>
      <td>0.000328</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PF</td>
      <td>ARG</td>
      <td>urban_population_pct</td>
      <td>macro</td>
      <td>0.951911</td>
      <td>0.007273</td>
      <td>0.006923</td>
      <td>0.000350</td>
      <td>0.000279</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PF</td>
      <td>ARG</td>
      <td>country_risk_premium</td>
      <td>institutional</td>
      <td>0.635858</td>
      <td>0.013000</td>
      <td>0.008266</td>
      <td>0.004734</td>
      <td>0.000207</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PF</td>
      <td>ARG</td>
      <td>unemployment_rate</td>
      <td>macro</td>
      <td>0.632010</td>
      <td>0.012727</td>
      <td>0.008044</td>
      <td>0.004684</td>
      <td>0.000190</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PF</td>
      <td>ARG</td>
      <td>government_effectiveness</td>
      <td>institutional</td>
      <td>0.582661</td>
      <td>0.019500</td>
      <td>0.011362</td>
      <td>0.008138</td>
      <td>0.000187</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PF</td>
      <td>ARG</td>
      <td>population_total</td>
      <td>macro</td>
      <td>0.717027</td>
      <td>0.007273</td>
      <td>0.005215</td>
      <td>0.002058</td>
      <td>0.000123</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_concentration_5</td>
      <td>financial</td>
      <td>0.628711</td>
      <td>0.008526</td>
      <td>0.005361</td>
      <td>0.003166</td>
      <td>0.000082</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PF</td>
      <td>ARG</td>
      <td>heritage_efi</td>
      <td>institutional</td>
      <td>0.623188</td>
      <td>0.007800</td>
      <td>0.004861</td>
      <td>0.002939</td>
      <td>0.000064</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PF</td>
      <td>ARG</td>
      <td>gdp_nominal</td>
      <td>macro</td>
      <td>0.581690</td>
      <td>0.010909</td>
      <td>0.006346</td>
      <td>0.004563</td>
      <td>0.000057</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>29</th>
      <td>PF</td>
      <td>ARG</td>
      <td>voice_accountability</td>
      <td>institutional</td>
      <td>0.663344</td>
      <td>0.005200</td>
      <td>0.003449</td>
      <td>0.001751</td>
      <td>0.000043</td>
      <td>0</td>
      <td>Driver</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PF</td>
      <td>ARG</td>
      <td>public_debt_gdp</td>
      <td>macro</td>
      <td>0.515770</td>
      <td>0.012727</td>
      <td>0.006564</td>
      <td>0.006163</td>
      <td>-0.000074</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PF</td>
      <td>ARG</td>
      <td>rule_of_law</td>
      <td>institutional</td>
      <td>0.532835</td>
      <td>0.022100</td>
      <td>0.011776</td>
      <td>0.010324</td>
      <td>-0.000104</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PF</td>
      <td>ARG</td>
      <td>political_stability</td>
      <td>institutional</td>
      <td>0.520374</td>
      <td>0.016900</td>
      <td>0.008794</td>
      <td>0.008106</td>
      <td>-0.000112</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PF</td>
      <td>ARG</td>
      <td>stock_market_cap_gdp</td>
      <td>financial</td>
      <td>0.283868</td>
      <td>0.008526</td>
      <td>0.002420</td>
      <td>0.006106</td>
      <td>-0.000286</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PF</td>
      <td>ARG</td>
      <td>fdi_net_inflows_gdp</td>
      <td>macro</td>
      <td>0.328804</td>
      <td>0.010909</td>
      <td>0.003587</td>
      <td>0.007322</td>
      <td>-0.000385</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>26</th>
      <td>PF</td>
      <td>ARG</td>
      <td>current_account_gdp</td>
      <td>macro</td>
      <td>0.366979</td>
      <td>0.012727</td>
      <td>0.004671</td>
      <td>0.008057</td>
      <td>-0.000430</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PF</td>
      <td>ARG</td>
      <td>regulatory_quality</td>
      <td>institutional</td>
      <td>0.510843</td>
      <td>0.028600</td>
      <td>0.014610</td>
      <td>0.013990</td>
      <td>-0.000436</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PF</td>
      <td>ARG</td>
      <td>control_of_corruption</td>
      <td>institutional</td>
      <td>0.391865</td>
      <td>0.016900</td>
      <td>0.006623</td>
      <td>0.010277</td>
      <td>-0.000652</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PF</td>
      <td>ARG</td>
      <td>bank_npl_ratio</td>
      <td>financial</td>
      <td>0.426004</td>
      <td>0.022737</td>
      <td>0.009686</td>
      <td>0.013051</td>
      <td>-0.000919</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PF</td>
      <td>ARG</td>
      <td>commercial_bank_branches_per_100k_adults</td>
      <td>digital_tech</td>
      <td>0.456250</td>
      <td>0.033600</td>
      <td>0.015330</td>
      <td>0.018270</td>
      <td>-0.001513</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PF</td>
      <td>ARG</td>
      <td>gdp_per_capita_ppp</td>
      <td>macro</td>
      <td>0.329687</td>
      <td>0.021818</td>
      <td>0.007193</td>
      <td>0.014625</td>
      <td>-0.001543</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PF</td>
      <td>ARG</td>
      <td>ict_goods_exports_pct_total_goods_exports</td>
      <td>digital_tech</td>
      <td>0.001684</td>
      <td>0.016800</td>
      <td>0.000028</td>
      <td>0.016772</td>
      <td>-0.002433</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PF</td>
      <td>ARG</td>
      <td>domestic_credit_private_gdp</td>
      <td>financial</td>
      <td>0.323269</td>
      <td>0.028421</td>
      <td>0.009188</td>
      <td>0.019233</td>
      <td>-0.002716</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>27</th>
      <td>PF</td>
      <td>ARG</td>
      <td>inflation_rate</td>
      <td>macro</td>
      <td>0.137314</td>
      <td>0.029091</td>
      <td>0.003995</td>
      <td>0.025096</td>
      <td>-0.005418</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial_system_deposits_gdp</td>
      <td>financial</td>
      <td>0.000000</td>
      <td>0.028421</td>
      <td>0.000000</td>
      <td>0.028421</td>
      <td>-0.007112</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>32</th>
      <td>PF</td>
      <td>ARG</td>
      <td>trade_openness</td>
      <td>macro</td>
      <td>0.033014</td>
      <td>0.040000</td>
      <td>0.001321</td>
      <td>0.038679</td>
      <td>-0.013483</td>
      <td>0</td>
      <td>Restriccion</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PF</td>
      <td>ARG</td>
      <td>personal_remittances_gdp</td>
      <td>financial</td>
      <td>0.036989</td>
      <td>0.079579</td>
      <td>0.002944</td>
      <td>0.076635</td>
      <td>-0.063374</td>
      <td>-1</td>
      <td>Restriccion</td>
    </tr>
  </tbody>
</table>
</div>




```python
pf_explainability["driver_class"] = pf_explainability.apply(
    classify_driver_robustness,
    axis=1,
)
```


```python
build_country_driver_table(
    explainability_df=pf_explainability,
    country_iso3="ARG",
    top_n=5,
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_line</th>
      <th>country_iso3</th>
      <th>removed_variable</th>
      <th>dimension_contribution</th>
      <th>normalized_value</th>
      <th>final_topsis_weight_contribution</th>
      <th>contribution</th>
      <th>shortfall</th>
      <th>score_effect</th>
      <th>rank_effect</th>
      <th>effect_type</th>
      <th>driver_side</th>
      <th>driver_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PF</td>
      <td>ARG</td>
      <td>digital_payment_adults_pct</td>
      <td>digital_tech</td>
      <td>0.681100</td>
      <td>0.134400</td>
      <td>0.091540</td>
      <td>0.042860</td>
      <td>0.045685</td>
      <td>9</td>
      <td>Driver</td>
      <td>Driver</td>
      <td>Driver robusto</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PF</td>
      <td>ARG</td>
      <td>internet_users_pct</td>
      <td>digital_tech</td>
      <td>0.872843</td>
      <td>0.067200</td>
      <td>0.058655</td>
      <td>0.008545</td>
      <td>0.021365</td>
      <td>1</td>
      <td>Driver</td>
      <td>Driver</td>
      <td>Driver robusto</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PF</td>
      <td>ARG</td>
      <td>mobile_subscriptions</td>
      <td>digital_tech</td>
      <td>0.674096</td>
      <td>0.092400</td>
      <td>0.062286</td>
      <td>0.030114</td>
      <td>0.017042</td>
      <td>5</td>
      <td>Driver</td>
      <td>Driver</td>
      <td>Driver robusto</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PF</td>
      <td>ARG</td>
      <td>account_ownership</td>
      <td>financial</td>
      <td>0.777665</td>
      <td>0.062526</td>
      <td>0.048625</td>
      <td>0.013902</td>
      <td>0.012999</td>
      <td>1</td>
      <td>Driver</td>
      <td>Driver</td>
      <td>Driver robusto</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PF</td>
      <td>ARG</td>
      <td>atms_per_100k_adults</td>
      <td>digital_tech</td>
      <td>0.658154</td>
      <td>0.046200</td>
      <td>0.030407</td>
      <td>0.015793</td>
      <td>0.003366</td>
      <td>0</td>
      <td>Driver</td>
      <td>Driver</td>
      <td>Driver descriptivo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PF</td>
      <td>ARG</td>
      <td>personal_remittances_gdp</td>
      <td>financial</td>
      <td>0.036989</td>
      <td>0.079579</td>
      <td>0.002944</td>
      <td>0.076635</td>
      <td>-0.063374</td>
      <td>-1</td>
      <td>Restriccion</td>
      <td>Restriccion</td>
      <td>Restriccion critica</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PF</td>
      <td>ARG</td>
      <td>trade_openness</td>
      <td>macro</td>
      <td>0.033014</td>
      <td>0.040000</td>
      <td>0.001321</td>
      <td>0.038679</td>
      <td>-0.013483</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>Restriccion</td>
      <td>Restriccion critica</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PF</td>
      <td>ARG</td>
      <td>financial_system_deposits_gdp</td>
      <td>financial</td>
      <td>0.000000</td>
      <td>0.028421</td>
      <td>0.000000</td>
      <td>0.028421</td>
      <td>-0.007112</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>Restriccion</td>
      <td>Driver moderado</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PF</td>
      <td>ARG</td>
      <td>inflation_rate</td>
      <td>macro</td>
      <td>0.137314</td>
      <td>0.029091</td>
      <td>0.003995</td>
      <td>0.025096</td>
      <td>-0.005418</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>Restriccion</td>
      <td>Driver moderado</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PF</td>
      <td>ARG</td>
      <td>domestic_credit_private_gdp</td>
      <td>financial</td>
      <td>0.323269</td>
      <td>0.028421</td>
      <td>0.009188</td>
      <td>0.019233</td>
      <td>-0.002716</td>
      <td>0</td>
      <td>Restriccion</td>
      <td>Restriccion</td>
      <td>Efecto marginal bajo</td>
    </tr>
  </tbody>
</table>
</div>



#### Cómo interpretar resultados

Caso 1: contribution alta y score_effect positivo alto
- Interpretación: la variable es un driver robusto. Tiene alto valor normalizado, alto peso y su remoción reduce el score.

Caso 2: contribution alta y score_effect bajo
- Interpretación: la variable describe bien el perfil del país, pero no es decisiva para el TOPSIS.

Caso 3: contribution baja, shortfall alta y score_effect negativo
- Interpretación: la variable es una restricción crítica. El país se beneficiaría si esa dimensión no pesara.

Caso 4: score_effect cercano a cero
- Interpretación: la variable no es material para la posición del país en esa línea.


TOPSIS es no lineal porque depende de:

- distancia al ideal positivo
- distancia al ideal negativo
- normalización relativa
- renormalización de peso

Por tanto: sum(score_effects) != score_full

El marginal es una prueba de robustez, no una atribución contable.

El análisis marginal score_full - score_without_variable complementa la contribución aditiva porque mide robustez, no solo aporte ponderado. Implementa leave-one-variable-out por línea, recalculando TOPSIS sin cada variable y renormalizando pesos. La lectura robusta surge al cruzar contribución, shortfall y efecto marginal: driver robusto, driver descriptivo, restricción crítica o efecto marginal bajo.



