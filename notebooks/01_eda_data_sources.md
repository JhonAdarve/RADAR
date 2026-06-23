# 01 - EDA de fuentes de datos | RADAR Cibest

**Fase ASUM-DM:** 2 - Entendimiento de datos
**lider:** Jhon Adarve 
**Fecha:** Marzo - Abril 2026

Este notebook explora la disponibilidad y calidad de las fuentes de datos antes de ejecutar el pipeline en produccion.

Objetivos:
1. Validar conectividad con cada API
2. Verificar cobertura por pais y variable
3. Identificar variables con datos incompletos
4. Generar reporte preliminar de calidad


```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
from src.utils import load_all_configs, setup_logger

configs = load_all_configs()
setup_logger(configs['settings'].get('logging'))
print(f"Proyecto: {configs['settings']['project']['name']}")
print(f"Paises en alcance: {len(configs['settings']['countries'])}")
print(f"Dimensiones: {configs['settings']['project']['dimensions']}")
```

    Proyecto: RADAR Cibest
    Paises en alcance: 30
    Dimensiones: ['macro', 'financial', 'institutional', 'digital_tech', 'proximity']
    

## 1. Inventario de variables por dimension


```python
from src.utils import get_variable_catalog

catalog = get_variable_catalog(configs['variables'])
df_catalog = pd.DataFrame.from_dict(catalog, orient='index')
print(f"Total variables: {len(df_catalog)}")
print(f"\nVariables por dimension:")
print(df_catalog.groupby('dimension').size())
print(f"\nVariables por fuente:")
print(df_catalog.groupby('source').size())
```

    Total variables: 45
    
    Variables por dimension:
    dimension
    digital_tech      7
    financial         9
    institutional     8
    macro            12
    proximity         9
    dtype: int64
    
    Variables por fuente:
    source
    complementary                     10
    damodaran_country_risk_premium     1
    world_bank                        34
    dtype: int64
    

## 2. Test de extraccion - Banco Mundial


```python
from src.data_extraction.world_bank import fetch_indicator

df_test = fetch_indicator(
    indicator_code='NY.GDP.PCAP.CD',
    countries=['COL', 'MEX', 'CHL', 'ESP']
)
df_test.head()
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
      <th>year</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHL</td>
      <td>2024</td>
      <td>NY.GDP.PCAP.CD</td>
      <td>16709.889397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COL</td>
      <td>2024</td>
      <td>NY.GDP.PCAP.CD</td>
      <td>7919.208868</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ESP</td>
      <td>2024</td>
      <td>NY.GDP.PCAP.CD</td>
      <td>35326.768307</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MEX</td>
      <td>2024</td>
      <td>NY.GDP.PCAP.CD</td>
      <td>14185.781225</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Pipeline completo de extraccion


```python

import importlib

import src.utils as utils
import src.data_extraction.world_bank as world_bank
import src.data_extraction.pipeline as pipeline
import src.scoring.hybrid_scorer as hybrid_scorer

importlib.invalidate_caches()

importlib.reload(world_bank)
importlib.reload(pipeline)
importlib.reload(hybrid_scorer)
importlib.reload(utils)


from src.utils import (
    load_all_configs,
    get_world_bank_variable_catalog,
    infer_world_bank_db,
)


from src.data_extraction.pipeline import run_extraction

configs = load_all_configs()

print("Tiene fetch_indicator_history:", hasattr(world_bank, "fetch_indicator_history"))

```

    Tiene fetch_indicator_history: True
    


```python
master, coverage = run_extraction(configs=configs, save_intermediate=True)

print(f"Master shape: {master.shape}")
print(f"Cobertura promedio: {coverage['pct_cobertura'].mean():.1f}%")

```

    2026-06-12 18:15:16 | INFO     | src.data_extraction.pipeline:run_extraction:116 | Fuente world_bank en modo latest_available
    2026-06-12 18:15:16 | INFO     | src.data_extraction.world_bank:fetch_all_indicators:254 | Indicadores World Bank agrupados por base: {2: 24, 32: 3, 3: 6, 28: 1}
    2026-06-12 18:15:16 | INFO     | src.data_extraction.world_bank:fetch_all_indicators:268 | Extrayendo indicadores World Bank: db=2, n_indicadores=24
    2026-06-12 18:15:16 | INFO     | src.data_extraction.world_bank:fetch_all_indicators:268 | Extrayendo indicadores World Bank: db=32, n_indicadores=3
    2026-06-12 18:15:16 | INFO     | src.data_extraction.world_bank:fetch_all_indicators:268 | Extrayendo indicadores World Bank: db=3, n_indicadores=6
    2026-06-12 18:15:17 | INFO     | src.data_extraction.world_bank:fetch_all_indicators:268 | Extrayendo indicadores World Bank: db=28, n_indicadores=1
    2026-06-12 18:15:17 | INFO     | src.data_extraction.pipeline:run_extraction:144 | Anexando histórico específico de gdp_growth para Trend: años 2022-2024
    2026-06-12 18:15:17 | INFO     | src.data_extraction.world_bank:fetch_indicator_history:619 | Usando caché histórico WB para gdp_growth: C:\Users\jadarve\OneDrive - Grupo Bancolombia\Bancolombia\GFEC-VEF\2026\Internacionalización\radar_cibest_v2\data\cache\wb_history_gdp_growth_2022_2024.parquet
    2026-06-12 18:15:17 | INFO     | src.data_extraction.world_bank:save_raw_data:823 | World Bank raw guardado: C:\Users\jadarve\OneDrive - Grupo Bancolombia\Bancolombia\GFEC-VEF\2026\Internacionalización\radar_cibest_v2\data\raw\world_bank_20260612.parquet
    2026-06-12 18:15:17 | INFO     | src.data_extraction.damodaran_country_risk_premium:_download_excel:51 | Descargando Country Risk Premium desde Damodaran: https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx
    C:\Users\jadarve\AppData\Roaming\Python\Python39\site-packages\urllib3\connectionpool.py:1097: InsecureRequestWarning: Unverified HTTPS request is being made to host 'pages.stern.nyu.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
    2026-06-12 18:15:19 | INFO     | src.data_extraction.damodaran_country_risk_premium:save_raw_data:137 | Damodaran raw guardado: C:\Users\jadarve\OneDrive - Grupo Bancolombia\Bancolombia\GFEC-VEF\2026\Internacionalización\radar_cibest_v2\data\raw\damodaran_country_risk_premium_20260612.parquet
    2026-06-12 18:15:21 | INFO     | src.data_extraction.complementary:save_raw_data:231 | Complementary raw guardado: C:\Users\jadarve\OneDrive - Grupo Bancolombia\Bancolombia\GFEC-VEF\2026\Internacionalización\radar_cibest_v2\data\raw\complementary_20260612.parquet
    2026-06-12 18:15:21 | INFO     | src.data_extraction.pipeline:run_extraction:303 | Master guardado: C:\Users\jadarve\OneDrive - Grupo Bancolombia\Bancolombia\GFEC-VEF\2026\Internacionalización\radar_cibest_v2\data\raw\master_raw_20260612.parquet | filas=1281 | países=30 | variables=45
    

    Master shape: (1281, 5)
    Cobertura promedio: 90.4%
    


```python
print("utils file:", utils.__file__)
print("Allowed DBs:", utils.WORLD_BANK_ALLOWED_DBS)
```

    utils file: c:\Users\jadarve\OneDrive - Grupo Bancolombia\Bancolombia\GFEC-VEF\2026\Internacionalización\radar_cibest_v2\src\utils.py
    Allowed DBs: {32, 2, 3, 28}
    

1. Qué significa cada línea
    World Bank guardado
    Plain TextWorld Bank raw guardado: ...\data\raw\world_bank_20260513.parquetMostrar más líneas
    Significa que world_bank.py extrajo los indicadores de:

    WDI / Findex db=2
    WGI db=3
    GFDD db=32

    y los guardó en:
    data/raw/world_bank_20260513.parquetMostrar más líneas

    Damodaran descargado y guardado
    Plain TextDescargando Country Risk Premium desde Damodaran...Damodaran raw guardado: ...\damodaran_country_risk_premium_20260513.parquetMostrar más líneas
    Significa que el script de Damodaran descargó el Excel de NYU Stern, procesó la prima de riesgo país y guardó el resultado.
    El warning:
    InsecureRequestWarning: Unverified HTTPS requestMostrar más líneas
    no es un error. Solo indica que estás descargando con verify_ssl=False. El pipeline continuó normalmente.

    Complementary guardado
    Plain TextComplementary raw guardado: ...\complementary_20260513.parquetMostrar más líneas
    Significa que complementary.py sí corrió y guardó las variables complementarias, como CEPII, Hofstede, Heritage o salidas de colombianos, según lo que esté activo en data_sources.yaml.

    Master shape
    Master shape: (916, 5)Mostrar más líneas
    Significa que el DataFrame maestro tiene:
    916 filas5 columnasMostrar más líneas
    Normalmente las columnas son:
    Plain Textcountry_iso3yearvariablevaluesourceMostrar más líneas
    Cada fila representa una observación tipo:
    país - año - variable - valor - fuenteMostrar más líneas

2. Qué significa cobertura promedio 87.2%
    Plain TextCobertura promedio: 87.2%Mostrar más líneas
    Esto quiere decir que, en promedio, cada país tiene datos para el 87.2% de las variables esperadas en variables.yaml.
    No quiere decir que el pipeline falló. Quiere decir que el universo de variables esperadas es más grande que los datos efectivamente disponibles para todos los países.
    La cobertura probablemente se está calculando así:
    Pythonpct_cobertura = variables_disponibles_para_el_pais / total_variables_esperadasMostrar más líneas
    Por ejemplo, si el catálogo tiene 31 variables y un país tiene datos para 27:
    27 / 31 = 87.1%Mostrar más líneas

3. Por qué no es 100%
    Hay varias razones normales:
    1. No todos los países tienen datos para todas las variables World Bank
        Algunos indicadores no tienen dato reciente o disponible para todos los países, especialmente en:

        países del Caribe,
        economías pequeñas,
        Cuba,
        Venezuela,
        algunos indicadores GFDD,
        algunos indicadores Findex.


    2. Algunas fuentes complementarias tienen cobertura parcial
        Variables como:

        Hofstede,
        Heritage EFI,
        CEPII,
        salidas de colombianos,
        Country Risk Premium,

        pueden no tener dato para todos los países del universo.
        Por ejemplo, Hofstede suele tener menos cobertura que World Bank. Heritage también puede no mapear todos los países si el nombre del país no coincide con el diccionario HERITAGE_COUNTRY_TO_ISO3.

    3. Damodaran puede no cubrir todos los países
        El Excel de Damodaran no necesariamente tiene prima de riesgo para todos los países del alcance, o puede tener nombres que pycountry no logra convertir automáticamente a ISO3.

    4. El denominador usa todo el catálogo de variables
        El reporte de cobertura compara cada país contra todas las variables de variables.yaml, no solo contra las que tienen datos disponibles en alguna fuente.
        Entonces, si hay una variable declarada en variables.yaml pero no llega desde ninguna fuente para un país, baja la cobertura.

    5. Posibles diferencias entre nombres de variables
        Si una fuente genera una variable con un nombre distinto al declarado en variables.yaml, el dato existe en master, pero el reporte de cobertura lo cuenta como faltante.
            Ejemplo:
            Plain TextDeclarado en variables.yaml: cultural_distance_hofstedeGenerado por complementary.py: hofstede_pdi, hofstede_idv, ...Mostrar más líneas
            Si el catálogo espera una variable y el extractor entrega otra, baja la cobertura.


```python
coverage.sort_values("pct_cobertura") #.head(10)
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
      <th>n_variables_total</th>
      <th>n_variables_disponibles</th>
      <th>pct_cobertura</th>
      <th>variables_faltantes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>CUB</td>
      <td>45</td>
      <td>23</td>
      <td>51.11</td>
      <td>account_ownership; atms_per_100k_adults; bank_...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>GUY</td>
      <td>45</td>
      <td>32</td>
      <td>71.11</td>
      <td>account_ownership; bank_capital_rwa; bank_npl_...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>HTI</td>
      <td>45</td>
      <td>33</td>
      <td>73.33</td>
      <td>bank_capital_rwa; bank_npl_ratio; country_risk...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BHS</td>
      <td>45</td>
      <td>34</td>
      <td>75.56</td>
      <td>account_ownership; bank_capital_rwa; bank_npl_...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BRB</td>
      <td>45</td>
      <td>35</td>
      <td>77.78</td>
      <td>account_ownership; bank_capital_rwa; digital_p...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BLZ</td>
      <td>45</td>
      <td>36</td>
      <td>80.00</td>
      <td>bank_capital_rwa; colombian_diaspora_stock; ho...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>SUR</td>
      <td>45</td>
      <td>36</td>
      <td>80.00</td>
      <td>account_ownership; bank_capital_rwa; bank_npl_...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NIC</td>
      <td>45</td>
      <td>37</td>
      <td>82.22</td>
      <td>hofstede_idv; hofstede_ivr; hofstede_lto; hofs...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>HND</td>
      <td>45</td>
      <td>41</td>
      <td>91.11</td>
      <td>hofstede_ivr; hofstede_lto; public_debt_gdp; s...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>JAM</td>
      <td>45</td>
      <td>41</td>
      <td>91.11</td>
      <td>bank_npl_ratio; hofstede_ivr; hofstede_lto; tr...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ECU</td>
      <td>45</td>
      <td>42</td>
      <td>93.33</td>
      <td>hofstede_ivr; interest_rate_spread; public_deb...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>VEN</td>
      <td>45</td>
      <td>42</td>
      <td>93.33</td>
      <td>bank_npl_ratio; colombian_diaspora_stock; publ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PAN</td>
      <td>45</td>
      <td>42</td>
      <td>93.33</td>
      <td>hofstede_ivr; hofstede_lto; public_debt_gdp</td>
    </tr>
    <tr>
      <th>16</th>
      <td>USA</td>
      <td>45</td>
      <td>43</td>
      <td>95.56</td>
      <td>colombian_diaspora_stock; interest_rate_spread</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BOL</td>
      <td>45</td>
      <td>43</td>
      <td>95.56</td>
      <td>colombian_diaspora_stock; stock_market_cap_gdp</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TTO</td>
      <td>45</td>
      <td>43</td>
      <td>95.56</td>
      <td>colombian_diaspora_stock; trade_openness</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CRI</td>
      <td>45</td>
      <td>43</td>
      <td>95.56</td>
      <td>hofstede_ivr; hofstede_lto</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GTM</td>
      <td>45</td>
      <td>43</td>
      <td>95.56</td>
      <td>hofstede_ivr; stock_market_cap_gdp</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ESP</td>
      <td>45</td>
      <td>43</td>
      <td>95.56</td>
      <td>heritage_efi; interest_rate_spread</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SLV</td>
      <td>45</td>
      <td>43</td>
      <td>95.56</td>
      <td>interest_rate_spread; stock_market_cap_gdp</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DOM</td>
      <td>45</td>
      <td>44</td>
      <td>97.78</td>
      <td>stock_market_cap_gdp</td>
    </tr>
    <tr>
      <th>8</th>
      <td>COL</td>
      <td>45</td>
      <td>44</td>
      <td>97.78</td>
      <td>colombian_diaspora_stock</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PRY</td>
      <td>45</td>
      <td>44</td>
      <td>97.78</td>
      <td>public_debt_gdp</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ARG</td>
      <td>45</td>
      <td>44</td>
      <td>97.78</td>
      <td>public_debt_gdp</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MEX</td>
      <td>45</td>
      <td>45</td>
      <td>100.00</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>CHL</td>
      <td>45</td>
      <td>45</td>
      <td>100.00</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>CAN</td>
      <td>45</td>
      <td>45</td>
      <td>100.00</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>URY</td>
      <td>45</td>
      <td>45</td>
      <td>100.00</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>BRA</td>
      <td>45</td>
      <td>45</td>
      <td>100.00</td>
      <td></td>
    </tr>
    <tr>
      <th>0</th>
      <td>PER</td>
      <td>45</td>
      <td>45</td>
      <td>100.00</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
master.groupby("year").size().sort_index()
```




    year
    1996      1
    1999      2
    2000      4
    2001      3
    2002      1
    2005      1
    2006      3
    2007      1
    2008      2
    2009      1
    2010      3
    2011      1
    2012     23
    2013      2
    2014      3
    2015      4
    2016      4
    2017      8
    2018      2
    2019      5
    2020    212
    2021     63
    2022     94
    2023    106
    2024    645
    2025     30
    2026     57
    dtype: int64




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
]

(
    master[master["variable"].isin(vars_check)]
    .groupby(["variable", "year"])
    .size()
    .reset_index(name="n")
    .sort_values(["variable", "year"])
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
      <th>variable</th>
      <th>year</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>account_ownership</td>
      <td>2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>account_ownership</td>
      <td>2021</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>account_ownership</td>
      <td>2024</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>domestic_credit_private_gdp</td>
      <td>2008</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>domestic_credit_private_gdp</td>
      <td>2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>domestic_credit_private_gdp</td>
      <td>2023</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>domestic_credit_private_gdp</td>
      <td>2024</td>
      <td>26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>gdp_growth</td>
      <td>2022</td>
      <td>30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gdp_growth</td>
      <td>2023</td>
      <td>30</td>
    </tr>
    <tr>
      <th>9</th>
      <td>gdp_growth</td>
      <td>2024</td>
      <td>30</td>
    </tr>
    <tr>
      <th>10</th>
      <td>gdp_nominal</td>
      <td>2020</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>gdp_nominal</td>
      <td>2024</td>
      <td>29</td>
    </tr>
    <tr>
      <th>12</th>
      <td>gdp_per_capita_ppp</td>
      <td>2011</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>gdp_per_capita_ppp</td>
      <td>2024</td>
      <td>28</td>
    </tr>
    <tr>
      <th>14</th>
      <td>government_effectiveness</td>
      <td>2024</td>
      <td>30</td>
    </tr>
    <tr>
      <th>15</th>
      <td>inflation_rate</td>
      <td>2016</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>inflation_rate</td>
      <td>2024</td>
      <td>28</td>
    </tr>
    <tr>
      <th>17</th>
      <td>internet_users_pct</td>
      <td>2023</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>internet_users_pct</td>
      <td>2024</td>
      <td>29</td>
    </tr>
    <tr>
      <th>19</th>
      <td>regulatory_quality</td>
      <td>2024</td>
      <td>30</td>
    </tr>
    <tr>
      <th>20</th>
      <td>rule_of_law</td>
      <td>2024</td>
      <td>30</td>
    </tr>
    <tr>
      <th>21</th>
      <td>secure_internet_servers_per_million</td>
      <td>2024</td>
      <td>30</td>
    </tr>
    <tr>
      <th>22</th>
      <td>unemployment_rate</td>
      <td>2025</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
master[master["variable"] == "gdp_growth"].sort_values(
    ["country_iso3", "year"]
).head(20)
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
      <th>year</th>
      <th>variable</th>
      <th>value</th>
      <th>source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>ARG</td>
      <td>2022</td>
      <td>gdp_growth</td>
      <td>6.020745</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ARG</td>
      <td>2023</td>
      <td>gdp_growth</td>
      <td>-1.855788</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ARG</td>
      <td>2024</td>
      <td>gdp_growth</td>
      <td>-1.342931</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>57</th>
      <td>BHS</td>
      <td>2022</td>
      <td>gdp_growth</td>
      <td>10.877901</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>58</th>
      <td>BHS</td>
      <td>2023</td>
      <td>gdp_growth</td>
      <td>3.048242</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>59</th>
      <td>BHS</td>
      <td>2024</td>
      <td>gdp_growth</td>
      <td>3.378666</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>95</th>
      <td>BLZ</td>
      <td>2022</td>
      <td>gdp_growth</td>
      <td>9.250513</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>96</th>
      <td>BLZ</td>
      <td>2023</td>
      <td>gdp_growth</td>
      <td>0.500128</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>97</th>
      <td>BLZ</td>
      <td>2024</td>
      <td>gdp_growth</td>
      <td>3.504664</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>134</th>
      <td>BOL</td>
      <td>2022</td>
      <td>gdp_growth</td>
      <td>3.747418</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>135</th>
      <td>BOL</td>
      <td>2023</td>
      <td>gdp_growth</td>
      <td>2.516366</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>136</th>
      <td>BOL</td>
      <td>2024</td>
      <td>gdp_growth</td>
      <td>-1.123356</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>180</th>
      <td>BRA</td>
      <td>2022</td>
      <td>gdp_growth</td>
      <td>3.016694</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>181</th>
      <td>BRA</td>
      <td>2023</td>
      <td>gdp_growth</td>
      <td>3.241655</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>182</th>
      <td>BRA</td>
      <td>2024</td>
      <td>gdp_growth</td>
      <td>3.419315</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>224</th>
      <td>BRB</td>
      <td>2022</td>
      <td>gdp_growth</td>
      <td>17.226463</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>225</th>
      <td>BRB</td>
      <td>2023</td>
      <td>gdp_growth</td>
      <td>2.914357</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>226</th>
      <td>BRB</td>
      <td>2024</td>
      <td>gdp_growth</td>
      <td>2.482262</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>264</th>
      <td>CAN</td>
      <td>2022</td>
      <td>gdp_growth</td>
      <td>4.189036</td>
      <td>world_bank</td>
    </tr>
    <tr>
      <th>265</th>
      <td>CAN</td>
      <td>2023</td>
      <td>gdp_growth</td>
      <td>1.528746</td>
      <td>world_bank</td>
    </tr>
  </tbody>
</table>
</div>




```python
(
    master[master["variable"] == "gdp_growth"]
    .groupby("year")
    .size()
    .sort_index()
)
```




    year
    2022    30
    2023    30
    2024    30
    dtype: int64




```python
master[master["variable"] == "gdp_growth"]["value"].describe()
```




    count    90.000000
    mean      4.570314
    std       8.770089
    min      -4.169634
    25%       1.734381
    50%       3.078248
    75%       4.303408
    max      63.334634
    Name: value, dtype: float64




```python
df_g = master[master["variable"] == "gdp_growth"].copy()
df_g["year"] = pd.to_numeric(df_g["year"], errors="coerce")
df_g["value"] = pd.to_numeric(df_g["value"], errors="coerce")

print(df_g.shape)
print(df_g["year"].value_counts().sort_index())
print(df_g["value"].describe())
print(df_g.groupby("country_iso3")["value"].mean().sort_values())
```

    (90, 5)
    year
    2022    30
    2023    30
    2024    30
    Name: count, dtype: int64
    count    90.000000
    mean      4.570314
    std       8.770089
    min      -4.169634
    25%       1.734381
    50%       3.078248
    75%       4.303408
    max      63.334634
    Name: value, dtype: float64
    country_iso3
    HTI    -2.571834
    CUB    -0.405484
    ARG     0.940676
    TTO     1.612309
    BOL     1.713476
    CHL     1.773224
    PER     1.903244
    ECU     1.951802
    SUR     2.181257
    CAN     2.424192
    USA     2.730978
    URY     2.778870
    MEX     2.830242
    JAM     2.883536
    SLV     3.031931
    PRY     3.141742
    COL     3.212996
    BRA     3.225888
    HND     3.757793
    GTM     3.789847
    NIC     3.855291
    ESP     4.095476
    DOM     4.127926
    BLZ     4.418435
    CRI     4.661546
    VEN     5.767533
    BHS     5.768269
    PAN     6.984254
    BRB     7.541027
    GUY    46.982985
    Name: value, dtype: float64
    


```python
from pathlib import Path

# Crear carpeta de reportes si no existe
output_dir = Path("data/reports")
output_dir.mkdir(parents=True, exist_ok=True)

# Ruta de salida
output_file = output_dir / "master_radar_cibest.xlsx"

# Exportar master a Excel
master.to_excel(output_file, index=False, sheet_name="master")

print(f"Master exportada correctamente en: {output_file}")
print(f"Shape exportado: {master.shape}")
```

    Master exportada correctamente en: data\reports\master_radar_cibest.xlsx
    Shape exportado: (1281, 5)
    


```python
# from ydata_profiling import ProfileReport
# master_wide = (
#     master
#     .sort_values("year")
#     .drop_duplicates(subset=["country_iso3", "variable"], keep="last")
#     .pivot(index="country_iso3", columns="variable", values="value")
#     .reset_index()
# )

# profile = ProfileReport(
#     master_wide,
#     title="Reporte de Perfilamiento - Master Dataset",
#     explorative=True,
#     minimal=False,
# )

# profile.to_file("master_profile_report.html")
```
