# 02 - Reporte de calidad de datos | RADAR Cibest

**Fase ASUM-DM:** 2 - Entendimiento de datos  
**Objetivo:** Perfilado estadistico de cada variable y deteccion de paises con cobertura insuficiente


```python
import sys
from pathlib import Path
import re
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
from src.utils import load_all_configs, resolve_data_path, get_variable_catalog
from src.data_preparation.cleaning import pivot_latest_value_and_year, apply_freshness_filter


catalog = get_variable_catalog(configs["variables"])

configs = load_all_configs()
raw_dir = resolve_data_path(configs['settings']['data']['raw_path'])



pattern = re.compile(r"^master_raw_\d{8}\.parquet$")

candidates = sorted(
    [
        p for p in raw_dir.glob("master_raw_*.parquet")
        if pattern.match(p.name)
    ],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not candidates:
    raise FileNotFoundError("Ejecute primero el pipeline de extraccion")

master = pd.read_parquet(candidates[0])

print(f"Archivo cargado: {candidates[0].name}")
print(f"Master cargado: {master.shape}")

```

    Archivo cargado: master_raw_20260612.parquet
    Master cargado: (1281, 5)
    

## 1. Perfilado por variable


```python
n_countries = master["country_iso3"].nunique()

stats = (
    master.dropna(subset=["value"])
    .groupby("variable")["value"]
    .agg(
        n_obs="count",
        mean="mean",
        std="std",
        min="min",
        max="max"
    )
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
summary["n_missing_countries"] = (
    summary["n_countries_expected"] - summary["n_countries_with_data"]
)
summary["pct_coverage"] = (
    summary["n_countries_with_data"] / summary["n_countries_expected"]
)
summary["pct_missing"] = 1 - summary["pct_coverage"]

summary = summary.sort_values("pct_missing", ascending=False)

summary_display = summary.copy()

summary_display["pct_coverage"] = (summary_display["pct_coverage"] * 100).round(1)
summary_display["pct_missing"] = (summary_display["pct_missing"] * 100).round(1)

summary_display[[
    "variable",
    "n_obs",
    "n_countries_with_data",
    "n_missing_countries",
    "pct_coverage",
    "pct_missing",
    "mean",
    "std",
    "min",
    "max"
]].style.format({
    "n_obs": "{:,.0f}",
    "n_countries_with_data": "{:,.0f}",
    "n_missing_countries": "{:,.0f}",
    "pct_coverage": "{:.1f}%",
    "pct_missing": "{:.1f}%",
    "mean": "{:,.2f}",
    "std": "{:,.2f}",
    "min": "{:,.2f}",
    "max": "{:,.2f}",
}).background_gradient(
    subset=["pct_coverage"],
    cmap="RdYlGn"
).background_gradient(
    subset=["pct_missing"],
    cmap="YlOrRd"
)
```




<style type="text/css">
#T_11dd2_row0_col4 {
  background-color: #a50026;
  color: #f1f1f1;
}
#T_11dd2_row0_col5 {
  background-color: #800026;
  color: #f1f1f1;
}
#T_11dd2_row1_col4, #T_11dd2_row2_col4 {
  background-color: #e34933;
  color: #f1f1f1;
}
#T_11dd2_row1_col5, #T_11dd2_row2_col5 {
  background-color: #c20325;
  color: #f1f1f1;
}
#T_11dd2_row3_col4 {
  background-color: #f57547;
  color: #f1f1f1;
}
#T_11dd2_row3_col5 {
  background-color: #d7121f;
  color: #f1f1f1;
}
#T_11dd2_row4_col4, #T_11dd2_row5_col4, #T_11dd2_row6_col4, #T_11dd2_row7_col4, #T_11dd2_row8_col4, #T_11dd2_row9_col4, #T_11dd2_row10_col4 {
  background-color: #feffbe;
  color: #000000;
}
#T_11dd2_row4_col5, #T_11dd2_row5_col5, #T_11dd2_row6_col5, #T_11dd2_row7_col5, #T_11dd2_row8_col5, #T_11dd2_row9_col5, #T_11dd2_row10_col5 {
  background-color: #fd8e3c;
  color: #f1f1f1;
}
#T_11dd2_row11_col4, #T_11dd2_row12_col4, #T_11dd2_row13_col4 {
  background-color: #c3e67d;
  color: #000000;
}
#T_11dd2_row11_col5, #T_11dd2_row12_col5, #T_11dd2_row13_col5 {
  background-color: #feb852;
  color: #000000;
}
#T_11dd2_row14_col4 {
  background-color: #6ec064;
  color: #000000;
}
#T_11dd2_row14_col5 {
  background-color: #fedf83;
  color: #000000;
}
#T_11dd2_row15_col4 {
  background-color: #39a758;
  color: #f1f1f1;
}
#T_11dd2_row15_col5 {
  background-color: #ffea9b;
  color: #000000;
}
#T_11dd2_row16_col4, #T_11dd2_row17_col4, #T_11dd2_row18_col4, #T_11dd2_row19_col4, #T_11dd2_row20_col4, #T_11dd2_row21_col4, #T_11dd2_row22_col4, #T_11dd2_row23_col4, #T_11dd2_row24_col4, #T_11dd2_row25_col4, #T_11dd2_row26_col4 {
  background-color: #128a49;
  color: #f1f1f1;
}
#T_11dd2_row16_col5, #T_11dd2_row17_col5, #T_11dd2_row18_col5, #T_11dd2_row19_col5, #T_11dd2_row20_col5, #T_11dd2_row21_col5, #T_11dd2_row22_col5, #T_11dd2_row23_col5, #T_11dd2_row24_col5, #T_11dd2_row25_col5, #T_11dd2_row26_col5 {
  background-color: #fff5b3;
  color: #000000;
}
#T_11dd2_row27_col4, #T_11dd2_row28_col4, #T_11dd2_row29_col4, #T_11dd2_row30_col4, #T_11dd2_row31_col4, #T_11dd2_row32_col4, #T_11dd2_row33_col4, #T_11dd2_row34_col4, #T_11dd2_row35_col4, #T_11dd2_row36_col4, #T_11dd2_row37_col4, #T_11dd2_row38_col4, #T_11dd2_row39_col4, #T_11dd2_row40_col4, #T_11dd2_row41_col4, #T_11dd2_row42_col4, #T_11dd2_row43_col4, #T_11dd2_row44_col4 {
  background-color: #006837;
  color: #f1f1f1;
}
#T_11dd2_row27_col5, #T_11dd2_row28_col5, #T_11dd2_row29_col5, #T_11dd2_row30_col5, #T_11dd2_row31_col5, #T_11dd2_row32_col5, #T_11dd2_row33_col5, #T_11dd2_row34_col5, #T_11dd2_row35_col5, #T_11dd2_row36_col5, #T_11dd2_row37_col5, #T_11dd2_row38_col5, #T_11dd2_row39_col5, #T_11dd2_row40_col5, #T_11dd2_row41_col5, #T_11dd2_row42_col5, #T_11dd2_row43_col5, #T_11dd2_row44_col5 {
  background-color: #ffffcc;
  color: #000000;
}
</style>
<table id="T_11dd2">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_11dd2_level0_col0" class="col_heading level0 col0" >variable</th>
      <th id="T_11dd2_level0_col1" class="col_heading level0 col1" >n_obs</th>
      <th id="T_11dd2_level0_col2" class="col_heading level0 col2" >n_countries_with_data</th>
      <th id="T_11dd2_level0_col3" class="col_heading level0 col3" >n_missing_countries</th>
      <th id="T_11dd2_level0_col4" class="col_heading level0 col4" >pct_coverage</th>
      <th id="T_11dd2_level0_col5" class="col_heading level0 col5" >pct_missing</th>
      <th id="T_11dd2_level0_col6" class="col_heading level0 col6" >mean</th>
      <th id="T_11dd2_level0_col7" class="col_heading level0 col7" >std</th>
      <th id="T_11dd2_level0_col8" class="col_heading level0 col8" >min</th>
      <th id="T_11dd2_level0_col9" class="col_heading level0 col9" >max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_11dd2_level0_row0" class="row_heading level0 row0" >22</th>
      <td id="T_11dd2_row0_col0" class="data row0 col0" >hofstede_ivr</td>
      <td id="T_11dd2_row0_col1" class="data row0 col1" >16</td>
      <td id="T_11dd2_row0_col2" class="data row0 col2" >16</td>
      <td id="T_11dd2_row0_col3" class="data row0 col3" >14</td>
      <td id="T_11dd2_row0_col4" class="data row0 col4" >53.3%</td>
      <td id="T_11dd2_row0_col5" class="data row0 col5" >46.7%</td>
      <td id="T_11dd2_row0_col6" class="data row0 col6" >67.06</td>
      <td id="T_11dd2_row0_col7" class="data row0 col7" >18.05</td>
      <td id="T_11dd2_row0_col8" class="data row0 col8" >44.00</td>
      <td id="T_11dd2_row0_col9" class="data row0 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row1" class="row_heading level0 row1" >39</th>
      <td id="T_11dd2_row1_col0" class="data row1 col0" >stock_market_cap_gdp</td>
      <td id="T_11dd2_row1_col1" class="data row1 col1" >18</td>
      <td id="T_11dd2_row1_col2" class="data row1 col2" >18</td>
      <td id="T_11dd2_row1_col3" class="data row1 col3" >12</td>
      <td id="T_11dd2_row1_col4" class="data row1 col4" >60.0%</td>
      <td id="T_11dd2_row1_col5" class="data row1 col5" >40.0%</td>
      <td id="T_11dd2_row1_col6" class="data row1 col6" >44.21</td>
      <td id="T_11dd2_row1_col7" class="data row1 col7" >56.26</td>
      <td id="T_11dd2_row1_col8" class="data row1 col8" >1.38</td>
      <td id="T_11dd2_row1_col9" class="data row1 col9" >216.29</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row2" class="row_heading level0 row2" >23</th>
      <td id="T_11dd2_row2_col0" class="data row2 col0" >hofstede_lto</td>
      <td id="T_11dd2_row2_col1" class="data row2 col1" >18</td>
      <td id="T_11dd2_row2_col2" class="data row2 col2" >18</td>
      <td id="T_11dd2_row2_col3" class="data row2 col3" >12</td>
      <td id="T_11dd2_row2_col4" class="data row2 col4" >60.0%</td>
      <td id="T_11dd2_row2_col5" class="data row2 col5" >40.0%</td>
      <td id="T_11dd2_row2_col6" class="data row2 col6" >23.33</td>
      <td id="T_11dd2_row2_col7" class="data row2 col7" >14.99</td>
      <td id="T_11dd2_row2_col8" class="data row2 col8" >0.00</td>
      <td id="T_11dd2_row2_col9" class="data row2 col9" >54.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row3" class="row_heading level0 row3" >35</th>
      <td id="T_11dd2_row3_col0" class="data row3 col0" >public_debt_gdp</td>
      <td id="T_11dd2_row3_col1" class="data row3 col1" >19</td>
      <td id="T_11dd2_row3_col2" class="data row3 col2" >19</td>
      <td id="T_11dd2_row3_col3" class="data row3 col3" >11</td>
      <td id="T_11dd2_row3_col4" class="data row3 col4" >63.3%</td>
      <td id="T_11dd2_row3_col5" class="data row3 col5" >36.7%</td>
      <td id="T_11dd2_row3_col6" class="data row3 col6" >64.88</td>
      <td id="T_11dd2_row3_col7" class="data row3 col7" >34.99</td>
      <td id="T_11dd2_row3_col8" class="data row3 col8" >13.15</td>
      <td id="T_11dd2_row3_col9" class="data row3 col9" >125.13</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row4" class="row_heading level0 row4" >21</th>
      <td id="T_11dd2_row4_col0" class="data row4 col0" >hofstede_idv</td>
      <td id="T_11dd2_row4_col1" class="data row4 col1" >23</td>
      <td id="T_11dd2_row4_col2" class="data row4 col2" >23</td>
      <td id="T_11dd2_row4_col3" class="data row4 col3" >7</td>
      <td id="T_11dd2_row4_col4" class="data row4 col4" >76.7%</td>
      <td id="T_11dd2_row4_col5" class="data row4 col5" >23.3%</td>
      <td id="T_11dd2_row4_col6" class="data row4 col6" >35.35</td>
      <td id="T_11dd2_row4_col7" class="data row4 col7" >17.78</td>
      <td id="T_11dd2_row4_col8" class="data row4 col8" >11.00</td>
      <td id="T_11dd2_row4_col9" class="data row4 col9" >72.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row5" class="row_heading level0 row5" >2</th>
      <td id="T_11dd2_row5_col0" class="data row5 col0" >bank_capital_rwa</td>
      <td id="T_11dd2_row5_col1" class="data row5 col1" >23</td>
      <td id="T_11dd2_row5_col2" class="data row5 col2" >23</td>
      <td id="T_11dd2_row5_col3" class="data row5 col3" >7</td>
      <td id="T_11dd2_row5_col4" class="data row5 col4" >76.7%</td>
      <td id="T_11dd2_row5_col5" class="data row5 col5" >23.3%</td>
      <td id="T_11dd2_row5_col6" class="data row5 col6" >17.06</td>
      <td id="T_11dd2_row5_col7" class="data row5 col7" >2.89</td>
      <td id="T_11dd2_row5_col8" class="data row5 col8" >12.70</td>
      <td id="T_11dd2_row5_col9" class="data row5 col9" >25.60</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row6" class="row_heading level0 row6" >4</th>
      <td id="T_11dd2_row6_col0" class="data row6 col0" >bank_npl_ratio</td>
      <td id="T_11dd2_row6_col1" class="data row6 col1" >23</td>
      <td id="T_11dd2_row6_col2" class="data row6 col2" >23</td>
      <td id="T_11dd2_row6_col3" class="data row6 col3" >7</td>
      <td id="T_11dd2_row6_col4" class="data row6 col4" >76.7%</td>
      <td id="T_11dd2_row6_col5" class="data row6 col5" >23.3%</td>
      <td id="T_11dd2_row6_col6" class="data row6 col6" >2.62</td>
      <td id="T_11dd2_row6_col7" class="data row6 col7" >1.34</td>
      <td id="T_11dd2_row6_col8" class="data row6 col8" >0.46</td>
      <td id="T_11dd2_row6_col9" class="data row6 col9" >5.84</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row7" class="row_heading level0 row7" >5</th>
      <td id="T_11dd2_row7_col0" class="data row7 col0" >colombian_diaspora_stock</td>
      <td id="T_11dd2_row7_col1" class="data row7 col1" >23</td>
      <td id="T_11dd2_row7_col2" class="data row7 col2" >23</td>
      <td id="T_11dd2_row7_col3" class="data row7 col3" >7</td>
      <td id="T_11dd2_row7_col4" class="data row7 col4" >76.7%</td>
      <td id="T_11dd2_row7_col5" class="data row7 col5" >23.3%</td>
      <td id="T_11dd2_row7_col6" class="data row7 col6" >11,013.04</td>
      <td id="T_11dd2_row7_col7" class="data row7 col7" >17,275.88</td>
      <td id="T_11dd2_row7_col8" class="data row7 col8" >23.00</td>
      <td id="T_11dd2_row7_col9" class="data row7 col9" >61,020.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row8" class="row_heading level0 row8" >26</th>
      <td id="T_11dd2_row8_col0" class="data row8 col0" >hofstede_uai</td>
      <td id="T_11dd2_row8_col1" class="data row8 col1" >23</td>
      <td id="T_11dd2_row8_col2" class="data row8 col2" >23</td>
      <td id="T_11dd2_row8_col3" class="data row8 col3" >7</td>
      <td id="T_11dd2_row8_col4" class="data row8 col4" >76.7%</td>
      <td id="T_11dd2_row8_col5" class="data row8 col5" >23.3%</td>
      <td id="T_11dd2_row8_col6" class="data row8 col6" >74.30</td>
      <td id="T_11dd2_row8_col7" class="data row8 col7" >21.50</td>
      <td id="T_11dd2_row8_col8" class="data row8 col8" >13.00</td>
      <td id="T_11dd2_row8_col9" class="data row8 col9" >98.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row9" class="row_heading level0 row9" >25</th>
      <td id="T_11dd2_row9_col0" class="data row9 col0" >hofstede_pdi</td>
      <td id="T_11dd2_row9_col1" class="data row9 col1" >23</td>
      <td id="T_11dd2_row9_col2" class="data row9 col2" >23</td>
      <td id="T_11dd2_row9_col3" class="data row9 col3" >7</td>
      <td id="T_11dd2_row9_col4" class="data row9 col4" >76.7%</td>
      <td id="T_11dd2_row9_col5" class="data row9 col5" >23.3%</td>
      <td id="T_11dd2_row9_col6" class="data row9 col6" >65.65</td>
      <td id="T_11dd2_row9_col7" class="data row9 col7" >17.29</td>
      <td id="T_11dd2_row9_col8" class="data row9 col8" >35.00</td>
      <td id="T_11dd2_row9_col9" class="data row9 col9" >95.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row10" class="row_heading level0 row10" >24</th>
      <td id="T_11dd2_row10_col0" class="data row10 col0" >hofstede_mas</td>
      <td id="T_11dd2_row10_col1" class="data row10 col1" >23</td>
      <td id="T_11dd2_row10_col2" class="data row10 col2" >23</td>
      <td id="T_11dd2_row10_col3" class="data row10 col3" >7</td>
      <td id="T_11dd2_row10_col4" class="data row10 col4" >76.7%</td>
      <td id="T_11dd2_row10_col5" class="data row10 col5" >23.3%</td>
      <td id="T_11dd2_row10_col6" class="data row10 col6" >49.13</td>
      <td id="T_11dd2_row10_col7" class="data row10 col7" >14.09</td>
      <td id="T_11dd2_row10_col8" class="data row10 col8" >21.00</td>
      <td id="T_11dd2_row10_col9" class="data row10 col9" >73.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row11" class="row_heading level0 row11" >29</th>
      <td id="T_11dd2_row11_col0" class="data row11 col0" >interest_rate_spread</td>
      <td id="T_11dd2_row11_col1" class="data row11 col1" >25</td>
      <td id="T_11dd2_row11_col2" class="data row11 col2" >25</td>
      <td id="T_11dd2_row11_col3" class="data row11 col3" >5</td>
      <td id="T_11dd2_row11_col4" class="data row11 col4" >83.3%</td>
      <td id="T_11dd2_row11_col5" class="data row11 col5" >16.7%</td>
      <td id="T_11dd2_row11_col6" class="data row11 col6" >7.16</td>
      <td id="T_11dd2_row11_col7" class="data row11 col7" >5.67</td>
      <td id="T_11dd2_row11_col8" class="data row11 col8" >1.48</td>
      <td id="T_11dd2_row11_col9" class="data row11 col9" >32.52</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row12" class="row_heading level0 row12" >0</th>
      <td id="T_11dd2_row12_col0" class="data row12 col0" >account_ownership</td>
      <td id="T_11dd2_row12_col1" class="data row12 col1" >25</td>
      <td id="T_11dd2_row12_col2" class="data row12 col2" >25</td>
      <td id="T_11dd2_row12_col3" class="data row12 col3" >5</td>
      <td id="T_11dd2_row12_col4" class="data row12 col4" >83.3%</td>
      <td id="T_11dd2_row12_col5" class="data row12 col5" >16.7%</td>
      <td id="T_11dd2_row12_col6" class="data row12 col6" >66.24</td>
      <td id="T_11dd2_row12_col7" class="data row12 col7" >20.34</td>
      <td id="T_11dd2_row12_col8" class="data row12 col8" >23.47</td>
      <td id="T_11dd2_row12_col9" class="data row12 col9" >98.40</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row13" class="row_heading level0 row13" >11</th>
      <td id="T_11dd2_row13_col0" class="data row13 col0" >digital_payment_adults_pct</td>
      <td id="T_11dd2_row13_col1" class="data row13 col1" >25</td>
      <td id="T_11dd2_row13_col2" class="data row13 col2" >25</td>
      <td id="T_11dd2_row13_col3" class="data row13 col3" >5</td>
      <td id="T_11dd2_row13_col4" class="data row13 col4" >83.3%</td>
      <td id="T_11dd2_row13_col5" class="data row13 col5" >16.7%</td>
      <td id="T_11dd2_row13_col6" class="data row13 col6" >56.22</td>
      <td id="T_11dd2_row13_col7" class="data row13 col7" >22.96</td>
      <td id="T_11dd2_row13_col8" class="data row13 col8" >16.39</td>
      <td id="T_11dd2_row13_col9" class="data row13 col9" >98.35</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row14" class="row_heading level0 row14" >40</th>
      <td id="T_11dd2_row14_col0" class="data row14 col0" >trade_openness</td>
      <td id="T_11dd2_row14_col1" class="data row14 col1" >27</td>
      <td id="T_11dd2_row14_col2" class="data row14 col2" >27</td>
      <td id="T_11dd2_row14_col3" class="data row14 col3" >3</td>
      <td id="T_11dd2_row14_col4" class="data row14 col4" >90.0%</td>
      <td id="T_11dd2_row14_col5" class="data row14 col5" >10.0%</td>
      <td id="T_11dd2_row14_col6" class="data row14 col6" >68.87</td>
      <td id="T_11dd2_row14_col7" class="data row14 col7" >36.54</td>
      <td id="T_11dd2_row14_col8" class="data row14 col8" >22.25</td>
      <td id="T_11dd2_row14_col9" class="data row14 col9" >194.35</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row15" class="row_heading level0 row15" >9</th>
      <td id="T_11dd2_row15_col0" class="data row15 col0" >country_risk_premium</td>
      <td id="T_11dd2_row15_col1" class="data row15 col1" >28</td>
      <td id="T_11dd2_row15_col2" class="data row15 col2" >28</td>
      <td id="T_11dd2_row15_col3" class="data row15 col3" >2</td>
      <td id="T_11dd2_row15_col4" class="data row15 col4" >93.3%</td>
      <td id="T_11dd2_row15_col5" class="data row15 col5" >6.7%</td>
      <td id="T_11dd2_row15_col6" class="data row15 col6" >0.06</td>
      <td id="T_11dd2_row15_col7" class="data row15 col7" >0.06</td>
      <td id="T_11dd2_row15_col8" class="data row15 col8" >0.00</td>
      <td id="T_11dd2_row15_col9" class="data row15 col9" >0.27</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row16" class="row_heading level0 row16" >6</th>
      <td id="T_11dd2_row16_col0" class="data row16 col0" >commercial_bank_branches_per_100k_adults</td>
      <td id="T_11dd2_row16_col1" class="data row16 col1" >29</td>
      <td id="T_11dd2_row16_col2" class="data row16 col2" >29</td>
      <td id="T_11dd2_row16_col3" class="data row16 col3" >1</td>
      <td id="T_11dd2_row16_col4" class="data row16 col4" >96.7%</td>
      <td id="T_11dd2_row16_col5" class="data row16 col5" >3.3%</td>
      <td id="T_11dd2_row16_col6" class="data row16 col6" >15.17</td>
      <td id="T_11dd2_row16_col7" class="data row16 col7" >11.23</td>
      <td id="T_11dd2_row16_col8" class="data row16 col8" >2.62</td>
      <td id="T_11dd2_row16_col9" class="data row16 col9" >62.62</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row17" class="row_heading level0 row17" >32</th>
      <td id="T_11dd2_row17_col0" class="data row17 col0" >personal_remittances_gdp</td>
      <td id="T_11dd2_row17_col1" class="data row17 col1" >29</td>
      <td id="T_11dd2_row17_col2" class="data row17 col2" >29</td>
      <td id="T_11dd2_row17_col3" class="data row17 col3" >1</td>
      <td id="T_11dd2_row17_col4" class="data row17 col4" >96.7%</td>
      <td id="T_11dd2_row17_col5" class="data row17 col5" >3.3%</td>
      <td id="T_11dd2_row17_col6" class="data row17 col6" >5.94</td>
      <td id="T_11dd2_row17_col7" class="data row17 col7" >8.48</td>
      <td id="T_11dd2_row17_col8" class="data row17 col8" >0.03</td>
      <td id="T_11dd2_row17_col9" class="data row17 col9" >26.64</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row18" class="row_heading level0 row18" >14</th>
      <td id="T_11dd2_row18_col0" class="data row18 col0" >financial_system_deposits_gdp</td>
      <td id="T_11dd2_row18_col1" class="data row18 col1" >29</td>
      <td id="T_11dd2_row18_col2" class="data row18 col2" >29</td>
      <td id="T_11dd2_row18_col3" class="data row18 col3" >1</td>
      <td id="T_11dd2_row18_col4" class="data row18 col4" >96.7%</td>
      <td id="T_11dd2_row18_col5" class="data row18 col5" >3.3%</td>
      <td id="T_11dd2_row18_col6" class="data row18 col6" >61.79</td>
      <td id="T_11dd2_row18_col7" class="data row18 col7" >28.14</td>
      <td id="T_11dd2_row18_col8" class="data row18 col8" >21.91</td>
      <td id="T_11dd2_row18_col9" class="data row18 col9" >119.72</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row19" class="row_heading level0 row19" >28</th>
      <td id="T_11dd2_row19_col0" class="data row19 col0" >inflation_rate</td>
      <td id="T_11dd2_row19_col1" class="data row19 col1" >29</td>
      <td id="T_11dd2_row19_col2" class="data row19 col2" >29</td>
      <td id="T_11dd2_row19_col3" class="data row19 col3" >1</td>
      <td id="T_11dd2_row19_col4" class="data row19 col4" >96.7%</td>
      <td id="T_11dd2_row19_col5" class="data row19 col5" >3.3%</td>
      <td id="T_11dd2_row19_col6" class="data row19 col6" >20.48</td>
      <td id="T_11dd2_row19_col7" class="data row19 col7" >60.50</td>
      <td id="T_11dd2_row19_col8" class="data row19 col8" >-0.41</td>
      <td id="T_11dd2_row19_col9" class="data row19 col9" >254.95</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row20" class="row_heading level0 row20" >27</th>
      <td id="T_11dd2_row20_col0" class="data row20 col0" >ict_goods_exports_pct_total_goods_exports</td>
      <td id="T_11dd2_row20_col1" class="data row20 col1" >29</td>
      <td id="T_11dd2_row20_col2" class="data row20 col2" >29</td>
      <td id="T_11dd2_row20_col3" class="data row20 col3" >1</td>
      <td id="T_11dd2_row20_col4" class="data row20 col4" >96.7%</td>
      <td id="T_11dd2_row20_col5" class="data row20 col5" >3.3%</td>
      <td id="T_11dd2_row20_col6" class="data row20 col6" >1.68</td>
      <td id="T_11dd2_row20_col7" class="data row20 col7" >4.20</td>
      <td id="T_11dd2_row20_col8" class="data row20 col8" >0.00</td>
      <td id="T_11dd2_row20_col9" class="data row20 col9" >17.82</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row21" class="row_heading level0 row21" >10</th>
      <td id="T_11dd2_row21_col0" class="data row21 col0" >current_account_gdp</td>
      <td id="T_11dd2_row21_col1" class="data row21 col1" >29</td>
      <td id="T_11dd2_row21_col2" class="data row21 col2" >29</td>
      <td id="T_11dd2_row21_col3" class="data row21 col3" >1</td>
      <td id="T_11dd2_row21_col4" class="data row21 col4" >96.7%</td>
      <td id="T_11dd2_row21_col5" class="data row21 col5" >3.3%</td>
      <td id="T_11dd2_row21_col6" class="data row21 col6" >-0.20</td>
      <td id="T_11dd2_row21_col7" class="data row21 col7" >4.04</td>
      <td id="T_11dd2_row21_col8" class="data row21 col8" >-6.65</td>
      <td id="T_11dd2_row21_col9" class="data row21 col9" >13.90</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row22" class="row_heading level0 row22" >13</th>
      <td id="T_11dd2_row22_col0" class="data row22 col0" >fdi_net_inflows_gdp</td>
      <td id="T_11dd2_row22_col1" class="data row22 col1" >29</td>
      <td id="T_11dd2_row22_col2" class="data row22 col2" >29</td>
      <td id="T_11dd2_row22_col3" class="data row22 col3" >1</td>
      <td id="T_11dd2_row22_col4" class="data row22 col4" >96.7%</td>
      <td id="T_11dd2_row22_col5" class="data row22 col5" >3.3%</td>
      <td id="T_11dd2_row22_col6" class="data row22 col6" >3.25</td>
      <td id="T_11dd2_row22_col7" class="data row22 col7" >6.50</td>
      <td id="T_11dd2_row22_col8" class="data row22 col8" >-4.86</td>
      <td id="T_11dd2_row22_col9" class="data row22 col9" >34.99</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row23" class="row_heading level0 row23" >1</th>
      <td id="T_11dd2_row23_col0" class="data row23 col0" >atms_per_100k_adults</td>
      <td id="T_11dd2_row23_col1" class="data row23 col1" >29</td>
      <td id="T_11dd2_row23_col2" class="data row23 col2" >29</td>
      <td id="T_11dd2_row23_col3" class="data row23 col3" >1</td>
      <td id="T_11dd2_row23_col4" class="data row23 col4" >96.7%</td>
      <td id="T_11dd2_row23_col5" class="data row23 col5" >3.3%</td>
      <td id="T_11dd2_row23_col6" class="data row23 col6" >69.23</td>
      <td id="T_11dd2_row23_col7" class="data row23 col7" >64.94</td>
      <td id="T_11dd2_row23_col8" class="data row23 col8" >3.81</td>
      <td id="T_11dd2_row23_col9" class="data row23 col9" >314.77</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row24" class="row_heading level0 row24" >12</th>
      <td id="T_11dd2_row24_col0" class="data row24 col0" >domestic_credit_private_gdp</td>
      <td id="T_11dd2_row24_col1" class="data row24 col1" >29</td>
      <td id="T_11dd2_row24_col2" class="data row24 col2" >29</td>
      <td id="T_11dd2_row24_col3" class="data row24 col3" >1</td>
      <td id="T_11dd2_row24_col4" class="data row24 col4" >96.7%</td>
      <td id="T_11dd2_row24_col5" class="data row24 col5" >3.3%</td>
      <td id="T_11dd2_row24_col6" class="data row24 col6" >46.55</td>
      <td id="T_11dd2_row24_col7" class="data row24 col7" >24.90</td>
      <td id="T_11dd2_row24_col8" class="data row24 col8" >3.96</td>
      <td id="T_11dd2_row24_col9" class="data row24 col9" >124.10</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row25" class="row_heading level0 row25" >20</th>
      <td id="T_11dd2_row25_col0" class="data row25 col0" >heritage_efi</td>
      <td id="T_11dd2_row25_col1" class="data row25 col1" >29</td>
      <td id="T_11dd2_row25_col2" class="data row25 col2" >29</td>
      <td id="T_11dd2_row25_col3" class="data row25 col3" >1</td>
      <td id="T_11dd2_row25_col4" class="data row25 col4" >96.7%</td>
      <td id="T_11dd2_row25_col5" class="data row25 col5" >3.3%</td>
      <td id="T_11dd2_row25_col6" class="data row25 col6" >59.50</td>
      <td id="T_11dd2_row25_col7" class="data row25 col7" >12.11</td>
      <td id="T_11dd2_row25_col8" class="data row25 col8" >25.20</td>
      <td id="T_11dd2_row25_col9" class="data row25 col9" >75.60</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row26" class="row_heading level0 row26" >17</th>
      <td id="T_11dd2_row26_col0" class="data row26 col0" >gdp_per_capita_ppp</td>
      <td id="T_11dd2_row26_col1" class="data row26 col1" >29</td>
      <td id="T_11dd2_row26_col2" class="data row26 col2" >29</td>
      <td id="T_11dd2_row26_col3" class="data row26 col3" >1</td>
      <td id="T_11dd2_row26_col4" class="data row26 col4" >96.7%</td>
      <td id="T_11dd2_row26_col5" class="data row26 col5" >3.3%</td>
      <td id="T_11dd2_row26_col6" class="data row26 col6" >29,212.24</td>
      <td id="T_11dd2_row26_col7" class="data row26 col7" >20,521.28</td>
      <td id="T_11dd2_row26_col8" class="data row26 col8" >3,193.67</td>
      <td id="T_11dd2_row26_col9" class="data row26 col9" >85,809.90</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row27" class="row_heading level0 row27" >3</th>
      <td id="T_11dd2_row27_col0" class="data row27 col0" >bank_concentration_5</td>
      <td id="T_11dd2_row27_col1" class="data row27 col1" >30</td>
      <td id="T_11dd2_row27_col2" class="data row27 col2" >30</td>
      <td id="T_11dd2_row27_col3" class="data row27 col3" >0</td>
      <td id="T_11dd2_row27_col4" class="data row27 col4" >100.0%</td>
      <td id="T_11dd2_row27_col5" class="data row27 col5" >0.0%</td>
      <td id="T_11dd2_row27_col6" class="data row27 col6" >83.94</td>
      <td id="T_11dd2_row27_col7" class="data row27 col7" >13.90</td>
      <td id="T_11dd2_row27_col8" class="data row27 col8" >49.68</td>
      <td id="T_11dd2_row27_col9" class="data row27 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row28" class="row_heading level0 row28" >41</th>
      <td id="T_11dd2_row28_col0" class="data row28 col0" >unemployment_rate</td>
      <td id="T_11dd2_row28_col1" class="data row28 col1" >30</td>
      <td id="T_11dd2_row28_col2" class="data row28 col2" >30</td>
      <td id="T_11dd2_row28_col3" class="data row28 col3" >0</td>
      <td id="T_11dd2_row28_col4" class="data row28 col4" >100.0%</td>
      <td id="T_11dd2_row28_col5" class="data row28 col5" >0.0%</td>
      <td id="T_11dd2_row28_col6" class="data row28 col6" >6.25</td>
      <td id="T_11dd2_row28_col7" class="data row28 col7" >3.04</td>
      <td id="T_11dd2_row28_col8" class="data row28 col8" >1.75</td>
      <td id="T_11dd2_row28_col9" class="data row28 col9" >14.94</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row29" class="row_heading level0 row29" >42</th>
      <td id="T_11dd2_row29_col0" class="data row29 col0" >urban_population_pct</td>
      <td id="T_11dd2_row29_col1" class="data row29 col1" >30</td>
      <td id="T_11dd2_row29_col2" class="data row29 col2" >30</td>
      <td id="T_11dd2_row29_col3" class="data row29 col3" >0</td>
      <td id="T_11dd2_row29_col4" class="data row29 col4" >100.0%</td>
      <td id="T_11dd2_row29_col5" class="data row29 col5" >0.0%</td>
      <td id="T_11dd2_row29_col6" class="data row29 col6" >71.06</td>
      <td id="T_11dd2_row29_col7" class="data row29 col7" >15.72</td>
      <td id="T_11dd2_row29_col8" class="data row29 col8" >26.48</td>
      <td id="T_11dd2_row29_col9" class="data row29 col9" >95.60</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row30" class="row_heading level0 row30" >38</th>
      <td id="T_11dd2_row30_col0" class="data row30 col0" >secure_internet_servers_per_million</td>
      <td id="T_11dd2_row30_col1" class="data row30 col1" >30</td>
      <td id="T_11dd2_row30_col2" class="data row30 col2" >30</td>
      <td id="T_11dd2_row30_col3" class="data row30 col3" >0</td>
      <td id="T_11dd2_row30_col4" class="data row30 col4" >100.0%</td>
      <td id="T_11dd2_row30_col5" class="data row30 col5" >0.0%</td>
      <td id="T_11dd2_row30_col6" class="data row30 col6" >25,990.54</td>
      <td id="T_11dd2_row30_col7" class="data row30 col7" >90,393.91</td>
      <td id="T_11dd2_row30_col8" class="data row30 col8" >7.73</td>
      <td id="T_11dd2_row30_col9" class="data row30 col9" >464,560.08</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row31" class="row_heading level0 row31" >37</th>
      <td id="T_11dd2_row31_col0" class="data row31 col0" >rule_of_law</td>
      <td id="T_11dd2_row31_col1" class="data row31 col1" >30</td>
      <td id="T_11dd2_row31_col2" class="data row31 col2" >30</td>
      <td id="T_11dd2_row31_col3" class="data row31 col3" >0</td>
      <td id="T_11dd2_row31_col4" class="data row31 col4" >100.0%</td>
      <td id="T_11dd2_row31_col5" class="data row31 col5" >0.0%</td>
      <td id="T_11dd2_row31_col6" class="data row31 col6" >-0.33</td>
      <td id="T_11dd2_row31_col7" class="data row31 col7" >0.87</td>
      <td id="T_11dd2_row31_col8" class="data row31 col8" >-2.20</td>
      <td id="T_11dd2_row31_col9" class="data row31 col9" >1.46</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row32" class="row_heading level0 row32" >43</th>
      <td id="T_11dd2_row32_col0" class="data row32 col0" >voice_accountability</td>
      <td id="T_11dd2_row32_col1" class="data row32 col1" >30</td>
      <td id="T_11dd2_row32_col2" class="data row32 col2" >30</td>
      <td id="T_11dd2_row32_col3" class="data row32 col3" >0</td>
      <td id="T_11dd2_row32_col4" class="data row32 col4" >100.0%</td>
      <td id="T_11dd2_row32_col5" class="data row32 col5" >0.0%</td>
      <td id="T_11dd2_row32_col6" class="data row32 col6" >0.06</td>
      <td id="T_11dd2_row32_col7" class="data row32 col7" >0.85</td>
      <td id="T_11dd2_row32_col8" class="data row32 col8" >-1.82</td>
      <td id="T_11dd2_row32_col9" class="data row32 col9" >1.50</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row33" class="row_heading level0 row33" >36</th>
      <td id="T_11dd2_row33_col0" class="data row33 col0" >regulatory_quality</td>
      <td id="T_11dd2_row33_col1" class="data row33 col1" >30</td>
      <td id="T_11dd2_row33_col2" class="data row33 col2" >30</td>
      <td id="T_11dd2_row33_col3" class="data row33 col3" >0</td>
      <td id="T_11dd2_row33_col4" class="data row33 col4" >100.0%</td>
      <td id="T_11dd2_row33_col5" class="data row33 col5" >0.0%</td>
      <td id="T_11dd2_row33_col6" class="data row33 col6" >-0.19</td>
      <td id="T_11dd2_row33_col7" class="data row33 col7" >0.78</td>
      <td id="T_11dd2_row33_col8" class="data row33 col8" >-1.90</td>
      <td id="T_11dd2_row33_col9" class="data row33 col9" >1.36</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row34" class="row_heading level0 row34" >7</th>
      <td id="T_11dd2_row34_col0" class="data row34 col0" >common_language_spanish</td>
      <td id="T_11dd2_row34_col1" class="data row34 col1" >30</td>
      <td id="T_11dd2_row34_col2" class="data row34 col2" >30</td>
      <td id="T_11dd2_row34_col3" class="data row34 col3" >0</td>
      <td id="T_11dd2_row34_col4" class="data row34 col4" >100.0%</td>
      <td id="T_11dd2_row34_col5" class="data row34 col5" >0.0%</td>
      <td id="T_11dd2_row34_col6" class="data row34 col6" >0.63</td>
      <td id="T_11dd2_row34_col7" class="data row34 col7" >0.49</td>
      <td id="T_11dd2_row34_col8" class="data row34 col8" >0.00</td>
      <td id="T_11dd2_row34_col9" class="data row34 col9" >1.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row35" class="row_heading level0 row35" >34</th>
      <td id="T_11dd2_row35_col0" class="data row35 col0" >population_total</td>
      <td id="T_11dd2_row35_col1" class="data row35 col1" >30</td>
      <td id="T_11dd2_row35_col2" class="data row35 col2" >30</td>
      <td id="T_11dd2_row35_col3" class="data row35 col3" >0</td>
      <td id="T_11dd2_row35_col4" class="data row35 col4" >100.0%</td>
      <td id="T_11dd2_row35_col5" class="data row35 col5" >0.0%</td>
      <td id="T_11dd2_row35_col6" class="data row35 col6" >36,267,628.40</td>
      <td id="T_11dd2_row35_col7" class="data row35 col7" >72,197,538.25</td>
      <td id="T_11dd2_row35_col8" class="data row35 col8" >282,467.00</td>
      <td id="T_11dd2_row35_col9" class="data row35 col9" >340,110,988.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row36" class="row_heading level0 row36" >33</th>
      <td id="T_11dd2_row36_col0" class="data row36 col0" >political_stability</td>
      <td id="T_11dd2_row36_col1" class="data row36 col1" >30</td>
      <td id="T_11dd2_row36_col2" class="data row36 col2" >30</td>
      <td id="T_11dd2_row36_col3" class="data row36 col3" >0</td>
      <td id="T_11dd2_row36_col4" class="data row36 col4" >100.0%</td>
      <td id="T_11dd2_row36_col5" class="data row36 col5" >0.0%</td>
      <td id="T_11dd2_row36_col6" class="data row36 col6" >0.02</td>
      <td id="T_11dd2_row36_col7" class="data row36 col7" >0.74</td>
      <td id="T_11dd2_row36_col8" class="data row36 col8" >-1.74</td>
      <td id="T_11dd2_row36_col9" class="data row36 col9" >1.28</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row37" class="row_heading level0 row37" >31</th>
      <td id="T_11dd2_row37_col0" class="data row37 col0" >mobile_subscriptions</td>
      <td id="T_11dd2_row37_col1" class="data row37 col1" >30</td>
      <td id="T_11dd2_row37_col2" class="data row37 col2" >30</td>
      <td id="T_11dd2_row37_col3" class="data row37 col3" >0</td>
      <td id="T_11dd2_row37_col4" class="data row37 col4" >100.0%</td>
      <td id="T_11dd2_row37_col5" class="data row37 col5" >0.0%</td>
      <td id="T_11dd2_row37_col6" class="data row37 col6" >114.34</td>
      <td id="T_11dd2_row37_col7" class="data row37 col7" >29.31</td>
      <td id="T_11dd2_row37_col8" class="data row37 col8" >65.19</td>
      <td id="T_11dd2_row37_col9" class="data row37 col9" >176.52</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row38" class="row_heading level0 row38" >30</th>
      <td id="T_11dd2_row38_col0" class="data row38 col0" >internet_users_pct</td>
      <td id="T_11dd2_row38_col1" class="data row38 col1" >30</td>
      <td id="T_11dd2_row38_col2" class="data row38 col2" >30</td>
      <td id="T_11dd2_row38_col3" class="data row38 col3" >0</td>
      <td id="T_11dd2_row38_col4" class="data row38 col4" >100.0%</td>
      <td id="T_11dd2_row38_col5" class="data row38 col5" >0.0%</td>
      <td id="T_11dd2_row38_col6" class="data row38 col6" >80.25</td>
      <td id="T_11dd2_row38_col7" class="data row38 col7" >12.25</td>
      <td id="T_11dd2_row38_col8" class="data row38 col8" >47.86</td>
      <td id="T_11dd2_row38_col9" class="data row38 col9" >95.76</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row39" class="row_heading level0 row39" >8</th>
      <td id="T_11dd2_row39_col0" class="data row39 col0" >control_of_corruption</td>
      <td id="T_11dd2_row39_col1" class="data row39 col1" >30</td>
      <td id="T_11dd2_row39_col2" class="data row39 col2" >30</td>
      <td id="T_11dd2_row39_col3" class="data row39 col3" >0</td>
      <td id="T_11dd2_row39_col4" class="data row39 col4" >100.0%</td>
      <td id="T_11dd2_row39_col5" class="data row39 col5" >0.0%</td>
      <td id="T_11dd2_row39_col6" class="data row39 col6" >-0.20</td>
      <td id="T_11dd2_row39_col7" class="data row39 col7" >0.93</td>
      <td id="T_11dd2_row39_col8" class="data row39 col8" >-1.59</td>
      <td id="T_11dd2_row39_col9" class="data row39 col9" >1.63</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row40" class="row_heading level0 row40" >19</th>
      <td id="T_11dd2_row40_col0" class="data row40 col0" >government_effectiveness</td>
      <td id="T_11dd2_row40_col1" class="data row40 col1" >30</td>
      <td id="T_11dd2_row40_col2" class="data row40 col2" >30</td>
      <td id="T_11dd2_row40_col3" class="data row40 col3" >0</td>
      <td id="T_11dd2_row40_col4" class="data row40 col4" >100.0%</td>
      <td id="T_11dd2_row40_col5" class="data row40 col5" >0.0%</td>
      <td id="T_11dd2_row40_col6" class="data row40 col6" >-0.03</td>
      <td id="T_11dd2_row40_col7" class="data row40 col7" >0.79</td>
      <td id="T_11dd2_row40_col8" class="data row40 col8" >-2.02</td>
      <td id="T_11dd2_row40_col9" class="data row40 col9" >1.76</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row41" class="row_heading level0 row41" >18</th>
      <td id="T_11dd2_row41_col0" class="data row41 col0" >geographic_distance_km</td>
      <td id="T_11dd2_row41_col1" class="data row41 col1" >30</td>
      <td id="T_11dd2_row41_col2" class="data row41 col2" >30</td>
      <td id="T_11dd2_row41_col3" class="data row41 col3" >0</td>
      <td id="T_11dd2_row41_col4" class="data row41 col4" >100.0%</td>
      <td id="T_11dd2_row41_col5" class="data row41 col5" >0.0%</td>
      <td id="T_11dd2_row41_col6" class="data row41 col6" >2,511.67</td>
      <td id="T_11dd2_row41_col7" class="data row41 col7" >1,742.70</td>
      <td id="T_11dd2_row41_col8" class="data row41 col8" >0.00</td>
      <td id="T_11dd2_row41_col9" class="data row41 col9" >8,030.00</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row42" class="row_heading level0 row42" >16</th>
      <td id="T_11dd2_row42_col0" class="data row42 col0" >gdp_nominal</td>
      <td id="T_11dd2_row42_col1" class="data row42 col1" >30</td>
      <td id="T_11dd2_row42_col2" class="data row42 col2" >30</td>
      <td id="T_11dd2_row42_col3" class="data row42 col3" >0</td>
      <td id="T_11dd2_row42_col4" class="data row42 col4" >100.0%</td>
      <td id="T_11dd2_row42_col5" class="data row42 col5" >0.0%</td>
      <td id="T_11dd2_row42_col6" class="data row42 col6" >1,320,375,045,601.12</td>
      <td id="T_11dd2_row42_col7" class="data row42 col7" >5,224,051,519,677.51</td>
      <td id="T_11dd2_row42_col8" class="data row42 col8" >3,203,631,800.00</td>
      <td id="T_11dd2_row42_col9" class="data row42 col9" >28,750,956,130,731.20</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row43" class="row_heading level0 row43" >15</th>
      <td id="T_11dd2_row43_col0" class="data row43 col0" >gdp_growth</td>
      <td id="T_11dd2_row43_col1" class="data row43 col1" >90</td>
      <td id="T_11dd2_row43_col2" class="data row43 col2" >30</td>
      <td id="T_11dd2_row43_col3" class="data row43 col3" >0</td>
      <td id="T_11dd2_row43_col4" class="data row43 col4" >100.0%</td>
      <td id="T_11dd2_row43_col5" class="data row43 col5" >0.0%</td>
      <td id="T_11dd2_row43_col6" class="data row43 col6" >4.57</td>
      <td id="T_11dd2_row43_col7" class="data row43 col7" >8.77</td>
      <td id="T_11dd2_row43_col8" class="data row43 col8" >-4.17</td>
      <td id="T_11dd2_row43_col9" class="data row43 col9" >63.33</td>
    </tr>
    <tr>
      <th id="T_11dd2_level0_row44" class="row_heading level0 row44" >44</th>
      <td id="T_11dd2_row44_col0" class="data row44 col0" >weighted_mean_applied_tariff_all_products</td>
      <td id="T_11dd2_row44_col1" class="data row44 col1" >30</td>
      <td id="T_11dd2_row44_col2" class="data row44 col2" >30</td>
      <td id="T_11dd2_row44_col3" class="data row44 col3" >0</td>
      <td id="T_11dd2_row44_col4" class="data row44 col4" >100.0%</td>
      <td id="T_11dd2_row44_col5" class="data row44 col5" >0.0%</td>
      <td id="T_11dd2_row44_col6" class="data row44 col6" >5.82</td>
      <td id="T_11dd2_row44_col7" class="data row44 col7" >4.74</td>
      <td id="T_11dd2_row44_col8" class="data row44 col8" >0.46</td>
      <td id="T_11dd2_row44_col9" class="data row44 col9" >18.05</td>
    </tr>
  </tbody>
</table>




## 2. Matriz de cobertura por pais y variable


```python
coverage_matrix = master.dropna(subset=['value']).groupby(['country_iso3', 'variable']).size().unstack(fill_value=0)
coverage_matrix = (coverage_matrix > 0).astype(int)
print(f'Cobertura media: {coverage_matrix.mean().mean()*100:.1f}%')
coverage_matrix#.head()
```

    Cobertura media: 90.4%
    




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
      <th>account_ownership</th>
      <th>atms_per_100k_adults</th>
      <th>bank_capital_rwa</th>
      <th>bank_concentration_5</th>
      <th>bank_npl_ratio</th>
      <th>colombian_diaspora_stock</th>
      <th>commercial_bank_branches_per_100k_adults</th>
      <th>common_language_spanish</th>
      <th>control_of_corruption</th>
      <th>country_risk_premium</th>
      <th>...</th>
      <th>public_debt_gdp</th>
      <th>regulatory_quality</th>
      <th>rule_of_law</th>
      <th>secure_internet_servers_per_million</th>
      <th>stock_market_cap_gdp</th>
      <th>trade_openness</th>
      <th>unemployment_rate</th>
      <th>urban_population_pct</th>
      <th>voice_accountability</th>
      <th>weighted_mean_applied_tariff_all_products</th>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CUB</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>VEN</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 45 columns</p>
</div>




```python
import plotly.express as px
fig = px.imshow(coverage_matrix.T, color_continuous_scale=['#FFD0D0', '#0D1B2A'],
                aspect='auto', title='Matriz de cobertura: pais x variable')
fig.update_layout(height=900)
fig.show()
```




```python
#variables y países están siendo castigados por datos antiguos.

wide_values, wide_years = pivot_latest_value_and_year(master)

wide_fresh, stale_report = apply_freshness_filter(
    wide_values=wide_values,
    wide_years=wide_years,
    variable_catalog=catalog,
    settings=configs["settings"],
)

stale_cells = (
    wide_values.notna()
    & wide_fresh.isna()
).sum().sum()

print("Celdas marcadas como stale:", stale_cells)

stale_by_variable = (
    wide_values.notna()
    & wide_fresh.isna()
).sum(axis=0).sort_values(ascending=False)

stale_by_country = (
    wide_values.notna()
    & wide_fresh.isna()
).sum(axis=1).sort_values(ascending=False)

display(stale_by_variable)
display(stale_by_country)
display(stale_report.sort_values("n_stale_variables", ascending=False))
```

    [32m2026-06-12 19:59:12.533[0m | [1mINFO    [0m | [36msrc.data_preparation.cleaning[0m:[36mpivot_latest_value_and_year[0m:[36m133[0m - [1mPivoteo ancho: 30 paises x 45 variables (estrategia=latest_available)[0m
    [32m2026-06-12 19:59:12.552[0m | [1mINFO    [0m | [36msrc.data_preparation.cleaning[0m:[36mapply_freshness_filter[0m:[36m200[0m - [1mControl de antigüedad aplicado: reference_year=2026, max_age=5, celdas stale=100[0m
    [32m2026-06-12 19:59:12.559[0m | [1mINFO    [0m | [36msrc.data_preparation.cleaning[0m:[36mapply_freshness_filter[0m:[36m225[0m - [1mVariables con más datos stale: {'bank_capital_rwa': 23, 'colombian_diaspora_stock': 23, 'public_debt_gdp': 9, 'stock_market_cap_gdp': 6, 'financial_system_deposits_gdp': 6, 'bank_concentration_5': 5, 'interest_rate_spread': 4, 'ict_goods_exports_pct_total_goods_exports': 4, 'atms_per_100k_adults': 3, 'personal_remittances_gdp': 2, 'current_account_gdp': 2, 'digital_payment_adults_pct': 2, 'domestic_credit_private_gdp': 2, 'commercial_bank_branches_per_100k_adults': 2, 'trade_openness': 2}[0m
    [32m2026-06-12 19:59:12.563[0m | [1mINFO    [0m | [36msrc.data_preparation.cleaning[0m:[36mapply_freshness_filter[0m:[36m230[0m - [1mPaíses con más datos stale: {'VEN': 12, 'BRB': 7, 'HTI': 6, 'TTO': 5, 'CAN': 5, 'CHL': 4, 'PAN': 4, 'CUB': 4, 'GTM': 3, 'USA': 3, 'URY': 3, 'PRY': 3, 'JAM': 3, 'BHS': 3, 'ARG': 3}[0m
    

    Celdas marcadas como stale: 100
    


    variable
    bank_capital_rwa                             23
    colombian_diaspora_stock                     23
    public_debt_gdp                               9
    stock_market_cap_gdp                          6
    financial_system_deposits_gdp                 6
    bank_concentration_5                          5
    interest_rate_spread                          4
    ict_goods_exports_pct_total_goods_exports     4
    atms_per_100k_adults                          3
    personal_remittances_gdp                      2
    current_account_gdp                           2
    digital_payment_adults_pct                    2
    domestic_credit_private_gdp                   2
    commercial_bank_branches_per_100k_adults      2
    trade_openness                                2
    inflation_rate                                1
    account_ownership                             1
    weighted_mean_applied_tariff_all_products     1
    gdp_per_capita_ppp                            1
    gdp_nominal                                   1
    geographic_distance_km                        0
    political_stability                           0
    voice_accountability                          0
    urban_population_pct                          0
    unemployment_rate                             0
    bank_npl_ratio                                0
    common_language_spanish                       0
    secure_internet_servers_per_million           0
    rule_of_law                                   0
    regulatory_quality                            0
    control_of_corruption                         0
    population_total                              0
    mobile_subscriptions                          0
    country_risk_premium                          0
    government_effectiveness                      0
    internet_users_pct                            0
    fdi_net_inflows_gdp                           0
    gdp_growth                                    0
    hofstede_uai                                  0
    hofstede_pdi                                  0
    hofstede_mas                                  0
    hofstede_lto                                  0
    hofstede_idv                                  0
    heritage_efi                                  0
    hofstede_ivr                                  0
    dtype: int64



    country_iso3
    VEN    12
    BRB     7
    HTI     6
    TTO     5
    CAN     5
    CHL     4
    PAN     4
    CUB     4
    GTM     3
    USA     3
    URY     3
    PRY     3
    JAM     3
    BHS     3
    ARG     3
    ECU     3
    DOM     3
    CRI     3
    ESP     2
    HND     2
    MEX     2
    NIC     2
    PER     2
    SLV     2
    SUR     2
    BRA     2
    BOL     2
    BLZ     2
    GUY     2
    COL     1
    dtype: int64



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
      <th>n_stale_variables</th>
      <th>pct_stale_variables</th>
    </tr>
    <tr>
      <th>country_iso3</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VEN</th>
      <td>12</td>
      <td>0.266667</td>
    </tr>
    <tr>
      <th>BRB</th>
      <td>7</td>
      <td>0.155556</td>
    </tr>
    <tr>
      <th>HTI</th>
      <td>6</td>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>TTO</th>
      <td>5</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>5</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>4</td>
      <td>0.088889</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>4</td>
      <td>0.088889</td>
    </tr>
    <tr>
      <th>CUB</th>
      <td>4</td>
      <td>0.088889</td>
    </tr>
    <tr>
      <th>GTM</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>URY</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>PRY</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>JAM</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>BHS</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>ARG</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>ECU</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>DOM</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>CRI</th>
      <td>3</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>HND</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>NIC</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>PER</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>SUR</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>BRA</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>BOL</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>BLZ</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>GUY</th>
      <td>2</td>
      <td>0.044444</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>1</td>
      <td>0.022222</td>
    </tr>
  </tbody>
</table>
</div>

