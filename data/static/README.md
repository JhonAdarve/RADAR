# data/static/ - Fuentes complementarias locales

Este directorio contiene únicamente los archivos locales que alimentan variables complementarias del proyecto **RADAR Cibest** y que no se extraen directamente desde `wbgapi`.

La lógica vigente del proyecto es:

- **World Bank / WDI / Findex** se extrae automáticamente con `wbgapi` usando `db=2`.
- **World Bank / WGI** se extrae automáticamente con `wbgapi` usando `db=3`.
- **World Bank / GFDD** se extrae automáticamente con `wbgapi` usando `db=32`.
- **Damodaran Country Risk Premium** se descarga desde Excel remoto de NYU Stern, no se almacena como CSV estático en este directorio.
- **Heritage EFI**, **Hofstede**, **CEPII GeoDist** y **CEPII idioma español** se manejan como fuentes complementarias locales.
- **Salidas de colombianos** se extrae desde `datos.gov.co`; no requiere CSV estático salvo que se quiera guardar una copia de respaldo.

## Archivos locales vigentes

| Archivo | Variable(s) asociada(s) | Fuente | Frecuencia | Esquema esperado | Estado |
|---|---|---|---|---|---|
| `cepii_geodist.csv` | `geographic_distance_km` | CEPII GeoDist | Estática | `country_iso3,value` o `country_iso3,year,value` | Requerido |
| `cepii_language_spanish.csv` | `common_language_spanish` | CEPII / idioma común | Estática | `country_iso3,value` o `country_iso3,year,value` | Requerido |
| `hofstede_country_scores.csv` | `hofstede_pdi`, `hofstede_idv`, `hofstede_mas`, `hofstede_uai`, `hofstede_lto`, `hofstede_ivr` | Hofstede country scores | Estática | `country,pdi,idv,mas,uai,lto,ivr` | Requerido |
| `heritage-index-of-economic-freedom-2026-05-02_2040.csv` | `heritage_efi` | Heritage Foundation | Anual | Archivo original descargado; el pipeline usa `skiprows=3` | Requerido |


## Esquemas esperados

### 1. CSV estándar para variables complementarias simples

Aplica para:

- `cepii_geodist.csv`
- `cepii_language_spanish.csv`

Formato mínimo:

```csv
country_iso3,value
COL,0
MEX,3490
PAN,810
```

También se acepta formato con año:

```csv
country_iso3,year,value
COL,2020,0
MEX,2020,3490
PAN,2020,810
```

Si no existe columna `year`, el pipeline asigna el año estructural definido en configuración, actualmente `2020`.

### 2. Hofstede country scores

Archivo esperado:

```text
hofstede_country_scores.csv
```

Columnas mínimas:

```csv
country,pdi,idv,mas,uai,lto,ivr
Colombia,67,13,64,80,13,83
Mexico,81,30,69,82,24,97
```

El pipeline transforma este archivo a formato largo:

```text
country_iso3 | year | variable | value
```

Variables generadas:

```text
hofstede_pdi
hofstede_idv
hofstede_mas
hofstede_uai
hofstede_lto
hofstede_ivr
```

### 3. Heritage EFI

Archivo esperado:

```text
heritage-index-of-economic-freedom-2026-05-02_2040.csv
```

El archivo se lee como fuente original con metadata inicial, usando:

```python
pd.read_csv(file_path, skiprows=3)
```

Columnas usadas:

```text
Country
Index Year
Overall Score
```

Salida estándar generada:

```text
country_iso3 | year | variable=heritage_efi | value
```

## Fuentes complementarias no locales

### Damodaran Country Risk Premium

No se guarda en `data/static` como CSV. Se descarga automáticamente desde:

```text
https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx
```

Hoja usada:

```text
ERPs by country
```

Columna usada:

```text
Country Risk Premium
```

Variable generada:

```text
country_risk_premium
```

### Salidas de colombianos

No requiere CSV estático. Se extrae desde `datos.gov.co` y se transforma a:

```text
country_iso3 | year | variable=colombian_diaspora_stock | value
```

La lógica suma los meses para obtener total anual por país y conserva el último año disponible.

## Indicadores World Bank que no deben estar en data/static

Los siguientes grupos se extraen automáticamente con `wbgapi` y no deben duplicarse como CSV locales:

### D1. Macroeconómica - `db=2`

```text
NY.GDP.MKTP.CD
NY.GDP.PCAP.PP.CD
NY.GDP.MKTP.KD.ZG
FP.CPI.TOTL.ZG
SP.POP.TOTL
SP.URB.TOTL.IN.ZS
```

### D2. Industria Financiera - `db=2`

```text
FD.AST.PRVT.GD.ZS
FX.OWN.TOTL.ZS
FR.INR.LNDP
FB.AST.NPER.ZS
CM.MKT.LCAP.GD.ZS
BX.TRF.PWKR.DT.GD.ZS
```

### D2. Industria Financiera GFDD - `db=32`

```text
GFDD.OI.06
GFDD.DI.08
GFDD.SI.05
```

### D3. Institucional WGI - `db=3`

```text
GOV_WGI_RQ.EST
GOV_WGI_GE.EST
GOV_WGI_RL.EST
GOV_WGI_PV.EST
GOV_WGI_VA.EST
GOV_WGI_CC.EST
```

### D4. Tecnológica - `db=2`

```text
IT.NET.USER.ZS
IT.CEL.SETS.P2
FG.TRAN.PAYM.ZS
```

## Países esperados

El proyecto trabaja con países de América más España. El alcance operativo principal es:

```text
ARG, BOL, BRA, CHL, COL, ECU, GUY, PRY, PER, SUR, URY, VEN,
BLZ, CRI, SLV, GTM, HND, NIC, PAN,
CAN, MEX, USA,
CUB, DOM, HTI, JAM, TTO, BHS, BRB,
ESP
```

Algunas fuentes complementarias pueden tener cobertura parcial. El pipeline debe manejar faltantes mediante la política de imputación y umbral de datos faltantes definidos en configuración.

## Notas operativas

- No se fuerza un año común entre variables. Para World Bank se usa el último dato disponible por país e indicador.
- Las variables estructurales sin año explícito usan `structural_year_reference`, actualmente `2020`.
- No incluir archivos excluidos en el flujo productivo aunque permanezcan en la carpeta por trazabilidad histórica.
- Si se conserva un archivo excluido para respaldo, se recomienda moverlo a una subcarpeta como `data/static/deprecated/`.
- Todos los CSV locales activos deben usar codificación UTF-8.
