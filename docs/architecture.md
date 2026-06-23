# Arquitectura del Sistema RADAR Cibest

## 1. Principios de diseno

El sistema se construye sobre cinco principios:

1. **Configuracion sobre codigo** - todos los parametros se externalizan
   en archivos YAML versionables
2. **Modularidad** - cada componente se ejecuta independiente o como
   parte del pipeline completo
3. **Tolerancia a fallos** - el pipeline nunca se detiene por fallo
   de una unica fuente
4. **Reproducibilidad** - cada ejecucion se versiona con hash de inputs
   y configuracion
5. **Patron Strategy** - tecnicas MCDM intercambiables sin reescribir
   el flujo principal

## 2. Diagrama de capas

```
+-----------------------------------------------------------------+
|                       DASHBOARD STREAMLIT                       |
|  Ranking | Perfil pais | Senales | Simulador | Tendencias       |
+-----------------------------------------------------------------+
                                 |
+-----------------------------------------------------------------+
|                          MOTOR DE SENALES                       |
|       Perfiles fortalezas/debilidades + Senales x linea         |
+-----------------------------------------------------------------+
                                 |
+-----------------------------------------------------------------+
|                   ORQUESTADOR HIBRIDO (scorer)                  |
|     RADAR = alpha * TOPSIS + beta * IPC + gamma * Tendencia     |
+--------+---------------+-----------------+----------------------+
         |               |                 |
+--------v------+ +------v-------+ +-------v---------+
|     BWM       | |    TOPSIS    | |  GRAVITACIONAL  |
|  (Pesos)      | |   (Ranking)  | |     (IPC)       |
+---------------+ +--------------+ +-----------------+
                                 |
+-----------------------------------------------------------------+
|                    PREPARACION DE DATOS                         |
|     Limpieza -> Feature Engineering -> Normalizacion            |
+-----------------------------------------------------------------+
                                 |
+-----------------------------------------------------------------+
|                       EXTRACCION DE DATOS                       |
|     World Bank | IMF | WGI | TI | Complementarias               |
+-----------------------------------------------------------------+
                                 |
+-----------------------------------------------------------------+
|                       CONFIGURACION YAML                        |
|  settings | variables | business_lines | data_sources | weights |
+-----------------------------------------------------------------+
```

## 3. Flujo de datos

```
APIs externas                                  CSV estaticos
     |                                              |
     v                                              v
[extraccion paralela] -----> data/cache/      [data/static/]
     |                                              |
     +----------------- consolidacion --------------+
                              |
                              v
                  [data/raw/master_raw_*.parquet]
                              |
                              v
                   [data/raw/coverage_report_*]
                              |
                              v
              limpieza + imputacion + features
                              |
                              v
                normalizacion + orientacion
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
          [TOPSIS]         [BWM]          [Gravity IPC]
        global+ x linea   pesos jerar.   proximidad
              |               |                |
              +---------------+----------------+
                              v
                    score RADAR compuesto
                              |
                              v
                       motor de senales
                              |
                              v
              [data/scores/*.parquet versionados]
                              |
                              v
                       Dashboard / Reporte
```

## 4. Decisiones tecnicas clave

### 4.1. Por que BWM y no AHP

- 2n-3 comparaciones vs n(n-1)/2 (Rezaei, 2015)
- Mayor consistencia estadistica documentada
- Carga cognitiva razonable para ejecutivos senior

### 4.2. Por que TOPSIS y no PROMETHEE

- Maneja sin problema 30+ alternativas
- Calculo transparente y explicable
- Estandar en literatura financiera (Cernevicience & Kabasinskas, 2022)

### 4.3. Consolidacion institucional

Las dimensiones regulatoria y de riesgo se consolidaron en una unica
dimension Institucional siguiendo:
- Marco de Kaufmann & Kraay (WGI) que integra ambas
- Reduccion de redundancia (mismas WGI alimentaban ambas)
- Simplificacion BWM a 5 dimensiones (9 comparaciones por ejecutivo)

### 4.4. Por que multiples ejecuciones TOPSIS por linea

Computacionalmente trivial y permite mantener una matriz de datos unica
mientras se varian los vectores de pesos segun la relevancia por linea
de negocio. Es la solucion al Gap 2 identificado en la revision de
literatura.

## 5. Persistencia y versionado

Cada ejecucion produce archivos Parquet con timestamp `YYYYMMDD`:

```
data/raw/
├── world_bank_20260415.parquet
├── imf_20260415.parquet
├── master_raw_20260415.parquet
└── coverage_report_20260415.parquet

data/scores/
├── global_ranking_20260415.parquet
├── radar_by_line_20260415.parquet
├── ipc_20260415.parquet
├── ranking_IB_20260415.parquet
├── ranking_PT_20260415.parquet
├── ...
└── index.yaml          # registro maestro de ejecuciones
```

El archivo `index.yaml` mantiene:
- Hash SHA1 del master_raw usado
- Hash de la configuracion de pesos
- Lista de archivos generados
- Metricas resumen

Esto permite reproducir cualquier resultado pasado y comparar versiones.

## 6. Manejo de errores

Jerarquia de excepciones en `src/utils.py`:

```
RadarCibestError
├── DataExtractionError
├── DataPreparationError
├── ConsistencyError
├── InsufficientDataError
├── ConfigurationError
└── ScoringError
```

Politica:
- Errores de fuente unica -> log + continuar pipeline
- Errores de configuracion -> detener inmediatamente
- Errores de consistencia BWM -> notificar al ejecutivo y solicitar revision
