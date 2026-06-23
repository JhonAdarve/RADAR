# RADAR Cibest

**Ranking de Atractivo y Diagnostico Analitico Regional**

Sistema analitico multicriterio hibrido para evaluar el atractivo de mercados
de America (Norte, Centro, Sur, Caribe) y Espana como destinos de la
internacionalizacion de Grupo Cibest, generando senales diferenciadas para
las cinco lineas de negocio del holding.

---

## 1. Vision general

El sistema implementa la arquitectura **BWM + TOPSIS + Gravitacional**
seleccionada en la revision sistematica de literatura del proyecto:

- **BWM (Best-Worst Method, Rezaei 2015)** para capturar la percepcion
  ponderada de la alta direccion sobre la importancia relativa de cinco
  dimensiones de atractivo de mercado.
- **TOPSIS (Hwang & Yoon, 1981)** para rankear los 30 paises del alcance
  con pesos diferenciados por linea de negocio.
- **Modelo gravitacional** para incorporar la afinidad bilateral de cada
  destino con Colombia.
- **Motor de senales** que cruza el perfil de cada pais con las variables
  criticas de cada linea, produciendo etiquetas de oportunidad en cuatro
  niveles.

El score RADAR final se construye como combinacion lineal:

```
RADAR(j, l) = alpha * CC_TOPSIS(j, l) + beta * IPC(j) + gamma * Tendencia(j)
```

Por defecto: `alpha = 0.6`, `beta = 0.3`, `gamma = 0.1` (configurable).

## 2. Lineas de negocio cubiertas

| Codigo | Linea | Foco |
|---|---|---|
| **IB** | Intermediacion Bancaria | Captacion y colocacion |
| **PF** | Pagos y Flujos | Tarjetas, recaudos, remesas |
| **AD** | Activos Digitales | Crypto, tokens, custodia digital |
| **BD** | Bancos Digitales | Operacion 100% digital |
| **CIB** | Corporate & Investment Banking | Estructuracion, WM/AM |

## 3. Dimensiones de evaluacion

El sistema utiliza cinco dimensiones consolidadas (35 variables en total):

1. **Macroeconomica** - tamano, crecimiento y estabilidad del mercado
2. **Financiera** - profundidad y eficiencia del sistema financiero
3. **Institucional** - calidad regulatoria y riesgo pais (consolidados)
4. **Digital-Tecnologica** - penetracion digital y adopcion fintech
5. **Proximidad** - afinidad geografica, cultural e institucional con Colombia

## 4. Estructura del repositorio

```
radar_cibest/
├── README.md
├── requirements.txt
├── config/                    # Configuracion externalizada (YAML)
│   ├── settings.yaml          # Parametros generales
│   ├── variables.yaml         # Diccionario de las 35 variables
│   ├── business_lines.yaml    # Pesos y senales por linea
│   ├── data_sources.yaml      # APIs y endpoints
│   └── weights.yaml           # Pesos BWM (output de elicitacion)
├── src/
│   ├── utils.py               # Excepciones, configs, logger
│   ├── data_extraction/       # Conectores a fuentes (WB, IMF, WGI, TI, complementarias)
│   ├── data_preparation/      # Limpieza, normalizacion, feature engineering
│   ├── scoring/               # BWM, TOPSIS, gravitacional, sensibilidad, hibrido
│   ├── signals/               # Perfiles y senales por linea
│   ├── dashboard/             # App Streamlit (5 paginas)
│   └── automation/            # Scheduler, alertas, versionado
├── notebooks/                 # Cuadernos por fase ASUM-DM
├── tests/                     # Suite pytest
├── data/                      # Crudos, procesados, scores, cache, estaticos
└── docs/                      # Documentacion adicional
```

## 5. Instalacion

### 5.1. Requisitos

- Python 3.9.12 o superior
- Acceso a internet para APIs publicas
- Aproximadamente 500 MB libres en disco

### 5.2. Pasos

```bash
# Clonar el repositorio (ajustar URL al repo interno de Cibest)
git clone <URL_REPO> radar_cibest
cd radar_cibest

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org

# (Opcional) Instalar el proyecto en modo editable
pip install -e .
```

### 5.3. Variables complementarias

Antes de la primera ejecucion productiva, completar los archivos CSV en
`data/static/` con los datos reales. Vea `data/static/README.md` para el
inventario completo.

## 6. Ejecucion

### 6.1. Pipeline completo

```bash
# Extraccion de datos
python -m src.data_extraction.pipeline

# Scoring hibrido (extraccion + scoring + senales)
python -m src.scoring.hybrid_scorer
```

### 6.2. Dashboard

```bash
streamlit run src/dashboard/app.py
```

El dashboard se sirve por defecto en `http://localhost:8501`.

### 6.3. Notebooks (orden recomendado)

```bash
jupyter lab notebooks/
```

1. `01_eda_data_sources.ipynb` - validacion de fuentes
2. `02_data_quality_report.ipynb` - perfilado y cobertura
3. `03_scoring_results.ipynb` - resultados del scoring
4. `04_sensitivity_analysis.ipynb` - analisis de sensibilidad

## 7. Personalizacion

| Cambio | Archivo a editar |
|---|---|
| Agregar pais | `config/settings.yaml` (lista `countries`) |
| Agregar variable | `config/variables.yaml` (dimension correspondiente) |
| Cambiar pesos BWM | `config/weights.yaml` (output de elicitacion) |
| Modificar pesos por linea | `config/business_lines.yaml` (`weight_profile`) |
| Cambiar umbrales de senal | `config/business_lines.yaml` (`signal_thresholds`) |
| Cambiar alpha/beta/gamma | `config/settings.yaml` (`scoring.composite_weights`) |

## 8. Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html   # con cobertura
```

## 9. Roadmap

- **Fase A (mar-may 2026):** MVP cuantitativo con dashboard interactivo
- **Fase B (may-jun 2026):** Narrativas ejecutivas enriquecidas con LLM
- **Mantenimiento (recurrente):** revision anual con la Junta Directiva,
  re-elicitacion BWM cada 2-3 anos

## 10 Licencia y uso

Uso interno exclusivo de Grupo Cibest. Todos los derechos reservados.
