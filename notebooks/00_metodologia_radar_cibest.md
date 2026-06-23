# 📊 Notebook 00 — Guía metodológica de RADAR Cibest

### *Ranking de Atractivo y Diagnóstico Analítico Regional*

---

**Proyecto:** RADAR Cibest — Técnicas de Selección Internacional de Mercados  
**Autor:** Jhon Adarve — Dirección de Estrategia  
**Audiencia:** este notebook está escrito para que **cualquier lector pueda entenderlo**, sea o no técnico.

---

## ¿Para qué sirve este notebook?

Cuando alguien pregunta *"¿cómo sabe RADAR Cibest que Panamá es más atractivo que Bolivia para nuestra línea de pagos y flujos?"*, este notebook responde la pregunta de principio a fin. Aquí se explica, en orden y con ejemplos, cada técnica que el sistema aplica: cómo se eligieron las dimensiones, cómo se ponderan, cómo se rankea cada país, cómo se incorpora la afinidad con Colombia, cómo se generan las señales por línea de negocio y cómo se valida la robustez del resultado.

La idea es que cualquier persona del Comité Ejecutivo o de la Junta Directiva pueda abrirlo, leerlo de corrido como un libro, y salir entendiendo qué hay detrás de cada cifra del reporte final. Para los analistas del equipo, sirve también como referencia técnica: todas las fórmulas están escritas formalmente y todos los pasos están reproducidos en código ejecutable.

## Cómo leerlo

El notebook se divide en diez capítulos. Los primeros tres dan el contexto del problema y el marco de trabajo. Los capítulos cuatro al ocho explican cada técnica analítica en el orden exacto en que el sistema la aplica. Los capítulos nueve y diez muestran cómo se valida la solidez del modelo y cierran con una recapitulación visual del flujo completo.

Cada capítulo metodológico tiene la misma estructura interna: primero una *intuición* en lenguaje cotidiano, luego la *fórmula* formal, después un *ejemplo numérico* con cinco países que se pueden seguir con calculadora si se quiere, y al final una *visualización interactiva* que permite jugar con los valores.

---


```python
# Configuración inicial del notebook

#Antes de empezar el recorrido cargamos las librerías necesarias y fijamos la paleta corporativa de Cibest para que todos los gráficos compartan el mismo lenguaje visual. Si esta celda corre sin errores significa que el entorno está listo.

# ---------------------------------------------------------------------------
# Importacion de librerias
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from IPython.display import HTML, display, Markdown
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paleta corporativa Cibest
# ---------------------------------------------------------------------------
CIBEST = {
    'gray':        '#2C2A28',   # gris profundo institucional
    'gray_light':  '#CCCAC7',   # gris claro para acentos
    'yellow':      '#FDD923',   # acento amarillo Cibest
    'gold':        '#D6B302',   # dorado caracteristico Cibest
    'gold_light':  '#FFF7D3',   # dorado muy claro para fondos y degradados
    'gold_dark':   '#8F7701',   # dorado profundo
    'gray_bg':     '#F5F5F5',   # fondo gris claro
    'gray_border': '#D0D0D0',   # bordes neutros
    'white':       '#FFFFFF',
    'green':       '#0BA682',   # senal alta oportunidad
    'amber':       '#FF7E41',   # senal baja oportunidad
    'red':         '#C62828',   # senal de riesgo

}

SIGNAL_COLORS = {
    'ALTA_OPORTUNIDAD':     CIBEST['green'],
    'OPORTUNIDAD_MODERADA': CIBEST['gold'],
    'BAJA_OPORTUNIDAD':     CIBEST['amber'],
    'RIESGO':               CIBEST['red'],
}

# Template global de Plotly
PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family='Arial, sans-serif', size=13, color=CIBEST['gray']),
        title=dict(font=dict(size=17, color=CIBEST['gray'])),
        plot_bgcolor=CIBEST['white'],
        paper_bgcolor=CIBEST['white'],
        xaxis=dict(gridcolor=CIBEST['gray_border'], linecolor=CIBEST['gray']),
        yaxis=dict(gridcolor=CIBEST['gray_border'], linecolor=CIBEST['gray']),
        colorway=[CIBEST['gray'], CIBEST['gold'], CIBEST['gray_light'],
                  CIBEST['gold_dark'], CIBEST['green'], CIBEST['amber']],
    )
)

# Degradado personalizado de gris claro a amarillo para heatmaps
cmap_custom = LinearSegmentedColormap.from_list("GrayToYellow", [CIBEST["gray_light"], CIBEST["yellow"]])

# Estilo de tablas pandas
def style_table(df, gradient_cols=None, gradient_cmap='YlGnBu', format_dict=None):
    """Aplica el estilo corporativo Cibest a un DataFrame."""
    styler = df.style.set_table_styles([
        {'selector': 'th',
         'props': [('background-color', CIBEST['gray']),
                   ('color', CIBEST['yellow']),
                   ('font-weight', 'bold'),
                   ('text-align', 'center'),
                   ('padding', '8px'),
                   ('font-family', 'Arial, sans-serif')]},
        {'selector': 'td',
         'props': [('padding', '6px 10px'),
                   ('font-family', 'Arial, sans-serif'),
                   ('border-bottom', f"1px solid {CIBEST['gray_border']}")]},
        {'selector': 'tbody tr:hover',
         'props': [('background-color', CIBEST['gray_bg'])]},
    ])
    if gradient_cols:
        styler = styler.background_gradient(subset=gradient_cols, cmap=gradient_cmap)
    if format_dict:
        styler = styler.format(format_dict)
    return styler

#print('Entorno listo — paleta corporativa Cibest activada')

```

---

# Capítulo 1 — El problema de negocio que RADAR Cibest resuelve

## La pregunta que origina todo

Grupo Cibest tiene operación en Colombia y mira hacia afuera. La región de América Latina, el Caribe, Norteamérica y España ofrece varios mercados potenciales, pero no todos son igualmente atractivos para entrar, ni son atractivos por las mismas razones, ni lo son para todas las líneas de negocio del holding. **Panamá puede ser excelente para Pagos y Flujos gracias a su rol de hub financiero y su diáspora, pero ser solo intermedio para Banca Digital porque el mercado es pequeño en términos de población.** Esa es la complejidad que el sistema tiene que capturar.

La pregunta que la Dirección de Estrategia necesita responder es engañosamente simple: *dados estos 30 mercados candidatos, ¿en qué orden debemos priorizarlos, y por qué razones, y qué tan robusta es esa priorización ante cambios en nuestras prioridades estratégicas?* Responderla bien requiere muchas cosas a la vez: integrar datos heterogéneos de fuentes distintas (Banco Mundial, FMI, indicadores de gobernanza, etc.), incorporar el juicio experto de la alta dirección sobre qué dimensiones importan más, generar señales diferenciadas para cada una de las cinco líneas de negocio, y producir un resultado que se pueda explicar al Comité y a la Junta sin pedirles que estudien optimización multicriterio.

## Las cinco líneas que el sistema atiende

Grupo Cibest agrupa su negocio en cinco líneas, y cada una busca cosas diferentes en un mercado destino. La **Intermediación Bancaria (IB)** es el negocio tradicional de captar depósitos y colocar crédito, y le importa sobre todo la profundidad financiera y la solidez institucional. **Pagos y Flujos (PF)** procesa tarjetas, recaudos y remesas, y necesita penetración digital alta y presencia de diáspora colombiana. **Activos Digitales (AD)** se mueve en cripto, stablecoins y tokens regulados, así que la claridad regulatoria es casi todo. **Bancos Digitales (BD)** opera sin sucursales y prospera donde hay grandes mercados jóvenes con alta digitalización. **Corporate & Investment Banking (CIB)** estructura operaciones para grandes corporativos y busca mercados grandes, estables y con buen rating soberano.

La consecuencia metodológica de esto es importante: **no basta con un solo ranking global**. RADAR Cibest produce un ranking global y además cinco rankings adicionales, uno por línea, cada uno con pesos diferenciados sobre las mismas variables.

## Las cinco dimensiones que se evalúan

Para evaluar cualquier mercado el sistema mira cinco dimensiones distintas que juntas capturan tanto el "tamaño del premio" como la "facilidad de capturarlo" y el "riesgo de intentarlo":


```python
# Tabla de las cinco dimensiones
dimensiones = pd.DataFrame([
    {'Dimensión': 'Macroeconómica',     'Qué responde': 'Qué tan grande, dinámico y estable es el mercado',
     'Ejemplos típicos': 'PIB, crecimiento, inflación, desempleo, integracion internacional (apertura comercial e inversión extrajera directa)'},
    {'Dimensión': 'Financiera',         'Qué responde': 'Qué tan profundo, accesible y eficiente es el sistema financiero',
     'Ejemplos típicos': 'Crédito/PIB, depósitos, cuentas bancarias, ratio de préstamos mororsos'},
    {'Dimensión': 'Institucional',      'Qué responde': 'Qué tan confiable es operar ahí en términos legales y regulatorios',
     'Ejemplos típicos': 'WGI, control de corrupción, calidad regulatoria, riesgo soberano'},
    {'Dimensión': 'Digital-Tecnológica','Qué responde': 'Qué tan preparado está el mercado para servicios digitales',
     'Ejemplos típicos': 'Usuarios de internet, móviles, pagos digitales, adopción cripto'},
    {'Dimensión': 'Proximidad',         'Qué responde': 'Qué tan cerca está de Colombia en sentido amplio',
     'Ejemplos típicos': 'Distancia, idioma, distancia cultural, diáspora, comercio bilateral'},
])
style_table(dimensiones)

```




<style type="text/css">
#T_66f9c th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_66f9c td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_66f9c tbody tr:hover {
  background-color: #F5F5F5;
}
</style>
<table id="T_66f9c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_66f9c_level0_col0" class="col_heading level0 col0" >Dimensión</th>
      <th id="T_66f9c_level0_col1" class="col_heading level0 col1" >Qué responde</th>
      <th id="T_66f9c_level0_col2" class="col_heading level0 col2" >Ejemplos típicos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_66f9c_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_66f9c_row0_col0" class="data row0 col0" >Macroeconómica</td>
      <td id="T_66f9c_row0_col1" class="data row0 col1" >Qué tan grande, dinámico y estable es el mercado</td>
      <td id="T_66f9c_row0_col2" class="data row0 col2" >PIB, crecimiento, inflación, desempleo, integracion internacional (apertura comercial e inversión extrajera directa)</td>
    </tr>
    <tr>
      <th id="T_66f9c_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_66f9c_row1_col0" class="data row1 col0" >Financiera</td>
      <td id="T_66f9c_row1_col1" class="data row1 col1" >Qué tan profundo, accesible y eficiente es el sistema financiero</td>
      <td id="T_66f9c_row1_col2" class="data row1 col2" >Crédito/PIB, depósitos, cuentas bancarias, ratio de préstamos mororsos</td>
    </tr>
    <tr>
      <th id="T_66f9c_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_66f9c_row2_col0" class="data row2 col0" >Institucional</td>
      <td id="T_66f9c_row2_col1" class="data row2 col1" >Qué tan confiable es operar ahí en términos legales y regulatorios</td>
      <td id="T_66f9c_row2_col2" class="data row2 col2" >WGI, control de corrupción, calidad regulatoria, riesgo soberano</td>
    </tr>
    <tr>
      <th id="T_66f9c_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_66f9c_row3_col0" class="data row3 col0" >Digital-Tecnológica</td>
      <td id="T_66f9c_row3_col1" class="data row3 col1" >Qué tan preparado está el mercado para servicios digitales</td>
      <td id="T_66f9c_row3_col2" class="data row3 col2" >Usuarios de internet, móviles, pagos digitales, adopción cripto</td>
    </tr>
    <tr>
      <th id="T_66f9c_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_66f9c_row4_col0" class="data row4 col0" >Proximidad</td>
      <td id="T_66f9c_row4_col1" class="data row4 col1" >Qué tan cerca está de Colombia en sentido amplio</td>
      <td id="T_66f9c_row4_col2" class="data row4 col2" >Distancia, idioma, distancia cultural, diáspora, comercio bilateral</td>
    </tr>
  </tbody>
</table>




La intuición que une las cinco dimensiones es la siguiente: las tres primeras describen *propiedades intrínsecas del mercado destino* (qué tan grande, qué tan desarrollado financieramente, qué tan confiable). La cuarta refleja la *preparación del mercado para los servicios modernos* que ofrece Cibest. La quinta es distinta y muy importante: no describe al mercado destino en sí, sino la *relación bilateral entre Colombia y ese mercado*. Un mercado igualmente atractivo "en abstracto" es más atractivo para Cibest si está cerca, si comparte idioma, si tiene un perfil cultural parecido y si recibe muchos colombianos al año. Esa dimensión bilateral es la que se modela con un *modelo gravitacional* en el capítulo seis.

Cada una de estas cinco dimensiones se descompone en variables observables. Las cinco juntas suman **40 variables** que el sistema extrae periódicamente de cinco fuentes distintas (Banco Mundial, indicadores de gobernanza WGI, prima de riesgo país Damodaran, índice de libertad económica Heritage, y conjuntos de datos complementarios estáticos como CEPII, Hofstede y registros de salidas de colombianos). El detalle completo de las variables se presenta en el capítulo tres, pero por ahora vale la pena tener en mente que la distribución del trabajo entre dimensiones **no es simétrica**: Macroecnomica,Proximidad y Financiera tienen más de nueve variables cada una, mientras que Digital-Tecnológica tiene solo siete debido a las limitadas fuentes y completitud (por la naturaleza descentralizada de las criptomonedas y la novedad de las finanzas abiertas y finctech, muchas de estas métricas no miden transacciones directas, sino que se utilizan "proxys" (indicadores indirectos) sobre adopción, regulación y comportamiento). Esa asimetría es una decisión metodológica deliberada que se explica en su momento, y refleja entre otras cosas que las seis dimensiones culturales de Hofstede se incluyen como variables individuales antes de agregarse en un índice de distancia cultural.

---

# Capítulo 2 — El marco metodológico: ASUM-DM

## ¿Por qué un marco metodológico y no improvisación?

Un proyecto analítico que va a llegar a la Junta Directiva y que va a guiar decisiones de inversión internacional no puede entregarse con una sola corrida exitosa de código. Tiene que ser *reproducible, auditable, robusto y revisable*. Para conseguir eso RADAR Cibest se apoya en una metodología estructurada llamada **ASUM-DM** (*Analytics Solutions Unified Method for Data Mining*), que es la evolución moderna de CRISP-DM adaptada por IBM para proyectos analíticos de gran escala.

ASUM-DM organiza el trabajo en cinco fases secuenciales pero iterativas. *Secuenciales* porque cada fase produce un entregable que alimenta la siguiente. *Iterativas* porque si una fase posterior revela un problema (por ejemplo, descubrir en la fase de modelado que faltan datos críticos), se vuelve atrás a la fase apropiada y se itera. El sistema RADAR está implementado siguiendo este flujo de cinco fases con checkpoints formales al cierre de cada una.

## Las cinco fases de ASUM-DM aplicadas a RADAR


```python
fases = pd.DataFrame([
    {'Fase': '1. Entendimiento del negocio',
     'Pregunta central': '¿Qué problema de negocio resolvemos y cómo se medirá éxito?',
     'Entregable en RADAR': 'Project Charter aprobado, matriz de riesgos, alcance de 30 países'},
    {'Fase': '2. Entendimiento de los datos',
     'Pregunta central': '¿Qué datos tenemos, cuáles necesitamos y qué tan buenos son?',
     'Entregable en RADAR': 'Revisión sistemática de literatura, diccionario de variables, mapa de fuentes'},
    {'Fase': '3. Preparación de los datos',
     'Pregunta central': '¿Cómo transformamos los datos crudos en una matriz lista para modelar?',
     'Entregable en RADAR': 'Pipeline de extracción, limpieza, imputación, normalización con dirección'},
    {'Fase': '4. Modelado',
     'Pregunta central': '¿Qué técnicas aplicamos y cómo las integramos?',
     'Entregable en RADAR': 'BWM (pesos) + TOPSIS (ranking) + Gravitacional (proximidad) + Score compuesto'},
    {'Fase': '5. Evaluación',
     'Pregunta central': '¿Qué tan robusto y útil es el resultado?',
     'Entregable en RADAR': 'Análisis de sensibilidad, validación cruzada TOPSIS-VIKOR, reporte ejecutivo'},
])
style_table(fases)

```




<style type="text/css">
#T_324fd th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_324fd td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_324fd tbody tr:hover {
  background-color: #F5F5F5;
}
</style>
<table id="T_324fd">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_324fd_level0_col0" class="col_heading level0 col0" >Fase</th>
      <th id="T_324fd_level0_col1" class="col_heading level0 col1" >Pregunta central</th>
      <th id="T_324fd_level0_col2" class="col_heading level0 col2" >Entregable en RADAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_324fd_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_324fd_row0_col0" class="data row0 col0" >1. Entendimiento del negocio</td>
      <td id="T_324fd_row0_col1" class="data row0 col1" >¿Qué problema de negocio resolvemos y cómo se medirá éxito?</td>
      <td id="T_324fd_row0_col2" class="data row0 col2" >Project Charter aprobado, matriz de riesgos, alcance de 30 países</td>
    </tr>
    <tr>
      <th id="T_324fd_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_324fd_row1_col0" class="data row1 col0" >2. Entendimiento de los datos</td>
      <td id="T_324fd_row1_col1" class="data row1 col1" >¿Qué datos tenemos, cuáles necesitamos y qué tan buenos son?</td>
      <td id="T_324fd_row1_col2" class="data row1 col2" >Revisión sistemática de literatura, diccionario de variables, mapa de fuentes</td>
    </tr>
    <tr>
      <th id="T_324fd_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_324fd_row2_col0" class="data row2 col0" >3. Preparación de los datos</td>
      <td id="T_324fd_row2_col1" class="data row2 col1" >¿Cómo transformamos los datos crudos en una matriz lista para modelar?</td>
      <td id="T_324fd_row2_col2" class="data row2 col2" >Pipeline de extracción, limpieza, imputación, normalización con dirección</td>
    </tr>
    <tr>
      <th id="T_324fd_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_324fd_row3_col0" class="data row3 col0" >4. Modelado</td>
      <td id="T_324fd_row3_col1" class="data row3 col1" >¿Qué técnicas aplicamos y cómo las integramos?</td>
      <td id="T_324fd_row3_col2" class="data row3 col2" >BWM (pesos) + TOPSIS (ranking) + Gravitacional (proximidad) + Score compuesto</td>
    </tr>
    <tr>
      <th id="T_324fd_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_324fd_row4_col0" class="data row4 col0" >5. Evaluación</td>
      <td id="T_324fd_row4_col1" class="data row4 col1" >¿Qué tan robusto y útil es el resultado?</td>
      <td id="T_324fd_row4_col2" class="data row4 col2" >Análisis de sensibilidad, validación cruzada TOPSIS-VIKOR, reporte ejecutivo</td>
    </tr>
  </tbody>
</table>




Vale la pena hacer una observación pedagógica importante aquí: **las técnicas analíticas concretas (BWM, TOPSIS, modelo gravitacional) aparecen únicamente en la fase 4**. Las tres primeras fases no tienen "matemática" en sentido estricto; son trabajo de definición, comprensión y preparación. Pero un proyecto que se salta esas tres fases para llegar rápido a la técnica suele producir resultados técnicamente funcionales pero conceptualmente equivocados. La calidad final de RADAR Cibest depende tanto del rigor de la fase 4 como de la solidez de las fases 1, 2 y 3 que la preceden.

## Cómo encaja todo en el flujo del sistema


```python
import plotly.graph_objects as go

# ============================================================
# Diagrama del flujo del sistema RADAR
# Proyecto analítico de selección internacional de mercados
# ============================================================

fig = go.Figure()


# ------------------------------------------------------------
# Definición de nodos
# x, y controlan la posición.
# w y h controlan tamaño visual de cada caja.
# ------------------------------------------------------------
nodos = [
    {
        "id": "fuentes",
        "texto": "Fuentes de datos<br><span style='font-size:11px'>WB, FMI, WGI, ITC, etc.</span>",
        "x": 0, "y": 4,
        "color": CIBEST["gray_bg"],
        "font": CIBEST["gray"],
        "grupo": "Datos"
    },
    {
        "id": "extraccion",
        "texto": "Extracción<br><span style='font-size:11px'>APIs y CSV estáticos</span>",
        "x": 1.5, "y": 4,
        "color": CIBEST["gray_light"],
        "font": CIBEST["gray"],
        "grupo": "Datos"
    },
    {
        "id": "preparacion",
        "texto": "Preparación<br><span style='font-size:11px'>Limpieza, imputación<br>y normalización</span>",
        "x": 3.0, "y": 4,
        "color": CIBEST["gray_light"],
        "font": CIBEST["gray"],
        "grupo": "Datos"
    },
    {
        "id": "matriz",
        "texto": "Matriz de decisión<br><span style='font-size:11px'>País × variable</span>",
        "x": 4.5, "y": 4,
        "color": CIBEST["gold_light"],
        "font": CIBEST["gray"],
        "grupo": "Modelo"
    },

    {
        "id": "bwm",
        "texto": "BWM<br><span style='font-size:11px'>Pesos del panel</span>",
        "x": 6.2, "y": 5.4,
        "color": CIBEST["gray"],
        "font": "white",
        "grupo": "Modelo"
    },
    {
        "id": "topsis",
        "texto": "TOPSIS<br><span style='font-size:11px'>Ranking competitivo</span>",
        "x": 6.2, "y": 4,
        "color": CIBEST["gray"],
        "font": "white",
        "grupo": "Modelo"
    },
    {
        "id": "gravitacional",
        "texto": "Modelo gravitacional<br><span style='font-size:11px'>IPC / potencial comercial</span>",
        "x": 6.2, "y": 2.6,
        "color": CIBEST["gray"],
        "font": "white",
        "grupo": "Modelo"
    },

    {
        "id": "score",
        "texto": "Score RADAR compuesto<br><span style='font-size:11px'>α·CC + β·IPC + γ·Tendencia</span>",
        "x": 8.2, "y": 4,
        "color": CIBEST["gold"],
        "font": CIBEST["gray"],
        "grupo": "Decisión"
    },
    {
        "id": "senales",
        "texto": "Motor de señales<br><span style='font-size:11px'>4 niveles × 5 líneas</span>",
        "x": 10.0, "y": 4,
        "color": CIBEST["gold_dark"],
        "font": "white",
        "grupo": "Decisión"
    },
    {
        "id": "reporte",
        "texto": "Reporte estratégico<br><span style='font-size:11px'>Dashboard + narrativa ejecutiva</span>",
        "x": 11.8, "y": 4,
        "color": CIBEST["green"],
        "font": "white",
        "grupo": "Salida"
    },
]

# Diccionario de posiciones para conectores
pos = {n["id"]: (n["x"], n["y"]) for n in nodos}

# ------------------------------------------------------------
# Función para crear cajas como anotaciones
# Esto evita problemas de texto sobre marcadores circulares
# ------------------------------------------------------------
def agregar_caja(fig, x, y, texto, color, font_color, ancho=1.25, alto=0.62):
    fig.add_annotation(
        x=x,
        y=y,
        text=texto,
        showarrow=False,
        align="center",
        font=dict(
            size=13,
            color=font_color,
            family="Arial"
        ),
        bgcolor=color,
        bordercolor=CIBEST["gray"],
        borderwidth=1.2,
        borderpad=8,
        opacity=1
    )

# ------------------------------------------------------------
# Función para agregar conectores
# Se usan desplazamientos para que las flechas no entren
# directamente al centro de las cajas
# ------------------------------------------------------------
def agregar_flecha(fig, origen, destino, color=CIBEST["gray"]):
    x0, y0 = pos[origen]
    x1, y1 = pos[destino]

    # Ajuste horizontal para que la flecha salga/entre por bordes
    dx = 0.52 if x1 >= x0 else -0.52
    

    fig.add_annotation(
        x=x1 - dx,
        y=y1,
        ax=x0 + dx,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=color,
        opacity=0.85
    )

# ------------------------------------------------------------
# Bandas de fondo por fase
# ------------------------------------------------------------
fases = [
    {
        "nombre": "1. Datos",
        "x0": -0.7, "x1": 3.75,
        "color": "#F8F9FA"
    },
    {
        "nombre": "2. Modelamiento analítico",
        "x0": 3.75, "x1": 7.35,
        "color": "#FFF8E8"
    },
    {
        "nombre": "3. Priorización y señales",
        "x0": 7.35, "x1": 10.9,
        "color": "#F7F2E8"
    },
    {
        "nombre": "4. Salida ejecutiva",
        "x0": 10.9, "x1": 12.6,
        "color": "#EAF6EF"
    }
]

for fase in fases:
    fig.add_shape(
        type="rect",
        x0=fase["x0"], x1=fase["x1"],
        y0=1.7, y1=6.15,
        line=dict(width=0),
        fillcolor=fase["color"],
        layer="below"
    )

    fig.add_annotation(
        x=(fase["x0"] + fase["x1"]) / 2,
        y=6.0,
        text=f"<b>{fase['nombre']}</b>",
        showarrow=False,
        font=dict(size=12, color=CIBEST["gray"]),
        align="center"
    )

# ------------------------------------------------------------
# Agregar nodos
# ------------------------------------------------------------
for n in nodos:
    agregar_caja(
        fig=fig,
        x=n["x"],
        y=n["y"],
        texto=n["texto"],
        color=n["color"],
        font_color=n["font"]
    )

# ------------------------------------------------------------
# Conectores principales
# ------------------------------------------------------------
conectores = [
    ("fuentes", "extraccion"),
    ("extraccion", "preparacion"),
    ("preparacion", "matriz"),

    ("matriz", "bwm"),
    ("matriz", "topsis"),
    ("matriz", "gravitacional"),

    ("bwm", "score"),
    ("topsis", "score"),
    ("gravitacional", "score"),

    ("score", "senales"),
    ("senales", "reporte"),
]

for origen, destino in conectores:
    agregar_flecha(fig, origen, destino)

# ------------------------------------------------------------
# Nota metodológica inferior
# ------------------------------------------------------------
fig.add_annotation(
    x=5.9,
    y=1.35,
    text=(
        "El sistema RADAR integra fuentes internacionales, técnicas multicriterio "
        "y señales estratégicas para priorizar mercados con mayor atractivo y viabilidad."
    ),
    showarrow=False,
    font=dict(size=11, color=CIBEST["gray"]),
    align="center"
)

# ------------------------------------------------------------
# Layout general
# ------------------------------------------------------------
fig.update_layout(
    title=dict(
        text=(
            "<b>Flujo del sistema RADAR Cibest</b>"
            "<br><sup>Proyecto analítico de selección internacional de mercados</sup>"
        ),
        x=0.5,
        xanchor="center",
        font=dict(size=22, color=CIBEST["gray"])
    ),
    xaxis=dict(
        visible=False,
        range=[-0.8, 12.7],
        fixedrange=True
    ),
    yaxis=dict(
        visible=False,
        range=[1.1, 6.45],
        fixedrange=True
    ),
    height=560,
    width=1250,
    plot_bgcolor=CIBEST["white"],
    paper_bgcolor=CIBEST["white"],
    font=dict(
        family="Arial",
        color=CIBEST["gray"]
    ),
    margin=dict(t=95, b=40, l=30, r=30)
)

fig.show()
```



La imagen anterior es el mapa mental del notebook completo. En los capítulos siguientes vamos a abrir cada uno de esos bloques uno por uno. Si en algún momento se pierde el hilo, esta es la figura a la que conviene regresar.

---

# Capítulo 3 — Las dimensiones y sus variables

## De dimensiones a variables medibles

En el capítulo uno presentamos las cinco dimensiones del análisis (Macro, Financiera, Institucional, Digital-Tecnológica y Proximidad). Pero esas dimensiones son *conceptos*, no son cosas que se puedan medir directamente. Para que el sistema pueda trabajar con ellas hay que descomponer cada una en *variables observables*, cada una con su fuente de datos, su unidad, su rango típico y su **dirección**.

La dirección de una variable es un concepto sutil pero crucial. Decir que una variable tiene *dirección positiva* significa que **valores más altos son mejores para el atractivo del mercado**. El PIB per cápita tiene dirección positiva: más alto es mejor. Decir que una variable tiene *dirección negativa* significa que valores más altos son peores. La inflación tiene dirección negativa: más alta es peor. La distancia geográfica a Bogotá también tiene dirección negativa: más lejos es peor (en términos de proximidad con Colombia). Sin clasificar bien la dirección de cada variable el sistema invertiría el sentido del ranking en algunas dimensiones, y producir un resultado erróneo.

Hay un tercer caso, sutilmente distinto: la dirección **neutral**. Una variable es neutral cuando ni "mucho" ni "poco" son mejores en términos absolutos; lo que importa es **la similitud con Colombia**. Las seis dimensiones culturales de Hofstede son el ejemplo perfecto: no es mejor tener un Power Distance alto o bajo en sentido absoluto; es mejor que el Power Distance del país destino sea *parecido al de Colombia*, porque la similitud cultural facilita la operación y reduce los costos de adaptación. El tratamiento de las variables neutrales requiere un paso de transformación adicional que veremos al final del capítulo: se calcula la **distancia cultural Kogut-Singh** entre cada destino y Colombia agregando las seis dimensiones Hofstede en un solo índice, y esa distancia agregada sí tiene dirección negativa (menor distancia es mejor).

## El diccionario completo de variables

A continuación se presenta el diccionario completo del proyecto. La tabla está organizada por dimensión y dentro de cada dimensión por orden de prioridad. Para cada variable se documenta el nombre técnico (usado en el código), la descripción en lenguaje natural, la fuente, el código del indicador en la fuente, la dirección y la frecuencia de actualización.


```python
# Diccionario completo de las 45 variables del proyecto RADAR Cibest
# Fuente: Excel "RADAR Diccionario Variables Actualizado" - version definitiva 2026


diccionario = pd.DataFrame([

    # ===== MACROECONOMICA - 12 variables =====
    {'Dim': 'Macro', '#':  1, 'Variable': 'gdp_nominal', 'Descripcion': 'PIB nominal (USD corrientes)',
     'Fuente': 'World Bank', 'Codigo': 'NY.GDP.MKTP.CD', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#':  2, 'Variable': 'gdp_per_capita_ppp', 'Descripcion': 'PIB per capita PPP (USD intl.)',
     'Fuente': 'World Bank', 'Codigo': 'NY.GDP.PCAP.PP.CD', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#':  3, 'Variable': 'gdp_growth', 'Descripcion': 'Crecimiento real anual del PIB',
     'Fuente': 'World Bank', 'Codigo': 'NY.GDP.MKTP.KD.ZG', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#':  4, 'Variable': 'inflation_rate', 'Descripcion': 'Inflacion anual IPC',
     'Fuente': 'World Bank', 'Codigo': 'FP.CPI.TOTL.ZG', 'Direccion': 'negative', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#':  5, 'Variable': 'population_total', 'Descripcion': 'Poblacion total',
     'Fuente': 'World Bank', 'Codigo': 'SP.POP.TOTL', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#':  6, 'Variable': 'urban_population_pct', 'Descripcion': 'Poblacion urbana (% del total)',
     'Fuente': 'World Bank', 'Codigo': 'SP.URB.TOTL.IN.ZS', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#':  7, 'Variable': 'unemployment_rate', 'Descripcion': 'Desempleo total (% fuerza laboral)',
     'Fuente': 'World Bank', 'Codigo': 'SL.UEM.TOTL.ZS', 'Direccion': 'negative', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#':  8, 'Variable': 'current_account_gdp', 'Descripcion': 'Cuenta corriente / PIB (%)',
     'Fuente': 'World Bank', 'Codigo': 'BN.CAB.XOKA.GD.ZS', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#':  9, 'Variable': 'public_debt_gdp', 'Descripcion': 'Deuda publica / PIB (%)',
     'Fuente': 'World Bank', 'Codigo': 'GC.DOD.TOTL.GD.ZS', 'Direccion': 'negative', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#': 10, 'Variable': 'trade_openness', 'Descripcion': 'Comercio total (% PIB)',
     'Fuente': 'World Bank', 'Codigo': 'NE.TRD.GNFS.ZS', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#': 11, 'Variable': 'fdi_net_inflows_gdp', 'Descripcion': 'IED neta / PIB (%)',
     'Fuente': 'World Bank', 'Codigo': 'BX.KLT.DINV.WD.GD.ZS', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Macro', '#': 12, 'Variable': 'weighted_mean_applied_tariff_all_products', 'Descripcion': 'Tarifa aplicada promedio ponderada (%)',
     'Fuente': 'World Bank', 'Codigo': 'TM.TAX.MRCH.WM.AR.ZS', 'Direccion': 'negative', 'Frecuencia': 'Anual'},


    # ===== FINANCIERA - 9 variables =====
    {'Dim': 'Financ.', '#': 13, 'Variable': 'domestic_credit_private_gdp',
     'Descripcion': 'Credito al sector privado (% PIB)', 'Fuente': 'World Bank',
     'Codigo': 'FD.AST.PRVT.GD.ZS', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Financ.', '#': 14, 'Variable': 'account_ownership',
     'Descripcion': 'Adultos con cuenta', 'Fuente': 'World Bank',
     'Codigo': 'FX.OWN.TOTL.ZS', 'Direccion': 'positive', 'Frecuencia': 'Trienal'},

    {'Dim': 'Financ.', '#': 15, 'Variable': 'interest_rate_spread',
     'Descripcion': 'Spread tasa activa - pasiva', 'Fuente': 'World Bank',
     'Codigo': 'FR.INR.LNDP', 'Direccion': 'negative', 'Frecuencia': 'Anual'},

    {'Dim': 'Financ.', '#': 16, 'Variable': 'bank_npl_ratio',
     'Descripcion': 'Cartera vencida (%)', 'Fuente': 'World Bank',
     'Codigo': 'FB.AST.NPER.ZS', 'Direccion': 'negative', 'Frecuencia': 'Anual'},

    {'Dim': 'Financ.', '#': 17, 'Variable': 'stock_market_cap_gdp',
     'Descripcion': 'Capitalizacion bursatil (% PIB)', 'Fuente': 'World Bank',
     'Codigo': 'CM.MKT.LCAP.GD.ZS', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Financ.', '#': 18, 'Variable': 'personal_remittances_gdp',
     'Descripcion': 'Remesas (% PIB)', 'Fuente': 'World Bank',
     'Codigo': 'BX.TRF.PWKR.DT.GD.ZS', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Financ.', '#': 19, 'Variable': 'bank_concentration_5',
     'Descripcion': 'Concentracion top-5 bancos', 'Fuente': 'World Bank',
     'Codigo': 'GFDD.OI.06', 'Direccion': 'negative', 'Frecuencia': 'Anual'},

    {'Dim': 'Financ.', '#': 20, 'Variable': 'financial_system_deposits_gc',
     'Descripcion': 'Depositos / PIB', 'Fuente': 'World Bank',
     'Codigo': 'GFDD.DI.08', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Financ.', '#': 21, 'Variable': 'bank_capital_rwa',
     'Descripcion': 'Capital regulatorio / RWA', 'Fuente': 'World Bank',
     'Codigo': 'GFDD.SI.05', 'Direccion': 'positive', 'Frecuencia': 'Anual'},


    # ===== INSTITUCIONAL - 8 variables =====
    {'Dim': 'Instit.', '#': 22, 'Variable': 'regulatory_quality', 'Descripcion': 'WGI: calidad regulatoria',
     'Fuente': 'World Bank', 'Codigo': 'GOV_WGI_RQ.EST', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Instit.', '#': 23, 'Variable': 'government_effectiveness', 'Descripcion': 'WGI: efectividad gubernamental',
     'Fuente': 'World Bank', 'Codigo': 'GOV_WGI_GE.EST', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Instit.', '#': 24, 'Variable': 'rule_of_law', 'Descripcion': 'WGI: estado de derecho',
     'Fuente': 'World Bank', 'Codigo': 'GOV_WGI_RL.EST', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Instit.', '#': 25, 'Variable': 'political_stability', 'Descripcion': 'WGI: estabilidad politica',
     'Fuente': 'World Bank', 'Codigo': 'GOV_WGI_PV.EST', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Instit.', '#': 26, 'Variable': 'voice_accountability', 'Descripcion': 'WGI: voz y rendicion de cuentas',
     'Fuente': 'World Bank', 'Codigo': 'GOV_WGI_VA.EST', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Instit.', '#': 27, 'Variable': 'control_of_corruption', 'Descripcion': 'WGI: control de corrupcion',
     'Fuente': 'World Bank', 'Codigo': 'GOV_WGI_CC.EST', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Instit.', '#': 28, 'Variable': 'country_risk_premium', 'Descripcion': 'Prima de riesgo pais',
     'Fuente': 'Damodaran', 'Codigo': 'COUNTRY_RISK_PREMIUM', 'Direccion': 'negative', 'Frecuencia': 'Anual'},

    {'Dim': 'Instit.', '#': 29, 'Variable': 'heritage_efi', 'Descripcion': 'Economic Freedom Index',
     'Fuente': 'Heritage', 'Codigo': 'HERITAGE_EFI', 'Direccion': 'positive', 'Frecuencia': 'Anual'},


    # ===== DIGITAL - 7 variables =====
    {'Dim': 'Digital', '#': 30, 'Variable': 'internet_users_pct', 'Descripcion': 'Usuarios de internet',
     'Fuente': 'World Bank', 'Codigo': 'IT.NET.USER.ZS', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Digital', '#': 31, 'Variable': 'mobile_subscriptions', 'Descripcion': 'Suscripciones moviles',
     'Fuente': 'World Bank', 'Codigo': 'IT.CEL.SETS.P2', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Digital', '#': 32, 'Variable': 'digital_payment_adults_pct', 'Descripcion': 'Pagos digitales',
     'Fuente': 'World Bank', 'Codigo': 'g20.any', 'Direccion': 'positive', 'Frecuencia': 'Trienal'},

    {'Dim': 'Digital', '#': 33, 'Variable': 'secure_internet_servers', 'Descripcion': 'Servidores seguros',
     'Fuente': 'World Bank', 'Codigo': 'IT.NET.SECR.P6', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Digital', '#': 34, 'Variable': 'bank_branches', 'Descripcion': 'Sucursales bancarias',
     'Fuente': 'World Bank', 'Codigo': 'FB.CBK.BRCH.P5', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Digital', '#': 35, 'Variable': 'atm_density', 'Descripcion': 'Cajeros automaticos',
     'Fuente': 'World Bank', 'Codigo': 'FB.ATM.TOTL.P5', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    {'Dim': 'Digital', '#': 36, 'Variable': 'ict_exports_pct', 'Descripcion': 'Exportaciones TIC',
     'Fuente': 'World Bank', 'Codigo': 'TX.VAL.ICTG.ZS.UN', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

    # ===== PROXIMIDAD - 9 variables =====
    {'Dim': 'Proxim.', '#': 37, 'Variable': 'geographic_distance_km', 'Descripcion': 'Distancia a Bogota',
     'Fuente': 'CEPII', 'Codigo': 'CEPII_DIST_BOG', 'Direccion': 'negative', 'Frecuencia': 'Estatica'},

    {'Dim': 'Proxim.', '#': 38, 'Variable': 'common_language_spanish', 'Descripcion': 'Idioma comun',
     'Fuente': 'CEPII', 'Codigo': 'CEPII_LANG_ES', 'Direccion': 'positive', 'Frecuencia': 'Estatica'},

    {'Dim': 'Proxim.', '#': 39, 'Variable': 'hofstede_pdi', 'Descripcion': 'Power distance',
     'Fuente': 'Hofstede', 'Codigo': 'pdi', 'Direccion': 'neutral', 'Frecuencia': 'Estatica'},

    {'Dim': 'Proxim.', '#': 40, 'Variable': 'hofstede_idv', 'Descripcion': 'Individualism',
     'Fuente': 'Hofstede', 'Codigo': 'idv', 'Direccion': 'neutral', 'Frecuencia': 'Estatica'},

    {'Dim': 'Proxim.', '#': 41, 'Variable': 'hofstede_mas', 'Descripcion': 'Masculinity',
     'Fuente': 'Hofstede', 'Codigo': 'mas', 'Direccion': 'neutral', 'Frecuencia': 'Estatica'},

    {'Dim': 'Proxim.', '#': 42, 'Variable': 'hofstede_uai', 'Descripcion': 'Uncertainty avoidance',
     'Fuente': 'Hofstede', 'Codigo': 'uai', 'Direccion': 'neutral', 'Frecuencia': 'Estatica'},

    {'Dim': 'Proxim.', '#': 43, 'Variable': 'hofstede_lto', 'Descripcion': 'Long term orientation',
     'Fuente': 'Hofstede', 'Codigo': 'lto', 'Direccion': 'neutral', 'Frecuencia': 'Estatica'},

    {'Dim': 'Proxim.', '#': 44, 'Variable': 'hofstede_ivr', 'Descripcion': 'Indulgence',
     'Fuente': 'Hofstede', 'Codigo': 'ivr', 'Direccion': 'neutral', 'Frecuencia': 'Estatica'},

    {'Dim': 'Proxim.', '#': 45, 'Variable': 'colombian_diaspora_stock', 'Descripcion': 'Flujo migratorio colombiano',
     'Fuente': 'Migracion', 'Codigo': 'SALIDAS_COL', 'Direccion': 'positive', 'Frecuencia': 'Anual'},

])

# Funcion para colorear la columna 'Direccion' con codigos visuales
def colorear_direccion(val):
    colors = {
        'positive': f'background-color: {CIBEST["green"]}; color: white; font-weight: 600',
        'negative': f'background-color: {CIBEST["red"]}; color: white; font-weight: 600',
        'neutral':  f'background-color: {CIBEST["gold"]}; color: {CIBEST["gray"]}; font-weight: 600',
    }
    return colors.get(val, '')

# Tabla estilizada
styled = diccionario.style.set_table_styles([
    {'selector': 'th',
     'props': [('background-color', CIBEST['gray']),
               ('color', CIBEST['yellow']),
               ('font-weight', 'bold'),
               ('text-align', 'center'),
               ('padding', '8px'),
               ('font-family', 'Arial, sans-serif'),
               ('font-size', '12px')]},
    {'selector': 'td',
     'props': [('padding', '5px 8px'),
               ('font-family', 'Arial, sans-serif'),
               ('font-size', '11px'),
               ('border-bottom', f"1px solid {CIBEST['gray_border']}")]},
    {'selector': 'tbody tr:hover',
     'props': [('background-color', CIBEST['gray_bg'])]},
]).map(colorear_direccion, subset=['Direccion']).hide(axis='index')

display(styled)

```


<style type="text/css">
#T_467bf th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
  font-size: 12px;
}
#T_467bf td {
  padding: 5px 8px;
  font-family: Arial, sans-serif;
  font-size: 11px;
  border-bottom: 1px solid #D0D0D0;
}
#T_467bf tbody tr:hover {
  background-color: #F5F5F5;
}
#T_467bf_row0_col6, #T_467bf_row1_col6, #T_467bf_row2_col6, #T_467bf_row4_col6, #T_467bf_row5_col6, #T_467bf_row7_col6, #T_467bf_row9_col6, #T_467bf_row10_col6, #T_467bf_row12_col6, #T_467bf_row13_col6, #T_467bf_row16_col6, #T_467bf_row17_col6, #T_467bf_row19_col6, #T_467bf_row20_col6, #T_467bf_row21_col6, #T_467bf_row22_col6, #T_467bf_row23_col6, #T_467bf_row24_col6, #T_467bf_row25_col6, #T_467bf_row26_col6, #T_467bf_row28_col6, #T_467bf_row29_col6, #T_467bf_row30_col6, #T_467bf_row31_col6, #T_467bf_row32_col6, #T_467bf_row33_col6, #T_467bf_row34_col6, #T_467bf_row35_col6, #T_467bf_row37_col6, #T_467bf_row44_col6 {
  background-color: #0BA682;
  color: white;
  font-weight: 600;
}
#T_467bf_row3_col6, #T_467bf_row6_col6, #T_467bf_row8_col6, #T_467bf_row11_col6, #T_467bf_row14_col6, #T_467bf_row15_col6, #T_467bf_row18_col6, #T_467bf_row27_col6, #T_467bf_row36_col6 {
  background-color: #C62828;
  color: white;
  font-weight: 600;
}
#T_467bf_row38_col6, #T_467bf_row39_col6, #T_467bf_row40_col6, #T_467bf_row41_col6, #T_467bf_row42_col6, #T_467bf_row43_col6 {
  background-color: #D6B302;
  color: #2C2A28;
  font-weight: 600;
}
</style>
<table id="T_467bf">
  <thead>
    <tr>
      <th id="T_467bf_level0_col0" class="col_heading level0 col0" >Dim</th>
      <th id="T_467bf_level0_col1" class="col_heading level0 col1" >#</th>
      <th id="T_467bf_level0_col2" class="col_heading level0 col2" >Variable</th>
      <th id="T_467bf_level0_col3" class="col_heading level0 col3" >Descripcion</th>
      <th id="T_467bf_level0_col4" class="col_heading level0 col4" >Fuente</th>
      <th id="T_467bf_level0_col5" class="col_heading level0 col5" >Codigo</th>
      <th id="T_467bf_level0_col6" class="col_heading level0 col6" >Direccion</th>
      <th id="T_467bf_level0_col7" class="col_heading level0 col7" >Frecuencia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_467bf_row0_col0" class="data row0 col0" >Macro</td>
      <td id="T_467bf_row0_col1" class="data row0 col1" >1</td>
      <td id="T_467bf_row0_col2" class="data row0 col2" >gdp_nominal</td>
      <td id="T_467bf_row0_col3" class="data row0 col3" >PIB nominal (USD corrientes)</td>
      <td id="T_467bf_row0_col4" class="data row0 col4" >World Bank</td>
      <td id="T_467bf_row0_col5" class="data row0 col5" >NY.GDP.MKTP.CD</td>
      <td id="T_467bf_row0_col6" class="data row0 col6" >positive</td>
      <td id="T_467bf_row0_col7" class="data row0 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row1_col0" class="data row1 col0" >Macro</td>
      <td id="T_467bf_row1_col1" class="data row1 col1" >2</td>
      <td id="T_467bf_row1_col2" class="data row1 col2" >gdp_per_capita_ppp</td>
      <td id="T_467bf_row1_col3" class="data row1 col3" >PIB per capita PPP (USD intl.)</td>
      <td id="T_467bf_row1_col4" class="data row1 col4" >World Bank</td>
      <td id="T_467bf_row1_col5" class="data row1 col5" >NY.GDP.PCAP.PP.CD</td>
      <td id="T_467bf_row1_col6" class="data row1 col6" >positive</td>
      <td id="T_467bf_row1_col7" class="data row1 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row2_col0" class="data row2 col0" >Macro</td>
      <td id="T_467bf_row2_col1" class="data row2 col1" >3</td>
      <td id="T_467bf_row2_col2" class="data row2 col2" >gdp_growth</td>
      <td id="T_467bf_row2_col3" class="data row2 col3" >Crecimiento real anual del PIB</td>
      <td id="T_467bf_row2_col4" class="data row2 col4" >World Bank</td>
      <td id="T_467bf_row2_col5" class="data row2 col5" >NY.GDP.MKTP.KD.ZG</td>
      <td id="T_467bf_row2_col6" class="data row2 col6" >positive</td>
      <td id="T_467bf_row2_col7" class="data row2 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row3_col0" class="data row3 col0" >Macro</td>
      <td id="T_467bf_row3_col1" class="data row3 col1" >4</td>
      <td id="T_467bf_row3_col2" class="data row3 col2" >inflation_rate</td>
      <td id="T_467bf_row3_col3" class="data row3 col3" >Inflacion anual IPC</td>
      <td id="T_467bf_row3_col4" class="data row3 col4" >World Bank</td>
      <td id="T_467bf_row3_col5" class="data row3 col5" >FP.CPI.TOTL.ZG</td>
      <td id="T_467bf_row3_col6" class="data row3 col6" >negative</td>
      <td id="T_467bf_row3_col7" class="data row3 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row4_col0" class="data row4 col0" >Macro</td>
      <td id="T_467bf_row4_col1" class="data row4 col1" >5</td>
      <td id="T_467bf_row4_col2" class="data row4 col2" >population_total</td>
      <td id="T_467bf_row4_col3" class="data row4 col3" >Poblacion total</td>
      <td id="T_467bf_row4_col4" class="data row4 col4" >World Bank</td>
      <td id="T_467bf_row4_col5" class="data row4 col5" >SP.POP.TOTL</td>
      <td id="T_467bf_row4_col6" class="data row4 col6" >positive</td>
      <td id="T_467bf_row4_col7" class="data row4 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row5_col0" class="data row5 col0" >Macro</td>
      <td id="T_467bf_row5_col1" class="data row5 col1" >6</td>
      <td id="T_467bf_row5_col2" class="data row5 col2" >urban_population_pct</td>
      <td id="T_467bf_row5_col3" class="data row5 col3" >Poblacion urbana (% del total)</td>
      <td id="T_467bf_row5_col4" class="data row5 col4" >World Bank</td>
      <td id="T_467bf_row5_col5" class="data row5 col5" >SP.URB.TOTL.IN.ZS</td>
      <td id="T_467bf_row5_col6" class="data row5 col6" >positive</td>
      <td id="T_467bf_row5_col7" class="data row5 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row6_col0" class="data row6 col0" >Macro</td>
      <td id="T_467bf_row6_col1" class="data row6 col1" >7</td>
      <td id="T_467bf_row6_col2" class="data row6 col2" >unemployment_rate</td>
      <td id="T_467bf_row6_col3" class="data row6 col3" >Desempleo total (% fuerza laboral)</td>
      <td id="T_467bf_row6_col4" class="data row6 col4" >World Bank</td>
      <td id="T_467bf_row6_col5" class="data row6 col5" >SL.UEM.TOTL.ZS</td>
      <td id="T_467bf_row6_col6" class="data row6 col6" >negative</td>
      <td id="T_467bf_row6_col7" class="data row6 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row7_col0" class="data row7 col0" >Macro</td>
      <td id="T_467bf_row7_col1" class="data row7 col1" >8</td>
      <td id="T_467bf_row7_col2" class="data row7 col2" >current_account_gdp</td>
      <td id="T_467bf_row7_col3" class="data row7 col3" >Cuenta corriente / PIB (%)</td>
      <td id="T_467bf_row7_col4" class="data row7 col4" >World Bank</td>
      <td id="T_467bf_row7_col5" class="data row7 col5" >BN.CAB.XOKA.GD.ZS</td>
      <td id="T_467bf_row7_col6" class="data row7 col6" >positive</td>
      <td id="T_467bf_row7_col7" class="data row7 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row8_col0" class="data row8 col0" >Macro</td>
      <td id="T_467bf_row8_col1" class="data row8 col1" >9</td>
      <td id="T_467bf_row8_col2" class="data row8 col2" >public_debt_gdp</td>
      <td id="T_467bf_row8_col3" class="data row8 col3" >Deuda publica / PIB (%)</td>
      <td id="T_467bf_row8_col4" class="data row8 col4" >World Bank</td>
      <td id="T_467bf_row8_col5" class="data row8 col5" >GC.DOD.TOTL.GD.ZS</td>
      <td id="T_467bf_row8_col6" class="data row8 col6" >negative</td>
      <td id="T_467bf_row8_col7" class="data row8 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row9_col0" class="data row9 col0" >Macro</td>
      <td id="T_467bf_row9_col1" class="data row9 col1" >10</td>
      <td id="T_467bf_row9_col2" class="data row9 col2" >trade_openness</td>
      <td id="T_467bf_row9_col3" class="data row9 col3" >Comercio total (% PIB)</td>
      <td id="T_467bf_row9_col4" class="data row9 col4" >World Bank</td>
      <td id="T_467bf_row9_col5" class="data row9 col5" >NE.TRD.GNFS.ZS</td>
      <td id="T_467bf_row9_col6" class="data row9 col6" >positive</td>
      <td id="T_467bf_row9_col7" class="data row9 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row10_col0" class="data row10 col0" >Macro</td>
      <td id="T_467bf_row10_col1" class="data row10 col1" >11</td>
      <td id="T_467bf_row10_col2" class="data row10 col2" >fdi_net_inflows_gdp</td>
      <td id="T_467bf_row10_col3" class="data row10 col3" >IED neta / PIB (%)</td>
      <td id="T_467bf_row10_col4" class="data row10 col4" >World Bank</td>
      <td id="T_467bf_row10_col5" class="data row10 col5" >BX.KLT.DINV.WD.GD.ZS</td>
      <td id="T_467bf_row10_col6" class="data row10 col6" >positive</td>
      <td id="T_467bf_row10_col7" class="data row10 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row11_col0" class="data row11 col0" >Macro</td>
      <td id="T_467bf_row11_col1" class="data row11 col1" >12</td>
      <td id="T_467bf_row11_col2" class="data row11 col2" >weighted_mean_applied_tariff_all_products</td>
      <td id="T_467bf_row11_col3" class="data row11 col3" >Tarifa aplicada promedio ponderada (%)</td>
      <td id="T_467bf_row11_col4" class="data row11 col4" >World Bank</td>
      <td id="T_467bf_row11_col5" class="data row11 col5" >TM.TAX.MRCH.WM.AR.ZS</td>
      <td id="T_467bf_row11_col6" class="data row11 col6" >negative</td>
      <td id="T_467bf_row11_col7" class="data row11 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row12_col0" class="data row12 col0" >Financ.</td>
      <td id="T_467bf_row12_col1" class="data row12 col1" >13</td>
      <td id="T_467bf_row12_col2" class="data row12 col2" >domestic_credit_private_gdp</td>
      <td id="T_467bf_row12_col3" class="data row12 col3" >Credito al sector privado (% PIB)</td>
      <td id="T_467bf_row12_col4" class="data row12 col4" >World Bank</td>
      <td id="T_467bf_row12_col5" class="data row12 col5" >FD.AST.PRVT.GD.ZS</td>
      <td id="T_467bf_row12_col6" class="data row12 col6" >positive</td>
      <td id="T_467bf_row12_col7" class="data row12 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row13_col0" class="data row13 col0" >Financ.</td>
      <td id="T_467bf_row13_col1" class="data row13 col1" >14</td>
      <td id="T_467bf_row13_col2" class="data row13 col2" >account_ownership</td>
      <td id="T_467bf_row13_col3" class="data row13 col3" >Adultos con cuenta</td>
      <td id="T_467bf_row13_col4" class="data row13 col4" >World Bank</td>
      <td id="T_467bf_row13_col5" class="data row13 col5" >FX.OWN.TOTL.ZS</td>
      <td id="T_467bf_row13_col6" class="data row13 col6" >positive</td>
      <td id="T_467bf_row13_col7" class="data row13 col7" >Trienal</td>
    </tr>
    <tr>
      <td id="T_467bf_row14_col0" class="data row14 col0" >Financ.</td>
      <td id="T_467bf_row14_col1" class="data row14 col1" >15</td>
      <td id="T_467bf_row14_col2" class="data row14 col2" >interest_rate_spread</td>
      <td id="T_467bf_row14_col3" class="data row14 col3" >Spread tasa activa - pasiva</td>
      <td id="T_467bf_row14_col4" class="data row14 col4" >World Bank</td>
      <td id="T_467bf_row14_col5" class="data row14 col5" >FR.INR.LNDP</td>
      <td id="T_467bf_row14_col6" class="data row14 col6" >negative</td>
      <td id="T_467bf_row14_col7" class="data row14 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row15_col0" class="data row15 col0" >Financ.</td>
      <td id="T_467bf_row15_col1" class="data row15 col1" >16</td>
      <td id="T_467bf_row15_col2" class="data row15 col2" >bank_npl_ratio</td>
      <td id="T_467bf_row15_col3" class="data row15 col3" >Cartera vencida (%)</td>
      <td id="T_467bf_row15_col4" class="data row15 col4" >World Bank</td>
      <td id="T_467bf_row15_col5" class="data row15 col5" >FB.AST.NPER.ZS</td>
      <td id="T_467bf_row15_col6" class="data row15 col6" >negative</td>
      <td id="T_467bf_row15_col7" class="data row15 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row16_col0" class="data row16 col0" >Financ.</td>
      <td id="T_467bf_row16_col1" class="data row16 col1" >17</td>
      <td id="T_467bf_row16_col2" class="data row16 col2" >stock_market_cap_gdp</td>
      <td id="T_467bf_row16_col3" class="data row16 col3" >Capitalizacion bursatil (% PIB)</td>
      <td id="T_467bf_row16_col4" class="data row16 col4" >World Bank</td>
      <td id="T_467bf_row16_col5" class="data row16 col5" >CM.MKT.LCAP.GD.ZS</td>
      <td id="T_467bf_row16_col6" class="data row16 col6" >positive</td>
      <td id="T_467bf_row16_col7" class="data row16 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row17_col0" class="data row17 col0" >Financ.</td>
      <td id="T_467bf_row17_col1" class="data row17 col1" >18</td>
      <td id="T_467bf_row17_col2" class="data row17 col2" >personal_remittances_gdp</td>
      <td id="T_467bf_row17_col3" class="data row17 col3" >Remesas (% PIB)</td>
      <td id="T_467bf_row17_col4" class="data row17 col4" >World Bank</td>
      <td id="T_467bf_row17_col5" class="data row17 col5" >BX.TRF.PWKR.DT.GD.ZS</td>
      <td id="T_467bf_row17_col6" class="data row17 col6" >positive</td>
      <td id="T_467bf_row17_col7" class="data row17 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row18_col0" class="data row18 col0" >Financ.</td>
      <td id="T_467bf_row18_col1" class="data row18 col1" >19</td>
      <td id="T_467bf_row18_col2" class="data row18 col2" >bank_concentration_5</td>
      <td id="T_467bf_row18_col3" class="data row18 col3" >Concentracion top-5 bancos</td>
      <td id="T_467bf_row18_col4" class="data row18 col4" >World Bank</td>
      <td id="T_467bf_row18_col5" class="data row18 col5" >GFDD.OI.06</td>
      <td id="T_467bf_row18_col6" class="data row18 col6" >negative</td>
      <td id="T_467bf_row18_col7" class="data row18 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row19_col0" class="data row19 col0" >Financ.</td>
      <td id="T_467bf_row19_col1" class="data row19 col1" >20</td>
      <td id="T_467bf_row19_col2" class="data row19 col2" >financial_system_deposits_gc</td>
      <td id="T_467bf_row19_col3" class="data row19 col3" >Depositos / PIB</td>
      <td id="T_467bf_row19_col4" class="data row19 col4" >World Bank</td>
      <td id="T_467bf_row19_col5" class="data row19 col5" >GFDD.DI.08</td>
      <td id="T_467bf_row19_col6" class="data row19 col6" >positive</td>
      <td id="T_467bf_row19_col7" class="data row19 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row20_col0" class="data row20 col0" >Financ.</td>
      <td id="T_467bf_row20_col1" class="data row20 col1" >21</td>
      <td id="T_467bf_row20_col2" class="data row20 col2" >bank_capital_rwa</td>
      <td id="T_467bf_row20_col3" class="data row20 col3" >Capital regulatorio / RWA</td>
      <td id="T_467bf_row20_col4" class="data row20 col4" >World Bank</td>
      <td id="T_467bf_row20_col5" class="data row20 col5" >GFDD.SI.05</td>
      <td id="T_467bf_row20_col6" class="data row20 col6" >positive</td>
      <td id="T_467bf_row20_col7" class="data row20 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row21_col0" class="data row21 col0" >Instit.</td>
      <td id="T_467bf_row21_col1" class="data row21 col1" >22</td>
      <td id="T_467bf_row21_col2" class="data row21 col2" >regulatory_quality</td>
      <td id="T_467bf_row21_col3" class="data row21 col3" >WGI: calidad regulatoria</td>
      <td id="T_467bf_row21_col4" class="data row21 col4" >World Bank</td>
      <td id="T_467bf_row21_col5" class="data row21 col5" >GOV_WGI_RQ.EST</td>
      <td id="T_467bf_row21_col6" class="data row21 col6" >positive</td>
      <td id="T_467bf_row21_col7" class="data row21 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row22_col0" class="data row22 col0" >Instit.</td>
      <td id="T_467bf_row22_col1" class="data row22 col1" >23</td>
      <td id="T_467bf_row22_col2" class="data row22 col2" >government_effectiveness</td>
      <td id="T_467bf_row22_col3" class="data row22 col3" >WGI: efectividad gubernamental</td>
      <td id="T_467bf_row22_col4" class="data row22 col4" >World Bank</td>
      <td id="T_467bf_row22_col5" class="data row22 col5" >GOV_WGI_GE.EST</td>
      <td id="T_467bf_row22_col6" class="data row22 col6" >positive</td>
      <td id="T_467bf_row22_col7" class="data row22 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row23_col0" class="data row23 col0" >Instit.</td>
      <td id="T_467bf_row23_col1" class="data row23 col1" >24</td>
      <td id="T_467bf_row23_col2" class="data row23 col2" >rule_of_law</td>
      <td id="T_467bf_row23_col3" class="data row23 col3" >WGI: estado de derecho</td>
      <td id="T_467bf_row23_col4" class="data row23 col4" >World Bank</td>
      <td id="T_467bf_row23_col5" class="data row23 col5" >GOV_WGI_RL.EST</td>
      <td id="T_467bf_row23_col6" class="data row23 col6" >positive</td>
      <td id="T_467bf_row23_col7" class="data row23 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row24_col0" class="data row24 col0" >Instit.</td>
      <td id="T_467bf_row24_col1" class="data row24 col1" >25</td>
      <td id="T_467bf_row24_col2" class="data row24 col2" >political_stability</td>
      <td id="T_467bf_row24_col3" class="data row24 col3" >WGI: estabilidad politica</td>
      <td id="T_467bf_row24_col4" class="data row24 col4" >World Bank</td>
      <td id="T_467bf_row24_col5" class="data row24 col5" >GOV_WGI_PV.EST</td>
      <td id="T_467bf_row24_col6" class="data row24 col6" >positive</td>
      <td id="T_467bf_row24_col7" class="data row24 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row25_col0" class="data row25 col0" >Instit.</td>
      <td id="T_467bf_row25_col1" class="data row25 col1" >26</td>
      <td id="T_467bf_row25_col2" class="data row25 col2" >voice_accountability</td>
      <td id="T_467bf_row25_col3" class="data row25 col3" >WGI: voz y rendicion de cuentas</td>
      <td id="T_467bf_row25_col4" class="data row25 col4" >World Bank</td>
      <td id="T_467bf_row25_col5" class="data row25 col5" >GOV_WGI_VA.EST</td>
      <td id="T_467bf_row25_col6" class="data row25 col6" >positive</td>
      <td id="T_467bf_row25_col7" class="data row25 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row26_col0" class="data row26 col0" >Instit.</td>
      <td id="T_467bf_row26_col1" class="data row26 col1" >27</td>
      <td id="T_467bf_row26_col2" class="data row26 col2" >control_of_corruption</td>
      <td id="T_467bf_row26_col3" class="data row26 col3" >WGI: control de corrupcion</td>
      <td id="T_467bf_row26_col4" class="data row26 col4" >World Bank</td>
      <td id="T_467bf_row26_col5" class="data row26 col5" >GOV_WGI_CC.EST</td>
      <td id="T_467bf_row26_col6" class="data row26 col6" >positive</td>
      <td id="T_467bf_row26_col7" class="data row26 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row27_col0" class="data row27 col0" >Instit.</td>
      <td id="T_467bf_row27_col1" class="data row27 col1" >28</td>
      <td id="T_467bf_row27_col2" class="data row27 col2" >country_risk_premium</td>
      <td id="T_467bf_row27_col3" class="data row27 col3" >Prima de riesgo pais</td>
      <td id="T_467bf_row27_col4" class="data row27 col4" >Damodaran</td>
      <td id="T_467bf_row27_col5" class="data row27 col5" >COUNTRY_RISK_PREMIUM</td>
      <td id="T_467bf_row27_col6" class="data row27 col6" >negative</td>
      <td id="T_467bf_row27_col7" class="data row27 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row28_col0" class="data row28 col0" >Instit.</td>
      <td id="T_467bf_row28_col1" class="data row28 col1" >29</td>
      <td id="T_467bf_row28_col2" class="data row28 col2" >heritage_efi</td>
      <td id="T_467bf_row28_col3" class="data row28 col3" >Economic Freedom Index</td>
      <td id="T_467bf_row28_col4" class="data row28 col4" >Heritage</td>
      <td id="T_467bf_row28_col5" class="data row28 col5" >HERITAGE_EFI</td>
      <td id="T_467bf_row28_col6" class="data row28 col6" >positive</td>
      <td id="T_467bf_row28_col7" class="data row28 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row29_col0" class="data row29 col0" >Digital</td>
      <td id="T_467bf_row29_col1" class="data row29 col1" >30</td>
      <td id="T_467bf_row29_col2" class="data row29 col2" >internet_users_pct</td>
      <td id="T_467bf_row29_col3" class="data row29 col3" >Usuarios de internet</td>
      <td id="T_467bf_row29_col4" class="data row29 col4" >World Bank</td>
      <td id="T_467bf_row29_col5" class="data row29 col5" >IT.NET.USER.ZS</td>
      <td id="T_467bf_row29_col6" class="data row29 col6" >positive</td>
      <td id="T_467bf_row29_col7" class="data row29 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row30_col0" class="data row30 col0" >Digital</td>
      <td id="T_467bf_row30_col1" class="data row30 col1" >31</td>
      <td id="T_467bf_row30_col2" class="data row30 col2" >mobile_subscriptions</td>
      <td id="T_467bf_row30_col3" class="data row30 col3" >Suscripciones moviles</td>
      <td id="T_467bf_row30_col4" class="data row30 col4" >World Bank</td>
      <td id="T_467bf_row30_col5" class="data row30 col5" >IT.CEL.SETS.P2</td>
      <td id="T_467bf_row30_col6" class="data row30 col6" >positive</td>
      <td id="T_467bf_row30_col7" class="data row30 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row31_col0" class="data row31 col0" >Digital</td>
      <td id="T_467bf_row31_col1" class="data row31 col1" >32</td>
      <td id="T_467bf_row31_col2" class="data row31 col2" >digital_payment_adults_pct</td>
      <td id="T_467bf_row31_col3" class="data row31 col3" >Pagos digitales</td>
      <td id="T_467bf_row31_col4" class="data row31 col4" >World Bank</td>
      <td id="T_467bf_row31_col5" class="data row31 col5" >g20.any</td>
      <td id="T_467bf_row31_col6" class="data row31 col6" >positive</td>
      <td id="T_467bf_row31_col7" class="data row31 col7" >Trienal</td>
    </tr>
    <tr>
      <td id="T_467bf_row32_col0" class="data row32 col0" >Digital</td>
      <td id="T_467bf_row32_col1" class="data row32 col1" >33</td>
      <td id="T_467bf_row32_col2" class="data row32 col2" >secure_internet_servers</td>
      <td id="T_467bf_row32_col3" class="data row32 col3" >Servidores seguros</td>
      <td id="T_467bf_row32_col4" class="data row32 col4" >World Bank</td>
      <td id="T_467bf_row32_col5" class="data row32 col5" >IT.NET.SECR.P6</td>
      <td id="T_467bf_row32_col6" class="data row32 col6" >positive</td>
      <td id="T_467bf_row32_col7" class="data row32 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row33_col0" class="data row33 col0" >Digital</td>
      <td id="T_467bf_row33_col1" class="data row33 col1" >34</td>
      <td id="T_467bf_row33_col2" class="data row33 col2" >bank_branches</td>
      <td id="T_467bf_row33_col3" class="data row33 col3" >Sucursales bancarias</td>
      <td id="T_467bf_row33_col4" class="data row33 col4" >World Bank</td>
      <td id="T_467bf_row33_col5" class="data row33 col5" >FB.CBK.BRCH.P5</td>
      <td id="T_467bf_row33_col6" class="data row33 col6" >positive</td>
      <td id="T_467bf_row33_col7" class="data row33 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row34_col0" class="data row34 col0" >Digital</td>
      <td id="T_467bf_row34_col1" class="data row34 col1" >35</td>
      <td id="T_467bf_row34_col2" class="data row34 col2" >atm_density</td>
      <td id="T_467bf_row34_col3" class="data row34 col3" >Cajeros automaticos</td>
      <td id="T_467bf_row34_col4" class="data row34 col4" >World Bank</td>
      <td id="T_467bf_row34_col5" class="data row34 col5" >FB.ATM.TOTL.P5</td>
      <td id="T_467bf_row34_col6" class="data row34 col6" >positive</td>
      <td id="T_467bf_row34_col7" class="data row34 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row35_col0" class="data row35 col0" >Digital</td>
      <td id="T_467bf_row35_col1" class="data row35 col1" >36</td>
      <td id="T_467bf_row35_col2" class="data row35 col2" >ict_exports_pct</td>
      <td id="T_467bf_row35_col3" class="data row35 col3" >Exportaciones TIC</td>
      <td id="T_467bf_row35_col4" class="data row35 col4" >World Bank</td>
      <td id="T_467bf_row35_col5" class="data row35 col5" >TX.VAL.ICTG.ZS.UN</td>
      <td id="T_467bf_row35_col6" class="data row35 col6" >positive</td>
      <td id="T_467bf_row35_col7" class="data row35 col7" >Anual</td>
    </tr>
    <tr>
      <td id="T_467bf_row36_col0" class="data row36 col0" >Proxim.</td>
      <td id="T_467bf_row36_col1" class="data row36 col1" >37</td>
      <td id="T_467bf_row36_col2" class="data row36 col2" >geographic_distance_km</td>
      <td id="T_467bf_row36_col3" class="data row36 col3" >Distancia a Bogota</td>
      <td id="T_467bf_row36_col4" class="data row36 col4" >CEPII</td>
      <td id="T_467bf_row36_col5" class="data row36 col5" >CEPII_DIST_BOG</td>
      <td id="T_467bf_row36_col6" class="data row36 col6" >negative</td>
      <td id="T_467bf_row36_col7" class="data row36 col7" >Estatica</td>
    </tr>
    <tr>
      <td id="T_467bf_row37_col0" class="data row37 col0" >Proxim.</td>
      <td id="T_467bf_row37_col1" class="data row37 col1" >38</td>
      <td id="T_467bf_row37_col2" class="data row37 col2" >common_language_spanish</td>
      <td id="T_467bf_row37_col3" class="data row37 col3" >Idioma comun</td>
      <td id="T_467bf_row37_col4" class="data row37 col4" >CEPII</td>
      <td id="T_467bf_row37_col5" class="data row37 col5" >CEPII_LANG_ES</td>
      <td id="T_467bf_row37_col6" class="data row37 col6" >positive</td>
      <td id="T_467bf_row37_col7" class="data row37 col7" >Estatica</td>
    </tr>
    <tr>
      <td id="T_467bf_row38_col0" class="data row38 col0" >Proxim.</td>
      <td id="T_467bf_row38_col1" class="data row38 col1" >39</td>
      <td id="T_467bf_row38_col2" class="data row38 col2" >hofstede_pdi</td>
      <td id="T_467bf_row38_col3" class="data row38 col3" >Power distance</td>
      <td id="T_467bf_row38_col4" class="data row38 col4" >Hofstede</td>
      <td id="T_467bf_row38_col5" class="data row38 col5" >pdi</td>
      <td id="T_467bf_row38_col6" class="data row38 col6" >neutral</td>
      <td id="T_467bf_row38_col7" class="data row38 col7" >Estatica</td>
    </tr>
    <tr>
      <td id="T_467bf_row39_col0" class="data row39 col0" >Proxim.</td>
      <td id="T_467bf_row39_col1" class="data row39 col1" >40</td>
      <td id="T_467bf_row39_col2" class="data row39 col2" >hofstede_idv</td>
      <td id="T_467bf_row39_col3" class="data row39 col3" >Individualism</td>
      <td id="T_467bf_row39_col4" class="data row39 col4" >Hofstede</td>
      <td id="T_467bf_row39_col5" class="data row39 col5" >idv</td>
      <td id="T_467bf_row39_col6" class="data row39 col6" >neutral</td>
      <td id="T_467bf_row39_col7" class="data row39 col7" >Estatica</td>
    </tr>
    <tr>
      <td id="T_467bf_row40_col0" class="data row40 col0" >Proxim.</td>
      <td id="T_467bf_row40_col1" class="data row40 col1" >41</td>
      <td id="T_467bf_row40_col2" class="data row40 col2" >hofstede_mas</td>
      <td id="T_467bf_row40_col3" class="data row40 col3" >Masculinity</td>
      <td id="T_467bf_row40_col4" class="data row40 col4" >Hofstede</td>
      <td id="T_467bf_row40_col5" class="data row40 col5" >mas</td>
      <td id="T_467bf_row40_col6" class="data row40 col6" >neutral</td>
      <td id="T_467bf_row40_col7" class="data row40 col7" >Estatica</td>
    </tr>
    <tr>
      <td id="T_467bf_row41_col0" class="data row41 col0" >Proxim.</td>
      <td id="T_467bf_row41_col1" class="data row41 col1" >42</td>
      <td id="T_467bf_row41_col2" class="data row41 col2" >hofstede_uai</td>
      <td id="T_467bf_row41_col3" class="data row41 col3" >Uncertainty avoidance</td>
      <td id="T_467bf_row41_col4" class="data row41 col4" >Hofstede</td>
      <td id="T_467bf_row41_col5" class="data row41 col5" >uai</td>
      <td id="T_467bf_row41_col6" class="data row41 col6" >neutral</td>
      <td id="T_467bf_row41_col7" class="data row41 col7" >Estatica</td>
    </tr>
    <tr>
      <td id="T_467bf_row42_col0" class="data row42 col0" >Proxim.</td>
      <td id="T_467bf_row42_col1" class="data row42 col1" >43</td>
      <td id="T_467bf_row42_col2" class="data row42 col2" >hofstede_lto</td>
      <td id="T_467bf_row42_col3" class="data row42 col3" >Long term orientation</td>
      <td id="T_467bf_row42_col4" class="data row42 col4" >Hofstede</td>
      <td id="T_467bf_row42_col5" class="data row42 col5" >lto</td>
      <td id="T_467bf_row42_col6" class="data row42 col6" >neutral</td>
      <td id="T_467bf_row42_col7" class="data row42 col7" >Estatica</td>
    </tr>
    <tr>
      <td id="T_467bf_row43_col0" class="data row43 col0" >Proxim.</td>
      <td id="T_467bf_row43_col1" class="data row43 col1" >44</td>
      <td id="T_467bf_row43_col2" class="data row43 col2" >hofstede_ivr</td>
      <td id="T_467bf_row43_col3" class="data row43 col3" >Indulgence</td>
      <td id="T_467bf_row43_col4" class="data row43 col4" >Hofstede</td>
      <td id="T_467bf_row43_col5" class="data row43 col5" >ivr</td>
      <td id="T_467bf_row43_col6" class="data row43 col6" >neutral</td>
      <td id="T_467bf_row43_col7" class="data row43 col7" >Estatica</td>
    </tr>
    <tr>
      <td id="T_467bf_row44_col0" class="data row44 col0" >Proxim.</td>
      <td id="T_467bf_row44_col1" class="data row44 col1" >45</td>
      <td id="T_467bf_row44_col2" class="data row44 col2" >colombian_diaspora_stock</td>
      <td id="T_467bf_row44_col3" class="data row44 col3" >Flujo migratorio colombiano</td>
      <td id="T_467bf_row44_col4" class="data row44 col4" >Migracion</td>
      <td id="T_467bf_row44_col5" class="data row44 col5" >SALIDAS_COL</td>
      <td id="T_467bf_row44_col6" class="data row44 col6" >positive</td>
      <td id="T_467bf_row44_col7" class="data row44 col7" >Anual</td>
    </tr>
  </tbody>
</table>



## Visualización del peso de cada dimensión

Una observación que salta a la vista en el diccionario es que **las dimensiones no tienen el mismo número de variables**. La dimensión Digital-Tecnológica tiene 7 variables e instritucional 8, mientras que las dimensiones Financiera y Proximidad tienen nueve cada una. La gráfica siguiente lo hace explícito:


```python
# Conteo de variables por dimension
conteo_dim = diccionario.groupby('Dim').size().reset_index(name='n_variables')
conteo_dim = conteo_dim.sort_values('n_variables', ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=conteo_dim['Dim'], x=conteo_dim['n_variables'],
    orientation='h',
    marker=dict(color=conteo_dim['n_variables'],
                colorscale=[[0, CIBEST['gray_light']], [1, CIBEST['yellow']]],
                #line=dict(color=CIBEST['gray'], width=1.5)
    ), 
    text=conteo_dim['n_variables'],
    textposition='outside',
    textfont=dict(size=14, color=CIBEST['gray'], family='Arial Black'),
    hovertemplate='<b>%{y}</b><br>Variables: %{x}<extra></extra>',
))

fig.update_layout(
    title='<b>Variables por dimensión (45 totales)</b><br>'
          '<sub>La asimetría refleja decisiones metodológicas, no un sesgo</sub>',
    xaxis=dict(title='Número de variables', range=[0, 15], gridcolor=CIBEST['gray_border']),
    yaxis_title='Dimensión',
    plot_bgcolor=CIBEST['white'], paper_bgcolor=CIBEST['white'],
    font=dict(family='Arial', color=CIBEST['gray']),
    height=380, width=850, showlegend=False, margin=dict(l=100, t=80),
)
fig.show()

```



¿Por qué Proximidad tiene más variables que Digital-Tecnológica? La respuesta es metodológica y conviene desglosarla. Seis de las nueve variables de Proximidad son las seis dimensiones culturales de Hofstede. La literatura de proximidad cultural (Kogut & Singh 1988, Ghemawat 2001) recomienda mantener las seis dimensiones por separado durante la extracción y construir el índice de distancia cultural agregada en una etapa posterior, en lugar de promediarlas a priori, porque cada dimensión tiene una varianza distinta entre países y la distancia debe ponderarse por esa varianza. En la práctica, esas seis variables se *agregan* en una única variable derivada llamada `kogut_singh_distance` antes de entrar a TOPSIS, por lo que en términos de poder explicativo Proximidad contribuye con cuatro variables efectivas (la distancia cultural agregada, la distancia geográfica, el idioma compartido y las salidas de colombianos), no con nueve.

Algo análogo pasa con Institucional, donde seis de las ocho variables son los seis indicadores WGI del Banco Mundial. Estos seis indicadores están altamente correlacionados entre sí (un país con mala calidad regulatoria suele también tener débil estado de derecho), por lo que el sistema los mantiene separados para diagnóstico pero su contribución efectiva al ranking es menor que lo que la cuenta sugiere. La dimensión Digital-Tecnológica con sus tres variables, por contraste, captura el fenómeno completo con muy poca redundancia: una variable para la *base instalada* (usuarios de internet), una para el *canal de distribución* (móviles), una para la *adopción específica del producto financiero digital* (pagos digitales) y tres más que denotan la infraestructura teconologica disponible (cajeros, sucursales y servidores seguros).

## Las variables culturales Hofstede y la distancia Kogut-Singh

La explicación más detallada del tratamiento de las seis variables Hofstede merece su propio espacio, porque es el único caso del sistema donde la dirección de las variables no es "positiva" ni "negativa" sino "neutral". La fórmula que el sistema aplica es la **distancia Kogut-Singh**, propuesta por Bruce Kogut y Harbir Singh en 1988 y considerada el estándar de la literatura en estudios de internacionalización. Para cada país $j$ y dimensión cultural $k$ (de las seis de Hofstede), se calcula:

$$
d_{k}(\text{COL}, j) = \frac{(I_{k,j} - I_{k,\text{COL}})^2}{\sigma_k^2}
$$

donde $I_{k,j}$ es el valor de Hofstede del país $j$ en la dimensión $k$, $I_{k,\text{COL}}$ es el valor de Colombia en esa misma dimensión, y $\sigma_k^2$ es la varianza de la dimensión $k$ a lo largo de todos los países. La normalización por varianza es lo que evita que una dimensión con escala más amplia (como Power Distance, que va de 11 a 104) domine sobre dimensiones con menos rango.

Luego se agregan las seis distancias parciales en un solo índice:

$$
\text{KS}(\text{COL}, j) = \frac{1}{6} \sum_{k=1}^{6} d_k(\text{COL}, j)
$$

Esa cantidad $\text{KS}$ es siempre no negativa, es cero cuando el país tiene exactamente el mismo perfil cultural que Colombia (caso teórico que aplica solo a Colombia consigo misma) y crece a medida que el perfil cultural se aleja. **Esa distancia agregada sí tiene dirección negativa** (menor distancia es mejor para Cibest), y es ella la que entra a TOPSIS como variable de proximidad cultural, no las seis variables originales.

## Las fuentes de datos: de dónde sale cada cifra


```python
# Conteo de variables por fuente y descripcion de cada una
fuentes = diccionario.groupby('Fuente').size().reset_index(name='n_variables')
fuentes_info = {
    'World Bank':  {'descripcion': 'World Development Indicators - API publica REST', 'metodo': 'wbgapi (DB=2)'},
    'WB GFDD':     {'descripcion': 'Global Financial Development Database',          'metodo': 'wbgapi (DB=32)'},
    'WGI':         {'descripcion': 'Worldwide Governance Indicators',                'metodo': 'wbgapi (DB=3)'},
    'Damodaran':   {'descripcion': 'NYU Stern Country Risk Premium',                  'metodo': 'requests + Excel'},
    'Heritage':    {'descripcion': 'Heritage Foundation Index of Economic Freedom',  'metodo': 'CSV / API complementaria'},
    'CEPII':       {'descripcion': 'CEPII GeoDist / Language',                        'metodo': 'CSV estatico'},
    'Hofstede':    {'descripcion': 'Hofstede Insights cultural dimensions',           'metodo': 'CSV estatico'},
    'Migracion':   {'descripcion': 'Migracion Colombia - registros de salidas',       'metodo': 'CSV / archivo'},
}
fuentes['Descripcion'] = fuentes['Fuente'].map(lambda f: fuentes_info.get(f, {}).get('descripcion', ''))
fuentes['Metodo de extraccion'] = fuentes['Fuente'].map(lambda f: fuentes_info.get(f, {}).get('metodo', ''))
fuentes = fuentes.sort_values('n_variables', ascending=False).reset_index(drop=True)
fuentes = fuentes[['Fuente', 'n_variables', 'Descripcion', 'Metodo de extraccion']]

print('🌐 Fuentes de datos del proyecto:')
style_table(fuentes, gradient_cols=['n_variables'], gradient_cmap=cmap_custom, format_dict={'n_variables': '{:.0f}'})

```

    🌐 Fuentes de datos del proyecto:
    




<style type="text/css">
#T_e3722 th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_e3722 td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_e3722 tbody tr:hover {
  background-color: #F5F5F5;
}
#T_e3722_row0_col1 {
  background-color: #fdd923;
  color: #000000;
}
#T_e3722_row1_col1 {
  background-color: #d3ccaf;
  color: #000000;
}
#T_e3722_row2_col1 {
  background-color: #cdcac2;
  color: #000000;
}
#T_e3722_row3_col1, #T_e3722_row4_col1, #T_e3722_row5_col1 {
  background-color: #cccac7;
  color: #000000;
}
</style>
<table id="T_e3722">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e3722_level0_col0" class="col_heading level0 col0" >Fuente</th>
      <th id="T_e3722_level0_col1" class="col_heading level0 col1" >n_variables</th>
      <th id="T_e3722_level0_col2" class="col_heading level0 col2" >Descripcion</th>
      <th id="T_e3722_level0_col3" class="col_heading level0 col3" >Metodo de extraccion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e3722_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e3722_row0_col0" class="data row0 col0" >World Bank</td>
      <td id="T_e3722_row0_col1" class="data row0 col1" >34</td>
      <td id="T_e3722_row0_col2" class="data row0 col2" >World Development Indicators - API publica REST</td>
      <td id="T_e3722_row0_col3" class="data row0 col3" >wbgapi (DB=2)</td>
    </tr>
    <tr>
      <th id="T_e3722_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e3722_row1_col0" class="data row1 col0" >Hofstede</td>
      <td id="T_e3722_row1_col1" class="data row1 col1" >6</td>
      <td id="T_e3722_row1_col2" class="data row1 col2" >Hofstede Insights cultural dimensions</td>
      <td id="T_e3722_row1_col3" class="data row1 col3" >CSV estatico</td>
    </tr>
    <tr>
      <th id="T_e3722_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e3722_row2_col0" class="data row2 col0" >CEPII</td>
      <td id="T_e3722_row2_col1" class="data row2 col1" >2</td>
      <td id="T_e3722_row2_col2" class="data row2 col2" >CEPII GeoDist / Language</td>
      <td id="T_e3722_row2_col3" class="data row2 col3" >CSV estatico</td>
    </tr>
    <tr>
      <th id="T_e3722_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_e3722_row3_col0" class="data row3 col0" >Damodaran</td>
      <td id="T_e3722_row3_col1" class="data row3 col1" >1</td>
      <td id="T_e3722_row3_col2" class="data row3 col2" >NYU Stern Country Risk Premium</td>
      <td id="T_e3722_row3_col3" class="data row3 col3" >requests + Excel</td>
    </tr>
    <tr>
      <th id="T_e3722_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_e3722_row4_col0" class="data row4 col0" >Heritage</td>
      <td id="T_e3722_row4_col1" class="data row4 col1" >1</td>
      <td id="T_e3722_row4_col2" class="data row4 col2" >Heritage Foundation Index of Economic Freedom</td>
      <td id="T_e3722_row4_col3" class="data row4 col3" >CSV / API complementaria</td>
    </tr>
    <tr>
      <th id="T_e3722_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_e3722_row5_col0" class="data row5 col0" >Migracion</td>
      <td id="T_e3722_row5_col1" class="data row5 col1" >1</td>
      <td id="T_e3722_row5_col2" class="data row5 col2" >Migracion Colombia - registros de salidas</td>
      <td id="T_e3722_row5_col3" class="data row5 col3" >CSV / archivo</td>
    </tr>
  </tbody>
</table>




## La codificación de relevancia por línea de negocio

El diccionario original del Excel incluye cinco columnas adicionales que no aparecen en la tabla principal de este notebook pero son cruciales para el funcionamiento del sistema: IB, PT, AD, BD y CIB. En cada celda de esas columnas aparece una letra **A** o **B** que codifica la *relevancia de la variable para esa línea de negocio*. La letra **A** indica relevancia **Alta** (la variable es crítica para entender el atractivo del mercado en esa línea, y por tanto su peso relativo dentro de su dimensión es elevado); la letra **B** indica relevancia **Básica** (la variable se incluye en el ranking pero con peso estándar).

Esta codificación es lo que permite que TOPSIS produzca rankings *diferenciados* por línea sin tener que mantener cinco diccionarios paralelos. El procedimiento operativo es el siguiente: durante la construcción del `weight_profile` para cada línea de negocio, las variables marcadas con A reciben un *multiplicador de relevancia* (típicamente alrededor de 1.5) sobre el peso base de la variable dentro de su dimensión, y las marcadas con B mantienen el peso base. Luego se renormalizan los pesos para que sumen uno dentro de la dimensión. El resultado es que en el ranking de Pagos y Flujos (PF), por ejemplo, las variables `personal_remittances_gdp`, `mobile_subscriptions` y `colombian_diaspora_stock` tienen efectivamente más peso que en el ranking de Intermediación Bancaria (IB), donde el peso se concentra en `account_ownership`, `regulatory_quality` y `stock_market_cap_gdp`. La codificación A/B se traduce así en señales sustantivamente distintas para mercados que en términos absolutos podrían parecer similares.

Para visualizar este patrón, vale la pena ver cuántas variables tienen relevancia alta por línea de negocio:


```python
# Relevancia A por linea de negocio (segun el Excel del diccionario)
# Conteo aproximado basado en el listado de variables marcadas como 'A'
relevancia_A = pd.DataFrame({
    'Linea de negocio': ['IB', 'PF', 'AD', 'BD', 'CIB'],
    'Etiqueta': ['Intermediacion Bancaria', 'Pagos y Flujos', 'Activos Digitales',
                 'Bancos Digitales', 'Corporate & Investment Banking'],
    'Variables A (alta)':  [27, 26, 19, 31, 27],
    'Variables B (basica)': [18, 19, 26, 14, 18],
})
relevancia_A['Total'] = relevancia_A['Variables A (alta)'] + relevancia_A['Variables B (basica)']

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Relevancia ALTA (A)', x=relevancia_A['Linea de negocio'],
    y=relevancia_A['Variables A (alta)'],
    marker_color=CIBEST['yellow'],
    text=relevancia_A['Variables A (alta)'], textposition='inside',
    textfont=dict(color=CIBEST['gray'], size=13, family='Arial Black'),
    hovertemplate='<b>%{x}</b><br>Variables A: %{y}<extra></extra>',
))
fig.add_trace(go.Bar(
    name='Relevancia BASICA (B)', x=relevancia_A['Linea de negocio'],
    y=relevancia_A['Variables B (basica)'],
    marker_color=CIBEST['gray_light'],
    text=relevancia_A['Variables B (basica)'], textposition='inside',
    textfont=dict(color='white', size=13),
    hovertemplate='<b>%{x}</b><br>Variables B: %{y}<extra></extra>',
))

fig.update_layout(
    title='<b>Distribución A/B de relevancia por línea de negocio</b><br>'
          '<sub>Cada línea usa las 45 variables, pero asigna más peso a las marcadas con A</sub>',
    barmode='stack',
    xaxis=dict(title='Línea de negocio'),
    yaxis=dict(title='Número de variables', range=[0, 40], gridcolor=CIBEST['gray_border']),
    plot_bgcolor=CIBEST['white'], paper_bgcolor=CIBEST['white'],
    font=dict(family='Arial', color=CIBEST['gray']),
    height=420, width=850,
    legend=dict(orientation='h', y=-0.18),
)
fig.show()

print('\n📋 Detalle por linea de negocio:')
style_table(relevancia_A, gradient_cols=['Variables A (alta)'], gradient_cmap=cmap_custom)

```



    
    📋 Detalle por linea de negocio:
    




<style type="text/css">
#T_666f5 th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_666f5 td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_666f5 tbody tr:hover {
  background-color: #F5F5F5;
}
#T_666f5_row0_col2, #T_666f5_row4_col2 {
  background-color: #edd45a;
  color: #000000;
}
#T_666f5_row1_col2 {
  background-color: #e9d367;
  color: #000000;
}
#T_666f5_row2_col2 {
  background-color: #cccac7;
  color: #000000;
}
#T_666f5_row3_col2 {
  background-color: #fdd923;
  color: #000000;
}
</style>
<table id="T_666f5">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_666f5_level0_col0" class="col_heading level0 col0" >Linea de negocio</th>
      <th id="T_666f5_level0_col1" class="col_heading level0 col1" >Etiqueta</th>
      <th id="T_666f5_level0_col2" class="col_heading level0 col2" >Variables A (alta)</th>
      <th id="T_666f5_level0_col3" class="col_heading level0 col3" >Variables B (basica)</th>
      <th id="T_666f5_level0_col4" class="col_heading level0 col4" >Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_666f5_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_666f5_row0_col0" class="data row0 col0" >IB</td>
      <td id="T_666f5_row0_col1" class="data row0 col1" >Intermediacion Bancaria</td>
      <td id="T_666f5_row0_col2" class="data row0 col2" >27</td>
      <td id="T_666f5_row0_col3" class="data row0 col3" >18</td>
      <td id="T_666f5_row0_col4" class="data row0 col4" >45</td>
    </tr>
    <tr>
      <th id="T_666f5_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_666f5_row1_col0" class="data row1 col0" >PF</td>
      <td id="T_666f5_row1_col1" class="data row1 col1" >Pagos y Flujos</td>
      <td id="T_666f5_row1_col2" class="data row1 col2" >26</td>
      <td id="T_666f5_row1_col3" class="data row1 col3" >19</td>
      <td id="T_666f5_row1_col4" class="data row1 col4" >45</td>
    </tr>
    <tr>
      <th id="T_666f5_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_666f5_row2_col0" class="data row2 col0" >AD</td>
      <td id="T_666f5_row2_col1" class="data row2 col1" >Activos Digitales</td>
      <td id="T_666f5_row2_col2" class="data row2 col2" >19</td>
      <td id="T_666f5_row2_col3" class="data row2 col3" >26</td>
      <td id="T_666f5_row2_col4" class="data row2 col4" >45</td>
    </tr>
    <tr>
      <th id="T_666f5_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_666f5_row3_col0" class="data row3 col0" >BD</td>
      <td id="T_666f5_row3_col1" class="data row3 col1" >Bancos Digitales</td>
      <td id="T_666f5_row3_col2" class="data row3 col2" >31</td>
      <td id="T_666f5_row3_col3" class="data row3 col3" >14</td>
      <td id="T_666f5_row3_col4" class="data row3 col4" >45</td>
    </tr>
    <tr>
      <th id="T_666f5_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_666f5_row4_col0" class="data row4 col0" >CIB</td>
      <td id="T_666f5_row4_col1" class="data row4 col1" >Corporate & Investment Banking</td>
      <td id="T_666f5_row4_col2" class="data row4 col2" >27</td>
      <td id="T_666f5_row4_col3" class="data row4 col3" >18</td>
      <td id="T_666f5_row4_col4" class="data row4 col4" >45</td>
    </tr>
  </tbody>
</table>




La tabla muestra una distribución relativamente homogénea en términos de volumen total, ya que **todas las líneas de negocio cuentan con 45 variables evaluadas**. Sin embargo, la composición entre variables de relevancia alta y básica sí evidencia diferencias importantes. La línea **BD (Bancos Digitales) es la que concentra el mayor número de variables A —31 de 45—**, lo que sugiere que este negocio requiere una lectura más fina y prioritaria de múltiples dimensiones del entorno país. Esto es consistente con su naturaleza digital, escalable y altamente dependiente de condiciones como infraestructura tecnológica, adopción financiera, estabilidad regulatoria, madurez del ecosistema digital y confianza institucional.

En un segundo nivel aparecen **IB (Intermediación Bancaria)** y **CIB (Corporate & Investment Banking)**, ambas con **27 variables de relevancia alta y 18 básicas**. Esta composición refleja que, aunque pertenecen a negocios financieros más tradicionales o institucionales, siguen dependiendo de un conjunto amplio de factores críticos para evaluar oportunidades de internacionalización, profundidad de mercado, riesgo país, solidez económica y condiciones regulatorias. Por su parte, **PF (Pagos y Flujos)** presenta una distribución muy cercana, con **26 variables A y 19 B**, lo que indica una necesidad importante de análisis, aunque ligeramente menos intensiva frente a IB y CIB.

La línea con menor concentración de variables de relevancia alta es **AD (Activos Digitales)**, con **19 variables A y 26 variables B**. Esto no implica menor importancia estratégica, sino una priorización más selectiva de los factores críticos para este negocio. En este caso, la evaluación parece enfocarse en un conjunto más acotado de variables altamente determinantes, posiblemente asociadas al entorno regulatorio, la infraestructura digital, la adopción tecnológica y la apertura del mercado a modelos financieros emergentes. En síntesis, la tabla sugiere que **Bancos Digitales demanda el análisis más amplio y sensible**, mientras que **Activos Digitales presenta una estructura más focalizada**, y las demás líneas mantienen un balance intermedio-alto en la cantidad de variables críticas para la toma de decisiones.


Una variable que aparece con relevancia A en **las cinco líneas simultáneamente** es `country_risk_premium` (Damodaran). Esa universalidad no es casual: la prima de riesgo país es una métrica que afecta directamente la rentabilidad esperada de cualquier negocio financiero internacional, sin importar su tipo, porque determina el costo de capital marginal de operar en ese mercado. Por eso es la única variable que el sistema trata como críticamente importante para todas las decisiones de internacionalización.

## Cinco países sintéticos para los ejemplos del notebook

A lo largo del resto del notebook vamos a usar un conjunto pequeño de cinco países sintéticos para que los ejemplos numéricos se puedan seguir con calculadora si se quiere. Los cinco son **Colombia, México, Chile, Panamá y España**, escogidos porque cubren regiones distintas (Sudamérica, Norteamérica, Centroamérica y Europa estratégica) y porque tienen perfiles muy diferentes entre sí, lo que hace que los rankings resultantes sean interesantes pedagógicamente.

Para mantener el ejemplo manejable, usamos solamente **siete variables** representativas de las cinco dimensiones, en lugar de las 35 del sistema completo. Los valores son aproximaciones realistas pero no datos oficiales: el objetivo es que el lector entienda el flujo de cálculo, no que extraiga conclusiones de negocio de este ejemplo reducido. Cuando se ejecuta el sistema real con las 35 variables y los 30 países del alcance, el resultado tiene magnitud distinta pero la lógica es exactamente la misma que veremos en los próximos capítulos.

## Cinco países sintéticos para los ejemplos del notebook

A lo largo del resto del notebook vamos a usar un conjunto pequeño de cinco países sintéticos para que los ejemplos numéricos se puedan seguir con calculadora si se quiere. Los cinco son **Colombia, México, Chile, Panamá y España**, escogidos porque cubren regiones distintas (Sudamérica, Norteamérica, Centroamérica y Europa estratégica) y porque tienen perfiles muy diferentes entre sí, lo que hace que los rankings resultantes sean interesantes pedagógicamente.

Para mantener el ejemplo manejable, usamos solamente **siete variables** que representan a las cinco dimensiones, en lugar de las ~35 del sistema completo. Los valores son aproximaciones realistas pero no datos oficiales: el objetivo es que el lector entienda el flujo de cálculo, no que extraiga conclusiones de negocio de este ejemplo reducido.


```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
from src.utils import load_all_configs, resolve_data_path

configs = load_all_configs()
raw_dir = resolve_data_path(configs['settings']['data']['raw_path'])
candidates = sorted(raw_dir.glob('master_raw_*.parquet'), reverse=True)
if not candidates:
    raise FileNotFoundError('Ejecute primero el pipeline de extraccion')
master = pd.read_parquet(candidates[0])

master_wide = (
    master
    .sort_values("year")
    .drop_duplicates(subset=["country_iso3", "variable"], keep="last")
    .pivot(index="country_iso3", columns="variable", values="value")
    .reset_index()
)

#print(master_wide.columns.tolist())
#print(f"Dimensiones en la matriz de decisión: {master_wide.shape[1] - 1} variables (sin contar el ISO3)")

numeric_cols = master_wide.select_dtypes(include="number").columns.tolist()
#print(f"Variables numéricas disponibles para modelado: {len(numeric_cols)}")


# Matriz de decision sintetica para los ejemplos del notebook
cols = [
    "country_iso3",
    "gdp_per_capita_ppp",
    "inflation_rate",
    "financial_system_deposits_gdp",
    "regulatory_quality",
    "internet_users_pct",
    "geographic_distance_km",
    "common_language_spanish",
]

ejemplo = (
    master_wide[cols]
    .dropna()
    .head(5)
    .set_index("country_iso3")
)

paises_demo = ["COL", "MEX", "CHL", "PAN", "ESP"]

ejemplo = (
    master_wide[
        master_wide["country_iso3"].isin(paises_demo)
    ][cols].dropna().set_index("country_iso3")
)


# Direccion de cada variable
direccion = {
    'gdp_per_capita_ppp':      'positive',
    'inflation_rate':          'negative',
    'financial_system_deposits_gdp': 'positive',
    'regulatory_quality':      'positive',
    'internet_users_pct':      'positive',
    'geographic_distance_km':  'negative',
    'common_language_spanish': 'positive',
}

# Dimension de cada variable
dimension_var = {
    'gdp_per_capita_ppp':      'Macro',
    'inflation_rate':          'Macro',
    'financial_system_deposits_gdp': 'Financiera',
    'regulatory_quality':      'Institucional',
    'internet_users_pct':      'Digital',
    'geographic_distance_km':  'Proximidad',
    'common_language_spanish': 'Proximidad',
}



print('Matriz de decisión sintética para el resto del notebook:')
# style_table(master_wide, gradient_cols=numeric_cols, gradient_cmap=cmap_custom, 
#             format_dict={col: '{:.2f}' for col in numeric_cols})


numeric_cols_ejemplo = ejemplo.select_dtypes(include="number").columns.tolist()

style_table(
    ejemplo,
    gradient_cols=numeric_cols_ejemplo,
    gradient_cmap=cmap_custom,
    format_dict={col: '{:.2f}' for col in numeric_cols_ejemplo}
).set_caption("Matriz de decisión sintética (ejemplo)")


```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[9], line 44
         28 #print(f"Variables numéricas disponibles para modelado: {len(numeric_cols)}")
         29 
         30 
         31 # Matriz de decision sintetica para los ejemplos del notebook
         32 cols = [
         33     "country_iso3",
         34     "gdp_per_capita_ppp",
       (...)
         40     "common_language_spanish",
         41 ]
         43 ejemplo = (
    ---> 44     master_wide[cols]
         45     .dropna()
         46     .head(5)
         47     .set_index("country_iso3")
         48 )
         50 paises_demo = ["COL", "MEX", "CHL", "PAN", "ESP"]
         52 ejemplo = (
         53     master_wide[
         54         master_wide["country_iso3"].isin(paises_demo)
         55     ][cols].dropna().set_index("country_iso3")
         56 )
    

    File ~\AppData\Roaming\Python\Python39\site-packages\pandas\core\frame.py:4108, in DataFrame.__getitem__(self, key)
       4106     if is_iterator(key):
       4107         key = list(key)
    -> 4108     indexer = self.columns._get_indexer_strict(key, "columns")[1]
       4110 # take() does not accept boolean indexers
       4111 if getattr(indexer, "dtype", None) == bool:
    

    File ~\AppData\Roaming\Python\Python39\site-packages\pandas\core\indexes\base.py:6200, in Index._get_indexer_strict(self, key, axis_name)
       6197 else:
       6198     keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
    -> 6200 self._raise_if_missing(keyarr, indexer, axis_name)
       6202 keyarr = self.take(indexer)
       6203 if isinstance(key, Index):
       6204     # GH 42790 - Preserve name from an Index
    

    File ~\AppData\Roaming\Python\Python39\site-packages\pandas\core\indexes\base.py:6252, in Index._raise_if_missing(self, key, indexer, axis_name)
       6249     raise KeyError(f"None of [{key}] are in the [{axis_name}]")
       6251 not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
    -> 6252 raise KeyError(f"{not_found} not in index")
    

    KeyError: "['gdp_per_capita_ppp', 'inflation_rate', 'financial_system_deposits_gdp', 'regulatory_quality', 'internet_users_pct'] not in index"


Mirando esta tabla de cerca ya se intuye el problema central que el sistema tiene que resolver: **ningún país domina a los demás en todas las variables**. España tiene el PIB per cápita más alto y la mejor digitalización, pero está muy lejos de Bogotá. Panamá está cerquita y comparte idioma, pero su mercado es pequeño. Colombia tiene la inflación más alta y un PIB per cápita bajo, pero la distancia es cero. Chile tiene la mejor calidad regulatoria pero está lejos. Esa pluralidad de fortalezas y debilidades es exactamente lo que hace que las técnicas multicriterio sean necesarias: no hay una respuesta obvia.

---

# Capítulo 4 — BWM: ¿qué tanto importa cada dimensión?

## La intuición

Antes de poder rankear países necesitamos saber *cuánto pesa cada dimensión en la decisión*. Una persona del Comité podría pensar que la dimensión Institucional es el doble de importante que la Digital, y otra persona podría pensar lo contrario. El sistema necesita un mecanismo para que la alta dirección exprese esas preferencias de forma estructurada, y luego convertir esas preferencias en pesos numéricos que se puedan usar en el algoritmo de ranking.

La técnica que RADAR Cibest usa para esto se llama [**BWM: Best-Worst Method**](https://bestworstmethod.com/), propuesta por Jafar Rezaei en 2015. Su gran ventaja sobre el método clásico AHP de Saaty es que requiere **muchas menos comparaciones** del experto. AHP pide $n(n-1)/2$ comparaciones, lo que para cinco dimensiones son diez. BWM pide solo $2n-3$ comparaciones, lo que para cinco dimensiones son siete. Esa reducción no es trivial: significa que un ejecutivo puede completar la elicitación BWM en una sesión de 90 minutos en lugar de necesitar varias sesiones, y los estudios muestran que los pesos resultantes son **más consistentes** estadísticamente que los del AHP.

## Cómo funciona BWM, paso a paso

El procedimiento [BWM](https://bestworstmethod.com/wp-content/uploads/2020/01/Best-Worst-Method-BWM-2019.pdf) tiene cuatro pasos muy concretos que el ejecutivo recorre con un facilitador. **Primero**, el ejecutivo identifica cuál dimensión considera *la más importante* (la "Best") y cuál considera *la menos importante* (la "Worst"). Esa elección no tiene que ser racional ni justificada con datos; refleja la intuición estratégica del ejecutivo y eso es exactamente lo que queremos capturar.

**Segundo**, el ejecutivo compara la dimensión Best con todas las demás, usando una escala entera de 1 a 9 donde 1 significa "igualmente importantes" y 9 significa "la Best es extremadamente más importante que esta otra". Estas comparaciones forman el llamado *vector Best-to-Others*. **Tercero**, hace lo simétrico: compara cada una de las demás dimensiones contra la Worst, formando el *vector Others-to-Worst*. **Cuarto**, los dos vectores entran a un modelo de optimización que produce los pesos finales.

## La fórmula del modelo de optimización

El modelo BWM se formula así: dado que $a_{Bj}$ es la comparación entre la mejor dimensión y la dimensión $j$, y $a_{jW}$ es la comparación entre la dimensión $j$ y la peor, los pesos $w_j$ se obtienen resolviendo:

$$
\min \; \xi
$$

$$
\text{sujeto a:} \quad |w_B - a_{Bj} \cdot w_j| \leq \xi \quad \forall j
$$

$$
|w_j - a_{jW} \cdot w_W| \leq \xi \quad \forall j
$$

$$
\sum_{j} w_j = 1, \quad w_j \geq 0
$$

La variable $\xi$ (xi) mide el *grado de inconsistencia* en los juicios del experto: si el experto fuera perfectamente consistente, sería posible encontrar pesos que satisfacen todas las restricciones con $\xi = 0$. En la práctica $\xi$ termina siendo positivo, y entre más pequeño, más consistente fue el experto. El cociente $\xi^* / CI$ (donde $CI$ es el *Consistency Index* tabulado por Rezaei) se llama *Consistency Ratio* y debe estar por debajo de 0.10 para que los pesos se consideren confiables.

## Ejemplo numérico completo

Vamos a simular la elicitación BWM con un solo ejecutivo, que considera que la dimensión Institucional es la *más importante* y la dimensión Digital-Tecnológica es la *menos importante*. Vamos a producir los pesos y validar su consistencia.


```python
# Juicios BWM simulados de un ejecutivo de Cibest
juicios_ejecutivo = {
    'criteria': ['macro', 'financial', 'institutional', 'digital_tech', 'proximity'],
    'best': 'institutional',         # la dimension MAS importante
    'worst': 'digital_tech',         # la dimension MENOS importante
    'best_to_others': {              # cuanto mas importante es la 'best' vs cada otra (1-9)
        'macro':          3,
        'financial':      2,
        'institutional':  1,         # comparacion consigo misma = 1
        'digital_tech':   7,
        'proximity':      4,
    },
    'others_to_worst': {             # cuanto mas importante es cada otra vs la 'worst' (1-9)
        'macro':          5,
        'financial':      6,
        'institutional':  7,
        'digital_tech':   1,         # comparacion consigo misma = 1
        'proximity':      4,
    },
}

# Mostrar los juicios como tabla
df_juicios = pd.DataFrame({
    'Dimensión': juicios_ejecutivo['criteria'],
    'Best-to-Others (vs Institucional)': [juicios_ejecutivo['best_to_others'][c] for c in juicios_ejecutivo['criteria']],
    'Others-to-Worst (vs Digital)':      [juicios_ejecutivo['others_to_worst'][c] for c in juicios_ejecutivo['criteria']],
})
print('Juicios del ejecutivo:')
print(f"   Best = {juicios_ejecutivo['best']}    |    Worst = {juicios_ejecutivo['worst']}\n")
style_table(df_juicios)

```

    Juicios del ejecutivo:
       Best = institutional    |    Worst = digital_tech
    
    




<style type="text/css">
#T_65639 th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_65639 td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_65639 tbody tr:hover {
  background-color: #F5F5F5;
}
</style>
<table id="T_65639">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_65639_level0_col0" class="col_heading level0 col0" >Dimensión</th>
      <th id="T_65639_level0_col1" class="col_heading level0 col1" >Best-to-Others (vs Institucional)</th>
      <th id="T_65639_level0_col2" class="col_heading level0 col2" >Others-to-Worst (vs Digital)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_65639_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_65639_row0_col0" class="data row0 col0" >macro</td>
      <td id="T_65639_row0_col1" class="data row0 col1" >3</td>
      <td id="T_65639_row0_col2" class="data row0 col2" >5</td>
    </tr>
    <tr>
      <th id="T_65639_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_65639_row1_col0" class="data row1 col0" >financial</td>
      <td id="T_65639_row1_col1" class="data row1 col1" >2</td>
      <td id="T_65639_row1_col2" class="data row1 col2" >6</td>
    </tr>
    <tr>
      <th id="T_65639_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_65639_row2_col0" class="data row2 col0" >institutional</td>
      <td id="T_65639_row2_col1" class="data row2 col1" >1</td>
      <td id="T_65639_row2_col2" class="data row2 col2" >7</td>
    </tr>
    <tr>
      <th id="T_65639_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_65639_row3_col0" class="data row3 col0" >digital_tech</td>
      <td id="T_65639_row3_col1" class="data row3 col1" >7</td>
      <td id="T_65639_row3_col2" class="data row3 col2" >1</td>
    </tr>
    <tr>
      <th id="T_65639_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_65639_row4_col0" class="data row4 col0" >proximity</td>
      <td id="T_65639_row4_col1" class="data row4 col1" >4</td>
      <td id="T_65639_row4_col2" class="data row4 col2" >4</td>
    </tr>
  </tbody>
</table>





```python
# Resolucion del modelo BWM con SLSQP
from scipy.optimize import minimize

def resolver_bwm(juicios):
    criteria = juicios['criteria']
    best = juicios['best']
    worst = juicios['worst']
    bto = juicios['best_to_others']
    otw = juicios['others_to_worst']
    n = len(criteria)
    idx = {c: i for i, c in enumerate(criteria)}

    def objective(x): return x[-1]   # minimizar xi (ultimo elemento)

    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n]) - 1.0}]
    for c in criteria:
        if c != best:
            j, b = idx[c], idx[best]
            a = bto[c]
            cons.append({'type': 'ineq', 'fun': lambda x, b=b, j=j, a=a: x[-1] - abs(x[b] - a * x[j])})
    for c in criteria:
        if c != worst:
            j, w = idx[c], idx[worst]
            a = otw[c]
            cons.append({'type': 'ineq', 'fun': lambda x, j=j, w=w, a=a: x[-1] - abs(x[j] - a * x[w])})

    x0 = np.concatenate([np.full(n, 1.0/n), [0.1]])
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    res = minimize(objective, x0=x0, method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
    pesos = res.x[:n] / res.x[:n].sum()
    xi_star = res.x[-1]
    return {c: float(pesos[idx[c]]) for c in criteria}, float(xi_star)

pesos, xi_star = resolver_bwm(juicios_ejecutivo)

# Tabla CI segun Rezaei (2015)
CI_TABLE = {1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30, 6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23}
a_BW = juicios_ejecutivo['best_to_others'][juicios_ejecutivo['worst']]
ci = CI_TABLE[a_BW]
cr = xi_star / ci

print(f'\n Resolución del modelo BWM:')
print(f'   ξ* = {xi_star:.4f}    (inconsistencia)')
print(f'   CI = {ci:.2f}        (a_BW = {a_BW}, según tabla Rezaei 2015)')
print(f'   CR = ξ*/CI = {cr:.4f}    {"✅ CONSISTENTE" if cr < 0.10 else "⚠️  INCONSISTENTE"}\n')

df_pesos = pd.DataFrame({
    'Dimensión': list(pesos.keys()),
    'Peso BWM':  list(pesos.values()),
    'Peso (%)':  [f'{v*100:.1f}%' for v in pesos.values()],
})
style_table(df_pesos, gradient_cols=['Peso BWM'], gradient_cmap=cmap_custom,
            format_dict={'Peso BWM': '{:.4f}'})

```

    
     Resolución del modelo BWM:
       ξ* = 0.0775    (inconsistencia)
       CI = 3.73        (a_BW = 7, según tabla Rezaei 2015)
       CR = ξ*/CI = 0.0208    ✅ CONSISTENTE
    
    




<style type="text/css">
#T_eb5cb th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_eb5cb td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_eb5cb tbody tr:hover {
  background-color: #F5F5F5;
}
#T_eb5cb_row0_col1 {
  background-color: #dbcf94;
  color: #000000;
}
#T_eb5cb_row1_col1 {
  background-color: #e7d26e;
  color: #000000;
}
#T_eb5cb_row2_col1 {
  background-color: #fdd923;
  color: #000000;
}
#T_eb5cb_row3_col1 {
  background-color: #cccac7;
  color: #000000;
}
#T_eb5cb_row4_col1 {
  background-color: #d6cda6;
  color: #000000;
}
</style>
<table id="T_eb5cb">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_eb5cb_level0_col0" class="col_heading level0 col0" >Dimensión</th>
      <th id="T_eb5cb_level0_col1" class="col_heading level0 col1" >Peso BWM</th>
      <th id="T_eb5cb_level0_col2" class="col_heading level0 col2" >Peso (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_eb5cb_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_eb5cb_row0_col0" class="data row0 col0" >macro</td>
      <td id="T_eb5cb_row0_col1" class="data row0 col1" >0.1646</td>
      <td id="T_eb5cb_row0_col2" class="data row0 col2" >16.5%</td>
    </tr>
    <tr>
      <th id="T_eb5cb_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_eb5cb_row1_col0" class="data row1 col0" >financial</td>
      <td id="T_eb5cb_row1_col1" class="data row1 col1" >0.2470</td>
      <td id="T_eb5cb_row1_col2" class="data row1 col2" >24.7%</td>
    </tr>
    <tr>
      <th id="T_eb5cb_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_eb5cb_row2_col0" class="data row2 col0" >institutional</td>
      <td id="T_eb5cb_row2_col1" class="data row2 col1" >0.4165</td>
      <td id="T_eb5cb_row2_col2" class="data row2 col2" >41.6%</td>
    </tr>
    <tr>
      <th id="T_eb5cb_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_eb5cb_row3_col0" class="data row3 col0" >digital_tech</td>
      <td id="T_eb5cb_row3_col1" class="data row3 col1" >0.0484</td>
      <td id="T_eb5cb_row3_col2" class="data row3 col2" >4.8%</td>
    </tr>
    <tr>
      <th id="T_eb5cb_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_eb5cb_row4_col0" class="data row4 col0" >proximity</td>
      <td id="T_eb5cb_row4_col1" class="data row4 col1" >0.1235</td>
      <td id="T_eb5cb_row4_col2" class="data row4 col2" >12.3%</td>
    </tr>
  </tbody>
</table>





```python
# Visualizacion interactiva de los pesos BWM
fig = go.Figure()
dims = list(pesos.keys())
vals = [pesos[d] for d in dims]
colors_bar = [CIBEST['yellow'] if d == juicios_ejecutivo['best']
              else CIBEST['red'] if d == juicios_ejecutivo['worst']
              else CIBEST['gray'] for d in dims]

fig.add_trace(go.Bar(
    x=dims, y=vals,
    marker=dict(color=colors_bar, line=dict(color=CIBEST['gray'], width=1.5)),
    text=[f'{v*100:.1f}%' for v in vals],
    textposition='outside',
    textfont=dict(size=13, color=CIBEST['gray']),
    hovertemplate='<b>%{x}</b><br>Peso: %{y:.4f}<br>%{text}<extra></extra>',
))
fig.update_layout(
    title=f'<b>Pesos BWM resultantes</b><br><sub>Best (dorado) = {juicios_ejecutivo["best"]}, '
          f'Worst (rojo) = {juicios_ejecutivo["worst"]} · CR = {cr:.3f}</sub>',
    xaxis_title='Dimensión',
    yaxis_title='Peso',
    yaxis=dict(range=[0, max(vals) * 1.25], gridcolor=CIBEST['gray_border']),
    plot_bgcolor=CIBEST['white'],
    paper_bgcolor=CIBEST['white'],
    font=dict(family='Arial', color=CIBEST['gray']),
    height=440, showlegend=False,
)
fig.show()

```



Lo que esta gráfica muestra es la "huella estratégica" del ejecutivo: cuánto pesa cada dimensión según su criterio. La dimensión Best (Institucional, en dorado) se llevó el peso más alto, la dimensión Worst (Digital, en rojo) el más bajo, y las demás quedaron ordenadas en función de las comparaciones intermedias. La suma de todos los pesos es exactamente uno por construcción, lo que permite usarlos directamente como vector de pesos en el algoritmo TOPSIS que viene en el próximo capítulo.

## ¿Y si hay varios ejecutivos en el panel?

En la sesión real con la alta dirección de Cibest no participa un solo ejecutivo, sino un panel. Cada miembro del panel completa su propia elicitación y produce su propio vector de pesos. Para integrar los vectores individuales en un vector consolidado el sistema usa la **media geométrica**, que es el estándar en la literatura de decisión multicriterio porque preserva las razones entre pesos:

$$
w_j^{\text{panel}} = \sqrt[m]{\prod_{k=1}^{m} w_{j,k}}
$$

donde $m$ es el número de ejecutivos en el panel y $w_{j,k}$ es el peso de la dimensión $j$ según el ejecutivo $k$. Después se renormaliza para que la suma sea uno.

## De pesos por dimensión a pesos por variable

Hay un último paso conceptualmente sencillo pero operativamente importante. BWM nos da pesos para las cinco *dimensiones*, pero TOPSIS necesita pesos para las ~35 *variables*. La conversión se hace en dos niveles. Dentro de cada dimensión, se asigna un peso a cada variable (puede ser uniforme, o derivado de otra elicitación específica). Luego el peso final de una variable es el producto:

$$
w_v^{\text{final}} = w_{\text{dim}(v)} \cdot w_v^{\text{dentro de dim}}
$$

Esto garantiza que la suma de todos los pesos de variables sea uno, y que las variables de una dimensión "más importante" tengan más peso total que las de una dimensión "menos importante".

---

# Capítulo 5 — TOPSIS: ranking de los países

## La intuición geométrica

Una vez que tenemos los pesos de las variables (gracias a BWM), necesitamos un método que nos diga qué tan atractivo es cada país. La técnica que usa RADAR Cibest se llama [**TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution)](https://www.youtube.com/watch?v=kfcN7MuYVeI), propuesta por Hwang y Yoon en 1981, y es probablemente la técnica multicriteria más usada en literatura financiera y bancaria.

La intuición de [TOPSIS](https://en.wikipedia.org/wiki/TOPSIS) es geométrica y muy hermosa. Imagine usted que cada país es un punto en un espacio donde cada eje es una variable (PIB, inflación, calidad regulatoria, etc.). En ese espacio existen dos puntos especiales: la **solución ideal positiva**, que sería un país hipotético que tiene el mejor valor en todas las variables a la vez (el PIB más alto, la inflación más baja, la mejor calidad regulatoria, etc.); y la **solución ideal negativa**, que sería el opuesto: peor valor en todo. Ningún país real va a ser ni el ideal positivo ni el ideal negativo, pero podemos medir, para cada país, qué tan cerca está del ideal positivo y qué tan cerca está del ideal negativo.

Un país es atractivo si está *cerca* del ideal positivo *y a la vez lejos* del ideal negativo. TOPSIS combina esas dos distancias en una sola métrica llamada *closeness coefficient* o coeficiente de cercanía, que está siempre entre 0 y 1 y que define el ranking final.

## El algoritmo completo, paso a paso

El procedimiento TOPSIS tiene seis pasos. Vamos a recorrerlos uno por uno con la matriz de cinco países y siete variables que definimos en el capítulo tres.

### Paso 1: normalización de la matriz y aplicación de dirección

Antes de cualquier cosa hay que poner todas las variables en la misma escala. Las unidades del PIB (USD), de la inflación (%) y de la distancia (km) son incomparables. La normalización **min-max** convierte cada variable a una escala de 0 a 1:

$$
R_{ij} = \frac{x_{ij} - \min_i(x_{ij})}{\max_i(x_{ij}) - \min_i(x_{ij})}
$$

Luego se aplica la **dirección**: si una variable tiene dirección negativa (como la inflación), se invierte para que "alto siga significando bueno":

$$
R_{ij}^{\text{orientado}} = 1 - R_{ij} \quad \text{si } \text{dirección}(j) = \text{negativa}
$$


```python
# Paso 1: normalizacion min-max + aplicacion de direccion
def normalizar_y_orientar(matriz, direccion):
    normalizada = pd.DataFrame(index=matriz.index, columns=matriz.columns, dtype=float)
    for col in matriz.columns:
        vmin, vmax = matriz[col].min(), matriz[col].max()
        if vmax - vmin == 0:
            normalizada[col] = 0.5
        else:
            normalizada[col] = (matriz[col] - vmin) / (vmax - vmin)
        if direccion[col] == 'negative':
            normalizada[col] = 1.0 - normalizada[col]
    return normalizada

matriz_norm = normalizar_y_orientar(ejemplo, direccion)
print('Matriz normalizada y orientada (todas en [0,1], mayor = mejor):')
style_table(matriz_norm.round(3),
            gradient_cols=matriz_norm.columns.tolist(), gradient_cmap=cmap_custom,
            format_dict={c: '{:.3f}' for c in matriz_norm.columns})

```

    Matriz normalizada y orientada (todas en [0,1], mayor = mejor):
    




<style type="text/css">
#T_54295 th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_54295 td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_54295 tbody tr:hover {
  background-color: #F5F5F5;
}
#T_54295_row0_col0, #T_54295_row0_col1 {
  background-color: #dfd087;
  color: #000000;
}
#T_54295_row0_col2 {
  background-color: #ddcf8c;
  color: #000000;
}
#T_54295_row0_col3, #T_54295_row1_col5, #T_54295_row2_col0, #T_54295_row2_col2, #T_54295_row2_col4, #T_54295_row4_col1 {
  background-color: #fdd923;
  color: #000000;
}
#T_54295_row0_col4 {
  background-color: #fdd924;
  color: #000000;
}
#T_54295_row0_col5 {
  background-color: #e3d17a;
  color: #000000;
}
#T_54295_row0_col6, #T_54295_row1_col0, #T_54295_row1_col1, #T_54295_row1_col2, #T_54295_row1_col6, #T_54295_row2_col5, #T_54295_row2_col6, #T_54295_row3_col3, #T_54295_row3_col6, #T_54295_row4_col4, #T_54295_row4_col6 {
  background-color: #cccac7;
  color: #000000;
}
#T_54295_row1_col3 {
  background-color: #d6cda4;
  color: #000000;
}
#T_54295_row1_col4 {
  background-color: #dace98;
  color: #000000;
}
#T_54295_row2_col1 {
  background-color: #ecd45d;
  color: #000000;
}
#T_54295_row2_col3 {
  background-color: #f5d73d;
  color: #000000;
}
#T_54295_row3_col0 {
  background-color: #d1ccb6;
  color: #000000;
}
#T_54295_row3_col1 {
  background-color: #dccf93;
  color: #000000;
}
#T_54295_row3_col2 {
  background-color: #cecbc0;
  color: #000000;
}
#T_54295_row3_col4 {
  background-color: #e2d17d;
  color: #000000;
}
#T_54295_row3_col5 {
  background-color: #e8d26a;
  color: #000000;
}
#T_54295_row4_col0 {
  background-color: #e6d270;
  color: #000000;
}
#T_54295_row4_col2 {
  background-color: #dbce96;
  color: #000000;
}
#T_54295_row4_col3 {
  background-color: #d2ccb2;
  color: #000000;
}
#T_54295_row4_col5 {
  background-color: #f8d833;
  color: #000000;
}
</style>
<table id="T_54295">
  <thead>
    <tr>
      <th class="index_name level0" >variable</th>
      <th id="T_54295_level0_col0" class="col_heading level0 col0" >gdp_per_capita_ppp</th>
      <th id="T_54295_level0_col1" class="col_heading level0 col1" >inflation_rate</th>
      <th id="T_54295_level0_col2" class="col_heading level0 col2" >financial_system_deposits_gdp</th>
      <th id="T_54295_level0_col3" class="col_heading level0 col3" >regulatory_quality</th>
      <th id="T_54295_level0_col4" class="col_heading level0 col4" >internet_users_pct</th>
      <th id="T_54295_level0_col5" class="col_heading level0 col5" >geographic_distance_km</th>
      <th id="T_54295_level0_col6" class="col_heading level0 col6" >common_language_spanish</th>
    </tr>
    <tr>
      <th class="index_name level0" >country_iso3</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_54295_level0_row0" class="row_heading level0 row0" >CHL</th>
      <td id="T_54295_row0_col0" class="data row0 col0" >0.388</td>
      <td id="T_54295_row0_col1" class="data row0 col1" >0.391</td>
      <td id="T_54295_row0_col2" class="data row0 col2" >0.356</td>
      <td id="T_54295_row0_col3" class="data row0 col3" >1.000</td>
      <td id="T_54295_row0_col4" class="data row0 col4" >0.993</td>
      <td id="T_54295_row0_col5" class="data row0 col5" >0.471</td>
      <td id="T_54295_row0_col6" class="data row0 col6" >0.500</td>
    </tr>
    <tr>
      <th id="T_54295_level0_row1" class="row_heading level0 row1" >COL</th>
      <td id="T_54295_row1_col0" class="data row1 col0" >0.000</td>
      <td id="T_54295_row1_col1" class="data row1 col1" >0.000</td>
      <td id="T_54295_row1_col2" class="data row1 col2" >0.000</td>
      <td id="T_54295_row1_col3" class="data row1 col3" >0.214</td>
      <td id="T_54295_row1_col4" class="data row1 col4" >0.286</td>
      <td id="T_54295_row1_col5" class="data row1 col5" >1.000</td>
      <td id="T_54295_row1_col6" class="data row1 col6" >0.500</td>
    </tr>
    <tr>
      <th id="T_54295_level0_row2" class="row_heading level0 row2" >ESP</th>
      <td id="T_54295_row2_col0" class="data row2 col0" >1.000</td>
      <td id="T_54295_row2_col1" class="data row2 col1" >0.648</td>
      <td id="T_54295_row2_col2" class="data row2 col2" >1.000</td>
      <td id="T_54295_row2_col3" class="data row2 col3" >0.836</td>
      <td id="T_54295_row2_col4" class="data row2 col4" >1.000</td>
      <td id="T_54295_row2_col5" class="data row2 col5" >0.000</td>
      <td id="T_54295_row2_col6" class="data row2 col6" >0.500</td>
    </tr>
    <tr>
      <th id="T_54295_level0_row3" class="row_heading level0 row3" >MEX</th>
      <td id="T_54295_row3_col0" class="data row3 col0" >0.108</td>
      <td id="T_54295_row3_col1" class="data row3 col1" >0.319</td>
      <td id="T_54295_row3_col2" class="data row3 col2" >0.045</td>
      <td id="T_54295_row3_col3" class="data row3 col3" >0.000</td>
      <td id="T_54295_row3_col4" class="data row3 col4" >0.450</td>
      <td id="T_54295_row3_col5" class="data row3 col5" >0.565</td>
      <td id="T_54295_row3_col6" class="data row3 col6" >0.500</td>
    </tr>
    <tr>
      <th id="T_54295_level0_row4" class="row_heading level0 row4" >PAN</th>
      <td id="T_54295_row4_col0" class="data row4 col0" >0.534</td>
      <td id="T_54295_row4_col1" class="data row4 col1" >1.000</td>
      <td id="T_54295_row4_col2" class="data row4 col2" >0.297</td>
      <td id="T_54295_row4_col3" class="data row4 col3" >0.125</td>
      <td id="T_54295_row4_col4" class="data row4 col4" >0.000</td>
      <td id="T_54295_row4_col5" class="data row4 col5" >0.899</td>
      <td id="T_54295_row4_col6" class="data row4 col6" >0.500</td>
    </tr>
  </tbody>
</table>




Note que ahora todos los valores están entre 0 y 1, y que la columna `inflacion` cambió de sentido: Colombia que tenía 6.6% (la peor) ahora tiene valor 0 en la versión normalizada y orientada (porque su inflación es la peor para el atractivo), y Panama que tenía 0.69% (la mejor) ahora tiene 1.

### Paso 2: aplicación de pesos

Cada columna se multiplica por su peso. Para el ejemplo vamos a usar pesos uniformes (cada variable pesa $1/7$), lo cual es una simplificación frente al sistema real donde los pesos vienen de BWM. Esto produce la *matriz de decisión ponderada* $V$:

$$
V_{ij} = w_j \cdot R_{ij}^{\text{orientado}}
$$


```python
# Paso 2: ponderacion
pesos_uniformes = {c: 1/len(matriz_norm.columns) for c in matriz_norm.columns}
matriz_ponderada = matriz_norm.copy()
for col in matriz_ponderada.columns:
    matriz_ponderada[col] = matriz_ponderada[col] * pesos_uniformes[col]

print(f'Matriz ponderada (peso uniforme = {1/7:.4f} por variable):')
style_table(matriz_ponderada.round(4),
            gradient_cols=matriz_ponderada.columns.tolist(), gradient_cmap=cmap_custom,
            format_dict={c: '{:.4f}' for c in matriz_ponderada.columns})

```

    Matriz ponderada (peso uniforme = 0.1429 por variable):
    




<style type="text/css">
#T_f2c3d th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_f2c3d td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_f2c3d tbody tr:hover {
  background-color: #F5F5F5;
}
#T_f2c3d_row0_col0, #T_f2c3d_row0_col1 {
  background-color: #dfd087;
  color: #000000;
}
#T_f2c3d_row0_col2 {
  background-color: #ddcf8c;
  color: #000000;
}
#T_f2c3d_row0_col3, #T_f2c3d_row1_col5, #T_f2c3d_row2_col0, #T_f2c3d_row2_col2, #T_f2c3d_row2_col4, #T_f2c3d_row4_col1 {
  background-color: #fdd923;
  color: #000000;
}
#T_f2c3d_row0_col4 {
  background-color: #fdd924;
  color: #000000;
}
#T_f2c3d_row0_col5 {
  background-color: #e3d17a;
  color: #000000;
}
#T_f2c3d_row0_col6, #T_f2c3d_row1_col0, #T_f2c3d_row1_col1, #T_f2c3d_row1_col2, #T_f2c3d_row1_col6, #T_f2c3d_row2_col5, #T_f2c3d_row2_col6, #T_f2c3d_row3_col3, #T_f2c3d_row3_col6, #T_f2c3d_row4_col4, #T_f2c3d_row4_col6 {
  background-color: #cccac7;
  color: #000000;
}
#T_f2c3d_row1_col3 {
  background-color: #d6cda4;
  color: #000000;
}
#T_f2c3d_row1_col4 {
  background-color: #dace98;
  color: #000000;
}
#T_f2c3d_row2_col1 {
  background-color: #ecd45d;
  color: #000000;
}
#T_f2c3d_row2_col3 {
  background-color: #f5d73d;
  color: #000000;
}
#T_f2c3d_row3_col0 {
  background-color: #d1ccb6;
  color: #000000;
}
#T_f2c3d_row3_col1 {
  background-color: #dccf93;
  color: #000000;
}
#T_f2c3d_row3_col2 {
  background-color: #cecbc0;
  color: #000000;
}
#T_f2c3d_row3_col4 {
  background-color: #e2d17d;
  color: #000000;
}
#T_f2c3d_row3_col5 {
  background-color: #e8d26a;
  color: #000000;
}
#T_f2c3d_row4_col0 {
  background-color: #e6d270;
  color: #000000;
}
#T_f2c3d_row4_col2 {
  background-color: #dace97;
  color: #000000;
}
#T_f2c3d_row4_col3 {
  background-color: #d2ccb3;
  color: #000000;
}
#T_f2c3d_row4_col5 {
  background-color: #f8d833;
  color: #000000;
}
</style>
<table id="T_f2c3d">
  <thead>
    <tr>
      <th class="index_name level0" >variable</th>
      <th id="T_f2c3d_level0_col0" class="col_heading level0 col0" >gdp_per_capita_ppp</th>
      <th id="T_f2c3d_level0_col1" class="col_heading level0 col1" >inflation_rate</th>
      <th id="T_f2c3d_level0_col2" class="col_heading level0 col2" >financial_system_deposits_gdp</th>
      <th id="T_f2c3d_level0_col3" class="col_heading level0 col3" >regulatory_quality</th>
      <th id="T_f2c3d_level0_col4" class="col_heading level0 col4" >internet_users_pct</th>
      <th id="T_f2c3d_level0_col5" class="col_heading level0 col5" >geographic_distance_km</th>
      <th id="T_f2c3d_level0_col6" class="col_heading level0 col6" >common_language_spanish</th>
    </tr>
    <tr>
      <th class="index_name level0" >country_iso3</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f2c3d_level0_row0" class="row_heading level0 row0" >CHL</th>
      <td id="T_f2c3d_row0_col0" class="data row0 col0" >0.0555</td>
      <td id="T_f2c3d_row0_col1" class="data row0 col1" >0.0558</td>
      <td id="T_f2c3d_row0_col2" class="data row0 col2" >0.0509</td>
      <td id="T_f2c3d_row0_col3" class="data row0 col3" >0.1429</td>
      <td id="T_f2c3d_row0_col4" class="data row0 col4" >0.1418</td>
      <td id="T_f2c3d_row0_col5" class="data row0 col5" >0.0672</td>
      <td id="T_f2c3d_row0_col6" class="data row0 col6" >0.0714</td>
    </tr>
    <tr>
      <th id="T_f2c3d_level0_row1" class="row_heading level0 row1" >COL</th>
      <td id="T_f2c3d_row1_col0" class="data row1 col0" >0.0000</td>
      <td id="T_f2c3d_row1_col1" class="data row1 col1" >0.0000</td>
      <td id="T_f2c3d_row1_col2" class="data row1 col2" >0.0000</td>
      <td id="T_f2c3d_row1_col3" class="data row1 col3" >0.0306</td>
      <td id="T_f2c3d_row1_col4" class="data row1 col4" >0.0409</td>
      <td id="T_f2c3d_row1_col5" class="data row1 col5" >0.1429</td>
      <td id="T_f2c3d_row1_col6" class="data row1 col6" >0.0714</td>
    </tr>
    <tr>
      <th id="T_f2c3d_level0_row2" class="row_heading level0 row2" >ESP</th>
      <td id="T_f2c3d_row2_col0" class="data row2 col0" >0.1429</td>
      <td id="T_f2c3d_row2_col1" class="data row2 col1" >0.0926</td>
      <td id="T_f2c3d_row2_col2" class="data row2 col2" >0.1429</td>
      <td id="T_f2c3d_row2_col3" class="data row2 col3" >0.1195</td>
      <td id="T_f2c3d_row2_col4" class="data row2 col4" >0.1429</td>
      <td id="T_f2c3d_row2_col5" class="data row2 col5" >0.0000</td>
      <td id="T_f2c3d_row2_col6" class="data row2 col6" >0.0714</td>
    </tr>
    <tr>
      <th id="T_f2c3d_level0_row3" class="row_heading level0 row3" >MEX</th>
      <td id="T_f2c3d_row3_col0" class="data row3 col0" >0.0154</td>
      <td id="T_f2c3d_row3_col1" class="data row3 col1" >0.0456</td>
      <td id="T_f2c3d_row3_col2" class="data row3 col2" >0.0064</td>
      <td id="T_f2c3d_row3_col3" class="data row3 col3" >0.0000</td>
      <td id="T_f2c3d_row3_col4" class="data row3 col4" >0.0643</td>
      <td id="T_f2c3d_row3_col5" class="data row3 col5" >0.0808</td>
      <td id="T_f2c3d_row3_col6" class="data row3 col6" >0.0714</td>
    </tr>
    <tr>
      <th id="T_f2c3d_level0_row4" class="row_heading level0 row4" >PAN</th>
      <td id="T_f2c3d_row4_col0" class="data row4 col0" >0.0763</td>
      <td id="T_f2c3d_row4_col1" class="data row4 col1" >0.1429</td>
      <td id="T_f2c3d_row4_col2" class="data row4 col2" >0.0424</td>
      <td id="T_f2c3d_row4_col3" class="data row4 col3" >0.0178</td>
      <td id="T_f2c3d_row4_col4" class="data row4 col4" >0.0000</td>
      <td id="T_f2c3d_row4_col5" class="data row4 col5" >0.1284</td>
      <td id="T_f2c3d_row4_col6" class="data row4 col6" >0.0714</td>
    </tr>
  </tbody>
</table>




### Paso 3: cálculo de soluciones ideal positiva e ideal negativa

La solución ideal positiva $A^+$ toma el máximo de cada columna; la ideal negativa $A^-$ toma el mínimo:

$$
A^+ = \langle \max_i V_{i1}, \max_i V_{i2}, \dots, \max_i V_{ij} \rangle
$$

$$
A^- = \langle \min_i V_{i1}, \min_i V_{i2}, \dots, \min_i V_{ij} \rangle
$$


```python
# Paso 3: soluciones ideales
A_pos = matriz_ponderada.max()
A_neg = matriz_ponderada.min()

ideales = pd.DataFrame({'A⁺ (ideal positivo)': A_pos.round(4),
                        'A⁻ (ideal negativo)': A_neg.round(4)})
print('Soluciones ideales:')
style_table(ideales, format_dict={c: '{:.4f}' for c in ideales.columns})

```

    Soluciones ideales:
    




<style type="text/css">
#T_cd8c0 th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_cd8c0 td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_cd8c0 tbody tr:hover {
  background-color: #F5F5F5;
}
</style>
<table id="T_cd8c0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_cd8c0_level0_col0" class="col_heading level0 col0" >A⁺ (ideal positivo)</th>
      <th id="T_cd8c0_level0_col1" class="col_heading level0 col1" >A⁻ (ideal negativo)</th>
    </tr>
    <tr>
      <th class="index_name level0" >variable</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cd8c0_level0_row0" class="row_heading level0 row0" >gdp_per_capita_ppp</th>
      <td id="T_cd8c0_row0_col0" class="data row0 col0" >0.1429</td>
      <td id="T_cd8c0_row0_col1" class="data row0 col1" >0.0000</td>
    </tr>
    <tr>
      <th id="T_cd8c0_level0_row1" class="row_heading level0 row1" >inflation_rate</th>
      <td id="T_cd8c0_row1_col0" class="data row1 col0" >0.1429</td>
      <td id="T_cd8c0_row1_col1" class="data row1 col1" >0.0000</td>
    </tr>
    <tr>
      <th id="T_cd8c0_level0_row2" class="row_heading level0 row2" >financial_system_deposits_gdp</th>
      <td id="T_cd8c0_row2_col0" class="data row2 col0" >0.1429</td>
      <td id="T_cd8c0_row2_col1" class="data row2 col1" >0.0000</td>
    </tr>
    <tr>
      <th id="T_cd8c0_level0_row3" class="row_heading level0 row3" >regulatory_quality</th>
      <td id="T_cd8c0_row3_col0" class="data row3 col0" >0.1429</td>
      <td id="T_cd8c0_row3_col1" class="data row3 col1" >0.0000</td>
    </tr>
    <tr>
      <th id="T_cd8c0_level0_row4" class="row_heading level0 row4" >internet_users_pct</th>
      <td id="T_cd8c0_row4_col0" class="data row4 col0" >0.1429</td>
      <td id="T_cd8c0_row4_col1" class="data row4 col1" >0.0000</td>
    </tr>
    <tr>
      <th id="T_cd8c0_level0_row5" class="row_heading level0 row5" >geographic_distance_km</th>
      <td id="T_cd8c0_row5_col0" class="data row5 col0" >0.1429</td>
      <td id="T_cd8c0_row5_col1" class="data row5 col1" >0.0000</td>
    </tr>
    <tr>
      <th id="T_cd8c0_level0_row6" class="row_heading level0 row6" >common_language_spanish</th>
      <td id="T_cd8c0_row6_col0" class="data row6 col0" >0.0714</td>
      <td id="T_cd8c0_row6_col1" class="data row6 col1" >0.0714</td>
    </tr>
  </tbody>
</table>




### Paso 4: distancias euclidianas a las soluciones ideales

Para cada país $i$ se calcula su distancia al ideal positivo y al ideal negativo:

$$
d_i^+ = \sqrt{\sum_{j} (V_{ij} - A_j^+)^2}, \quad d_i^- = \sqrt{\sum_{j} (V_{ij} - A_j^-)^2}
$$


```python
# Paso 4: distancias euclidianas
d_pos = np.sqrt(((matriz_ponderada - A_pos) ** 2).sum(axis=1))
d_neg = np.sqrt(((matriz_ponderada - A_neg) ** 2).sum(axis=1))

distancias = pd.DataFrame({'d⁺ (distancia al ideal positivo)': d_pos.round(4),
                           'd⁻ (distancia al ideal negativo)': d_neg.round(4)})
print('Distancias euclidianas:')
style_table(distancias, format_dict={c: '{:.4f}' for c in distancias.columns})

```

    Distancias euclidianas:
    




<style type="text/css">
#T_01ddc th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_01ddc td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_01ddc tbody tr:hover {
  background-color: #F5F5F5;
}
</style>
<table id="T_01ddc">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_01ddc_level0_col0" class="col_heading level0 col0" >d⁺ (distancia al ideal positivo)</th>
      <th id="T_01ddc_level0_col1" class="col_heading level0 col1" >d⁻ (distancia al ideal negativo)</th>
    </tr>
    <tr>
      <th class="index_name level0" >country_iso3</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_01ddc_level0_row0" class="row_heading level0 row0" >CHL</th>
      <td id="T_01ddc_row0_col0" class="data row0 col0" >0.1714</td>
      <td id="T_01ddc_row0_col1" class="data row0 col1" >0.2320</td>
    </tr>
    <tr>
      <th id="T_01ddc_level0_row1" class="row_heading level0 row1" >COL</th>
      <td id="T_01ddc_row1_col0" class="data row1 col0" >0.2902</td>
      <td id="T_01ddc_row1_col1" class="data row1 col1" >0.1517</td>
    </tr>
    <tr>
      <th id="T_01ddc_level0_row2" class="row_heading level0 row2" >ESP</th>
      <td id="T_01ddc_row2_col0" class="data row2 col0" >0.1532</td>
      <td id="T_01ddc_row2_col1" class="data row2 col1" >0.2900</td>
    </tr>
    <tr>
      <th id="T_01ddc_level0_row3" class="row_heading level0 row3" >MEX</th>
      <td id="T_01ddc_row3_col0" class="data row3 col0" >0.2734</td>
      <td id="T_01ddc_row3_col1" class="data row3 col1" >0.1141</td>
    </tr>
    <tr>
      <th id="T_01ddc_level0_row4" class="row_heading level0 row4" >PAN</th>
      <td id="T_01ddc_row4_col0" class="data row4 col0" >0.2253</td>
      <td id="T_01ddc_row4_col1" class="data row4 col1" >0.2118</td>
    </tr>
  </tbody>
</table>




### Paso 5: cálculo del coeficiente de cercanía

El coeficiente $C^*$ combina ambas distancias:

$$
C_i^* = \frac{d_i^-}{d_i^+ + d_i^-}
$$

La interpretación es elegante: $C^*$ vale 1 si el país coincide con el ideal positivo, 0 si coincide con el ideal negativo, y valores intermedios en proporción a qué tan cerca está de uno u otro. **Mayor $C^*$ significa mayor atractivo del mercado**.

### Paso 6: ranking final

Se ordenan los países por $C^*$ descendente:


```python
# Pasos 5 y 6: coeficiente de cercania y ranking
cc = d_neg / (d_pos + d_neg)
ranking_final = pd.DataFrame({
    'd⁺':     d_pos.round(4),
    'd⁻':     d_neg.round(4),
    'C*':     cc.round(4),
}).sort_values('C*', ascending=False)
ranking_final.insert(0, 'Rank', range(1, len(ranking_final) + 1))

print('Ranking final TOPSIS:')
style_table(ranking_final, gradient_cols=['C*'], gradient_cmap=cmap_custom,
            format_dict={'d⁺': '{:.4f}', 'd⁻': '{:.4f}', 'C*': '{:.4f}', 'Rank': '{:.0f}'})

```

    Ranking final TOPSIS:
    




<style type="text/css">
#T_c16ab th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_c16ab td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_c16ab tbody tr:hover {
  background-color: #F5F5F5;
}
#T_c16ab_row0_col3 {
  background-color: #fdd923;
  color: #000000;
}
#T_c16ab_row1_col3 {
  background-color: #f2d647;
  color: #000000;
}
#T_c16ab_row2_col3 {
  background-color: #e6d270;
  color: #000000;
}
#T_c16ab_row3_col3 {
  background-color: #d3ccb1;
  color: #000000;
}
#T_c16ab_row4_col3 {
  background-color: #cccac7;
  color: #000000;
}
</style>
<table id="T_c16ab">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c16ab_level0_col0" class="col_heading level0 col0" >Rank</th>
      <th id="T_c16ab_level0_col1" class="col_heading level0 col1" >d⁺</th>
      <th id="T_c16ab_level0_col2" class="col_heading level0 col2" >d⁻</th>
      <th id="T_c16ab_level0_col3" class="col_heading level0 col3" >C*</th>
    </tr>
    <tr>
      <th class="index_name level0" >country_iso3</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c16ab_level0_row0" class="row_heading level0 row0" >ESP</th>
      <td id="T_c16ab_row0_col0" class="data row0 col0" >1</td>
      <td id="T_c16ab_row0_col1" class="data row0 col1" >0.1532</td>
      <td id="T_c16ab_row0_col2" class="data row0 col2" >0.2900</td>
      <td id="T_c16ab_row0_col3" class="data row0 col3" >0.6542</td>
    </tr>
    <tr>
      <th id="T_c16ab_level0_row1" class="row_heading level0 row1" >CHL</th>
      <td id="T_c16ab_row1_col0" class="data row1 col0" >2</td>
      <td id="T_c16ab_row1_col1" class="data row1 col1" >0.1714</td>
      <td id="T_c16ab_row1_col2" class="data row1 col2" >0.2320</td>
      <td id="T_c16ab_row1_col3" class="data row1 col3" >0.5751</td>
    </tr>
    <tr>
      <th id="T_c16ab_level0_row2" class="row_heading level0 row2" >PAN</th>
      <td id="T_c16ab_row2_col0" class="data row2 col0" >3</td>
      <td id="T_c16ab_row2_col1" class="data row2 col1" >0.2253</td>
      <td id="T_c16ab_row2_col2" class="data row2 col2" >0.2118</td>
      <td id="T_c16ab_row2_col3" class="data row2 col3" >0.4845</td>
    </tr>
    <tr>
      <th id="T_c16ab_level0_row3" class="row_heading level0 row3" >COL</th>
      <td id="T_c16ab_row3_col0" class="data row3 col0" >4</td>
      <td id="T_c16ab_row3_col1" class="data row3 col1" >0.2902</td>
      <td id="T_c16ab_row3_col2" class="data row3 col2" >0.1517</td>
      <td id="T_c16ab_row3_col3" class="data row3 col3" >0.3433</td>
    </tr>
    <tr>
      <th id="T_c16ab_level0_row4" class="row_heading level0 row4" >MEX</th>
      <td id="T_c16ab_row4_col0" class="data row4 col0" >5</td>
      <td id="T_c16ab_row4_col1" class="data row4 col1" >0.2734</td>
      <td id="T_c16ab_row4_col2" class="data row4 col2" >0.1141</td>
      <td id="T_c16ab_row4_col3" class="data row4 col3" >0.2944</td>
    </tr>
  </tbody>
</table>





```python
# Visualizacion interactiva del ranking TOPSIS
fig = go.Figure()
df_plot = ranking_final.reset_index().sort_values('C*', ascending=True)
fig.add_trace(go.Bar(
    x=df_plot['C*'], y=df_plot.iloc[:, 0],  # primera columna = pais (index reseteado)
    orientation='h',
    marker=dict(color=df_plot['C*'], colorscale=[[0, CIBEST['red']],
                                                   [0.5, CIBEST['gold']],
                                                   [1.0, CIBEST['green']]],
                line=dict(color=CIBEST['gray'], width=1)),
    text=df_plot['C*'].round(3),
    textposition='outside',
    textfont=dict(color=CIBEST['gray'], size=12),
    hovertemplate='<b>%{y}</b><br>C* = %{x:.4f}<extra></extra>',
))
fig.update_layout(
    title='<b>Ranking TOPSIS — Coeficiente de cercanía C*</b><br>'
          '<sub>Países ordenados por proximidad al ideal positivo</sub>',
    xaxis_title='C* (coeficiente de cercanía)',
    yaxis_title='País',
    xaxis=dict(range=[0, 1.0], gridcolor=CIBEST['gray_border']),
    plot_bgcolor=CIBEST['white'], paper_bgcolor=CIBEST['white'],
    font=dict(family='Arial', color=CIBEST['gray']),
    height=400, showlegend=False, margin=dict(l=100),
)
fig.show()

```



## Una visualización geométrica para fijar la intuición

Para ver geométricamente lo que TOPSIS está haciendo, podemos proyectar el problema a dos dimensiones. Si tomamos solo dos variables (por ejemplo PIB per cápita y Calidad Regulatoria), podemos dibujar cada país como un punto en el plano y marcar dónde están las soluciones ideales:


```python
# Proyeccion 2D para intuicion geometrica de TOPSIS
fig = go.Figure()

# Solo dos variables para visualizacion 2D
xs = matriz_norm['gdp_per_capita_ppp']
ys = matriz_norm['regulatory_quality']

# Marcadores de cada pais
fig.add_trace(go.Scatter(
    x=xs, y=ys, mode='markers+text',
    marker=dict(size=22, color=CIBEST['gray'], line=dict(color=CIBEST['gold'], width=2)),
    text=xs.index, textposition='top center',
    textfont=dict(size=12, color=CIBEST['gray']),
    name='Países', hovertemplate='<b>%{text}</b><br>PIB PPP norm.: %{x:.3f}<br>Calidad reg. norm.: %{y:.3f}<extra></extra>',
))

# Ideal positivo (en la esquina superior derecha del cuadrado [0,1]x[0,1])
fig.add_trace(go.Scatter(
    x=[xs.max()], y=[ys.max()], mode='markers+text',
    marker=dict(size=28, color=CIBEST['green'], symbol='star', line=dict(color=CIBEST['gray'], width=2)),
    text=['A⁺'], textposition='top center', textfont=dict(size=15, color=CIBEST['green']),
    name='Ideal positivo (A⁺)',
))

# Ideal negativo (esquina inferior izquierda)
fig.add_trace(go.Scatter(
    x=[xs.min()], y=[ys.min()], mode='markers+text',
    marker=dict(size=28, color=CIBEST['red'], symbol='star', line=dict(color=CIBEST['gray'], width=2)),
    text=['A⁻'], textposition='bottom center', textfont=dict(size=15, color=CIBEST['red']),
    name='Ideal negativo (A⁻)',
))

# Lineas hacia los ideales para el primer pais del ranking
top_country = ranking_final.index[0]
fig.add_trace(go.Scatter(
    x=[xs[top_country], xs.max()], y=[ys[top_country], ys.max()],
    mode='lines', line=dict(color=CIBEST['green'], width=2, dash='dash'),
    name=f'd⁺ desde {top_country}', showlegend=True,
    hoverinfo='skip',
))
fig.add_trace(go.Scatter(
    x=[xs[top_country], xs.min()], y=[ys[top_country], ys.min()],
    mode='lines', line=dict(color=CIBEST['red'], width=2, dash='dash'),
    name=f'd⁻ desde {top_country}', showlegend=True,
    hoverinfo='skip',
))

fig.update_layout(
    title='<b>Visualización geométrica de TOPSIS</b><br>'
          '<sub>Proyección 2D: PIB per cápita PPP vs. Calidad regulatoria (normalizadas)</sub>',
    xaxis=dict(title='PIB per cápita PPP (normalizado)', range=[-0.05, 1.1], gridcolor=CIBEST['gray_border']),
    yaxis=dict(title='Calidad regulatoria (normalizada)', range=[-0.05, 1.1], gridcolor=CIBEST['gray_border']),
    plot_bgcolor=CIBEST['white'], paper_bgcolor=CIBEST['white'],
    font=dict(family='Arial', color=CIBEST['gray']),
    height=550, legend=dict(orientation='h', y=-0.15),
)
fig.show()

```



La figura ilustra perfectamente la intuición de TOPSIS. Los cinco países están en distintos puntos del cuadrado unitario. La estrella verde arriba a la derecha es el ideal positivo (mejor en todo), la estrella roja abajo a la izquierda es el ideal negativo (peor en todo). Las líneas punteadas muestran las dos distancias del país top: a más corta sea la distancia verde (al ideal positivo) y a más larga sea la distancia roja (al ideal negativo), mejor el ranking. En el sistema real esto sucede no en dos dimensiones sino en ~35 dimensiones, pero la idea es la misma.

## Scores parciales por dimensión: una sutileza importante

Hay un detalle del TOPSIS implementado en RADAR Cibest que merece mencionarse explícitamente. Además del coeficiente $C^*$ global, el sistema ejecuta TOPSIS también sobre cada **subconjunto** de variables que pertenece a una misma dimensión, generando así *scores parciales por dimensión*. Estos scores parciales son los que alimentan el motor de señales y el perfil de fortalezas/debilidades de cada país. Sin esos scores parciales el sistema no podría decir cosas como "Panamá es fuerte en proximidad pero débil en macro": solo podría decir "Panamá tiene C* = 0.6 globalmente".

## Múltiples ejecuciones por línea de negocio

La otra sutileza importante es que TOPSIS se ejecuta **seis veces en total** en el sistema real: una vez con pesos globales (BWM agregado del panel) y cinco veces más, una por cada línea de negocio, donde los pesos cambian según el `weight_profile` de la línea. Esto es lo que produce los rankings diferenciados por línea de los que hablábamos al inicio. Computacionalmente es trivial; conceptualmente es la respuesta a la pregunta "¿cuáles son los mercados más atractivos *para esta línea en particular*?".

---

# Capítulo 6 — Modelo gravitacional: el factor de proximidad con Colombia

## La intuición física

Hay una observación empírica muy robusta en economía internacional: **los flujos de bienes, capital, personas y servicios entre dos países disminuyen con la distancia entre ellos y aumentan con sus tamaños económicos**. La fórmula que captura esta regularidad se llama, por analogía con la ley de gravitación universal de Newton, *modelo gravitacional*. En la versión clásica de Anderson y van Wincoop (2003), el flujo bilateral entre dos países $i$ y $j$ tiene la forma:

$$
F_{ij} \propto \frac{Y_i^{\alpha} \cdot Y_j^{\beta}}{D_{ij}^{\gamma}}
$$

donde $Y_i$ e $Y_j$ son los PIB de los dos países y $D_{ij}$ es la distancia entre ellos. Los exponentes $\alpha$, $\beta$ y $\gamma$ se estiman empíricamente y suelen estar cerca de 1.

En el contexto bancario específicamente, Brei y von Peter (2018) muestran que duplicar la distancia entre dos países reduce los flujos bancarios bilaterales entre 30% y 50%. Ese hallazgo es central para RADAR Cibest: la distancia importa muchísimo para el negocio bancario. Pero "distancia" en sentido económico no es solo distancia geográfica; incluye también *distancia cultural*, *distancia institucional* y *distancia lingüística*. Ghemawat (2001) formalizó esto en el llamado *marco CAGE* (Cultural, Administrative, Geographic, Economic).

## El Índice de Proximidad Compuesto (IPC)

RADAR Cibest no estima una regresión gravitacional completa porque no necesita predecir flujos bilaterales en valores; lo que necesita es un **índice resumen** de qué tan "cerca" está cada país de Colombia en sentido amplio. Ese índice se llama **Índice de Proximidad Compuesto (IPC)** y se construye así. Para cada componente $k$ de proximidad (distancia geográfica, distancia cultural Hofstede, idioma compartido, comercio bilateral, stock de diáspora), se normaliza el componente a la escala [0,1] aplicando su dirección, y luego se hace una combinación lineal:

$$
\text{IPC}_j = \sum_{k} w_k \cdot \text{prox}_k(\text{COL}, j)
$$

donde $w_k$ son los pesos de cada componente (por defecto equiponderados) y $\text{prox}_k(\text{COL}, j)$ es el componente $k$ normalizado y orientado de modo que mayor valor signifique mayor proximidad. Por construcción Colombia tiene $\text{IPC} = 1$ (es la "máxima proximidad consigo misma").

## Ejemplo numérico con los cinco países


```python
# Calculo del IPC para los cinco paises del ejemplo
# Usamos las dos variables de proximidad de nuestra matriz: distancia y idioma
componentes_ipc = pd.DataFrame({
    'distancia_km':          ejemplo['geographic_distance_km'],
    'idioma_espanol':        ejemplo['common_language_spanish'],
})

# Normalizar y orientar: distancia (negativa) -> invertir
def norm_y_orientar_componente(serie, direccion):
    vmin, vmax = serie.min(), serie.max()
    if vmax - vmin == 0:
        n = pd.Series(0.5, index=serie.index)
    else:
        n = (serie - vmin) / (vmax - vmin)
    if direccion == 'negative':
        n = 1.0 - n
    return n

prox_distancia = norm_y_orientar_componente(componentes_ipc['distancia_km'], 'negative')
prox_idioma    = norm_y_orientar_componente(componentes_ipc['idioma_espanol'], 'positive')

# IPC equiponderado entre los dos componentes
ipc = 0.5 * prox_distancia + 0.5 * prox_idioma
ipc.loc['COL'] = 1.0   # por definicion

df_ipc = pd.DataFrame({
    'Distancia (km)':                componentes_ipc['distancia_km'].astype(int),
    'Proximidad geográfica (norm.)': prox_distancia.round(3),
    'Idioma español (1/0)':          componentes_ipc['idioma_espanol'].astype(int),
    'IPC':                           ipc.round(3),
}).sort_values('IPC', ascending=False)

print('Índice de Proximidad Compuesto (IPC):')
style_table(df_ipc, gradient_cols=['IPC'], gradient_cmap='YlOrBr',
            format_dict={'IPC': '{:.3f}', 'Proximidad geográfica (norm.)': '{:.3f}'})

```

    Índice de Proximidad Compuesto (IPC):
    




<style type="text/css">
#T_d3fa6 th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_d3fa6 td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_d3fa6 tbody tr:hover {
  background-color: #F5F5F5;
}
#T_d3fa6_row0_col3 {
  background-color: #662506;
  color: #f1f1f1;
}
#T_d3fa6_row1_col3 {
  background-color: #f07818;
  color: #f1f1f1;
}
#T_d3fa6_row2_col3 {
  background-color: #fec34f;
  color: #000000;
}
#T_d3fa6_row3_col3 {
  background-color: #fed36f;
  color: #000000;
}
#T_d3fa6_row4_col3 {
  background-color: #ffffe5;
  color: #000000;
}
</style>
<table id="T_d3fa6">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_d3fa6_level0_col0" class="col_heading level0 col0" >Distancia (km)</th>
      <th id="T_d3fa6_level0_col1" class="col_heading level0 col1" >Proximidad geográfica (norm.)</th>
      <th id="T_d3fa6_level0_col2" class="col_heading level0 col2" >Idioma español (1/0)</th>
      <th id="T_d3fa6_level0_col3" class="col_heading level0 col3" >IPC</th>
    </tr>
    <tr>
      <th class="index_name level0" >country_iso3</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d3fa6_level0_row0" class="row_heading level0 row0" >COL</th>
      <td id="T_d3fa6_row0_col0" class="data row0 col0" >0</td>
      <td id="T_d3fa6_row0_col1" class="data row0 col1" >1.000</td>
      <td id="T_d3fa6_row0_col2" class="data row0 col2" >1</td>
      <td id="T_d3fa6_row0_col3" class="data row0 col3" >1.000</td>
    </tr>
    <tr>
      <th id="T_d3fa6_level0_row1" class="row_heading level0 row1" >PAN</th>
      <td id="T_d3fa6_row1_col0" class="data row1 col0" >810</td>
      <td id="T_d3fa6_row1_col1" class="data row1 col1" >0.899</td>
      <td id="T_d3fa6_row1_col2" class="data row1 col2" >1</td>
      <td id="T_d3fa6_row1_col3" class="data row1 col3" >0.700</td>
    </tr>
    <tr>
      <th id="T_d3fa6_level0_row2" class="row_heading level0 row2" >MEX</th>
      <td id="T_d3fa6_row2_col0" class="data row2 col0" >3490</td>
      <td id="T_d3fa6_row2_col1" class="data row2 col1" >0.565</td>
      <td id="T_d3fa6_row2_col2" class="data row2 col2" >1</td>
      <td id="T_d3fa6_row2_col3" class="data row2 col3" >0.533</td>
    </tr>
    <tr>
      <th id="T_d3fa6_level0_row3" class="row_heading level0 row3" >CHL</th>
      <td id="T_d3fa6_row3_col0" class="data row3 col0" >4250</td>
      <td id="T_d3fa6_row3_col1" class="data row3 col1" >0.471</td>
      <td id="T_d3fa6_row3_col2" class="data row3 col2" >1</td>
      <td id="T_d3fa6_row3_col3" class="data row3 col3" >0.485</td>
    </tr>
    <tr>
      <th id="T_d3fa6_level0_row4" class="row_heading level0 row4" >ESP</th>
      <td id="T_d3fa6_row4_col0" class="data row4 col0" >8030</td>
      <td id="T_d3fa6_row4_col1" class="data row4 col1" >0.000</td>
      <td id="T_d3fa6_row4_col2" class="data row4 col2" >1</td>
      <td id="T_d3fa6_row4_col3" class="data row4 col3" >0.250</td>
    </tr>
  </tbody>
</table>





```python
# Visualizacion del IPC
fig = go.Figure()
df_sorted = df_ipc.sort_values('IPC')
fig.add_trace(go.Bar(
    y=df_sorted.index, x=df_sorted['IPC'],
    orientation='h',
    marker=dict(color=df_sorted['IPC'],
                colorscale=[[0, CIBEST['gray_bg']], [0.5, CIBEST['gray_light']], [1, CIBEST['yellow']]],
                line=dict(color=CIBEST['gray'], width=1)),
    text=df_sorted['IPC'].round(3),
    textposition='outside',
    textfont=dict(color=CIBEST['gray']),
    hovertemplate='<b>%{y}</b><br>IPC = %{x:.3f}<extra></extra>',
))
fig.update_layout(
    title='<b>Índice de Proximidad Compuesto (IPC)</b><br>'
          '<sub>Afinidad de cada país con Colombia · Colombia = 1.0 por construcción</sub>',
    xaxis=dict(title='IPC', range=[0, 1.1], gridcolor=CIBEST['gray_border']),
    yaxis_title='País',
    plot_bgcolor=CIBEST['white'], paper_bgcolor=CIBEST['white'],
    font=dict(family='Arial', color=CIBEST['gray']),
    height=400, margin=dict(l=100), showlegend=False,
)
fig.show()

```



El IPC del ejemplo confirma lo esperado: Panamá está muy cerca de Colombia (corto vuelo, mismo idioma), México intermedio (mismo idioma pero más lejos), y España queda al final pese a compartir idioma porque la distancia geográfica es enorme. En el sistema real el IPC integra más componentes (distancia cultural Hofstede, comercio bilateral, stock de diáspora), lo que afina aún más la medida.

## ¿Por qué incorporar proximidad si TOPSIS ya considera variables de proximidad?

Esta es una pregunta legítima y conviene responderla. TOPSIS efectivamente puede incluir la distancia y el idioma como variables más, y de hecho lo hace en el sistema real. Entonces ¿para qué tener un IPC separado? La respuesta tiene dos partes. La primera es **interpretabilidad**: tener un IPC explícito permite responder al Comité preguntas del tipo "¿qué tanto pesa la proximidad con Colombia en este ranking?" con una cifra concreta ($\beta$ del score compuesto que veremos en el siguiente capítulo). La segunda es **flexibilidad estratégica**: si en algún momento Cibest quiere experimentar con una estrategia "más globalista" (priorizar atractivo absoluto) o "más regionalista" (priorizar proximidad), se puede hacer ajustando $\beta$ sin tocar la estructura de pesos de TOPSIS. Esa flexibilidad es valiosa.

---

# Capítulo 7 — El score RADAR compuesto

## Tres componentes que se integran en uno

Hasta ahora tenemos tres salidas separadas. TOPSIS produce un coeficiente de cercanía $C^*$ para cada país, que mide su atractivo absoluto según las variables y pesos. El modelo gravitacional produce el IPC, que mide la afinidad bilateral con Colombia. Y por separado, el sistema calcula un **factor de tendencia** que captura el dinamismo reciente del mercado (típicamente el promedio del crecimiento real del PIB en los últimos tres años, normalizado).

El score RADAR final integra los tres componentes en una sola cifra mediante una combinación lineal sencilla:

$$
\text{RADAR}(j, \ell) = \alpha \cdot C^*(j, \ell) + \beta \cdot \text{IPC}(j) + \gamma \cdot \text{Tendencia}(j)
$$

donde $j$ es el país, $\ell$ es la línea de negocio (para el score por línea) o nulo (para el score global), y los pesos $\alpha + \beta + \gamma = 1$. Los valores por defecto del sistema son $\alpha = 0.55$, $\beta = 0.30$ y $\gamma = 0.15$, derivados de la mediana de la literatura sistematizada en la revisión bibliográfica del proyecto.

## La interpretación de los tres pesos

Cada uno de los tres pesos cuenta una historia estratégica diferente sobre cómo Cibest mira los mercados. **$\alpha$ pondera el atractivo absoluto del país** medido con TOPSIS. Si Cibest fuera puramente "globalista", $\alpha$ tendería a 1 y los otros dos tenderían a 0: solo importaría qué tan grande, desarrollado e institucionalmente sólido es cada mercado. **$\beta$ pondera la afinidad bilateral con Colombia**. Si Cibest fuera puramente "regionalista", $\beta$ tendería a 1: solo importaría qué tan cercano está el mercado en sentido amplio. **$\gamma$ pondera el dinamismo reciente**. Su valor por defecto es bajo (15%) porque la decisión de internacionalización es de largo plazo y no debería estar dominada por el ciclo, pero suficiente para que un mercado en plena recuperación pueda destacarse frente a otro estancado con scores similares.

Los valores por defecto $(0.55, 0.30, 0.15)$ representan una postura intermedia: el atractivo absoluto pesa más de la mitad, la proximidad es relevante pero no decisiva, y el dinamismo modula. Estos valores se pueden ajustar en el dashboard mediante sliders, lo que permite a la dirección explorar escenarios "qué pasa si fuéramos más globalistas" o "qué pasa si priorizáramos más a los vecinos".

## Cálculo del score RADAR sobre nuestro ejemplo


```python
# Score RADAR para los cinco paises
# Componente 1: C* viene de TOPSIS (calculado en capitulo 5)
C_star = cc.copy()
# Componente 2: IPC (calculado en capitulo 6)
IPC = ipc.copy()
# Componente 3: tendencia. Para el ejemplo usamos crecimiento del PIB sintetico
tendencia_raw = pd.Series({
    'COL': 2.5, 'MEX': 3.0, 'CHL': 1.8, 'PAN': 7.5, 'ESP': 2.2,
})
# Normalizar tendencia a [0,1]
tendencia = (tendencia_raw - tendencia_raw.min()) / (tendencia_raw.max() - tendencia_raw.min())

# Pesos por defecto del sistema
alpha, beta, gamma = 0.55, 0.30, 0.15

score_radar = alpha * C_star + beta * IPC + gamma * tendencia

resumen = pd.DataFrame({
    'C* (TOPSIS)': C_star.round(3),
    'IPC (Prox.)': IPC.round(3),
    'Tendencia':   tendencia.round(3),
    'RADAR':       score_radar.round(3),
})
resumen = resumen.sort_values('RADAR', ascending=False)
resumen.insert(0, 'Rank', range(1, len(resumen) + 1))

print(f'Score RADAR compuesto (α={alpha}, β={beta}, γ={gamma}):')
style_table(resumen,
            gradient_cols=['C* (TOPSIS)', 'IPC (Prox.)', 'Tendencia', 'RADAR'],
            gradient_cmap=cmap_custom,
            format_dict={'C* (TOPSIS)': '{:.3f}', 'IPC (Prox.)': '{:.3f}',
                         'Tendencia': '{:.3f}', 'RADAR': '{:.3f}', 'Rank': '{:.0f}'})

```

    Score RADAR compuesto (α=0.55, β=0.3, γ=0.15):
    




<style type="text/css">
#T_b3f48 th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_b3f48 td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_b3f48 tbody tr:hover {
  background-color: #F5F5F5;
}
#T_b3f48_row0_col1 {
  background-color: #e6d270;
  color: #000000;
}
#T_b3f48_row0_col2 {
  background-color: #e9d365;
  color: #000000;
}
#T_b3f48_row0_col3, #T_b3f48_row0_col4, #T_b3f48_row1_col2, #T_b3f48_row3_col1 {
  background-color: #fdd923;
  color: #000000;
}
#T_b3f48_row1_col1 {
  background-color: #d3ccb1;
  color: #000000;
}
#T_b3f48_row1_col3 {
  background-color: #d2ccb3;
  color: #000000;
}
#T_b3f48_row1_col4 {
  background-color: #e8d26a;
  color: #000000;
}
#T_b3f48_row2_col1 {
  background-color: #f2d647;
  color: #000000;
}
#T_b3f48_row2_col2 {
  background-color: #dbcf94;
  color: #000000;
}
#T_b3f48_row2_col3, #T_b3f48_row3_col2, #T_b3f48_row4_col1, #T_b3f48_row4_col4 {
  background-color: #cccac7;
  color: #000000;
}
#T_b3f48_row2_col4 {
  background-color: #e0d085;
  color: #000000;
}
#T_b3f48_row3_col3 {
  background-color: #cfcbbc;
  color: #000000;
}
#T_b3f48_row3_col4 {
  background-color: #ddcf90;
  color: #000000;
}
#T_b3f48_row4_col2 {
  background-color: #ded089;
  color: #000000;
}
#T_b3f48_row4_col3 {
  background-color: #d6cda4;
  color: #000000;
}
</style>
<table id="T_b3f48">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b3f48_level0_col0" class="col_heading level0 col0" >Rank</th>
      <th id="T_b3f48_level0_col1" class="col_heading level0 col1" >C* (TOPSIS)</th>
      <th id="T_b3f48_level0_col2" class="col_heading level0 col2" >IPC (Prox.)</th>
      <th id="T_b3f48_level0_col3" class="col_heading level0 col3" >Tendencia</th>
      <th id="T_b3f48_level0_col4" class="col_heading level0 col4" >RADAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b3f48_level0_row0" class="row_heading level0 row0" >PAN</th>
      <td id="T_b3f48_row0_col0" class="data row0 col0" >1</td>
      <td id="T_b3f48_row0_col1" class="data row0 col1" >0.485</td>
      <td id="T_b3f48_row0_col2" class="data row0 col2" >0.700</td>
      <td id="T_b3f48_row0_col3" class="data row0 col3" >1.000</td>
      <td id="T_b3f48_row0_col4" class="data row0 col4" >0.626</td>
    </tr>
    <tr>
      <th id="T_b3f48_level0_row1" class="row_heading level0 row1" >COL</th>
      <td id="T_b3f48_row1_col0" class="data row1 col0" >2</td>
      <td id="T_b3f48_row1_col1" class="data row1 col1" >0.343</td>
      <td id="T_b3f48_row1_col2" class="data row1 col2" >1.000</td>
      <td id="T_b3f48_row1_col3" class="data row1 col3" >0.123</td>
      <td id="T_b3f48_row1_col4" class="data row1 col4" >0.507</td>
    </tr>
    <tr>
      <th id="T_b3f48_level0_row2" class="row_heading level0 row2" >CHL</th>
      <td id="T_b3f48_row2_col0" class="data row2 col0" >3</td>
      <td id="T_b3f48_row2_col1" class="data row2 col1" >0.575</td>
      <td id="T_b3f48_row2_col2" class="data row2 col2" >0.485</td>
      <td id="T_b3f48_row2_col3" class="data row2 col3" >0.000</td>
      <td id="T_b3f48_row2_col4" class="data row2 col4" >0.462</td>
    </tr>
    <tr>
      <th id="T_b3f48_level0_row3" class="row_heading level0 row3" >ESP</th>
      <td id="T_b3f48_row3_col0" class="data row3 col0" >4</td>
      <td id="T_b3f48_row3_col1" class="data row3 col1" >0.654</td>
      <td id="T_b3f48_row3_col2" class="data row3 col2" >0.250</td>
      <td id="T_b3f48_row3_col3" class="data row3 col3" >0.070</td>
      <td id="T_b3f48_row3_col4" class="data row3 col4" >0.445</td>
    </tr>
    <tr>
      <th id="T_b3f48_level0_row4" class="row_heading level0 row4" >MEX</th>
      <td id="T_b3f48_row4_col0" class="data row4 col0" >5</td>
      <td id="T_b3f48_row4_col1" class="data row4 col1" >0.294</td>
      <td id="T_b3f48_row4_col2" class="data row4 col2" >0.533</td>
      <td id="T_b3f48_row4_col3" class="data row4 col3" >0.211</td>
      <td id="T_b3f48_row4_col4" class="data row4 col4" >0.353</td>
    </tr>
  </tbody>
</table>





```python
# Visualizacion stacked del score RADAR mostrando contribucion de cada componente
fig = go.Figure()
df_stack = resumen.sort_values('RADAR')

fig.add_trace(go.Bar(
    name='α · C* (TOPSIS)',
    y=df_stack.index, x=alpha * df_stack['C* (TOPSIS)'],
    orientation='h', marker_color=CIBEST['gray'],
    hovertemplate='<b>%{y}</b><br>α·C* = %{x:.3f}<extra></extra>',
))
fig.add_trace(go.Bar(
    name='β · IPC (Proximidad)',
    y=df_stack.index, x=beta * df_stack['IPC (Prox.)'],
    orientation='h', marker_color=CIBEST['yellow'],
    hovertemplate='<b>%{y}</b><br>β·IPC = %{x:.3f}<extra></extra>',
))
fig.add_trace(go.Bar(
    name='γ · Tendencia',
    y=df_stack.index, x=gamma * df_stack['Tendencia'],
    orientation='h', marker_color=CIBEST['gray_light'],
    hovertemplate='<b>%{y}</b><br>γ·Tendencia = %{x:.3f}<extra></extra>',
))

fig.update_layout(
    title=f'<b>Descomposición del score RADAR</b><br>'
          f'<sub>RADAR = α·C* + β·IPC + γ·Tendencia · '
          f'α={alpha}, β={beta}, γ={gamma}</sub>',
    barmode='stack',
    xaxis=dict(title='Contribución al score RADAR', range=[0, 1], gridcolor=CIBEST['gray_border']),
    yaxis_title='País',
    plot_bgcolor=CIBEST['white'], paper_bgcolor=CIBEST['white'],
    font=dict(family='Arial', color=CIBEST['gray']),
    height=400, legend=dict(orientation='h', y=-0.2),
    margin=dict(l=100),
)
fig.show()

```



La gráfica anterior es muy informativa. La parte oscura es lo que aporta el atractivo absoluto del país (C*), la parte amarilla es lo que aporta la proximidad con Colombia (IPC), y la gris es lo que aporta el dinamismo. Mirar el patrón de las barras ayuda a entender por qué cada país está donde está: España rankea alto principalmente por su C* (atractivo absoluto), Panamá rankea alto principalmente por su IPC (proximidad), Colombia rankea muy alto en proximidad por construcción pero su C* es modesto. Esa decomposición visual es exactamente lo que el dashboard ofrece a la alta dirección para que pueda explicar el ranking y argumentar decisiones.

---

# Capítulo 8 — Motor de señales por línea de negocio

## De scores numéricos a etiquetas accionables

Hasta este punto el sistema produce números: scores RADAR entre 0 y 1, rankings de 1 a 30. Esos números son útiles para los analistas, pero para que el resultado sea verdaderamente *accionable* por la alta dirección, conviene convertirlos en **etiquetas categóricas claras**: ¿este país es una alta oportunidad, una oportunidad moderada, una baja oportunidad, o un riesgo? La transformación de scores en etiquetas la hace el **motor de señales**, que aplica un sistema simple de cuatro niveles basado en percentiles, con una capa adicional de "overrides de riesgo" para garantizar que ningún mercado con problemas institucionales severos se etiquete como oportunidad alta solo por tener un score numérico bueno.

## Las cuatro etiquetas y sus umbrales por defecto

Los umbrales del sistema están definidos sobre el percentil del score por línea, no sobre el valor absoluto del score. Esa decisión es importante: si los umbrales fueran absolutos (digamos "ALTA si C* > 0.7"), un cambio uniforme en la matriz de decisión podría hacer que ningún país calificara como ALTA en una corrida y todos en la siguiente. Usar percentiles asegura que las señales sean *relativas al conjunto evaluado* y por tanto más estables operativamente.


```python
umbrales_senales = pd.DataFrame([
    {'Señal': 'ALTA_OPORTUNIDAD',     'Color': '🟢',
     'Condición': 'Percentil del score ≥ 0.70 y sin overrides activos',
     'Interpretación estratégica': 'Mercado prioritario para esta línea — abrir conversaciones'},
    {'Señal': 'OPORTUNIDAD_MODERADA', 'Color': '🟡',
     'Condición': '0.45 ≤ Percentil < 0.70 y sin overrides',
     'Interpretación estratégica': 'Mercado de segundo orden — monitorear y profundizar análisis'},
    {'Señal': 'BAJA_OPORTUNIDAD',     'Color': '🟠',
     'Condición': '0.25 ≤ Percentil < 0.45',
     'Interpretación estratégica': 'Mercado no prioritario en el ciclo actual'},
    {'Señal': 'RIESGO',               'Color': '🔴',
     'Condición': 'Percentil < 0.25 ó cualquier override activo',
     'Interpretación estratégica': 'Mercado a evitar o descartar — riesgo institucional o de score'},
])
style_table(umbrales_senales)

```




<style type="text/css">
#T_3628d th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_3628d td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_3628d tbody tr:hover {
  background-color: #F5F5F5;
}
</style>
<table id="T_3628d">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3628d_level0_col0" class="col_heading level0 col0" >Señal</th>
      <th id="T_3628d_level0_col1" class="col_heading level0 col1" >Color</th>
      <th id="T_3628d_level0_col2" class="col_heading level0 col2" >Condición</th>
      <th id="T_3628d_level0_col3" class="col_heading level0 col3" >Interpretación estratégica</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3628d_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_3628d_row0_col0" class="data row0 col0" >ALTA_OPORTUNIDAD</td>
      <td id="T_3628d_row0_col1" class="data row0 col1" >🟢</td>
      <td id="T_3628d_row0_col2" class="data row0 col2" >Percentil del score ≥ 0.70 y sin overrides activos</td>
      <td id="T_3628d_row0_col3" class="data row0 col3" >Mercado prioritario para esta línea — abrir conversaciones</td>
    </tr>
    <tr>
      <th id="T_3628d_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_3628d_row1_col0" class="data row1 col0" >OPORTUNIDAD_MODERADA</td>
      <td id="T_3628d_row1_col1" class="data row1 col1" >🟡</td>
      <td id="T_3628d_row1_col2" class="data row1 col2" >0.45 ≤ Percentil < 0.70 y sin overrides</td>
      <td id="T_3628d_row1_col3" class="data row1 col3" >Mercado de segundo orden — monitorear y profundizar análisis</td>
    </tr>
    <tr>
      <th id="T_3628d_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_3628d_row2_col0" class="data row2 col0" >BAJA_OPORTUNIDAD</td>
      <td id="T_3628d_row2_col1" class="data row2 col1" >🟠</td>
      <td id="T_3628d_row2_col2" class="data row2 col2" >0.25 ≤ Percentil < 0.45</td>
      <td id="T_3628d_row2_col3" class="data row2 col3" >Mercado no prioritario en el ciclo actual</td>
    </tr>
    <tr>
      <th id="T_3628d_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_3628d_row3_col0" class="data row3 col0" >RIESGO</td>
      <td id="T_3628d_row3_col1" class="data row3 col1" >🔴</td>
      <td id="T_3628d_row3_col2" class="data row3 col2" >Percentil < 0.25 ó cualquier override activo</td>
      <td id="T_3628d_row3_col3" class="data row3 col3" >Mercado a evitar o descartar — riesgo institucional o de score</td>
    </tr>
  </tbody>
</table>




## Los overrides de riesgo

Existen dos *overrides* automáticos que fuerzan la señal a RIESGO sin importar el score numérico. Si la **estabilidad política** (indicador WGI `PV.EST`) está en el percentil inferior al 15%, el país se etiqueta como RIESGO. Lo mismo si el **control de la corrupción** (indicador WGI `CC.EST`) está en el percentil inferior al 15%. Estos overrides son una salvaguarda: aseguran que un país con problemas institucionales severos no pueda terminar como "oportunidad alta" porque sus otras variables sean buenas.

## Cinco rankings, cinco señales: el resultado diferenciado por línea

Como recordatorio: TOPSIS se ejecuta una vez por cada línea con su `weight_profile` correspondiente. Eso produce cinco scores por país. Y el motor de señales se aplica cinco veces sobre cada uno de esos scores. El resultado final es una **matriz país × línea con etiquetas categóricas**, que es justamente lo que el Comité necesita para tomar decisiones: en lugar de ver "Panamá tiene C* de 0.62 para PT y 0.48 para BD", ve "Panamá es ALTA para PT y MODERADA para BD".


```python
# Ejemplo de matriz de senales con los cinco paises del notebook (sinteticos)
# Para no inventar pesos por linea complejos en este ejemplo, asumimos scores hipoteticos
np.random.seed(42)
scores_por_linea = pd.DataFrame({
    'IB':  [0.55, 0.45, 0.78, 0.62, 0.85],   # Colombia, Mexico, Chile, Panama, Espana
    'PF':  [0.50, 0.70, 0.60, 0.82, 0.75],
    'AD':  [0.45, 0.55, 0.40, 0.50, 0.70],
    'BD':  [0.52, 0.68, 0.65, 0.48, 0.80],
    'CIB': [0.48, 0.72, 0.75, 0.58, 0.88],
}, index=['COL', 'MEX', 'CHL', 'PAN', 'ESP'])

def clasificar_senal(percentil):
    if percentil >= 0.70: return 'ALTA_OPORTUNIDAD'
    if percentil >= 0.45: return 'OPORTUNIDAD_MODERADA'
    if percentil >= 0.25: return 'BAJA_OPORTUNIDAD'
    return 'RIESGO'

senales = pd.DataFrame(index=scores_por_linea.index, columns=scores_por_linea.columns)
for linea in scores_por_linea.columns:
    percentiles = scores_por_linea[linea].rank(pct=True)
    senales[linea] = percentiles.apply(clasificar_senal)

print('📡 Matriz de señales país × línea de negocio:')
display(senales)

```

    📡 Matriz de señales país × línea de negocio:
    


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
      <th>COL</th>
      <td>BAJA_OPORTUNIDAD</td>
      <td>RIESGO</td>
      <td>BAJA_OPORTUNIDAD</td>
      <td>BAJA_OPORTUNIDAD</td>
      <td>RIESGO</td>
    </tr>
    <tr>
      <th>MEX</th>
      <td>RIESGO</td>
      <td>OPORTUNIDAD_MODERADA</td>
      <td>ALTA_OPORTUNIDAD</td>
      <td>ALTA_OPORTUNIDAD</td>
      <td>OPORTUNIDAD_MODERADA</td>
    </tr>
    <tr>
      <th>CHL</th>
      <td>ALTA_OPORTUNIDAD</td>
      <td>BAJA_OPORTUNIDAD</td>
      <td>RIESGO</td>
      <td>OPORTUNIDAD_MODERADA</td>
      <td>ALTA_OPORTUNIDAD</td>
    </tr>
    <tr>
      <th>PAN</th>
      <td>OPORTUNIDAD_MODERADA</td>
      <td>ALTA_OPORTUNIDAD</td>
      <td>OPORTUNIDAD_MODERADA</td>
      <td>RIESGO</td>
      <td>BAJA_OPORTUNIDAD</td>
    </tr>
    <tr>
      <th>ESP</th>
      <td>ALTA_OPORTUNIDAD</td>
      <td>ALTA_OPORTUNIDAD</td>
      <td>ALTA_OPORTUNIDAD</td>
      <td>ALTA_OPORTUNIDAD</td>
      <td>ALTA_OPORTUNIDAD</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Heatmap categorico de las senales
emoji_map = {'ALTA_OPORTUNIDAD': '🟢', 'OPORTUNIDAD_MODERADA': '🟡',
             'BAJA_OPORTUNIDAD': '🟠', 'RIESGO': '🔴'}
codigo_map = {'ALTA_OPORTUNIDAD': 4, 'OPORTUNIDAD_MODERADA': 3,
              'BAJA_OPORTUNIDAD': 2, 'RIESGO': 1}
senales_num = senales.replace(codigo_map).astype(int)
senales_text = senales.apply(lambda c: c.map(lambda v: emoji_map[v]))

fig = go.Figure(data=go.Heatmap(
    z=senales_num.values,
    x=senales_num.columns, y=senales_num.index,
    text=senales_text.values, texttemplate='%{text}',
    textfont=dict(size=24),
    colorscale=[[0, CIBEST['red']], [0.33, CIBEST['amber']],
                [0.66, CIBEST['gold']], [1, CIBEST['green']]],
    showscale=False,
    hovertemplate='<b>%{y}</b> · <b>%{x}</b><br>Señal: %{text}<extra></extra>',
))
fig.update_layout(
    title='<b>Heatmap de señales · país × línea de negocio</b><br>'
          '<sub>🟢 Alta · 🟡 Moderada · 🟠 Baja · 🔴 Riesgo</sub>',
    xaxis=dict(title='Línea de negocio', side='top'),
    yaxis=dict(title='País', autorange='reversed'),
    height=400, width=700,
    plot_bgcolor=CIBEST['white'], paper_bgcolor=CIBEST['white'],
    font=dict(family='Arial', color=CIBEST['gray']),
)
fig.show()

```



Esta es la salida más usada del dashboard ejecutivo. Permite ver, de un solo golpe de vista, qué países son prioritarios para qué líneas. Las narrativas automáticas que se generan en la fase B del proyecto (mayo-junio 2026) toman esta matriz como insumo principal y la enriquecen con contexto cualitativo de fuentes externas.

---

# Capítulo 9 — Validación: análisis de sensibilidad y TOPSIS vs VIKOR

## ¿Por qué validar?

Un ranking que cambia drásticamente cuando los pesos cambian un poco es un ranking frágil, y un ranking frágil no es base para una decisión estratégica de largo plazo. La validación de RADAR Cibest tiene dos componentes complementarios: **análisis de sensibilidad** y **validación cruzada con un método alternativo**.

## Análisis de sensibilidad: perturbar pesos y ver qué cambia

La idea del análisis de sensibilidad es muy simple: tomamos los pesos óptimos producidos por BWM y los perturbamos sistemáticamente. Por ejemplo, multiplicamos el peso de la dimensión Macro por 0.80 (lo reducimos 20%) y renormalizamos para que la suma siga siendo 1; recalculamos TOPSIS con esos pesos perturbados; comparamos el ranking resultante con el original. Si los rankings son muy similares (correlación de Spearman ≥ 0.85) y los top-N comparten al menos 70-80% de países, el ranking es *robusto* ante esa perturbación. Si los rankings se mezclan drásticamente, el ranking es *frágil* en esa dirección y conviene reportar el hallazgo a la dirección.

El sistema repite este ejercicio para cada dimensión y para varios niveles de perturbación (±10%, ±20%), produciendo una tabla resumen que muestra qué tan robusto es el ranking en cada dirección.


```python
# Simulacion del analisis de sensibilidad sobre el ejemplo
def topsis_score(matriz_norm, pesos):
    """TOPSIS abreviado para la simulacion."""
    ponderada = matriz_norm.copy()
    for c in ponderada.columns:
        ponderada[c] = ponderada[c] * pesos[c]
    A_pos = ponderada.max()
    A_neg = ponderada.min()
    d_p = np.sqrt(((ponderada - A_pos) ** 2).sum(axis=1))
    d_n = np.sqrt(((ponderada - A_neg) ** 2).sum(axis=1))
    return d_n / (d_p + d_n)

# Pesos base (uniformes para el ejemplo)
pesos_base = {c: 1/len(matriz_norm.columns) for c in matriz_norm.columns}
score_base = topsis_score(matriz_norm, pesos_base)

# Perturbar cada variable +/- 20%
resultados_sens = []
for var in matriz_norm.columns:
    for pert in [0.80, 0.90, 1.10, 1.20]:
        pesos_p = pesos_base.copy()
        pesos_p[var] = pesos_p[var] * pert
        s = sum(pesos_p.values())
        pesos_p = {k: v/s for k, v in pesos_p.items()}
        score_p = topsis_score(matriz_norm, pesos_p)
        corr = score_base.corr(score_p, method='spearman')
        resultados_sens.append({
            'Variable perturbada': var,
            'Perturbación':        f'×{pert:.2f}',
            'Corr. Spearman':      round(corr, 4),
        })

df_sens = pd.DataFrame(resultados_sens)
print('🔬 Resultados del análisis de sensibilidad (5 variables × 4 perturbaciones):')
style_table(df_sens.head(15), gradient_cols=['Corr. Spearman'], gradient_cmap=cmap_custom,
            format_dict={'Corr. Spearman': '{:.4f}'})

```

    🔬 Resultados del análisis de sensibilidad (5 variables × 4 perturbaciones):
    




<style type="text/css">
#T_53054 th {
  background-color: #2C2A28;
  color: #FDD923;
  font-weight: bold;
  text-align: center;
  padding: 8px;
  font-family: Arial, sans-serif;
}
#T_53054 td {
  padding: 6px 10px;
  font-family: Arial, sans-serif;
  border-bottom: 1px solid #D0D0D0;
}
#T_53054 tbody tr:hover {
  background-color: #F5F5F5;
}
#T_53054_row0_col2, #T_53054_row1_col2, #T_53054_row2_col2, #T_53054_row3_col2, #T_53054_row4_col2, #T_53054_row5_col2, #T_53054_row6_col2, #T_53054_row7_col2, #T_53054_row8_col2, #T_53054_row9_col2, #T_53054_row10_col2, #T_53054_row11_col2, #T_53054_row12_col2, #T_53054_row13_col2, #T_53054_row14_col2 {
  background-color: #cccac7;
  color: #000000;
}
</style>
<table id="T_53054">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_53054_level0_col0" class="col_heading level0 col0" >Variable perturbada</th>
      <th id="T_53054_level0_col1" class="col_heading level0 col1" >Perturbación</th>
      <th id="T_53054_level0_col2" class="col_heading level0 col2" >Corr. Spearman</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_53054_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_53054_row0_col0" class="data row0 col0" >gdp_per_capita_ppp</td>
      <td id="T_53054_row0_col1" class="data row0 col1" >×0.80</td>
      <td id="T_53054_row0_col2" class="data row0 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_53054_row1_col0" class="data row1 col0" >gdp_per_capita_ppp</td>
      <td id="T_53054_row1_col1" class="data row1 col1" >×0.90</td>
      <td id="T_53054_row1_col2" class="data row1 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_53054_row2_col0" class="data row2 col0" >gdp_per_capita_ppp</td>
      <td id="T_53054_row2_col1" class="data row2 col1" >×1.10</td>
      <td id="T_53054_row2_col2" class="data row2 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_53054_row3_col0" class="data row3 col0" >gdp_per_capita_ppp</td>
      <td id="T_53054_row3_col1" class="data row3 col1" >×1.20</td>
      <td id="T_53054_row3_col2" class="data row3 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_53054_row4_col0" class="data row4 col0" >inflation_rate</td>
      <td id="T_53054_row4_col1" class="data row4 col1" >×0.80</td>
      <td id="T_53054_row4_col2" class="data row4 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_53054_row5_col0" class="data row5 col0" >inflation_rate</td>
      <td id="T_53054_row5_col1" class="data row5 col1" >×0.90</td>
      <td id="T_53054_row5_col2" class="data row5 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_53054_row6_col0" class="data row6 col0" >inflation_rate</td>
      <td id="T_53054_row6_col1" class="data row6 col1" >×1.10</td>
      <td id="T_53054_row6_col2" class="data row6 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_53054_row7_col0" class="data row7 col0" >inflation_rate</td>
      <td id="T_53054_row7_col1" class="data row7 col1" >×1.20</td>
      <td id="T_53054_row7_col2" class="data row7 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_53054_row8_col0" class="data row8 col0" >financial_system_deposits_gdp</td>
      <td id="T_53054_row8_col1" class="data row8 col1" >×0.80</td>
      <td id="T_53054_row8_col2" class="data row8 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_53054_row9_col0" class="data row9 col0" >financial_system_deposits_gdp</td>
      <td id="T_53054_row9_col1" class="data row9 col1" >×0.90</td>
      <td id="T_53054_row9_col2" class="data row9 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_53054_row10_col0" class="data row10 col0" >financial_system_deposits_gdp</td>
      <td id="T_53054_row10_col1" class="data row10 col1" >×1.10</td>
      <td id="T_53054_row10_col2" class="data row10 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_53054_row11_col0" class="data row11 col0" >financial_system_deposits_gdp</td>
      <td id="T_53054_row11_col1" class="data row11 col1" >×1.20</td>
      <td id="T_53054_row11_col2" class="data row11 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_53054_row12_col0" class="data row12 col0" >regulatory_quality</td>
      <td id="T_53054_row12_col1" class="data row12 col1" >×0.80</td>
      <td id="T_53054_row12_col2" class="data row12 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_53054_row13_col0" class="data row13 col0" >regulatory_quality</td>
      <td id="T_53054_row13_col1" class="data row13 col1" >×0.90</td>
      <td id="T_53054_row13_col2" class="data row13 col2" >1.0000</td>
    </tr>
    <tr>
      <th id="T_53054_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_53054_row14_col0" class="data row14 col0" >regulatory_quality</td>
      <td id="T_53054_row14_col1" class="data row14 col1" >×1.10</td>
      <td id="T_53054_row14_col2" class="data row14 col2" >1.0000</td>
    </tr>
  </tbody>
</table>




Una correlación de Spearman cercana a 1 entre el ranking original y el ranking perturbado significa que las posiciones de los países cambian poco bajo la perturbación: el ranking es robusto en esa dirección. Una correlación que baja de 0.85 sugiere que la dimensión perturbada tiene poder discriminante alto y vale la pena reportar al Comité ese hecho explícitamente para que sepan dónde está la "fragilidad" del modelo.

## Validación cruzada: TOPSIS contra VIKOR

La segunda capa de validación es metodológica: ¿qué pasaría si en lugar de TOPSIS hubiéramos usado un método multicriterio diferente? Si el ranking resultante fuera muy distinto, eso sugeriría que el ranking depende fuertemente de la elección del método y no del fenómeno subyacente, lo cual debilitaría las conclusiones. El sistema usa **VIKOR** (Opricovic y Tzeng, 2004) como método alternativo para esta validación cruzada porque comparte filosofía con TOPSIS (busca el "compromiso" más cercano al ideal) pero usa una métrica de distancia diferente.

VIKOR define dos medidas para cada país: $S_i$ (la "utilidad agregada" que suma las distancias ponderadas al ideal en todas las dimensiones) y $R_i$ (la "máxima distancia" individual al ideal en cualquier dimensión). El score de compromiso $Q_i$ combina ambas:

$$
Q_i = v \cdot \frac{S_i - S^*}{S^- - S^*} + (1-v) \cdot \frac{R_i - R^*}{R^- - R^*}
$$

donde $v$ es un parámetro de estrategia (típicamente 0.5) y $S^*, S^-, R^*, R^-$ son los mejores y peores valores observados. La correlación de Spearman entre el ranking TOPSIS y el ranking VIKOR debe ser **al menos 0.85** para que el resultado se considere robusto metodológicamente. Si fuera más baja, habría que reportar al Comité que la elección del método multicriterio está influyendo en las conclusiones y conviene profundizar.

## Lo que la validación logra y lo que no

Es importante ser honesto sobre los límites de la validación. Lo que la validación **sí** logra es identificar si el ranking es robusto ante cambios menores en los pesos y ante cambios de método. Eso da seguridad de que las conclusiones no son artefactos de decisiones técnicas particulares. Lo que la validación **no** logra es validar que las *variables* elegidas son las correctas o que la *teoría* subyacente es adecuada para el problema de negocio. Esa parte se valida en las fases 1 y 2 de ASUM-DM (entendimiento del negocio y entendimiento de los datos), no en la fase 5 de evaluación. Es por eso que el rigor metodológico de las primeras fases es tan importante como el rigor analítico de las posteriores.

---

# Capítulo 10 — Recapitulación visual del flujo completo

## El recorrido entero, en una sola figura

Hemos llegado al final del notebook. Para cerrar, vale la pena recapitular el flujo completo del sistema en una sola visualización que sintetiza todo lo que hemos visto. La figura siguiente muestra cómo cada técnica que estudiamos en los capítulos anteriores se conecta con las demás y produce, al final, la información accionable que llega al Comité Ejecutivo y a la Junta Directiva.


```python
# Diagrama de capas con la metodologia completa
fig = make_subplots(rows=1, cols=1)

capas = [
    ('1. Project Charter + revisión de literatura',                    0,  CIBEST['gray_bg'],    CIBEST['gray']),
    ('2. Diccionario de variables + extracción de datos',              1,  CIBEST['gray_light'], 'white'),
    ('3. Limpieza · imputación · normalización · dirección',          2,  CIBEST['gray_light'], 'white'),
    ('4a. BWM con panel ejecutivo → pesos de dimensiones y variables', 3,  CIBEST['gray'],       CIBEST['yellow']),
    ('4b. TOPSIS global + cinco TOPSIS por línea de negocio',         4,  CIBEST['gray'],       CIBEST['yellow']),
    ('4c. Modelo gravitacional → IPC',                                 5,  CIBEST['gray'],       CIBEST['yellow']),
    ('4d. Score compuesto: RADAR = α·C* + β·IPC + γ·Tendencia',       6,  CIBEST['gold'],       CIBEST['gray']),
    ('4e. Motor de señales (4 niveles × 5 líneas)',                    7,  CIBEST['gold_dark'],  'white'),
    ('5. Sensibilidad + validación cruzada TOPSIS↔VIKOR',             8,  CIBEST['green'],      'white'),
    ('Entrega: reporte estratégico + dashboard + narrativas IA',       9,  CIBEST['red'],        'white'),
]

for nombre, y, color, text_color in capas:
    fig.add_shape(
        type='rect', x0=0, x1=10, y0=y - 0.4, y1=y + 0.4,
        line=dict(color=CIBEST['gray'], width=1.5),
        fillcolor=color,
    )
    fig.add_annotation(
        x=5, y=y, text=f'<b>{nombre}</b>', showarrow=False,
        font=dict(family='Arial', size=13, color=text_color),
    )

# Flechas entre capas
for y in range(len(capas) - 1):
    fig.add_annotation(
        x=5, y=y + 0.55, ax=5, ay=y + 0.4,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
        arrowcolor=CIBEST['gray'],
    )

fig.update_layout(
    title='<b>RADAR Cibest · arquitectura metodológica completa</b><br>'
          '<sub>De Project Charter (abajo) a entrega ejecutiva (arriba)</sub>',
    xaxis=dict(visible=False, range=[-0.5, 10.5]),
    yaxis=dict(visible=False, range=[-1, len(capas)]),
    plot_bgcolor=CIBEST['white'], paper_bgcolor=CIBEST['white'],
    height=720, width=950, showlegend=False,
    margin=dict(t=80, b=30, l=30, r=30),
)
fig.show()

```



## Las ideas que vale la pena llevarse del notebook

La primera idea es que **RADAR Cibest no es una sola técnica sino una integración cuidadosa de varias técnicas**, cada una eligida para resolver un sub-problema específico. BWM resuelve el problema de capturar el juicio experto de forma estructurada con poca carga cognitiva. TOPSIS resuelve el problema de combinar muchas variables heterogéneas en un ranking interpretable. El modelo gravitacional resuelve el problema de incorporar la dimensión bilateral con Colombia. El motor de señales resuelve el problema de hacer accionable un resultado numérico para audiencias ejecutivas. Sustituir cualquiera de esas piezas por otra cambiaría la naturaleza del sistema.

**¿Por qué un modelo multicriterio hibrido? ¿No es demasiado complejo? ¿Por qué no usar simplemente un enfoque de proximidad (países cercanos a Colombia)?**

El negocio ya es complejo. El error es simplificarlo mal.

Un holding financiero:
opera bajo regulación,
gestiona riesgo,
depende de estructura bancaria,
y, en tres líneas, de adopción digital.

La literatura es clara:

Los modelos de selección de mercados de una sola dimensión fallan sistemáticamente en servicios financieros.
la proximidad reduce fricción, pero no maximiza rentabilidad.

**Nuestro problema no es dónde es más fácil entrar**, sino dónde tenemos mayor probabilidad de generar retornos sostenibles por línea de negocio.

La evidencia muestra que:

- La distancia sí importa, pero explica fricción, *no atractivo económico total*. La distancia reduce flujos, pero no los explica completamente.
- El tamaño del mercado, la calidad institucional y la profundidad financiera explican más varianza en flujos financieros que la cercanía por sí sola. Bancos exitosos como BBVA o Santander combinaron proximidad cultural con mercados grandes y financieramente profundos.

RADAR no elimina la proximidad: la integra como un componente explícito, sin permitir que opaque mercados grandes, rentables o estratégicos.

La segunda idea es que **las cinco líneas de negocio se atienden simultáneamente** gracias a la ejecución múltiple de TOPSIS con perfiles de pesos diferenciados. Esa decisión metodológica responde directamente al Gap 2 identificado en la revisión sistemática de literatura del proyecto, que documentó que ningún sistema publicado de evaluación de mercados produce señales diferenciadas por tipo de negocio financiero. Ese diferencial es propio de RADAR Cibest.

La tercera idea es que **la robustez del ranking se valida formalmente** mediante análisis de sensibilidad y validación cruzada con VIKOR. Esa validación no es un trámite: es lo que da soporte a la dirección para presentar el resultado ante la Junta sabiendo que no se desmoronará ante una pregunta del tipo "¿y si nos equivocamos un poco con los pesos?".

La cuarta idea es que **el sistema es modular y configurable**. Todas las decisiones que se pueden razonablemente discutir (qué variables incluir, qué pesos asignar, cuáles son los umbrales de señal, qué valores tienen $\alpha$, $\beta$ y $\gamma$) están expuestas en archivos YAML editables. Cambiar la estrategia de Cibest hacia una postura más globalista o más regionalista no requiere reescribir código: requiere ajustar un par de números en `config/settings.yaml` y volver a correr el pipeline. Esa modularidad es lo que va a permitir que el sistema sobreviva al ciclo estratégico anual de revisión con la Junta Directiva durante años.

---

## Próximos pasos en el proyecto

Para cerrar, vale la pena recordar dónde estamos en el cronograma del proyecto. La fase A (MVP cuantitativo, marzo-mayo 2026) está prácticamente concluida con la entrega del sistema completo. La fase B (narrativas ejecutivas enriquecidas con LLM, mayo-junio 2026) se inicia en las próximas semanas. La sesión de elicitación BWM con la alta dirección está sujeta a disponibilidad, usando exclusivamente herramientas Microsoft Office.

Gracias por leer este notebook hasta el final.

---

*Notebook 00 — versión 1.0 · Dirección de Estrategia · Grupo Cibest · 2026*
