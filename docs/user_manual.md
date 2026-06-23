# Manual de Usuario - Dashboard RADAR Cibest

## Acceso

Inicie el dashboard desde la raiz del proyecto:

```bash
streamlit run src/dashboard/app.py
```

Por defecto se abre en `http://localhost:8501` en su navegador.

## Pagina 1 - Ranking General

Vista de entrada al sistema. Muestra:

- **Mapa coropletico** de America + Espana coloreado por score RADAR global.
  Mayor intensidad de color dorado indica mayor atractivo.
- **Top N** del ranking en grafico de barras horizontal.
- **Tabla detallada** ordenable y descargable como CSV.

**Filtros laterales:**
- Region: filtra el conjunto de paises mostrados.
- Top N: ajusta el numero de paises en el grafico de barras.

## Pagina 2 - Perfil de Pais

Diagnostico individual de un mercado. Permite:

- Seleccionar pais por nombre.
- Ver KPIs principales (rank y score).
- Comparar el perfil dimensional del pais con la mediana regional via
  **radar chart**.
- Ver gauges de score por cada linea de negocio.
- Leer la **narrativa ejecutiva** auto-generada.
- Detalle dimensional con brechas vs mediana regional.

## Pagina 3 - Senales por Linea de Negocio

Vision focalizada en una linea de negocio. Para la linea seleccionada
(IB / PF / AD / BD / CIB):

- Mapa coropletico de senales con codigo de colores:
  - Verde: ALTA_OPORTUNIDAD
  - Amarillo: OPORTUNIDAD_MODERADA
  - Naranja: BAJA_OPORTUNIDAD
  - Rojo: RIESGO
- Top 5 mercados con justificacion narrativa.
- Tabla completa descargable.

## Pagina 4 - Simulador Que-Pasa-Si

Permite explorar el impacto de cambiar los pesos:

- Sliders para los pesos de las 5 dimensiones.
- Sliders para alpha, beta y gamma (componentes del score compuesto).
- Comparacion lado a lado: ranking original vs simulado.
- Destacado en dorado de paises que **suben** mas de 3 posiciones.
- Destacado en rosa de paises que **bajan** mas de 3 posiciones.
- Top 5 mayores ascensos y descensos.

Util durante la sesion de elicitacion BWM y para responder preguntas
estrategicas en tiempo real.

## Pagina 5 - Tendencias Historicas

Series de tiempo por variable y pais. Permite:

- Seleccionar variable (organizada por dimension).
- Comparar hasta 6 paises simultaneamente.
- Ver alertas de variaciones >10% YoY en el ultimo ano disponible.

## Tips operativos

- Use Ctrl+Shift+R (Cmd+Shift+R en Mac) para forzar recarga si los
  resultados parecen desactualizados.
- Los archivos de resultados se cargan automaticamente del Parquet mas
  reciente en `data/scores/`. Si no encuentra resultados, ejecute
  primero `python -m src.scoring.hybrid_scorer`.
- Todas las tablas son descargables en CSV.
- Los graficos Plotly tienen su propio menu de exportacion (camera icon).
