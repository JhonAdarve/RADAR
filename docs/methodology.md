# Documentacion Metodologica - RADAR Cibest

## 1. Marco general

RADAR Cibest implementa un enfoque multicriteria hibrido que combina tres
componentes seleccionados a partir de una revision sistematica de literatura
con 43 referencias verificadas:

1. **Componente de ponderacion** - BWM (Best-Worst Method)
2. **Componente de ranking** - TOPSIS
3. **Componente de proximidad** - Modelo gravitacional adaptado

Cada componente responde a una sub-pregunta especifica del problema de
seleccion de mercados: que importa, que tan bueno es cada mercado, y que
tan accesible es desde Colombia.

## 2. Componente de ponderacion: BWM

### 2.1. Justificacion de la seleccion

La revision sistematica compare 9 tecnicas MCDM (AHP, TOPSIS, VIKOR,
PROMETHEE, BWM, ELECTRE, Fuzzy AHP-TOPSIS, DEA, BWM-TOPSIS hibrido).
BWM fue seleccionado como metodo de ponderacion por:

- **Eficiencia cognitiva:** 2n-3 comparaciones vs n(n-1)/2 del AHP. Para
  5 dimensiones, son 7 comparaciones en lugar de 10.
- **Consistencia estadistica superior** documentada en Rezaei (2015) y
  Kumar et al. (2021).
- **Aplicabilidad al panel ejecutivo:** una sesion de 90 minutos basta
  para completar la elicitacion de cada ejecutivo.

### 2.2. Estructura de juicios

Para cada ejecutivo se solicita:

1. Identificar la dimension MEJOR (mas importante)
2. Identificar la dimension PEOR (menos importante)
3. Comparar la mejor con todas las otras (vector Best-to-Others)
4. Comparar todas las otras con la peor (vector Others-to-Worst)

Las comparaciones usan escala entera 1-9 (Saaty).

### 2.3. Resolucion del modelo

El modelo de optimizacion linealizado minimiza el maximo absoluto:

```
min xi
s.a. |w_B - a_Bj * w_j| <= xi   para todo j
     |w_j - a_jW * w_W| <= xi   para todo j
     sum(w) = 1
     w_j >= 0
```

Implementado con SLSQP (`scipy.optimize.minimize`).

### 2.4. Validacion de consistencia

```
CR = xi* / CI(a_BW)
```

Donde `CI(a_BW)` se consulta en la tabla tabulada de Rezaei (2015) segun
la comparacion entre el mejor y el peor criterio.

Aceptacion: `CR < 0.10`. Por encima de este umbral el ejecutivo es
solicitado a revisar sus juicios.

### 2.5. Agregacion del panel

Pesos individuales agregados con **media geometrica** (estandar en
literatura MCDM, preserva razones).

### 2.6. Pesos jerarquicos finales

Para cada variable v en dimension d:

```
w_v_final = w_d * w_v_dentro_de_d
```

Garantizando `sum(w_v_final) = 1` sobre todas las variables.

## 3. Componente de ranking: TOPSIS

### 3.1. Justificacion

- Metodo MCDM mas usado en literatura financiera (Cernevicience &
  Kabasinskas, 2022)
- Maneja sin restriccion 30+ alternativas
- Compatible directamente con pesos externos del BWM
- Validado en aplicaciones bancarias (Mandic et al., 2014; Dincer &
  Hacioglu, 2015)

### 3.2. Algoritmo

1. **Matriz de decision normalizada** R con todas las variables ya
   orientadas (mayor = mejor).
2. **Matriz ponderada** V donde `V_ij = w_j * R_ij`.
3. **Solucion ideal positiva** A+ = max por columna.
4. **Solucion ideal negativa** A- = min por columna.
5. **Distancias euclidianas** de cada pais a A+ y A-.
6. **Closeness Coefficient:** `C* = d- / (d+ + d-)`, en [0, 1].
7. **Ranking** descendente por C*.

### 3.3. Scores parciales por dimension

CRITICO para el motor de senales: TOPSIS se ejecuta tambien sobre cada
subconjunto de variables de la misma dimension, generando un score
parcial por dimension que alimenta el perfil de fortalezas/debilidades.

### 3.4. Multiples ejecuciones por linea de negocio

Para generar senales diferenciadas, TOPSIS se ejecuta cinco veces
adicionales con vectores de pesos especificos por linea (definidos en
`business_lines.yaml > weight_profile`). Computacionalmente trivial y
mantiene la matriz de datos unica.

## 4. Componente de proximidad: modelo gravitacional

### 4.1. Justificacion

- Brei & von Peter (2018) demuestran que duplicar la distancia reduce
  flujos bancarios 30-50%.
- Papaioannou (2009) identifica la calidad institucional como el
  predictor mas fuerte de flujos financieros.
- Ghemawat (2001) muestra que los servicios financieros son
  particularmente sensibles a distancia cultural e institucional.

### 4.2. Indice de Proximidad Compuesto (IPC)

Formula:

```
IPC(j) = sum_k w_k * proximidad_k(COL, j)
```

Donde k son los componentes:
- distancia geografica (negativa, log)
- distancia cultural Hofstede (negativa)
- idioma compartido (positiva, dummy)
- comercio bilateral (positiva, log)
- stock de diaspora colombiana (positiva, log)

Cada componente se normaliza a [0, 1] aplicando su direccion antes de
ponderar. El pais origen (Colombia) tiene IPC = 1 por definicion.

## 5. Score RADAR compuesto

```
RADAR(j, l) = alpha * CC_TOPSIS(j, l) + beta * IPC(j) + gamma * Tendencia(j)
```

Pesos por defecto (configurables en `settings.yaml`):

- `alpha = 0.55` - peso del atractivo absoluto
- `beta = 0.30` - peso de la afinidad bilateral
- `gamma = 0.15` - peso del dinamismo reciente

`Tendencia(j)` es por defecto el promedio normalizado del crecimiento
real del PIB en los ultimos 3 anos. Puede extenderse a un compuesto de
varias variables.

## 6. Motor de senales

Para cada par (pais, linea) se asigna una etiqueta de cuatro niveles:

| Etiqueta | Condicion |
|---|---|
| ALTA_OPORTUNIDAD | percentil del score por linea >= 70 |
| OPORTUNIDAD_MODERADA | 45 <= percentil < 70 |
| BAJA_OPORTUNIDAD | 25 <= percentil < 45 |
| RIESGO | percentil < 25 o overrides criticos |

**Overrides de riesgo:** un pais con estabilidad politica o control de
corrupcion en el percentil 15 inferior se marca como RIESGO sin importar
el score numerico.

## 7. Validacion del modelo

Tres niveles recomendados (implementados en `src/scoring/sensitivity.py`):

1. **Consistencia BWM** - CR < 0.10 por ejecutivo
2. **Sensibilidad de pesos** - perturbar +/-20% cada dimension y verificar
   que el top-10 mantenga al menos 70% de solapamiento
3. **Validacion cruzada TOPSIS-VIKOR** - correlacion de Spearman entre
   ambos rankings debe ser >= 0.85

## 8. Limitaciones reconocidas

- Modelo compensatorio: una variable muy alta puede compensar otra muy
  baja. Mitigado parcialmente con los overrides de riesgo.
- Snapshot temporal: el modelo se actualiza periodicamente, no en
  tiempo real.
- Dependencia de juicio experto: la calidad de los pesos BWM determina
  la calidad del ranking. Mitigada con la agregacion del panel y la
  validacion de consistencia.

## 9. Referencias clave

- Anderson, J.E. & van Wincoop, E. (2003). Gravity with gravitas. *AER*.
- Brei, M. & von Peter, G. (2018). The distance effect in banking.
  *JIMF*.
- Cernevicience, J. & Kabasinskas, A. (2022). Review of MCDM in finance.
  *Frontiers in AI*.
- Hwang, C.L. & Yoon, K. (1981). *Multiple Attribute Decision Making*.
- Kaufmann, D. & Kraay, A. (2024). WGI methodology. *World Bank*.
- Papaioannou, E. (2009). What drives international financial flows.
  *JDE*.
- Rezaei, J. (2015). Best-worst multi-criteria decision-making method.
  *Omega*.
- Saaty, T.L. (1980). *The Analytic Hierarchy Process*.
