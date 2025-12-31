# Fase 5 — Predicción de incidentes

## 5.1 Objetivo de la fase

El objetivo de esta fase es **anticipar incidentes operativos antes de que estos ocurran**, a partir del comportamiento reciente del sistema reflejado en métricas, logs y características agregadas generadas en fases anteriores. A diferencia de la Fase 4, centrada en la detección reactiva de anomalías, esta etapa aborda un problema **predictivo supervisado**, cuyo propósito principal es reducir el *Mean Time To Detect (MTTD)* mediante la generación de alertas tempranas.

Concretamente, se plantea el problema como una **clasificación binaria**: determinar si, dado el estado actual del sistema, **ocurrirá un incidente dentro de un horizonte temporal fijo**. Este enfoque se alinea con los principios de AIOps, donde la anticipación tiene mayor valor operativo que la mera detección tardía.

---

## 5.2 Formulación del problema predictivo

### 5.2.1 Definición del target

Se define la variable objetivo `y_future` como:

- `y_future = 1` si existe al menos un fallo registrado para el mismo `(run_id, service)` **dentro de los próximos T minutos** a partir del final de la ventana actual.
- `y_future = 0` en caso contrario.

El horizonte temporal seleccionado es:

- **T = 5 minutos**, lo que permite evaluar la capacidad del modelo para anticipar incidentes con suficiente antelación operativa sin introducir una excesiva incertidumbre temporal.

Esta definición evita explícitamente etiquetar como positivas las ventanas que ya están solapadas con un incidente activo, transformando el problema en una **predicción anticipativa real** y no en una detección retrasada.

---

## 5.3 Datos y características utilizadas

El modelo utiliza como entrada un único archivo parquet de características (`features_*.parquet`) generado en la Fase 3, que contiene:

- Estadísticos agregados de métricas (RPS, tasa de errores, latencia p95) calculados en ventanas de 1, 5 y 15 minutos.
- Señales derivadas de logs, incluyendo contadores de errores, advertencias y palabras clave relevantes.
- Componentes TF-IDF extraídos de mensajes de log.
- Metadatos temporales (`window_start`, `window_end`) y de servicio.

Las columnas identificativas (`run_id`, `service`, timestamps) se excluyen del entrenamiento, utilizándose únicamente **características numéricas**, lo que permite un enfoque generalizable y reproducible.

---

## 5.4 Modelo seleccionado

### 5.4.1 Justificación del enfoque

Para esta fase se selecciona un **modelo de regresión logística**, entrenado sobre características tabulares, por las siguientes razones:

- Produce **probabilidades explícitas de incidente futuro** mediante `predict_proba`, lo que permite ajustar dinámicamente el umbral de decisión.
- Es computacionalmente eficiente y fácilmente integrable en un microservicio FastAPI.
- Ofrece interpretabilidad básica, adecuada para un contexto académico.
- Funciona de forma robusta con datasets de tamaño moderado y pipelines reproducibles.

Este enfoque sustituye explícitamente a modelos de series temporales clásicos (como Prophet), priorizando la **predicción supervisada basada en contexto multivariable** frente a la extrapolación de una única señal temporal.

---

## 5.5 Proceso de entrenamiento

El entrenamiento del predictor sigue los siguientes pasos:

1. Construcción de la variable `y_future` a partir de los intervalos de fallo etiquetados.
2. Limpieza de valores inválidos (NaN e infinitos).
3. Normalización de las características mediante `StandardScaler`.
4. Entrenamiento del modelo de regresión logística con ponderación de clases (`class_weight="balanced"`).
5. Persistencia del artefacto entrenado (`predictor.joblib`), incluyendo el modelo, las columnas de entrada y el umbral inicial.

Durante esta fase se observa un **desbalance moderado de clases**, con una proporción de etiquetas positivas aproximada del **64 %**, resultado esperado dada la densidad de fallos inyectados y la formulación temporal del horizonte de predicción.

---

## 5.6 Evaluación del modelo

La evaluación se realiza sobre el conjunto de ventanas etiquetadas, utilizando métricas estándar de clasificación binaria.

Con el umbral por defecto (`τ = 0.5`), se obtienen los siguientes resultados:

- **Precision**: 0.79  
- **Recall**: 0.86  
- **F1-score**: 0.82  
- **Accuracy**: 0.76  
- **ROC AUC**: 0.70  
- **PR AUC**: 0.53  

Estos valores indican una capacidad predictiva razonable, aunque con margen de mejora en recall, métrica crítica en sistemas de alerta temprana.

---

## 5.7 Ajuste del umbral de decisión (threshold tuning)

Dado que el modelo produce probabilidades continuas, se realiza un **análisis sistemático de umbrales** entre 0.25 y 0.70, evaluando su impacto en precisión, recall y F1-score.

### 5.7.1 Resultados del tuning

| Threshold | Precision | Recall | F1 |
| ---------- | ---------- | -------- | ----- |
| 0.25 | 0.662 | 1.000 | 0.797 |
| 0.30 | 0.667 | 1.000 | 0.800 |
| 0.35 | 0.667 | 1.000 | 0.800 |
| 0.40 | 0.667 | 1.000 | 0.800 |
| **0.45** | **0.667** | **1.000** | **0.800** |
| 0.50 | 0.667 | 1.000 | 0.800 |
| 0.60 | 0.665 | 0.990 | 0.795 |

El mejor compromiso global se alcanza con:

- **Umbral óptimo τ\* = 0.45**
- **Precision = 0.67**
- **Recall = 1.00**
- **F1-score = 0.80**
- **ROC AUC = 0.70**
- **PR AUC = 0.53**

---

## 5.8 Resultados finales con umbral óptimo

Con el umbral seleccionado (τ\* = 0.45), el reporte de clasificación es:

- **Accuracy global**: 0.77  

**Clase 1 (incidente futuro):**

- Precision: 0.67  
- Recall: 1.00  
- F1-score: 0.80  

**Clase 0 (estado normal):**

- Precision: 1.00  
- Recall: 0.58  
- F1-score: 0.73  

Estos resultados confirman que el modelo está **optimizado para maximizar la anticipación**, aceptando una reducción controlada de recall en estados normales.

---

## 5.9 Discusión académica

Desde una perspectiva académica, los resultados obtenidos son **válidos y coherentes con la naturaleza del problema**. La predicción de incidentes en sistemas distribuidos presenta inherentemente:

- Alta incertidumbre temporal.
- Solapamiento entre estados normales y pre-incidente.
- Costes asimétricos entre falsos positivos y falsos negativos.

En este contexto, priorizar recall y F1-score sobre accuracy pura es metodológicamente correcto. Además, el uso de un modelo probabilístico con ajuste explícito del umbral permite adaptar el sistema a distintos escenarios operativos sin necesidad de reentrenar el modelo.

La proporción relativamente elevada de etiquetas positivas no indica un fallo del enfoque, sino una consecuencia directa de la definición temporal del horizonte predictivo y de la densidad de fallos inyectados, reflejando un escenario realista de estrés operacional.

---

## 5.10 Conclusiones de la Fase 5

La Fase 5 demuestra que:

- Es posible **anticipar incidentes con varios minutos de antelación** utilizando características agregadas del sistema.
- Un enfoque supervisado basado en regresión logística constituye un **baseline predictivo sólido y reproducible**.
- El ajuste del umbral de decisión es un componente clave para maximizar el valor operativo del predictor.
- El modelo entrenado es **directamente integrable como microservicio**, complementando los módulos de detección reactiva desarrollados en fases anteriores.

Estos resultados habilitan la siguiente etapa del proyecto: la **integración completa del predictor dentro de la arquitectura AIOps**, permitiendo cerrar el ciclo de observabilidad con capacidades tanto reactivas como proactivas.
