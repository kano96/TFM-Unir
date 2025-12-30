# Fase 4 — Detección de anomalías en sistemas observables

## 1. Objetivo de la fase

El objetivo de la Fase 4 es implementar, evaluar y comparar distintos enfoques de detección de anomalías aplicados a los datos de métricas y logs generados en fases anteriores. Esta fase se centra en la identificación temprana de comportamientos anómalos asociados a fallos inyectados de forma controlada, evaluando tanto enfoques basados en reglas como modelos de aprendizaje automático no supervisados.

---

## 2. Datos utilizados

Los modelos de detección se entrenan y evalúan utilizando:

- Ventanas temporales agregadas de métricas y logs (1, 5 y 15 minutos).
- Etiquetas de fallo derivadas del proceso de inyección controlada:
  - `run_id`
  - `service`
  - `start_ts`, `end_ts`
  - `fault_type`

Cada ventana se clasifica como **normal** o **anómala** en función de su solapamiento con intervalos de fallo.

---

## 3. Baselines de detección

### 3.1 Detector por umbral fijo

El primer baseline evaluado consiste en un detector basado en reglas simples, que marca una ventana como anómala cuando el valor medio del ratio de errores (`error_ratio_mean_1min`) supera un umbral fijo de 0.1.

Este enfoque se utiliza como referencia mínima, dada su simplicidad y bajo coste computacional.

---

### 3.2 Detector estadístico basado en z-score

Como segundo baseline se implementa un detector estadístico basado en **z-score móvil**, que identifica anomalías cuando la métrica `error_ratio_mean_1min` presenta una desviación significativa respecto a su media histórica (z ≥ 3.0) en una ventana de 12 pasos.

Este método busca mejorar la adaptabilidad respecto al umbral fijo, aunque sigue siendo un enfoque univariante.

---

## 4. Modelo de aprendizaje automático: Isolation Forest

El modelo principal evaluado es **Isolation Forest**, un algoritmo no supervisado que detecta anomalías mediante particiones aleatorias del espacio de características. El modelo se entrena exclusivamente con ventanas normales y produce un score continuo de normalidad, que se transforma en una decisión binaria mediante un umbral configurable.

Parámetros principales:

- `contamination = 0.05`
- Umbral de score: `score < 0.1`

---

## 5. Metodología de evaluación

La evaluación se realiza comparando las predicciones de cada detector con las etiquetas reales de fallo, calculando las siguientes métricas:

- Precisión (Precision)
- Exhaustividad (Recall)
- F1-score
- Área bajo la curva ROC (ROC-AUC)
- Área bajo la curva Precision-Recall (PR-AUC)

Además, se analiza la latencia de detección (MTTD) como métrica temporal clave.

---

## 6. Resultados cuantitativos

La Tabla siguiente resume los resultados obtenidos para cada enfoque evaluado:

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC | Detalles |
| ------ | ----------: | -------: | ---: | --------: | -------: | --------- |
| Baseline umbral | 0.431 | 0.705 | 0.534 | 0.516 | 0.633 | `error_ratio_mean_1min ≥ 0.1` |
| Baseline z-score | 0.000 | 0.000 | 0.000 | 0.551 | 0.666 | `z ≥ 3.0`, ventana 12 |
| Isolation Forest | 0.000 | 0.000 | 0.000 | 0.572 | 0.662 | `score_thr=0.1`, contamination=0.05 |

---

## 7. Análisis comparativo

### 7.1 Comparativa Baseline vs Isolation Forest

Los resultados muestran diferencias claras entre los enfoques evaluados:

- El **baseline por umbral fijo** obtiene el mejor F1-score, destacando por su alto recall, aunque con una precisión moderada, lo que indica una mayor tasa de falsos positivos.
- El **baseline estadístico (z-score)** no logra identificar correctamente ventanas anómalas en este escenario experimental, a pesar de mostrar valores de ROC-AUC y PR-AUC razonables, lo que indica una separación parcial en el score pero una mala calibración del umbral.
- **Isolation Forest**, en la configuración evaluada, tampoco alcanza una clasificación binaria efectiva, aunque presenta métricas ROC-AUC y PR-AUC superiores al baseline por umbral, lo que sugiere que el modelo aprende una estructura latente útil pero requiere un ajuste más fino del umbral o del preprocesamiento.

Estos resultados evidencian que, si bien los modelos no supervisados capturan información relevante, la **selección del umbral de decisión es crítica** para su desempeño operativo.

---

## 8. Análisis de latencia de detección (MTTD)

La latencia de detección se define como el tiempo transcurrido entre el inicio real de un fallo y la primera ventana marcada como anómala.

Observaciones principales:

- El **baseline por umbral** tiende a detectar fallos rápidamente cuando el error ratio supera el umbral, lo que explica su alto recall.
- Sin embargo, esta rapidez se logra a costa de una menor precisión.
- Los modelos basados en scores continuos (z-score e Isolation Forest) requieren un ajuste adicional para optimizar la latencia sin degradar la precisión.

En conjunto, los resultados muestran que existe un **trade-off claro entre rapidez de detección y fiabilidad**, especialmente relevante en entornos AIOps.

---

## 9. Discusión

Los experimentos realizados permiten extraer varias conclusiones relevantes:

- Los enfoques basados en reglas siguen siendo competitivos como baseline fuerte en escenarios controlados.
- Los modelos no supervisados requieren:
  - Normalización adecuada de features.
  - Calibración explícita de umbrales.
  - Posiblemente estrategias semi-supervisadas.
- Métricas como ROC-AUC y PR-AUC son insuficientes por sí solas para evaluar detectores operativos; es imprescindible considerar métricas de decisión binaria y latencia.

---

## 10. Conclusiones de la Fase 4

La Fase 4 demuestra que la detección de anomalías en sistemas observables es un problema multifactorial que no puede resolverse únicamente mediante métricas agregadas o reglas simples. Aunque los baselines ofrecen un punto de partida sólido, los modelos de aprendizaje automático proporcionan una base más flexible y extensible, especialmente cuando se integran en pipelines AIOps de mayor complejidad.

Estos resultados motivan la siguiente fase del trabajo, centrada en la **predicción y análisis de causa raíz**, donde la información temporal y estructural adquirirá un papel aún más relevante.
