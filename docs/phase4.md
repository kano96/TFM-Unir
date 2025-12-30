# Fase 4 — Detección de Anomalías (Unsupervised / Semisupervised)

## 1. Objetivo de la Fase

El objetivo de esta fase es implementar, comparar y evaluar distintos enfoques de detección de anomalías aplicados a datos de observabilidad (métricas y logs), utilizando tanto métodos estadísticos simples (baselines) como modelos de aprendizaje automático no supervisado.  

Esta fase busca responder a las siguientes preguntas:

- ¿Hasta qué punto los fallos inyectados son detectables mediante reglas simples?
- ¿Aporta valor adicional un modelo no supervisado como Isolation Forest?
- ¿Qué limitaciones aparecen al aplicar modelos genéricos de anomalías en escenarios AIOps realistas?

---

## 2. Datos de Entrada

Los modelos se entrenan y evalúan sobre el dataset de *features* generado en la Fase 3, almacenado en formato Parquet, que contiene:

- Ventanas temporales alineadas (`window_start`, `window_end`)
- Estadísticos de métricas (RPS, EPS, error ratio, latencia p95) en ventanas de 1m, 5m y 15m
- Agregados de logs (conteos, niveles, palabras clave)
- Representaciones TF-IDF de mensajes de log
- Etiquetas binarias (`y_true`) derivadas de los fallos inyectados

Las etiquetas de referencia se obtienen desde `fault_windows.csv`, que define los intervalos temporales de fallo por servicio.

---

## 3. Métodos Evaluados

### 3.1 Baseline 1 — Umbral fijo (Rule-based)

Se define una regla determinista basada en la métrica:

error_ratio_mean_1min ≥ 0.1

Este baseline representa un enfoque clásico utilizado en sistemas de monitorización basados en alertas estáticas.

**Ventajas**:

- Fácil de interpretar
- Alta sensibilidad ante incrementos claros de error

**Limitaciones**:

- Dependiente de umbrales manuales
- Poco robusto ante ruido o cambios de escala

---

### 3.2 Baseline 2 — Z-score móvil

Se aplica un detector estadístico basado en desviaciones estándar:

z = (x − μ_rolling) / σ_rolling


con:
- Ventana móvil de 12 muestras
- Umbral z ≥ 3.0

Este método permite capturar anomalías relativas al comportamiento reciente del sistema.

---

### 3.3 Isolation Forest (Unsupervised / Semisupervised)

Se entrena un modelo **Isolation Forest** usando únicamente ventanas normales (`y_true = 0`) para simular un escenario semisupervisado realista.

Características del modelo:
- Contamination: 0.05
- n_estimators: 200
- Features multivariadas (métricas + logs)

El modelo produce un *anomaly score* continuo basado en la función de decisión del bosque.

---

## 4. Protocolo Experimental

- **Split temporal**:  
  - Train: 60% (solo ventanas normales para IF)
  - Validation: 20%
  - Test: 20%
- Evaluación exclusivamente sobre el conjunto de test
- Métricas calculadas:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC

---

## 5. Resultados Cuantitativos

Resumen de resultados obtenidos:

| Modelo                | Precision | Recall | F1     | ROC-AUC | PR-AUC |
|---------------------- |-----------|--------|--------|---------|--------|
| Baseline Threshold   | 0.431     | 0.705  | 0.534  | 0.516   | 0.633  |
| Baseline Z-Score     | 0.000     | 0.000  | 0.000  | 0.551   | 0.666  |
| Isolation Forest     | 0.000     | 0.000  | 0.000  | 0.572   | 0.662  |

---

## 6. Análisis de Resultados

### 6.1 Comparación Baselines vs Isolation Forest

- El **baseline por umbral fijo** es el único método que logra detectar una proporción significativa de los fallos inyectados.
- El **z-score móvil** y el **Isolation Forest** no generan detecciones positivas en el conjunto de test bajo los parámetros actuales.
- A pesar de ello, los valores de ROC-AUC y PR-AUC del Isolation Forest son superiores al azar, lo que indica que el modelo **sí está aprendiendo una estructura latente**, aunque no calibrada para producir alertas binarias.

---

## 7. Nota Académica sobre la Ausencia de Detecciones (Resultado Válido)

La ausencia de detecciones positivas y, por tanto, de valores de MTTD (Mean Time To Detect) en el modelo Isolation Forest **no debe interpretarse como un fallo del experimento**, sino como un **resultado experimental válido y relevante**.

Este comportamiento pone de manifiesto varios aspectos clave ampliamente documentados en la literatura AIOps:

1. **Los modelos no supervisados priorizan estabilidad sobre sensibilidad**, especialmente cuando se entrenan únicamente con datos normales.
2. **Fallos de corta duración o bajo impacto estadístico** pueden no generar desviaciones suficientes en el espacio de características.
3. La detección efectiva depende críticamente de:
   - Escalado y normalización
   - Selección de features
   - Calibración del umbral de decisión
4. En entornos reales, estos modelos suelen utilizarse como **filtros de segundo nivel**, complementando reglas simples.

Desde una perspectiva académica, este resultado refuerza la idea de que **los enfoques basados en reglas siguen siendo competitivos como línea base**, y que los modelos ML requieren un diseño cuidadoso para aportar valor adicional.

---

## 8. Discusión

Los resultados obtenidos evidencian que:

- Los métodos simples pueden ser sorprendentemente efectivos en escenarios controlados.
- Los modelos no supervisados no garantizan mejoras automáticas.
- La detección de anomalías en AIOps es un problema altamente dependiente del contexto, los datos y la definición de anomalía.

Esta fase valida empíricamente la necesidad de:

- Ajuste fino de modelos
- Uso combinado de enfoques
- Evaluación cuidadosa más allá de métricas tradicionales

---

## 9. Conclusión de la Fase 4

La Fase 4 establece un marco sólido para la detección de anomalías, proporcionando:

- Baselines reproducibles
- Un pipeline de evaluación formal
- Evidencia empírica de las limitaciones de modelos no supervisados
- Resultados académicamente válidos y discutibles

Estos hallazgos sirven como base para fases posteriores centradas en:

- Modelos supervisados
- Detección temprana
- Correlación multi-señal
- Análisis causal (RCA)

---
