# Fase 3 — Análisis Exploratorio de Datos (EDA) y Feature Engineering

## 1. Objetivo de la Fase 3

El objetivo de la Fase 3 es comprender en profundidad los datos generados durante la Fase 1 y canalizados en la Fase 2, así como preparar un conjunto de características (features) robustas y reproducibles que sirvan como entrada para los modelos de detección, predicción y análisis de causa raíz (RCA) en fases posteriores.

Esta fase se centra en tres fuentes principales de observabilidad:

- **Métricas** (Prometheus)
- **Logs** (Loki)
- **Trazas** (Jaeger)

---

## 2. Análisis Exploratorio de Datos (EDA)

### 2.1 EDA de métricas

El análisis exploratorio de métricas se realiza sobre los datasets exportados desde Prometheus (`data/raw/metrics/`).

En los notebooks de métricas se analizan:

- Series temporales de:
  - Tasa de peticiones (RPS)
  - Tasa de errores (EPS)
  - Ratio de errores
  - Latencia p95
- Comparación entre servicios (`user`, `auth`, `orders`)
- Diferencias entre periodos normales y ventanas con fallos inyectados

**Hallazgos principales:**
- Los fallos de latencia generan incrementos claros en p95.
- Los fallos de errores producen picos sostenidos en EPS y error ratio.
- Las ventanas etiquetadas permiten una separación clara entre comportamiento normal y anómalo.

---

### 2.2 EDA de logs

El EDA de logs se realiza sobre los archivos exportados desde Loki (`data/raw/logs/`).

Se analizan:

- Volumen de logs por servicio
- Distribución de severidad (`INFO`, `WARN`, `ERROR`)
- Evolución temporal del número de errores
- Palabras clave frecuentes en mensajes de error

**Hallazgos principales:**

- Los fallos inyectados producen un aumento significativo de logs `ERROR`.
- Los servicios presentan patrones diferenciados de logging.
- Los mensajes de error contienen vocabulario consistente, útil para técnicas TF-IDF o embeddings.

---

### 2.3 EDA de trazas

El análisis exploratorio de trazas se basa en los datos recolectados por Jaeger.

Se estudian:

- Número de spans por petición
- Duración media de spans
- Latencias de cola (tail latency)
- Relaciones entre servicios (dependencias)

**Hallazgos principales:**

- Durante fallos, aumenta el número medio de spans y su duración.
- Se observan patrones de propagación de latencia entre servicios.
- Las dependencias permiten modelar el sistema como un grafo dirigido.

---

## 3. Feature Engineering

La extracción de características se implementa en el script:

```text
models/features.py
```

### 3.1 Features de métricas

Para cada servicio y ventana temporal se calculan estadísticas agregadas:

Ventanas deslizantes de:

- 1 minuto
- 5 minutos
- 15 minutos

Estadísticos:

- Media
- Desviación estándar
- Máximo
- Mínimo

Estas features capturan cambios abruptos y tendencias progresivas en el comportamiento del sistema.

### 3.2 Features de logs

A partir de los logs estructurados se generan:

- Conteo de logs por ventana
- Conteo de logs ERROR y WARN
- Conteo de palabras clave asociadas a errores
- Representaciones TF-IDF de mensajes de log agregados por ventana

Estas características permiten modelar señales semánticas y volumétricas de incidentes.

### 3.3 Features de trazas

A partir de los datos de trazas se derivan:

- Número de spans por ventana
- Duración media de spans
- Latencia máxima observada
- Features de grafo de dependencias:
  - In-degree
  - Out-degree

Estas features capturan la complejidad estructural y el impacto de fallos en el flujo de peticiones.

## 4. Etiquetado de ventanas

Las ventanas temporales se etiquetan utilizando los fallos inyectados durante la Fase 1:

- incident = 1: ventana que se solapa con un fallo
- incident = 0: ventana sin fallo

Este etiquetado permite entrenar modelos supervisados de detección y predicción.

## 5. Uso del código

### 5.1 Requisitos

Instalar dependencias principales:

``` bash
pip install pandas numpy scikit-learn pyarrow networkx matplotlib seaborn
```

### 5.2 Ejecución del feature engineering

Ejemplo de ejecución completa:

``` bash
python models/features.py --run-id 20251214T195904Z
```

El script:

1. Carga métricas, logs y trazas.
2. Alinea las fuentes temporalmente.
3. Calcula features por ventana.
4. Genera datasets listos para modelado.

### 5.3 Salidas generadas

Los resultados se almacenan en formatos reproducibles:

- Parquet/CSV de features
- Datasets alineados por timestamp
- Listos para ser utilizados por:
  - Modelos de detección
  - Modelos de predicción
  - Módulo de RCA

## 6. Resultados de la Fase 3

Al finalizar la Fase 3 se dispone de:

- Comprensión profunda de métricas, logs y trazas.
- Pipelines reproducibles de feature engineering.
- Datasets etiquetados y alineados temporalmente.
- Base sólida para entrenamiento y evaluación de modelos AIOps.

## 7. Conclusión

La Fase 3 transforma datos crudos de observabilidad en representaciones estructuradas y semánticas, alineadas con prácticas modernas de ingeniería de datos para AIOps. Los resultados obtenidos permiten avanzar con garantías hacia las fases de detección, predicción y análisis de causa raíz, manteniendo reproducibilidad, trazabilidad y rigor experimental.
