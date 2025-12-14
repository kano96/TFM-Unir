# Fase 1 — Simulación y configuración experimental

## 1. Objetivo

El objetivo de la Fase 1 es construir un **entorno experimental controlado y reproducible** capaz de generar datos operacionales realistas — métricas, logs y trazas — tanto en condiciones normales como bajo la presencia de fallos. Este entorno constituye la base del proyecto y permite disponer de datasets etiquetados necesarios para el entrenamiento, validación y evaluación de los modelos de detección de anomalías, predicción y análisis de causa raíz (RCA).

En particular, esta fase persigue los siguientes objetivos:

- Simular múltiples microservicios independientes.
- Instrumentar dichos servicios con herramientas estándar de observabilidad cloud-native.
- Inyectar fallos controlados y reproducibles.
- Generar y persistir datasets alineados temporalmente y etiquetados.

---

## 2. Arquitectura de microservicios simulados

Se implementaron tres servicios simulados utilizando **FastAPI**, cada uno desplegado como un contenedor independiente:

- `simulator-user`
- `simulator-auth`
- `simulator-orders`

Todos los simuladores comparten la misma base de código, diferenciándose mediante la variable de entorno `SERVICE_NAME`. Esta decisión permite mantener consistencia en el comportamiento y, al mismo tiempo, generar señales observables diferenciadas por servicio.

Cada simulador expone los siguientes endpoints:

- `GET /health`: verificación de estado del servicio.
- `GET /simulate`: generación de tráfico normal o afectado por fallos.
- `GET /metrics`: exposición de métricas en formato Prometheus.
- `POST /fault/*`: inyección de fallos controlados.

---

## 3. Instrumentación de observabilidad

### 3.1 Métricas (Prometheus)

Cada servicio expone métricas mediante el cliente oficial de Prometheus, incluyendo:

- Contadores de peticiones.
- Contadores de errores.
- Histogramas de latencia.
- Indicadores del estado de fallos activos.

Las métricas se etiquetan por `service` y `endpoint`, lo que permite analizar el comportamiento individual de cada microservicio. Prometheus recolecta estas métricas periódicamente mediante scraping.

---

### 3.2 Logs (Loki y Promtail)

Los servicios generan **logs estructurados en formato JSON** que se envían a la salida estándar. Cada evento de log contiene:

- Timestamp en UTC.
- Identificador del servicio.
- Nivel de severidad.
- Mensaje semántico.
- Campos contextuales (latencia, tipo de fallo, endpoint, etc.).

Promtail se encarga de recolectar los logs de los contenedores Docker y enviarlos a Loki, que los almacena indexando únicamente metadatos. Esta estrategia reduce el coste de almacenamiento y facilita la consulta posterior.

---

### 3.3 Trazas (OpenTelemetry y Jaeger)

Los simuladores están instrumentados con **OpenTelemetry**, lo que permite generar trazas automáticamente para cada petición HTTP. Las trazas se exportan mediante el protocolo OTLP hacia Jaeger.

El uso de trazas permite analizar:

- Latencias end-to-end.
- Comportamiento de los servicios durante la inyección de fallos.
- Alineación temporal entre métricas, logs y eventos anómalos.

---

## 4. Estrategia de inyección de fallos

La inyección de fallos se realiza mediante el script `inject_faults.py`, que orquesta escenarios de fallo durante la ejecución de un experimento. Los tipos de fallos simulados incluyen:

- Incremento artificial de latencia.
- Aumento de la tasa de errores.
- Saturación de CPU.
- Fugas de memoria controladas.

Cada fallo se caracteriza por:

- Servicio afectado.
- Tipo de fallo.
- Timestamp de inicio.
- Timestamp de fin.
- Identificador único del experimento (`run_id`).

Toda esta información se almacena en el archivo `data/labels/fault_windows.csv`, que actúa como **fuente de verdad (ground truth)** para el etiquetado de los datos.

---

## 5. Generación de tráfico

Para garantizar la generación continua de señales observables, se implementó el script `traffic_generator.py`, encargado de enviar peticiones HTTP a los simuladores durante el experimento.

Las características del tráfico incluyen:

- Tasa de peticiones configurable (RPS).
- Distribución uniforme o ponderada entre servicios.
- Duración total del experimento configurable.

Este enfoque asegura que los efectos de los fallos sean observables bajo carga y que los experimentos puedan reproducirse de forma consistente.

---

## 6. Orquestación del experimento

El flujo completo de la Fase 1 se coordina mediante el script `run_experiment.py`, que ejecuta secuencialmente:

1. Generación de tráfico.
2. Inyección de fallos.
3. Exportación de métricas.
4. Exportación de logs.

De este modo, todos los artefactos generados quedan alineados bajo un mismo `run_id`.

Ejemplo de ejecución:

```bash
python scripts/run_experiment.py \
  --duration 300 \
  --rps 2 \
  --services user,auth,orders \
  --weighted
```

## 7. Exportación y persistencia de datos

### 7.1 Exportación de métricas

Las métricas se extraen desde Prometheus utilizando ventanas temporales derivadas de los fallos inyectados. Los datasets resultantes se almacenan en:

data/raw/metrics/

Estos datos se emplearán posteriormente para la extracción de características y el entrenamiento de modelos.

---

### 7.2 Exportación de logs

Los logs se exportan desde Loki utilizando consultas alineadas con los intervalos de fallo. Los resultados incluyen archivos **JSONL** con eventos de log individuales y archivos **CSV** de resumen por servicio y nivel de severidad.

Los datos se almacenan en:

data/raw/logs/

---

## 8. Resultados de la Fase 1

Al finalizar la Fase 1 se dispone de:

- Ventanas de fallos etiquetadas.
- Datasets de métricas alineados temporalmente.
- Datasets de logs estructurados.
- Trazas accesibles desde Jaeger para análisis exploratorio.
- Un entorno Docker Compose completamente reproducible.

Estos resultados constituyen la base para las fases posteriores de análisis exploratorio, modelado y evaluación.

---

## 9. Resumen

La Fase 1 establece un entorno experimental realista y alineado con prácticas modernas de observabilidad en sistemas *cloud-native*. La combinación de simulación de microservicios, inyección de fallos, instrumentación completa y exportación automatizada de datos proporciona una base sólida para el desarrollo y validación de la plataforma AIOps propuesta en este trabajo.
