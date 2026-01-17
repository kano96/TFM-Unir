# Fase 7 — Visualización y Alerting

En esta fase se completó la capa de observabilidad orientada a operación, incorporando tableros (dashboards) en Grafana y un sistema de alertas automático. La validación se ejecutó en entorno local mediante Docker Compose, priorizando la reproducibilidad y el control de costos. El resultado fue un flujo operativo consistente en: instrumentación → recolección (Prometheus/Loki) → visualización (Grafana) → alerting y notificación.

---

## 7.1 Implementación de dashboards en Grafana

Se integró Grafana como plataforma de visualización central, con dashboards provisionados desde archivos JSON dentro del repositorio. Esta aproximación permitió que los tableros se desplegaran de forma determinista en cada ejecución del entorno experimental, sin depender de configuración manual en la interfaz.

### 7.1.1 Estructura y provisión de dashboards

Los dashboards se almacenaron en el directorio del repositorio y se montaron en el contenedor de Grafana mediante volúmenes. El provisioning se configuró usando un provider de tipo `file`, apuntando al path interno donde Grafana detecta los JSON.

**Archivo de provisioning de dashboards:**

- `observability/grafana/provisioning/dashboards/dashboards.yml`

**Directorio de dashboards:**

- `observability/grafana/dashboards/`

### 7.1.2 Dashboards implementados

Los tableros implementados cubrieron la trazabilidad operacional de los componentes AIOps y de los microservicios simulados, incluyendo:

- **Overview (service health):** visión general del estado por servicio y señales principales.
- **Per-service metrics:** métricas clave de cada microservicio (p. ej., latencia, errores, throughput).
- **Anomaly stream:** series temporales y eventos asociados a detección de anomalías.
- **Prediction timeline:** evolución temporal de probabilidad y banderas de predicción de incidentes.
- **RCA suggestions:** ranking y componentes del análisis de causa raíz, junto con trazas y/o logs de soporte.

> Recomendación de evidencias (capturas): incluir una captura por dashboard mostrando (i) el selector de servicio, (ii) al menos un panel con series renderizadas y (iii) la granularidad temporal empleada (por ejemplo “últimas 6h”).

---

## 7.2 Integración de fuentes de datos (Prometheus, Loki y Jaeger)

Se utilizaron tres fuentes de datos (datasources) para cubrir las tres señales de observabilidad:

- **Prometheus:** métricas numéricas (series temporales).
- **Loki:** logs centralizados.
- **Jaeger:** trazas distribuidas.

### 7.2.1 Provisioning de datasources con UID fijo

Para evitar inconsistencias entre reinicios de Grafana y garantizar que los dashboards apuntaran siempre a las mismas fuentes, se configuraron UIDs fijos. Esto permitió que los dashboards JSON referenciaran los datasources por UID, evitando fallos por recreación del contenedor o cambios internos de Grafana.

**Archivo de provisioning de datasources:**

- `observability/grafana/provisioning/datasources/datasources.yml`

> Recomendación de evidencias (capturas): incluir captura del archivo `datasources.yml` (sin información sensible) y de la pantalla “Connections → Data sources” mostrando Prometheus/Loki/Jaeger en estado `OK`.

---

## 7.3 Instrumentación y exposición de métricas (Predictor y RCA)

Como parte del alineamiento con los requisitos de observabilidad, los servicios AIOps (Predictor y RCA) expusieron métricas en formato Prometheus mediante el endpoint `/metrics`. Estas métricas se utilizaron tanto para visualización como para reglas de alerting.

### 7.3.1 Predictor: métricas de predicción

El servicio Predictor publicó métricas asociadas a la salida del modelo supervisado:

- **Probabilidad de incidente:** `aiops_incident_probability{service="..."}`  
- **Bandera de incidente (umbral):** `aiops_incident_flag{service="..."}`

Estas series permitieron representar en el dashboard de predicción tanto la tendencia de probabilidad como el evento de cruce del umbral.

> Recomendación de evidencias (capturas): panel con la probabilidad y el umbral (línea horizontal), más un panel binario (flag) donde se observe el cambio a `1`.

### 7.3.2 RCA: métricas de ranking y componentes

El servicio RCA publicó métricas para materializar el resultado del ranking en forma de series de Prometheus (top-k), permitiendo su visualización directa en Grafana:

- `aiops_rca_candidate_score{incident_id,service,rank}`
- `aiops_rca_candidate_magnitude{incident_id,service,rank}`
- `aiops_rca_candidate_centrality{incident_id,service,rank}`
- `aiops_rca_candidate_earliness{incident_id,service,rank}`

Adicionalmente, se publicaron métricas de operación del servicio (conteo de requests, latencia, errores), utilizadas para observabilidad del propio componente.

> Nota operacional: al tratarse de métricas tipo `Gauge` derivadas del resultado del algoritmo, estas series se materializaron al ejecutar el endpoint `/rca` al menos una vez para un `incident_id` válido.

---

## 7.4 Generación de tráfico sintético para poblar series

Durante la validación se observó que, al generar valores completamente aleatorios para el Predictor, el modelo podía saturar predicciones cercanas a 0 o 1 (por estar fuera de distribución). Para obtener curvas más informativas en la visualización, se utilizó una estrategia de generación sintética basada en:

1. búsqueda automática de un vector “semilla” por servicio que produjera probabilidades en un rango objetivo;  
2. generación de variaciones leves (jitter) alrededor de dicha semilla.

Esto permitió poblar series temporales con valores “naturales” (no saturados), facilitando el análisis visual y la demostración del funcionamiento del dashboard de predicción.

> Recomendación de evidencias (capturas): consola del generador mostrando probabilidades variables por servicio, y el dashboard “Prediction timeline” con series activas.

---

## 7.5 Sistema de alerting en Grafana

Se implementó un sistema de alertas basado en PromQL utilizando Grafana Alerting. Las reglas cubrieron tanto condiciones clásicas por umbral (métricas de servicio) como condiciones basadas en modelos (salidas del Predictor). Las alertas se evaluaron de forma periódica con una ventana de retención suficiente para evitar falsos positivos por picos instantáneos.

### 7.5.1 Reglas de alerta definidas

Las reglas se organizaron en grupos para mantener consistencia operativa y facilitar su mantenimiento.

| Regla | Tipo | Expresión (PromQL) | Severidad |
| ------ | ------ | --------------------- | ---------- |
| Predictor Incident Flag | Modelo | `max_over_time(aiops_incident_flag[1m]) == 1` | critical |
| Predictor High Probability | Modelo | `max_over_time(aiops_incident_probability[1m]) > 0.75` | warning |
| High Error Ratio | Umbral | `avg_over_time(error_ratio[1m]) > 0.05` | warning |
| High P95 Latency | Umbral | `avg_over_time(p95_latency[1m]) > 400` | warning |

> Recomendación de evidencias (capturas): pantalla de “Alert rules” con las reglas creadas; una instancia en estado `Firing` mostrando labels/annotations; y el panel asociado donde se aprecie la métrica cruzando el umbral.

### 7.5.2 Provisioning de reglas de alerta (reproducibilidad)

Para evitar recreación manual de alertas tras reinicios y garantizar reproducibilidad del entorno, las reglas se provisionaron por archivo YAML. Esto permitió que el conjunto de reglas se desplegara automáticamente al iniciar Grafana, asegurando comportamiento consistente entre ejecuciones.

**Archivo de provisioning de reglas:**

- `observability/grafana/provisioning/alerting/alert-rules.yml`

> Observación técnica: en el provisioning de Grafana Alerting, cada query debe declarar un `relativeTimeRange` válido; de lo contrario Grafana puede fallar al iniciar con errores de rango temporal inválido.

---

## 7.6 Canal de notificación y verificación con MailHog

Dado que en el entorno local no se dispuso de un servidor SMTP real, se integró MailHog como servidor SMTP de pruebas. Esta decisión permitió validar el flujo completo de alerting y notificación (end-to-end) sin dependencia de credenciales externas y manteniendo reproducibilidad.

### 7.6.1 Configuración del servidor SMTP de pruebas

MailHog se expuso como servicio dentro del mismo `docker-compose` y se configuró en Grafana como backend SMTP. Las notificaciones se verificaron mediante la interfaz web de MailHog.

**Interfaz de MailHog:**

- `http://localhost:8025`

> Recomendación de evidencias (capturas): (i) configuración del contact point en Grafana, (ii) regla en estado `Firing`, (iii) correo recibido en MailHog con el contenido de la alerta.

---

## 7.7 Entregables de la fase

Los artefactos generados en esta fase fueron:

- Dashboards Grafana en formato JSON, provisionados por archivo.
- Datasources provisionados con UID fijo para evitar inconsistencias en reinicios.
- Reglas de alerting definidas y provisionadas por YAML.
- Canal de notificación validado con SMTP de pruebas (MailHog).
- Scripts de generación de tráfico sintético para poblar series y facilitar la demostración operacional.

---
