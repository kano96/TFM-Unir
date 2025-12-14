# 🧠 Plataforma AIOps para Detección y Predicción de Incidentes en Entornos de Microservicios

**Autor:** Kevin Torres  
**Máster:** Ingeniería de Software y Sistemas Informáticos — UNIR  
**Año:** 2025

---

## Descripción general

Este repositorio contiene la implementación de una **plataforma experimental AIOps (Artificial Intelligence for IT Operations)** orientada a la **detección y predicción de incidentes** en arquitecturas basadas en microservicios. La plataforma integra recolección de métricas, logs y trazas, modelos de aprendizaje automático para detección y predicción, mecanismos de correlación y diagnóstico, y visualización en tiempo real con alertas automáticas. Está diseñada con tecnologías *open-source* y su despliegue objetivo es AWS (EKS, S3, ECR).

---

## Objetivo del proyecto

Diseñar e implementar una plataforma de detección y predicción de incidentes en entornos de microservicios mediante técnicas de AIOps, con el fin de **reducir los tiempos de diagnóstico y anticipar fallos potenciales**, mejorando la disponibilidad y resiliencia de los sistemas. El trabajo demostrará la efectividad de algoritmos de aprendizaje automático para la identificación proactiva de anomalías y su visualización en tiempo real.

### Objetivos específicos

- Analizar limitaciones de enfoques tradicionales de monitoreo en entornos cloud-native.  
- Identificar técnicas de IA/ML aplicables a detección y predicción de anomalías.  
- Desarrollar una plataforma experimental basada en AIOps con herramientas open-source y servicios en la nube.  
- Evaluar la efectividad mediante métricas (Precision, Recall, F1-score, MTTD, MTTR).

---

## Arquitectura general (resumida)

**Componentes principales:**

- **Microservicios simulados**: generan métricas, logs y trazas (OpenTelemetry).  
- **Ingesta**: Prometheus (métricas), Loki/Elasticsearch (logs), Jaeger (trazas), Kafka (opcional, canal de eventos).  
- **Procesamiento/ML**: servicios de detección (Isolation Forest, Autoencoders), predicción (LSTM/Prophet), correlación (clustering, NLP), empacados como APIs (FastAPI).  
- **Visualización**: Grafana/Kibana dashboards.  
- **Alerting**: Prometheus Alertmanager / Grafana alerting → AWS SNS / Slack.  
- **Infraestructura**: Docker, Kubernetes (EKS AWS), Terraform para IAC, S3 para datasets y modelos.

---

## Requisitos previos (desarrollo local)

- Git  
- Docker & Docker Compose  
- Python 3.9+ (virtualenv recomendado)  
- Node.js (opcional, para microservicios ejemplo)  
- AWS CLI (para despliegue en nube)  
- Terraform (si vas a desplegar infra en AWS)  

---

## Despliegue local (modo rápido)

> **Objetivo:** levantar un entorno local mínimo con métricas, logs y trazas.

1. Clonar repositorio:

```bash
git clone https://github.com/usuario/aiops-platform.git
cd aiops-platform
```

2. Copiar variables de ejemplo:

```bash
cp .env.example .env
```

3. Levantar servicios con Docker Compose (incluye Prometheus, Grafana, Jaeger, Loki y microservicios ejemplo):

```bash
docker compose up -d
```

4. Accesos:

- Grafana: <http://localhost:3000>
 (user: admin / pass: admin por defecto)
- Prometheus: <http://localhost:9090>
- Jaeger: <http://localhost:16686>
- API (FastAPI): <http://localhost:8000/docs>

```bash
python scripts/run_experiment.py --duration 300 --rps 2 --services user,auth,orders --weighted
```

## Licencia

MIT License — copia y modifica libremente citando la fuente.
