# Esquema de eventos (Fase 2)

Este documento define el formato estándar de los eventos que consumirán los módulos de ML (feature extractor, detector y predictor). El objetivo es unificar métricas, logs y trazas en una estructura común.

## 1. Evento de métrica (MetricEvent)

Representa una muestra puntual (o agregada por ventana) proveniente de Prometheus.

```json
{
  "event_type": "metric",
  "timestamp": "2025-12-14T20:13:19.664Z",
  "service": "user",
  "metric_name": "app_request_latency_seconds_sum",
  "metric_value": 12.345,
  "labels": {
    "endpoint": "/simulate",
    "instance": "simulator-user:8000",
    "job": "simulators"
  }
}
```

## 2. Evento de log (LogEvent)

Representa una línea de log estructurado exportada desde Loki.

```json
{
  "event_type": "log",
  "timestamp": "2025-12-14T20:13:19.664Z",
  "service": "user",
  "level": "ERROR",
  "message": "request_failed",
  "labels": {
    "container": "/repo-simulator-user-1"
  },
  "raw": "{\"timestamp\": \"...\", \"service\": \"user\", ...}"
}
```

## 3. Evento de traza (TraceEvent)

Representa un span (o resumen de span) proveniente de OpenTelemetry/Jaeger.

```json
{
  "event_type": "trace",
  "timestamp": "2025-12-14T20:13:19.664Z",
  "service": "user",
  "trace_id": "a1b2c3...",
  "span_id": "d4e5f6...",
  "operation": "GET /simulate",
  "duration_ms": 120.5,
  "attributes": {
    "http.method": "GET",
    "http.route": "/simulate",
    "http.status_code": 200
  }
}
```

## 4. Notas de uso para ML

- Para detección: se usarán ventanas temporales (ej. 60s) agregando métricas y conteos de logs.

- Para predicción: se trabajará con series temporales agregadas por servicio (latencia, errores por minuto, etc.).

- service debe ser consistente entre todas las fuentes.

---
