# Fase 6 — Correlación de eventos y RCA (Root Cause Analysis)

## 6.1 Objetivo de la fase

El objetivo de la Fase 6 es **agrupar alertas en incidentes coherentes** y, a partir de estos incidentes, **priorizar candidatos de causa raíz** (*root cause candidates*) utilizando señales combinadas de métricas/logs y la estructura de dependencias entre servicios (derivada de trazas). Este paso busca reducir el esfuerzo manual de diagnóstico y producir una salida accionable para análisis posterior o visualización.

---

## 6.2 Entrada de datos

La fase utiliza dos artefactos principales generados a partir de fases previas:

1. **Alertas/Incidentes**
   - `data/processed/incidents_20251215T013009Z.parquet`
   - Contiene incidentes como agrupaciones temporales de alertas, con información por servicio (p. ej., magnitud de anomalía agregada).

2. **Grafo de dependencias**
   - `data/processed/service_graph_20251215T013009Z.json`
   - Representa dependencias entre servicios inferidas desde trazas (llamadas entre microservicios).
   - Se emplea para estimar una medida de centralidad (relevancia estructural) por servicio.

---

## 6.3 Metodología

### 6.3.1 Agrupación de alertas (Alert Grouping)

Las alertas se agrupan en incidentes usando un enfoque de *clustering* (p. ej., DBSCAN), donde cada alerta se representa como un vector que combina:

- **Embeddings TF-IDF** de logs agregados por ventana.
- **Componente temporal** (peso `time_weight`) para favorecer agrupaciones compactas en el tiempo.

De esta forma, alertas cercanas en tiempo y con contenido semántico similar tienden a pertenecer al mismo incidente.

### 6.3.2 RCA basado en grafo y magnitud de anomalía

Para cada incidente, se calcula un ranking de servicios candidatos a causa raíz combinando:

- **Magnitud**: severidad agregada de señales anómalas del servicio durante el incidente.
- **Centralidad**: importancia del servicio en el grafo de dependencias (servicios “centrales” tienden a propagar fallos a múltiples dependientes).
- **Earliness (anticipación)**: prioriza servicios cuya anomalía aparece antes dentro del incidente (cuando aplica).

La salida se expresa como una tabla de candidatos ordenada por `score`, junto con componentes interpretables (`magnitude`, `centrality`, `earliness`).

---

## 6.4 Ejecución y resultados obtenidos

Comando ejecutado:

```bash
python models/rca/rca.py \
  --incidents data/processed/incidents_20251215T013009Z.parquet \
  --graph data/processed/service_graph_20251215T013009Z.json \
  --incident-id 20251215T013009Z_inc0 \
  --out-dir models/rca/out \
  --top-k 3
```

Salida:

- Artefacto generado:

  - models/rca/out/rca_20251215T013009Z_inc0.json

- Ranking de candidatos (Top-3):

| service | score | magnitude | centrality | earliness |
| ---------- | ---------- | -------- | ----- | ----- |
| auth | 1.203368 | 0.203368 | 1.000000 | 1.0 |
| user | 1.116977 | 0.121859 | 0.991863 | 1.0 |
| orders | 0.813115 | 0.105736 | 0.512299 | 1.0 |

### Interpretación

- auth es el candidato más probable: combina la mayor magnitud (0.203368) con la centralidad máxima (1.0).

- user aparece como segundo candidato: aunque su magnitud es menor, mantiene una centralidad muy alta (~0.992).

- orders queda tercero: magnitud comparable pero centralidad significativamente más baja (~0.512), lo cual reduce su prioridad como origen del incidente.

## 6.5 Conclusiones de la fase

La Fase 6 valida que el sistema puede:

1. Agrupar alertas en incidentes de forma automática con criterios temporales y semánticos.

2. Proponer candidatos de causa raíz con una señal interpretable y justificable (magnitud/centralidad/temporalidad).

3. Generar artefactos persistentes (JSON) para trazabilidad y reproducibilidad.

Estos resultados constituyen una base sólida para integrar un módulo de RCA como microservicio y alimentar una interfaz de visualización o un flujo de respuesta automatizada.
