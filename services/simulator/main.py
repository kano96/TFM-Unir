from fastapi import FastAPI
import random
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI(title="Simulator Service")

# Configuración de logs
logging.basicConfig(level=logging.INFO, filename="service.log",
                    format="%(asctime)s %(message)s")

# Métricas Prometheus
REQUEST_COUNT = Counter('app_requests_total', 'Total de peticiones')
ERROR_COUNT = Counter('app_errors_total', 'Errores simulados')
LATENCY = Histogram('app_request_latency_seconds', 'Latencia de solicitudes')


@app.get("/simulate")
def simulate_request():
    REQUEST_COUNT.inc()
    latency = random.uniform(0.05, 0.5)
    time.sleep(latency)
    if random.random() < 0.1:
        ERROR_COUNT.inc()
        logging.error("Error simulado en procesamiento de petición.")
        return {"status": "error", "latency": latency}
    else:
        logging.info("Petición exitosa procesada correctamente.")
        return {"status": "ok", "latency": latency}


@app.get("/metrics")
def metrics():
    return generate_latest()
